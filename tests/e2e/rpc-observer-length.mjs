#!/usr/bin/env node
import { ModelServer, assert, createWorkspace, launchPi, omSettings, prepareWorkspace, sendSse, stopPi, waitFor } from "./harness.mjs";

const OBSERVATION = "The observer recovered after a provider output-length stop.";
const started = Date.now();
const log = (text) => console.log(`[observer-length-e2e +${((Date.now() - started) / 1000).toFixed(1)}s] ${text}`);
const state = { main: 0, observer: 0, firstLength: false, freshRetry: false, recorded: false, done: false };

const server = new ModelServer(async (request, res) => {
	if (request.role === "main") {
		state.main++;
		return sendSse(res, { text: "MAIN_SOURCE_COMPLETE", outputTokens: 100 });
	}
	assert(request.role === "observer", `Unexpected request role: ${request.role}`);
	state.observer++;
	const messages = request.body.messages ?? [];
	const toolMessages = messages.filter((message) => message.role === "tool");

	// The extension's 160k observer allowance is an upper bound. Pi's provider
	// request must honor this provider's smaller advertised 16k output maximum.
	const requestedMax = request.body.max_tokens ?? request.body.max_completion_tokens;
	assert(requestedMax === 16_000, `Expected observer max output 16000, got ${JSON.stringify(requestedMax)}`);

	if (state.observer === 1) {
		state.firstLength = true;
		return sendSse(res, { reasoning: "A long unfinished analysis that never called a tool.", finishReason: "length", outputTokens: 16_000 });
	}

	if (!state.recorded) {
		assert(/previous response reached the provider output limit/i.test(request.text), "Observer did not issue the explicit fresh length recovery prompt");
		assert(messages.some((message) => message.role === "assistant" && typeof message.content === "string" && message.content.includes("[Incomplete analysis from the preceding truncated response]\nA long unfinished analysis")), "Observer recovery did not replay partial thinking as portable assistant text");
		state.freshRetry = true;
		const sourceId = request.text.match(/Source entry id:\s*([\w-]+)/)?.[1];
		assert(sourceId, "Observer recovery prompt did not preserve the source entry id");
		state.recorded = true;
		return sendSse(res, { tool: { id: "observer-record-after-length", name: "record_observations", arguments: { observations: [{ timestamp: "2026-08-22 08:00", content: OBSERVATION, relevance: "high", retention: "contextual", sourceEntryIds: [sourceId] }] } } });
	}

	assert(toolMessages.length > 0, "Observer did not receive its record_observations result before completion");
	state.done = true;
	return sendSse(res, { tool: { id: "observer-done-after-length", name: "done", arguments: {} } });
});

console.log("RPC observer-length E2E: provider length stop, advertised max-token clipping, continuation retry, tool recovery, and coverage commit");
const workspace = await createWorkspace("pi-observer-length-e2e-");
let pi;
try {
	const port = await server.start();
	await prepareWorkspace(
		workspace,
		port,
		omSettings({ contemplatorEnabled: false, reviewerEnabled: false, summarizerEnabled: false, compactionObserverEnabled: false }),
		[{ id: "mock-model", contextWindow: 256_000, maxTokens: 16_000 }],
	);
	pi = await launchPi(workspace);
	const before = pi.rpc.events.length;
	await pi.rpc.command({ type: "prompt", message: "Create source material for the observer length-recovery test." });
	await pi.rpc.waitSettled(before);
	await waitFor(async () => (await pi.rpc.entries())
		.filter((entry) => entry.customType === "om.observations.recorded")
		.flatMap((entry) => entry.data?.observations ?? [])
		.some((observation) => observation.content === OBSERVATION), "observer recovery observation commit", 30_000);

	assert(state.firstLength, "Provider length response was not exercised");
	assert(state.freshRetry, "Observer did not make a continuation recovery request");
	assert(state.recorded && state.done, "Observer did not record and finish after recovery");
	assert(state.observer >= 3, `Expected at least three observer server requests (length, record, done), got ${state.observer}`);
	assert(!pi.rpc.events.some((event) => event.type === "extension_error"), "Extension error during observer length recovery");
	await stopPi(pi); pi = undefined;
	log("PASS provider length ended the first agent loop; wrapper preserved the partial response, continued, and committed an observation");
} catch (error) {
	console.error(`State: ${JSON.stringify(state)}; requests: ${server.requests.map((request) => `${request.role}:${request.body.max_tokens ?? request.body.max_completion_tokens}`).join(",")}`);
	if (pi) console.error(`stderr: ${pi.rpc.stderr}`);
	throw error;
} finally {
	if (pi?.child.exitCode === null) pi.child.kill("SIGKILL");
	await server.close().catch(() => {});
	await workspace.cleanup();
}
