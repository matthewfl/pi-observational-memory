#!/usr/bin/env node
import { ModelServer, assert, createWorkspace, launchPi, omSettings, prepareWorkspace, sendSse, stopPi, textOf, waitFor } from "./harness.mjs";

const MARKER = "E2E_HUGE_MEMORY_EDGE";
const COLLISION_CONTENT = "E2E deterministic duplicate observation collision";
const started = Date.now();
const log = (text) => console.log(`[memory-edges-e2e +${((Date.now() - started) / 1000).toFixed(1)}s] ${text}`);
const state = { main: 0, observerInitials: 0, emptyProseAttempted: false, reminderDoneSeen: false, lengthAttempted: false, lengthRetries: 0, invalidAttempted: false, sawOmission: false, maxObserverText: 0, maxObserverChunkText: 0, recallCollision: false };

const server = new ModelServer(async (request, res) => {
	const toolMessages = (request.body.messages ?? []).filter((message) => message.role === "tool");
	if (request.role === "observer") {
		if (/previous response reached the provider output limit/i.test(request.text)) {
			state.lengthRetries++;
			return sendSse(res, { text: "still unproductive observer reasoning", finishReason: "length", outputTokens: 128_000 });
		}
		if (/Observations recorded so far:\s*0|call done now/i.test(request.text)) {
			assert(state.emptyProseAttempted, "Observer reminder arrived before the deliberate prose-only stop");
			assert(request.body.tool_choice !== "required", "Observer reminder retry unexpectedly required a tool call");
			state.reminderDoneSeen = true;
			return sendSse(res, { tool: { id: "confirm-empty-coverage", name: "done", arguments: {} } });
		}
		if (toolMessages.length) {
			const result = JSON.stringify(toolMessages.at(-1));
			if (state.invalidAttempted && /invalid|unknown|source|not found|rejected/i.test(result)) {
				const sourceId = request.text.match(/Source entry id:\s*([\w-]+)/)?.[1];
				assert(sourceId, "Observer retry lacked its original valid source id");
				return sendSse(res, { tool: { id: "valid-retry", name: "record_observations", arguments: { observations: [{ timestamp: "2026-08-15 03:00", content: COLLISION_CONTENT, relevance: "high", retention: "contextual", sourceEntryIds: [sourceId] }] } } });
			}
			return sendSse(res, { text: "observer batch complete" });
		}
		state.observerInitials++;
		state.maxObserverText = Math.max(state.maxObserverText, request.text.length);
		const renderedPrompt = (request.body.messages ?? []).map(textOf).join("\n");
		const chunkMarker = "NEW CONVERSATION CHUNK:";
		const chunkStart = renderedPrompt.lastIndexOf(chunkMarker);
		const renderedChunk = chunkStart >= 0 ? renderedPrompt.slice(chunkStart + chunkMarker.length) : "";
		state.maxObserverChunkText = Math.max(state.maxObserverChunkText, renderedChunk.length);
		state.sawOmission ||= /omitted|truncat|bounded/i.test(renderedChunk);
		if (!state.emptyProseAttempted) {
			state.emptyProseAttempted = true;
			return sendSse(res, { text: "This source has no useful durable information." });
		}
		if (state.reminderDoneSeen && !state.lengthAttempted) {
			state.lengthAttempted = true;
			return sendSse(res, { text: "unproductive observer reasoning", finishReason: "length", outputTokens: 128_000 });
		}
		const sourceId = request.text.match(/Source entry id:\s*([\w-]+)/)?.[1];
		assert(sourceId, "Observer prompt lacked a source id");
		if (!state.invalidAttempted) {
			state.invalidAttempted = true;
			return sendSse(res, { tool: { id: "invalid-source-first", name: "record_observations", arguments: { observations: [{ timestamp: "2026-08-15 03:00", content: "MUST_NOT_PERSIST_INVALID", relevance: "high", retention: "contextual", sourceEntryIds: ["definitely-missing-entry"] }] } } });
		}
		return sendSse(res, { tool: { id: `valid-${state.observerInitials}`, name: "record_observations", arguments: { observations: [{ timestamp: "2026-08-15 03:00", content: COLLISION_CONTENT, relevance: "high", retention: "contextual", sourceEntryIds: [sourceId] }] } } });
	}
	assert(request.role === "main", `Unexpected role in memory-edge scenario: ${request.role}`);
	state.main++;
	const messages = request.body.messages ?? [];
	const latestUserIndex = messages.findLastIndex((message) => message.role === "user");
	const mainToolMessages = messages.slice(latestUserIndex + 1).filter((message) => message.role === "tool");
	const toolText = mainToolMessages.map((message) => JSON.stringify(message)).join("\n");
	if (request.text.includes("Use search_memories and recall")) {
		if (mainToolMessages.length === 0) return sendSse(res, { tool: { id: "search-collision", name: "search_memories", arguments: { query: COLLISION_CONTENT, limit: 10 } } });
		if (mainToolMessages.length === 1) {
			const id = toolText.match(/\[([a-f0-9]{12})\]/)?.[1];
			assert(id, `Collision search returned no id: ${toolText}`);
			return sendSse(res, { tool: { id: "recall-collision", name: "recall", arguments: { id } } });
		}
		state.recallCollision = /matched (?:multiple observations|more than one durable record)|"collision":true/.test(toolText) && toolText.includes(COLLISION_CONTENT);
		if (!state.recallCollision) console.error("Recall tool transcript:", toolText);
		assert(state.recallCollision, `Recall did not safely return all colliding observation sources: ${toolText}`);
		return sendSse(res, { text: "MEMORY_EDGE_COMPLETE" });
	}
	if (request.text.includes("Create an independent second source")) {
		if (mainToolMessages.length === 0) return sendSse(res, { tool: { id: "second-source", name: "bash", arguments: { command: "echo second-collision-source" } }, outputTokens: 100 });
		return sendSse(res, { text: "SECOND_SOURCE_COMPLETE", outputTokens: 100 });
	}
	if (mainToolMessages.length === 0) return sendSse(res, { tool: { id: "huge-output", name: "bash", arguments: { command: `node -e "process.stdout.write('${MARKER}\\n' + 'x'.repeat(30000))"` } }, outputTokens: 200 });
	return sendSse(res, { text: "HUGE_SOURCE_COMPLETE", outputTokens: 100 });
});

console.log("RPC memory-edge E2E: bounded backlog draining, empty/done and failed-length coverage, malformed source retry, collisions, search, and recall");
const workspace = await createWorkspace("pi-memory-edges-e2e-");
let pi;
try {
	const port = await server.start();
	await prepareWorkspace(workspace, port, omSettings({ observerChunkMaxTokens: 1_000, contemplatorEnabled: false, reviewerEnabled: false, compactionObserverEnabled: false }));
	pi = await launchPi(workspace);
	const first = pi.rpc.events.length;
	await pi.rpc.command({ type: "prompt", message: `Generate a huge source containing ${MARKER}.` });
	await pi.rpc.waitSettled(first);
	await waitFor(async () => (await pi.rpc.entries()).filter((entry) => entry.customType === "om.observations.recorded").flatMap((entry) => entry.data?.observations ?? []).some((observation) => observation.content === COLLISION_CONTENT), "valid retry after malformed source", 30_000);
	const secondSource = pi.rpc.events.length;
	await pi.rpc.command({ type: "prompt", message: "Create an independent second source for the collision test." });
	await pi.rpc.waitSettled(secondSource);
	await waitFor(async () => (await pi.rpc.entries()).filter((entry) => entry.customType === "om.observations.recorded").flatMap((entry) => entry.data?.observations ?? []).filter((observation) => observation.content === COLLISION_CONTENT).length >= 2, "two colliding valid observations from separate source entries", 30_000);
	const before = await pi.rpc.entries();
	const observationEntries = before.filter((entry) => entry.customType === "om.observations.recorded");
	const observations = observationEntries.flatMap((entry) => entry.data?.observations ?? []);
	assert(state.reminderDoneSeen, "Observer prose-only stop was not retried with a count reminder and explicit done");
	const emptyCoverage = observationEntries.filter((entry) => Array.isArray(entry.data?.observations) && entry.data.observations.length === 0 && entry.data?.coversUpToId);
	assert(emptyCoverage.length >= 2, `Expected clean-done and failed-length coverage markers, got ${emptyCoverage.length}`);
	assert(state.lengthAttempted && state.lengthRetries === 3, `Observer output-length retry/fail-forward scenario expected 3 retries, got ${state.lengthRetries}`);
	assert(!observations.some((observation) => observation.content === "MUST_NOT_PERSIST_INVALID"), "Observer persisted a record with an unknown source id");
	assert(new Set(observations.filter((observation) => observation.content === COLLISION_CONTENT).map((observation) => observation.id)).size === 1, "Deterministic duplicate content did not create the intended id collision");
	assert(state.sawOmission || state.maxObserverChunkText < 5_000, `Huge source chunk was not bounded (${state.maxObserverChunkText} chars; full request ${state.maxObserverText} chars)`);
	const second = pi.rpc.events.length;
	await pi.rpc.command({ type: "prompt", message: "Use search_memories and recall to recover all colliding sources." });
	await pi.rpc.waitSettled(second);
	assert(state.recallCollision, "Main-agent memory tools did not complete collision recovery");
	assert(!pi.rpc.events.some((event) => event.type === "extension_error"), "Extension error in memory-edge scenario");
	await stopPi(pi); pi = undefined;
	log("PASS backlog drained in bounded chunks, empty coverage used a normal done reminder, a length failure advanced safely, invalid ids retried, and colliding memories remained recallable");
} catch (error) {
	console.error(`State: ${JSON.stringify(state)}; requests: ${server.requests.map((request) => request.role).join(",")}`);
	console.error("Main request flow:", server.requests.filter((request) => request.role === "main").map((request) => {
		const messages = request.body.messages ?? [];
		const user = [...messages].reverse().find((message) => message.role === "user");
		return { user: JSON.stringify(user).slice(0, 100), tools: messages.filter((message) => message.role === "tool").length };
	}));
	if (pi) console.error(`Recorded observations: ${JSON.stringify((await pi.rpc.entries()).filter((entry) => entry.customType === "om.observations.recorded").map((entry) => entry.data))}`);
	throw error;
} finally {
	if (pi?.child.exitCode === null) pi.child.kill("SIGKILL");
	await server.close().catch(() => {});
	await workspace.cleanup();
}
