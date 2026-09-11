#!/usr/bin/env node
import { once } from "node:events";
import { createServer } from "node:http";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import process from "node:process";

const ROOT = resolve(dirname(new URL(import.meta.url).pathname), "../..");
const PI = join(ROOT, "node_modules/.bin/pi");
const EXTENSION = join(ROOT, "src/index.ts");
const PROBE_TEXT = "Memory evidence shows repeated assumptions; what direct check would distinguish the current approach from the alternative?";
const PROBE_RESPONSE_TEXT = "PROBE_RESPONSE_WITH_DIRECT_CHECK_RECORDED_BY_MAIN_AGENT";
const PROBE_FEEDBACK_OBSERVATION = "The contemplator probe reached the primary agent, which responded with a concrete direct-check acknowledgement.";
const SLEEP_OUTPUT = "sleep-tool-finished:SCENARIO_SLEEP";
const SCENARIOS = ["SCENARIO_PROBE", "SCENARIO_SLEEP", "SCENARIO_FEEDBACK", "SCENARIO_PROPOSAL", "SCENARIO_REJECT"];
const SCENARIO_NAMES = {
	SCENARIO_PROBE: "Probe delivery during a long-running primary-agent run",
	SCENARIO_SLEEP: "Probe queued during a sleeping bash call and drained immediately afterward",
	SCENARIO_FEEDBACK: "Observer feedback loop from delivered probe and response back to contemplator",
	SCENARIO_PROPOSAL: "Accepted workflow review and same-run proposal delivery",
	SCENARIO_REJECT: "Rejected software review with no primary-agent notice",
};
const TEST_PLAN = [
	"Launch an isolated real Pi RPC session with only the provider and contemplator extensions",
	"Run three real bash tool/model rounds per scenario and preserve every result in later contexts",
	"Run observer memory generation concurrently with the primary agent",
	"Persist observer memories while contemplator and reviewer workers run concurrently",
	"Deliver observer memory updates to the contemplator",
	"Delay the contemplator for two seconds while a primary provider request remains open",
	"Deliver and acknowledge a probe as an immediate same-run steer",
	"Queue a contemplator probe during a real sleeping bash call and deliver it after the tool finishes",
	"Exercise contemplator search_memories and recall calls against real persisted observer memory",
	"Observe a delivered probe together with its primary-agent response and return that memory to the contemplator",
	"Produce, persist, and deliver an accepted workflow-review proposal",
	"Persist a software review's no-proposal outcome without emitting a proposal notice",
	"Verify durable observer, contemplator, and reviewer transcripts with no extension errors",
];
const e2eStartedAt = Date.now();

function progress(message) {
	const elapsed = ((Date.now() - e2eStartedAt) / 1000).toFixed(1).padStart(5);
	console.log(`[e2e +${elapsed}s] ${message}`);
}

function assert(condition, message) {
	if (!condition) throw new Error(message);
}

function sleep(ms) {
	return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

async function waitFor(check, message, timeoutMs = 15_000) {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		const value = await check();
		if (value) return value;
		await sleep(25);
	}
	throw new Error(`Timed out: ${message}`);
}

async function waitForBackgroundQuiet(rpc, server, quietMs = 500, timeoutMs = 15_000) {
	const deadline = Date.now() + timeoutMs;
	let signature = "";
	let stableSince = Date.now();
	while (Date.now() < deadline) {
		const entries = await rpc.entries();
		const next = `${entries.length}:${server.requests.length}`;
		if (next !== signature) {
			signature = next;
			stableSince = Date.now();
		} else if (Date.now() - stableSince >= quietMs) {
			return;
		}
		await sleep(50);
	}
	throw new Error("Timed out waiting for observer/contemplator/reviewer background work to settle");
}

function messageText(message) {
	if (typeof message?.content === "string") return message.content;
	if (!Array.isArray(message?.content)) return "";
	return message.content.map((part) => typeof part?.text === "string" ? part.text : "").join("\n");
}

function latestScenario(body) {
	const text = JSON.stringify(body.messages ?? []);
	let selected;
	let selectedAt = -1;
	for (const scenario of SCENARIOS) {
		const at = text.lastIndexOf(scenario);
		if (at > selectedAt) {
			selected = scenario;
			selectedAt = at;
		}
	}
	return selected;
}

function toolNames(body) {
	return new Set((body.tools ?? []).map((tool) => tool?.function?.name).filter(Boolean));
}

function usage(outputTokens = 1) {
	return { prompt_tokens: 20, completion_tokens: outputTokens, total_tokens: 20 + outputTokens };
}

async function sendSse(res, { text, tool, outputTokens = 1, delayMs = 0 }) {
	if (delayMs > 0) await sleep(delayMs);
	res.writeHead(200, {
		"content-type": "text/event-stream",
		"cache-control": "no-cache",
		connection: "keep-alive",
	});
	const id = `chatcmpl-${Math.random().toString(16).slice(2)}`;
	const emit = (value) => res.write(`data: ${JSON.stringify(value)}\n\n`);
	const base = { id, object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model: "mock-model" };
	if (tool) {
		emit({ ...base, choices: [{ index: 0, delta: { role: "assistant", tool_calls: [{ index: 0, id: tool.id, type: "function", function: { name: tool.name, arguments: JSON.stringify(tool.arguments) } }] }, finish_reason: null }] });
		emit({ ...base, choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage: usage(outputTokens) });
	} else {
		emit({ ...base, choices: [{ index: 0, delta: { role: "assistant", content: text ?? "ok" }, finish_reason: null }] });
		emit({ ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: usage(outputTokens) });
	}
	res.end("data: [DONE]\n\n");
}

class MockModelServer {
	server = createServer((req, res) => {
		this.activeRequests++;
		this.maxConcurrentRequests = Math.max(this.maxConcurrentRequests, this.activeRequests);
		void this.handle(req, res)
			.catch((error) => {
				res.writeHead(500, { "content-type": "application/json" });
				res.end(JSON.stringify({ error: { message: String(error) } }));
			})
			.finally(() => { this.activeRequests--; });
	});
	requests = [];
	mainCounts = new Map();
	heldMain = new Map();
	interventionsSent = new Set();
	feedbackObserverSawProbeAndResponse = false;
	feedbackObservationReachedContemplator = false;
	backgroundWhileMainHeld = new Set();
	sleepToolIssued = new Set();
	contemplatorToolPhases = new Map();
	reviewerToolPhases = new Map();
	memorySearchUsed = false;
	memoryRecallUsed = false;
	recallReturnedSource = false;
	activeRequests = 0;
	maxConcurrentRequests = 0;

	async start() {
		this.server.listen(0, "127.0.0.1");
		await once(this.server, "listening");
		return this.server.address().port;
	}

	async close() {
		this.server.close();
		await once(this.server, "close");
	}

	async releaseMain(scenario) {
		const held = this.heldMain.get(scenario);
		assert(held, `No held main request for ${scenario}`);
		this.heldMain.delete(scenario);
		await sendSse(held, { text: `Background work for ${scenario} can now be incorporated.`, delayMs: 500 });
	}

	async handle(req, res) {
		let raw = "";
		for await (const chunk of req) raw += chunk;
		const body = JSON.parse(raw || "{}");
		const tools = toolNames(body);
		const scenario = latestScenario(body);
		const role = tools.has("record_observations")
			? "observer"
			: tools.has("summarize") && tools.has("fix_summary") && tools.has("done")
				? "summarizer"
				: tools.has("send_probe")
						? "contemplator"
						: tools.has("submit_workflow_proposal") || tools.has("submit_software_proposal") || tools.has("review_concluded_no_proposal")
							? "reviewer"
							: "main";
		this.requests.push({ role, scenario, body, startedAt: Date.now() });
		if (role !== "main" && scenario && this.heldMain.has(scenario)) this.backgroundWhileMainHeld.add(scenario);

		const hasToolResult = (body.messages ?? []).some((message) => message.role === "tool");
		const serializedMessages = JSON.stringify(body.messages ?? []);
		if (role === "observer") {
			if (hasToolResult) return sendSse(res, { text: "Observation coverage complete.", delayMs: 250 });
			const sourceIds = [...serializedMessages.matchAll(/Source entry id:\s*([a-zA-Z0-9_-]+)/g)].map((match) => match[1]);
			assert(sourceIds.length > 0, "Observer request did not contain a source entry id");
			const seesProbe = serializedMessages.includes(PROBE_TEXT);
			const seesProbeResponse = serializedMessages.includes(PROBE_RESPONSE_TEXT);
			const isProbeFeedback = scenario === "SCENARIO_FEEDBACK" && seesProbe && seesProbeResponse;
			if (isProbeFeedback) this.feedbackObserverSawProbeAndResponse = true;
			return sendSse(res, {
				delayMs: 250,
				tool: {
					id: `observe-${scenario}-${this.requests.length}`,
					name: "record_observations",
					arguments: {
						observations: [{
							timestamp: "2026-08-15 00:00",
							content: isProbeFeedback
								? PROBE_FEEDBACK_OBSERVATION
								: scenario === "SCENARIO_FEEDBACK" && seesProbe
									? PROBE_TEXT
									: `${scenario}: the primary agent repeatedly depends on an assumption that needs an independent check.`,
							relevance: "high",
							retention: "contextual",
							sourceEntryIds: isProbeFeedback ? sourceIds : [sourceIds[0]],
						}],
					},
				},
			});
		}

		if (role === "contemplator") {
			if (serializedMessages.includes(PROBE_FEEDBACK_OBSERVATION)) this.feedbackObservationReachedContemplator = true;
			const phase = this.contemplatorToolPhases.get(scenario);
			if (scenario === "SCENARIO_PROBE" && phase === "invalid_probe") {
				const resultText = (body.messages ?? []).filter((message) => message.role === "tool").map(messageText).join("\n");
				assert(resultText.includes("WARNING: memory deadbeef not found") && resultText.includes("call send_probe again to replace the probe"), "send_probe did not warn that the invalid parenthesized memory citation must be replaced");
				assert(resultText.includes("Probe will be delivered at the end of your turn"), "Invalid probe was not held as a replaceable end-of-turn intervention");
				this.contemplatorToolPhases.set(scenario, "searched");
				return sendSse(res, { tool: { id: "search-probe-memory", name: "search_memories", arguments: { query: "SCENARIO_PROBE independent check", limit: 5 } } });
			}
			if (scenario === "SCENARIO_PROBE" && phase === "searched") {
				const resultText = (body.messages ?? []).filter((message) => message.role === "tool").map(messageText).join("\n");
				assert(resultText.includes("Found ") && resultText.includes("SCENARIO_PROBE"), "search_memories did not return the persisted probe-scenario memory");
				const memoryId = resultText.match(/\[([a-f0-9]{12})\]/)?.[1];
				assert(memoryId, "search_memories result did not expose a recallable memory id");
				this.memorySearchUsed = true;
				this.contemplatorToolPhases.set(scenario, "recalled");
				return sendSse(res, { tool: { id: "recall-probe-memory", name: "recall", arguments: { id: memoryId } } });
			}
			if (scenario === "SCENARIO_PROBE" && phase === "recalled") {
				const resultText = (body.messages ?? []).filter((message) => message.role === "tool").map(messageText).join("\n");
				assert(resultText.includes("SCENARIO_PROBE") && /Source|User|Assistant|Tool result/.test(resultText), "recall did not return exact source context for the searched memory");
				this.memoryRecallUsed = true;
				this.recallReturnedSource = true;
				this.interventionsSent.add(scenario);
				this.contemplatorToolPhases.set(scenario, "probe_sent");
				const memoryId = serializedMessages.match(/\[([a-f0-9]{12})\]/)?.[1] ?? "000000000000";
				return sendSse(res, { tool: { id: `probe-call-${scenario}`, name: "send_probe", arguments: { question: `[${memoryId}] ${PROBE_TEXT}` } } });
			}
			if (this.interventionsSent.has(scenario)) {
				return sendSse(res, { tool: { id: `no-intervention-${scenario}-${this.requests.length}`, name: "no_intervention", arguments: {} } });
			}
			// Simulate a realistically slow contemplator. The primary agent must finish
			// three independent model/tool rounds and either enter another provider
			// request or begin the long-running sleep tool before intervention.
			await waitFor(
				() => scenario === "SCENARIO_SLEEP" ? this.sleepToolIssued.has(scenario) : this.heldMain.has(scenario),
				`${scenario} active primary work before contemplator response`,
			);
			// Keep both requests open long enough to exercise real asynchronous races
			// between the primary run and the background contemplator.
			await sleep(scenario === "SCENARIO_SLEEP" ? 1_000 : 2_000);
			const memoryId = JSON.stringify(body.messages ?? []).match(/\[([a-f0-9]{12})\]/)?.[1] ?? "000000000000";
			if (scenario === "SCENARIO_PROBE") {
				this.contemplatorToolPhases.set(scenario, "invalid_probe");
				return sendSse(res, { tool: { id: "invalid-probe-memory", name: "send_probe", arguments: { question: `(deadbeef) ${PROBE_TEXT}` } } });
			}
			this.interventionsSent.add(scenario);
			if (scenario === "SCENARIO_SLEEP" || scenario === "SCENARIO_FEEDBACK") {
				return sendSse(res, { tool: { id: `probe-call-${scenario}`, name: "send_probe", arguments: { question: `[${memoryId}] ${PROBE_TEXT}` } } });
			}
			return sendSse(res, {
				tool: {
					id: `review-call-${scenario}`,
					name: "request_review",
					arguments: {
						scope: scenario === "SCENARIO_PROPOSAL" ? "workflow" : "software",
						evidence: `[${memoryId}] ${scenario} recurring evidence`,
						concern: `${scenario}: determine whether the recurring assumption indicates a durable structural issue.`,
						review_focus: "Independently evaluate the evidence and reach one terminal outcome.",
						constraints: "Remain conceptual and preserve the user's requested behavior.",
					},
				},
			});
		}

		if (role === "reviewer") {
			const reviewerPhase = this.reviewerToolPhases.get(scenario);
			if (scenario === "SCENARIO_PROPOSAL" && reviewerPhase === "invalid_terminal") {
				const resultText = (body.messages ?? []).filter((message) => message.role === "tool").map(messageText).join("\n");
				assert(resultText.includes("WARNING: memory or primary-chat entry deadbeef not found"), "Reviewer terminal tool did not warn about an invalid curly-braced citation");
				assert(resultText.includes("call submit_workflow_proposal again to replace this review, or end your turn and it will be delivered as-is"), "Reviewer warning did not explain both correction and as-is delivery choices");
				const memoryId = serializedMessages.match(/\[([a-f0-9]{12})\]/)?.[1];
				assert(memoryId, "Reviewer correction context lacked a valid cited memory id");
				this.reviewerToolPhases.set(scenario, "corrected_terminal");
				return sendSse(res, {
					tool: {
						id: "proposal-terminal-corrected",
						name: "submit_workflow_proposal",
						arguments: {
							title: "Reusable evidence checkpoint",
							summary: "Preserve a reusable checkpoint that tests the recurring assumption before repeated work continues.",
							evidence: `[${memoryId}] The cited memory shows repeated reconstruction around the same uncertainty.`,
							inefficiency: "The primary agent repeatedly spends work without obtaining distinguishing evidence.",
							conceptual_design: "Maintain a reusable evidence checkpoint that records the question, direct test, and result for later rounds.",
							inputs: "The active assumption and available direct evidence.", outputs: "A durable result that later reasoning can reuse.", integration: "Consult and refresh the checkpoint when the same uncertainty recurs.", expected_effect: "Reduce repeated investigation and improve reviewability.", uncertainties: "The primary agent must decide which checks are stable enough to preserve.",
						},
					},
				});
			}
			if (hasToolResult) return sendSse(res, { text: "Terminal review complete.", delayMs: 300 });
			if (scenario === "SCENARIO_PROPOSAL") {
				this.reviewerToolPhases.set(scenario, "invalid_terminal");
				return sendSse(res, {
					delayMs: 1_200,
					tool: {
						id: "proposal-terminal-invalid",
						name: "submit_workflow_proposal",
						arguments: {
							title: "Unverified checkpoint",
							summary: "Initial outcome with a bad citation.", evidence: "{deadbeef} repeated reconstruction", inefficiency: "Repeated reconstruction", conceptual_design: "Maintain a checkpoint.", expected_effect: "Improve reviewability.", uncertainties: "Evidence must be corrected.",
						},
					},
				});
			}
			return sendSse(res, {
				delayMs: 1_200,
				tool: {
					id: "reject-terminal",
					name: "review_concluded_no_proposal",
					arguments: {
						reason: "The evidence is currently a single local symptom and does not justify a durable software proposal.",
						evidence_reviewed: "The cited observation and current primary-chat context were reviewed.",
						reconsider_if: "Reconsider if the same structural symptom recurs across independent changes.",
					},
				},
			});
		}

		assert(scenario, "Main request did not contain a scenario marker");
		const count = (this.mainCounts.get(scenario) ?? 0) + 1;
		this.mainCounts.set(scenario, count);
		const serialized = serializedMessages;
		const bashRounds = [
			`printf 'round-one:${scenario}:%s\\n' "$PI_MODEL"`,
			`printf 'round-two:${scenario}\\n'; pwd`,
			`printf 'round-three:${scenario}:'; test -f fixture.txt && wc -c < fixture.txt`,
		];
		if (count <= bashRounds.length) {
			return sendSse(res, {
				delayMs: 600,
				outputTokens: count === 1 ? 200 : 1,
				tool: { id: `bash-${scenario}-${count}`, name: "bash", arguments: { command: bashRounds[count - 1] } },
			});
		}
		if (count === 4) {
			for (const round of ["round-one", "round-two", "round-three"]) {
				assert(serialized.includes(`${round}:${scenario}`), `${scenario} fourth request did not contain ${round} bash output`);
			}
			if (scenario === "SCENARIO_SLEEP") {
				this.sleepToolIssued.add(scenario);
				return sendSse(res, {
					tool: { id: "bash-sleep-probe-race", name: "bash", arguments: { command: `sleep 3; printf '${SLEEP_OUTPUT}\\n'` } },
				});
			}
			assert(!this.heldMain.has(scenario), `Main request already held for ${scenario}`);
			this.heldMain.set(scenario, res);
			return;
		}
		if (scenario === "SCENARIO_SLEEP") {
			assert(serialized.includes(SLEEP_OUTPUT), "Next provider request arrived without the completed sleep-tool output");
			assert(serialized.includes(PROBE_TEXT), "Probe queued during sleep was absent from the first post-tool provider request");
			return sendSse(res, { text: "SLEEP_TOOL_PROBE_RECEIVED_BY_MAIN_AGENT" });
		}
		if (scenario === "SCENARIO_PROBE" || scenario === "SCENARIO_FEEDBACK") {
			assert(serialized.includes(PROBE_TEXT), "Probe was absent from the next main-agent provider request");
			return sendSse(res, {
				text: scenario === "SCENARIO_FEEDBACK" ? PROBE_RESPONSE_TEXT : "PROBE_RECEIVED_BY_MAIN_AGENT",
				// Let the probe-only observer run finish without advancing coverage;
				// turn_end then launches another observer over the complete exchange.
				delayMs: scenario === "SCENARIO_FEEDBACK" ? 1_000 : 0,
			});
		}
		if (scenario === "SCENARIO_PROPOSAL") {
			assert(serialized.includes("BACKGROUND WORKFLOW REVIEW PROPOSAL"), "Review proposal notice was absent from the next main-agent request");
			assert(serialized.includes("only an abbreviated notice, not the full proposal"), "Review notice did not distinguish its summary from the full proposal");
			assert(serialized.includes("call the recall tool with memory id"), "Review notice did not direct the main agent to recall the full proposal");
			return sendSse(res, { text: "PROPOSAL_RECEIVED_BY_MAIN_AGENT" });
		}
		throw new Error(`Unexpected extra main request for ${scenario}`);
	}
}

class RpcClient {
	constructor(child) {
		this.child = child;
		this.nextId = 1;
		this.pending = new Map();
		this.events = [];
		this.stderr = "";
		child.stderr.on("data", (chunk) => { this.stderr += chunk; });
		this.attach(child.stdout);
	}

	attach(stream) {
		const decoder = new StringDecoder("utf8");
		let buffer = "";
		stream.on("data", (chunk) => {
			buffer += decoder.write(chunk);
			while (true) {
				const newline = buffer.indexOf("\n");
				if (newline < 0) break;
				const line = buffer.slice(0, newline).replace(/\r$/, "");
				buffer = buffer.slice(newline + 1);
				if (line) this.handle(JSON.parse(line));
			}
		});
	}

	handle(message) {
		if (message.type === "response" && message.id && this.pending.has(message.id)) {
			const { resolve: resolveResponse, reject } = this.pending.get(message.id);
			this.pending.delete(message.id);
			if (message.success) resolveResponse(message.data);
			else reject(new Error(message.error ?? `RPC ${message.command} failed`));
			return;
		}
		this.events.push(message);
	}

	command(command) {
		const id = `e2e-${this.nextId++}`;
		return new Promise((resolveResponse, reject) => {
			this.pending.set(id, { resolve: resolveResponse, reject });
			this.child.stdin.write(`${JSON.stringify({ id, ...command })}\n`);
		});
	}

	async entries() {
		return (await this.command({ type: "get_entries" })).entries;
	}

	async waitSettled(afterIndex) {
		await waitFor(() => this.events.slice(afterIndex).some((event) => event.type === "agent_settled"), "agent_settled RPC event", 20_000);
	}
}

async function run() {
	console.log("RPC contemplator E2E test plan:");
	TEST_PLAN.forEach((test, index) => console.log(`  ${index + 1}. ${test}`));
	console.log("");
	progress("Creating isolated project, agent configuration, and session directories");
	const workspace = await mkdtemp(join(tmpdir(), "pi-contemplator-e2e-"));
	const project = join(workspace, "project");
	const agentDir = join(workspace, "agent");
	const sessions = join(workspace, "sessions");
	const providerExtension = join(workspace, "mock-provider.ts");
	const server = new MockModelServer();
	let child;
	let rpc;
	try {
		await mkdir(join(project, ".pi"), { recursive: true });
		await mkdir(agentDir, { recursive: true });
		await mkdir(sessions, { recursive: true });
		await writeFile(join(project, "fixture.txt"), "deterministic fixture for the real read tool\n");
		await writeFile(join(project, ".pi/settings.json"), JSON.stringify({
			"observational-memory": {
				observeAfterTokens: 1,
				compactAfterTokens: 1000000,
				// Summary graph behavior is exercised in rpc-summarizer.mjs. Keeping it
				// isolated avoids perturbing this suite's deliberate probe races.
				summarizerEnabled: false,
				agentMaxTurns: 4,
				model: { provider: "e2e", id: "mock-model", thinking: "off" },
				contemplatorEnabled: true,
				contemplatorModel: { provider: "e2e", id: "mock-model", thinking: "off" },
				contemplatorMinNewObservations: 1,
				contemplatorMinTurns: 1,
				showWorkerNotifications: false,
				showContemplatorMessages: true,
				reviewerEnabled: true,
				reviewerModel: { provider: "e2e", id: "mock-model", thinking: "off" },
				compactionObserverEnabled: false,
				debugLog: true,
			},
		}, null, 2));

		const port = await server.start();
		progress(`Mock OpenAI-compatible streaming server listening on 127.0.0.1:${port}`);
		await writeFile(providerExtension, `export default function (pi) {\n  pi.registerProvider("e2e", {\n    name: "E2E Mock",\n    baseUrl: "http://127.0.0.1:${port}/v1",\n    apiKey: "e2e-test-key",\n    api: "openai-completions",\n    models: [{ id: "mock-model", name: "Mock Model", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 4096 }]\n  });\n}\n`);

		progress("Launching real pi --mode rpc subprocess with normal plugins disabled");
		child = spawn(PI, [
			"--mode", "rpc",
			"--provider", "e2e",
			"--model", "mock-model",
			"--thinking", "off",
			"--session-dir", sessions,
			"--offline",
			"--approve",
			"--no-extensions",
			"-e", providerExtension,
			"-e", EXTENSION,
		], {
			cwd: project,
			env: { ...process.env, PI_CODING_AGENT_DIR: agentDir, PI_OFFLINE: "1", PI_SKIP_VERSION_CHECK: "1" },
			stdio: ["pipe", "pipe", "pipe"],
		});
		rpc = new RpcClient(child);
		const exited = once(child, "exit").then(([code, signal]) => ({ code, signal }));

		await waitFor(async () => {
			try {
				const state = await rpc.command({ type: "get_state" });
				return state?.model?.provider === "e2e";
			} catch {
				return false;
			}
		}, "RPC startup with mock model");
		progress("Pi RPC session is ready and connected to e2e/mock-model");

		for (let scenarioIndex = 0; scenarioIndex < SCENARIOS.length; scenarioIndex++) {
			const scenario = SCENARIOS[scenarioIndex];
			console.log("");
			progress(`TEST ${scenarioIndex + 1}/${SCENARIOS.length}: ${SCENARIO_NAMES[scenario]}`);
			const eventStart = rpc.events.length;
			const probeIdsBeforeScenario = new Set((await rpc.entries())
				.filter((entry) => entry.customType === "om.contemplator.suggestion" && typeof entry.data?.probeId === "string")
				.map((entry) => entry.data.probeId));
			await rpc.command({ type: "prompt", message: `${scenario}: perform a multi-round task and keep working through tool results.` });
			if (scenario === "SCENARIO_SLEEP") {
				await waitFor(() => server.sleepToolIssued.has(scenario), `${scenario} sleeping bash tool call issued`);
				progress(`${scenario}: three bash rounds completed; fourth bash call is sleeping for three seconds`);
			} else {
				await waitFor(() => server.heldMain.has(scenario), `${scenario} active fourth main request`);
				progress(`${scenario}: three bash rounds completed; fourth primary-model request is held open`);
			}
			const heldState = await rpc.command({ type: "get_state" });
			assert(heldState.isStreaming === true, `${scenario} primary agent was not streaming while background work ran`);
			if (scenario === "SCENARIO_SLEEP") {
				await waitFor(() => rpc.events.slice(eventStart).some((event) => event.type === "tool_execution_start" && event.toolName === "bash" && event.toolCallId === "bash-sleep-probe-race"), "sleeping bash execution start");
				progress(`${scenario}: real bash sleep is running while the contemplator works`);
			} else {
				await waitFor(() => server.backgroundWhileMainHeld.has(scenario), `${scenario} background model request while the primary request is held open`);
				progress(`${scenario}: primary and background model requests are concurrently active`);
			}

			if (scenario === "SCENARIO_PROBE" || scenario === "SCENARIO_SLEEP" || scenario === "SCENARIO_FEEDBACK") {
				await waitFor(async () => (await rpc.entries()).some((entry) => entry.customType === "om.contemplator.suggestion" && entry.data?.delivered === false && typeof entry.data?.probeId === "string" && !probeIdsBeforeScenario.has(entry.data.probeId) && JSON.stringify(entry.data).includes(PROBE_TEXT)), "new pending contemplator probe");
				const beforeDrain = await rpc.entries();
				assert(!beforeDrain.some((entry) => entry.type === "custom_message" && entry.customType === "om.contemplator.suggestion" && typeof entry.message?.details?.probeId === "string" && !probeIdsBeforeScenario.has(entry.message.details.probeId)), "Probe was persisted/displayed before Pi drained the steer");
				progress(`${scenario}: delayed contemplator queued a probe without prematurely displaying it`);
				if (scenario === "SCENARIO_SLEEP") {
					assert(!rpc.events.slice(eventStart).some((event) => event.type === "tool_execution_end" && event.toolCallId === "bash-sleep-probe-race"), "Sleeping bash call ended before the contemplator probe was queued");
					assert(server.mainCounts.get(scenario) === 4, "Main provider was called again before the sleeping tool completed");
					progress(`${scenario}: probe is pending while bash is still asleep; no premature provider request occurred`);
				}
			} else {
				const expectedOutcome = scenario === "SCENARIO_PROPOSAL" ? "proposal" : "no_proposal";
				await waitFor(async () => (await rpc.entries()).some((entry) => entry.customType === "om.review.result" && entry.data?.result?.outcome === expectedOutcome), `${expectedOutcome} review result`, 20_000);
				if (scenario === "SCENARIO_PROPOSAL") {
					await waitFor(async () => (await rpc.entries()).some((entry) => entry.customType === "om.reviewer.notice"), "queued reviewer proposal notice");
					const beforeDrain = await rpc.entries();
					assert(!beforeDrain.some((entry) => entry.type === "custom_message" && entry.customType === "om.review.proposal"), "Proposal notice was persisted/displayed before Pi drained the steer");
					progress(`${scenario}: reviewer accepted the proposal and queued its notice without premature display`);
				} else {
					progress(`${scenario}: reviewer persisted no_proposal and correctly queued no notice`);
				}
			}

			if (scenario === "SCENARIO_SLEEP") {
				progress(`${scenario}: waiting for bash completion and automatic steer draining`);
			} else {
				progress(`${scenario}: releasing the held primary-model response to test steer draining`);
				await server.releaseMain(scenario);
			}
			await rpc.waitSettled(eventStart);
			if (scenario === "SCENARIO_SLEEP") {
				assert(rpc.events.slice(eventStart).some((event) => event.type === "tool_execution_end" && event.toolCallId === "bash-sleep-probe-race"), "Sleeping bash call never completed");
				progress(`${scenario}: bash completed and the very next provider request contained its output plus the queued probe`);
			}
			const expectedMainRequests = scenario === "SCENARIO_REJECT" ? 4 : 5;
			assert(
				server.mainCounts.get(scenario) === expectedMainRequests,
				`${scenario} made ${server.mainCounts.get(scenario)} main requests; expected ${expectedMainRequests}. ` +
				"The probe/proposal must steer the current run, not wait for a later user prompt.",
			);
			if (scenario === "SCENARIO_FEEDBACK") {
				await waitFor(() => server.feedbackObserverSawProbeAndResponse, "observer request containing both the delivered probe and its primary-agent response", 30_000);
				progress(`${scenario}: observer received the delivered probe and primary-agent response together`);
				await waitFor(async () => (await rpc.entries()).some((entry) =>
					entry.customType === "om.observations.recorded" && JSON.stringify(entry.data).includes(PROBE_FEEDBACK_OBSERVATION)
				), "persisted probe-response observation", 30_000);
				// A finite observer snapshot can legitimately let contemplation run on the
				// delivered probe before source produced concurrently (the response) is
				// observed. The response observation then obeys the response-spaced minimum:
				// drive one real subsequent turn and verify it reaches contemplation.
				await rpc.command({ type: "prompt", message: `${scenario}: continue after the observer feedback checkpoint.` });
				await rpc.waitSettled(eventStart);
				await waitFor(() => server.feedbackObservationReachedContemplator, "probe-response observation delivered back to contemplator", 30_000);
				progress(`${scenario}: resulting observation reached a subsequent contemplator update after one response-spaced turn`);
			}
			// agent_settled covers the primary run, not fire-and-forget memory workers.
			// Do not begin the next scenario while a prior consolidation still owns
			// Runtime.consolidationPromise or its turn-end trigger can be skipped.
			await waitForBackgroundQuiet(rpc, server);
			progress(`PASS ${scenarioIndex + 1}/${SCENARIOS.length}: ${SCENARIO_NAMES[scenario]}`);
		}

		console.log("");
		progress("Running aggregate ledger, transcript, acknowledgement, concurrency, and error assertions");
		const entries = await rpc.entries();
		const observations = entries.filter((entry) => entry.customType === "om.observations.recorded");

		const probeTracking = entries.filter((entry) => entry.customType === "om.contemplator.suggestion" && typeof entry.data?.probeId === "string");
		const deliveredProbes = probeTracking.filter((entry) => entry.data?.delivered === true);
		const reviewRequests = entries.filter((entry) => entry.customType === "om.review.request");
		const reviewResultEntries = entries.filter((entry) => entry.customType === "om.review.result");
		const reviewResults = reviewResultEntries.map((entry) => entry.data?.result?.outcome);
		const reviewerMessages = entries.filter((entry) => entry.customType === "om.reviewer.message");
		const contemplatorMessages = entries.filter((entry) => entry.customType === "om.contemplator.message");
		const reviewerNotices = entries.filter((entry) => entry.customType === "om.reviewer.notice");
		const customMessages = entries.filter((entry) => entry.type === "custom_message");
		const bashResults = entries.filter((entry) => entry.type === "message" && entry.message?.role === "toolResult" && entry.message?.toolName === "bash");
		const latestProbeState = new Map(probeTracking.map((entry) => [entry.data.probeId, entry.data.delivered === true]));
		assert(observations.length >= 5, `Expected initial memories for every scenario plus probe feedback, got ${observations.length}`);

		assert(observations.some((entry) => JSON.stringify(entry.data).includes(PROBE_FEEDBACK_OBSERVATION)), "Probe/response feedback observation was not persisted");
		assert(contemplatorMessages.some((entry) => JSON.stringify(entry.data).includes(PROBE_FEEDBACK_OBSERVATION)), "Persisted contemplator transcript did not receive the probe/response feedback observation");
		assert(bashResults.length === 16, `Expected sixteen real bash tool rounds, got ${bashResults.length}`);
		for (const scenario of SCENARIOS) {
			for (const round of ["round-one", "round-two", "round-three"]) {
				assert(bashResults.some((entry) => JSON.stringify(entry.message.content).includes(`${round}:${scenario}`)), `Missing persisted ${round} bash result for ${scenario}`);
			}
		}
		assert(server.maxConcurrentRequests >= 2, "Expected overlapping primary and background model requests");
		assert(contemplatorMessages.length >= 6, "Expected persisted contemplator prompts and responses");
		assert(deliveredProbes.length === 3, `Expected exactly three acknowledged probes, got ${deliveredProbes.length}`);
		assert(!probeTracking.some((entry) => JSON.stringify(entry.data).includes("deadbeef")), "Superseded invalid probe escaped the contemplator turn");
		assert(customMessages.filter((entry) => entry.customType === "om.contemplator.suggestion").every((entry) => JSON.stringify(entry).includes("Referenced memories can be reviewed using the recall tool")), "A primary-agent probe omitted recall-tool guidance");
		assert(server.memorySearchUsed, "Contemplator never completed search_memories against persisted memory");
		assert(server.memoryRecallUsed && server.recallReturnedSource, "Contemplator never recalled exact source context from a search result");
		assert(server.requests.some((request) => request.role === "contemplator" && JSON.stringify(request.body.messages ?? []).includes("search-probe-memory")), "Mock server never saw the contemplator's search_memories tool round");
		assert(server.requests.some((request) => request.role === "contemplator" && JSON.stringify(request.body.messages ?? []).includes("recall-probe-memory")), "Mock server never saw the contemplator's recall tool round");
		assert([...latestProbeState.values()].every(Boolean), "Every queued probe must finish acknowledged");
		assert(reviewRequests.length === 2, `Expected exactly two review requests, got ${reviewRequests.length}`);
		assert(reviewResultEntries.length === 2, `Expected exactly two review results, got ${reviewResultEntries.length}`);
		assert(reviewResults.includes("proposal"), "Expected a reviewer proposal result");
		assert(reviewResults.includes("no_proposal"), "Expected a reviewer no-proposal result");
		assert(server.reviewerToolPhases.get("SCENARIO_PROPOSAL") === "corrected_terminal", "Reviewer did not receive an in-turn correction chance after its warning");
		assert(reviewResultEntries.some((entry) => entry.data?.result?.title === "Reusable evidence checkpoint" && !JSON.stringify(entry.data?.result).includes("deadbeef")), "Corrected reviewer outcome did not replace the warned candidate");
		assert(reviewerMessages.length >= 2, "Expected durable reviewer transcripts");
		assert(reviewerNotices.length === 1, `Only the accepted proposal should queue a main-agent notice; got ${reviewerNotices.length}`);
		assert(customMessages.some((entry) => entry.customType === "om.contemplator.suggestion"), "Expected probe insertion in the main conversation stream");
		assert(customMessages.filter((entry) => entry.customType === "om.review.proposal").length === 1, "Expected exactly one proposal notice in the main conversation stream");
		assert(!rpc.events.some((event) => event.type === "extension_error"), "The real Pi harness reported an extension error");
		assert(server.requests.some((request) => request.role === "observer"), "Observer never reached the mock server");
		assert(!server.requests.some((request) => request.role === "summarizer"), "An isolated summarizer unexpectedly reached this probe-race suite");
		assert(server.requests.some((request) => request.role === "contemplator"), "Contemplator never reached the mock server");
		assert(server.requests.some((request) => request.role === "reviewer"), "Reviewer never reached the mock server");

		child.kill("SIGTERM");
		const result = await exited;
		assert(result.code === 0 || result.code === 143 || result.signal === "SIGTERM", `Pi exited unexpectedly: ${JSON.stringify(result)}\n${rpc.stderr}`);
		progress(`ALL TESTS PASSED: ${observations.length} observation batches, 16 bash rounds including sleep race, ${deliveredProbes.length} delivered probes, search + recall, complete probe feedback loop, proposal + no-proposal reviewer outcomes`);
	} catch (error) {
		console.error("E2E failure:", error?.stack ?? error);
		console.error("Mock requests:", server.requests.map((request) => `${request.role}:${request.scenario}`).join(", "));
		if (rpc) {
			console.error("Recent RPC events:", JSON.stringify(rpc.events.slice(-12), null, 2));
			console.error("Pi stderr:", rpc.stderr);
		}
		throw error;
	} finally {
		if (child && child.exitCode === null) child.kill("SIGKILL");
		await server.close().catch(() => {});
		if (process.env.E2E_KEEP) console.error(`Preserved E2E workspace: ${workspace}`);
		else await rm(workspace, { recursive: true, force: true });
	}
}

run().catch((error) => {
	console.error(error?.stack ?? error);
	process.exitCode = 1;
});
