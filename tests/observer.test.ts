import { describe, expect, it } from "vitest";

import { normalizeSourceEntryIds, OBSERVATION_TIMESTAMP_PATTERN, OBSERVER_MAX_LENGTH_ATTEMPTS, ObserverStreamError, runObserver } from "../src/agents/observer/agent.js";
import { estimateStringTokens } from "../src/tokens.js";
import { OBSERVER_AGENT_LOOP_MAX_TOKENS } from "../src/model-budget.js";

function fakeAgentLoop(handler: (prompts: any[], context: any, config: any) => Promise<void> | void): any {
	return ((prompts: any[], context: any, config: any) => ({
		async *[Symbol.asyncIterator]() {
			// No streaming events needed for these tests.
		},
		result: async () => {
			await handler(prompts, context, config);
			await context.tools.find((tool: any) => tool.name === "done")?.execute("done", {});
			return {};
		},
	})) as any;
}

describe("OBSERVATION_TIMESTAMP_PATTERN", () => {
	it("matches local minute timestamps without regex shorthand escapes", () => {
		expect(OBSERVATION_TIMESTAMP_PATTERN).not.toContain("\\d");
		const pattern = new RegExp(OBSERVATION_TIMESTAMP_PATTERN);
		expect(pattern.test("2026-05-02 10:30")).toBe(true);
		expect(pattern.test("2026-5-02 10:30")).toBe(false);
		expect(pattern.test("2026-05-02T10:30")).toBe(false);
		expect(pattern.test("2026-05-02 10:30:00")).toBe(false);
	});
});

describe("runObserver", () => {
	const baseArgs = {
		model: {} as any,
		apiKey: "test",
		priorSummaries: [],
		priorObservations: [],
		chunk: "[Source entry id: entry-a]\nUser asked for a memory update.",
		allowedSourceEntryIds: ["entry-a"],
	};

	it("ends the source chunk with an explicit tool-call instruction", async () => {
		let userPrompt = "";
		const loop = fakeAgentLoop((prompts) => {
			userPrompt = prompts[0].content[0].text;
		});

		await runObserver({ ...baseArgs, agentLoop: loop });

		expect(userPrompt).toContain("END NEW CONVERSATION CHUNK");
		expect(userPrompt).toMatch(/END NEW CONVERSATION CHUNK[\s\S]*Now call record_observations/);
	});

	it("publishes the live observer transcript including streamed thinking", async () => {
		const snapshots: any[][] = [];
		const assistant = { role: "assistant", content: [{ type: "thinking", thinking: "working through the chunk" }], stopReason: "stop" };
		const loop = ((prompts: any[], context: any) => ({
			async *[Symbol.asyncIterator]() {
				yield { type: "message_start", message: assistant };
				yield { type: "message_update", message: assistant };
				yield { type: "message_end", message: assistant };
			},
			result: async () => {
				await context.tools.find((tool: any) => tool.name === "done").execute("done", {});
				return [...prompts, assistant];
			},
		})) as any;

		await runObserver({ ...baseArgs, agentLoop: loop, onMessages: (messages) => snapshots.push(messages.slice() as any[]) });

		expect(snapshots[0][0].role).toBe("user");
		expect(snapshots.some((messages) => messages.some((message) => message.role === "assistant" && message.content?.[0]?.thinking === "working through the chunk"))).toBe(true);
	});

	it("keeps core observer prompt rules", async () => {
		let systemPrompt = "";
		const loop = fakeAgentLoop((_prompts, context) => {
			systemPrompt = context.systemPrompt;
		});

		await runObserver({ ...baseArgs, agentLoop: loop });

		expect(systemPrompt).toContain("Preserve user assertions exactly");
		expect(systemPrompt).toContain("Detail preservation");
		expect(systemPrompt).toContain("Preserve useful concrete details first");
		expect(systemPrompt).toContain("not a high-level conversation-summary task");
		expect(systemPrompt).toContain("useful inputs, outputs, discoveries, and final conclusion");
		expect(systemPrompt).toContain("A vague topic summary is not a substitute");
		expect(systemPrompt).toContain("Frame state changes as supersession");
		expect(systemPrompt).toContain("sourceEntryIds");
		expect(systemPrompt).toContain("zero observations");
		expect(systemPrompt).toContain("The summarizer considers relevance together with retention");
		expect(systemPrompt).toContain("Retention horizons");
		expect(systemPrompt).toContain("highest-resistance, load-bearing observations");
		expect(systemPrompt).not.toContain("will NEVER be dropped");
		expect(systemPrompt).not.toContain("pruner");
	});

	it("defaults omitted retention without rejecting the observation batch", async () => {
		const content = "User asked for a memory update.";
		const loop = fakeAgentLoop(async (_prompts, context) => {
			expect(context.tools[0].parameters.properties.observations.items.required).not.toContain("retention");
			await context.tools[0].execute("tool-1", {
				observations: [{ timestamp: "2026-05-02 10:30", content, relevance: "high", sourceEntryIds: ["entry-a"] }],
			});
		});

		const observations = await runObserver({ ...baseArgs, agentLoop: loop });

		expect(observations).toHaveLength(1);
		expect(observations?.[0]).toMatchObject({
			content,
			timestamp: "2026-05-02 10:30",
			relevance: "high",
			retention: "contextual",
			sourceEntryIds: ["entry-a"],
			tokenCount: estimateStringTokens(content),
		});
		expect(observations?.[0].id).toMatch(/^[a-f0-9]{12}$/);
	});

	it("rejects invented source ids and returns no observations", async () => {
		const loop = fakeAgentLoop(async (_prompts, context) => {
			await context.tools[0].execute("tool-1", {
				observations: [{ timestamp: "2026-05-02 10:30", content: "Bad source", relevance: "medium", sourceEntryIds: ["missing"] }],
			});
		});

		await expect(runObserver({ ...baseArgs, agentLoop: loop })).rejects.toThrow(ObserverStreamError);
	});

	it("dedupes deterministic ids", async () => {
		const loop = fakeAgentLoop(async (_prompts, context) => {
			await context.tools[0].execute("tool-1", {
				observations: [
					{ timestamp: "2026-05-02 10:30", content: "Same content", relevance: "medium", sourceEntryIds: ["entry-a"] },
					{ timestamp: "2026-05-02 10:31", content: "Same content", relevance: "high", sourceEntryIds: ["entry-a"] },
				],
			});
		});

		const observations = await runObserver({ ...baseArgs, agentLoop: loop });

		expect(observations).toHaveLength(1);
		expect(observations?.[0].content).toBe("Same content");
	});

	it("commits recorded observations even when done is omitted", async () => {
		const loop = ((_prompts: any[], context: any) => ({
			async *[Symbol.asyncIterator]() {},
			result: async () => {
				await context.tools.find((tool: any) => tool.name === "record_observations").execute("record", {
					observations: [{ timestamp: "2026-05-02 10:30", content: "Useful result without done", relevance: "medium", sourceEntryIds: ["entry-a"] }],
				});
				return [];
			},
		})) as any;

		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toEqual([
			expect.objectContaining({ content: "Useful result without done" }),
		]);
	});

	it("returns undefined when done confirms a zero-observation chunk", async () => {
		const loop = fakeAgentLoop(() => {});
		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();
	});

	it("injects a normal reminder with the recorded count when the first response omits done", async () => {
		const promptsSeen: string[] = [];
		const configs: any[] = [];
		let invocation = 0;
		const loop = ((prompts: any[], context: any, config: any) => ({
			async *[Symbol.asyncIterator]() {},
			result: async () => {
				promptsSeen.push(prompts[0].content[0].text);
				configs.push(config);
				invocation++;
				if (invocation === 2) await context.tools.find((tool: any) => tool.name === "done").execute("done", {});
				return [];
			},
		})) as any;

		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();
		expect(promptsSeen).toHaveLength(2);
		expect(promptsSeen[1]).toContain("Observations recorded so far: 0");
		expect(promptsSeen[1]).toContain("call done now");
		expect(configs[0].toolChoice).toBeUndefined();
		expect(configs[1].toolChoice).toBeUndefined();
		expect(configs[1].onPayload).toBeUndefined();
	});

	it("accepts an empty result when the model ignores done after the reminder", async () => {
		let invocations = 0;
		const loop = (() => ({
			async *[Symbol.asyncIterator]() {},
			result: async () => {
				invocations++;
				return [];
			},
		})) as any;

		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();
		expect(invocations).toBe(2);
	});

	it("continues a first output-length stop with its partial work and minimal reasoning", async () => {
		let invocation = 0;
		const prompts: string[] = [];
		const reasonings: unknown[] = [];
		const contextMessages: any[][] = [];
		const llmMessages: any[][] = [];
		const loop = ((input: any[], context: any, config: any) => ({
			async *[Symbol.asyncIterator]() {},
			result: async () => {
				invocation++;
				prompts.push(input[0].content[0].text);
				reasonings.push(config.reasoning);
				contextMessages.push(context.messages);
				llmMessages.push(config.convertToLlm([...context.messages, ...input]));
				if (invocation === 1) return [{ role: "assistant", content: [{ type: "thinking", thinking: "partial analysis" }], stopReason: "length" }];
				await context.tools.find((tool: any) => tool.name === "record_observations").execute("record", {
					observations: [{ timestamp: "2026-05-02 10:30", content: "Recovered after length", relevance: "medium", sourceEntryIds: ["entry-a"] }],
				});
				return [];
			},
		})) as any;

		const result = await runObserver({ ...baseArgs, model: { reasoning: true } as any, thinkingLevel: "high", agentLoop: loop });

		expect(result?.[0].content).toBe("Recovered after length");
		expect(prompts).toHaveLength(2);
		expect(prompts[1]).toContain("previous response reached the provider output limit");
		expect(prompts[1]).not.toContain("NEW CONVERSATION CHUNK");
		expect(contextMessages[1]).toEqual(expect.arrayContaining([
			expect.objectContaining({ role: "assistant", stopReason: "length" }),
		]));
		expect(JSON.stringify(llmMessages[1])).toContain("[Incomplete analysis from the preceding truncated response]\\npartial analysis");
		expect(llmMessages[1].flatMap((message: any) => message.role === "assistant" ? message.content : []).some((part: any) => part.type === "thinking")).toBe(false);
		expect(reasonings).toEqual(["high", "minimal"]);
	});

	it("rejects after four zero-observation output-limit attempts instead of retrying forever", async () => {
		let invocations = 0;
		const loop = (() => ({
			async *[Symbol.asyncIterator]() {
				yield { type: "message_end", message: { role: "assistant", content: [], stopReason: "length" } };
			},
			result: async () => {
				invocations++;
				return [];
			},
		})) as any;
		await expect(runObserver({ ...baseArgs, agentLoop: loop })).rejects.toMatchObject({
			stopReason: "length",
			message: expect.stringContaining(`reached the output limit ${OBSERVER_MAX_LENGTH_ATTEMPTS} times`),
		});
		expect(invocations).toBe(OBSERVER_MAX_LENGTH_ATTEMPTS);
	});

	it("uses the expanded observer output allowance when the model supports it", async () => {
		let configuredMaxTokens: number | undefined;
		const loop = fakeAgentLoop((_prompts, _context, config) => {
			configuredMaxTokens = config.maxTokens;
		});

		await runObserver({ ...baseArgs, model: { maxTokens: 256_000 } as any, agentLoop: loop });

		expect(configuredMaxTokens).toBe(OBSERVER_AGENT_LOOP_MAX_TOKENS);
	});

	it("uses maxTurns as an observer turn cap", async () => {
		let stopResults: boolean[] = [];
		const loop = fakeAgentLoop((_prompts, _context, config) => {
			expect(config.shouldStopAfterTurn).toBeTypeOf("function");
			stopResults = [config.shouldStopAfterTurn({}), config.shouldStopAfterTurn({})];
		});

		await runObserver({ ...baseArgs, agentLoop: loop, maxTurns: 2 });

		expect(stopResults).toEqual([false, true]);
	});

	it("uses configured observer thinking level for reasoning models", async () => {
		let seenReasoning: unknown;
		const loop = fakeAgentLoop((_prompts, _context, config) => {
			seenReasoning = config.reasoning;
		});

		await runObserver({ ...baseArgs, model: { reasoning: true } as any, agentLoop: loop, thinkingLevel: "minimal" });

		expect(seenReasoning).toBe("minimal");
	});

	it("omits observer reasoning when thinkingLevel is off", async () => {
		let seenReasoning: unknown = "unset";
		const loop = fakeAgentLoop((_prompts, _context, config) => {
			seenReasoning = config.reasoning;
		});

		await runObserver({ ...baseArgs, model: { reasoning: true } as any, agentLoop: loop, thinkingLevel: "off" });

		expect(seenReasoning).toBeUndefined();
	});

	it("reports agentLoop usage through recordUsage", async () => {
		const usage = { input: 4000, output: 500, cacheRead: 0, cacheWrite: 0, cost: { total: 0.002 } };
		const loop = ((_prompts: any, context: any, _config: any) => ({
			async *[Symbol.asyncIterator]() {},
			result: async () => {
				await context.tools.find((tool: any) => tool.name === "done").execute("done", {});
				return [{ role: "assistant", content: [{ type: "text", text: "ok" }], usage, stopReason: "stop", timestamp: Date.now() }];
			},
		})) as any;
		const recorded: unknown[] = [];

		await runObserver({ ...baseArgs, agentLoop: loop, recordUsage: (u) => recorded.push(u) });

		expect(recorded).toEqual([usage]);
	});
});

describe("normalizeSourceEntryIds", () => {
	const allowed = ["entry-a", "entry-b", "entry-c"];

	it("accepts source ids from the allowed chunk and orders them by branch order", () => {
		expect(normalizeSourceEntryIds(["entry-c", "entry-a"], allowed)).toEqual(["entry-a", "entry-c"]);
	});

	it("dedupes repeated source ids", () => {
		expect(normalizeSourceEntryIds(["entry-b", "entry-b", "entry-a"], allowed)).toEqual(["entry-a", "entry-b"]);
	});

	it("rejects missing, empty, or hallucinated source ids", () => {
		expect(normalizeSourceEntryIds(undefined, allowed)).toBeUndefined();
		expect(normalizeSourceEntryIds([], allowed)).toBeUndefined();
		expect(normalizeSourceEntryIds(["entry-a", "not-in-the-chunk"], allowed)).toBeUndefined();
		expect(normalizeSourceEntryIds(["entry-a"], [])).toBeUndefined();
	});
});
