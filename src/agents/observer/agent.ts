import { agentLoop, type AgentContext, type AgentLoopConfig, type AgentMessage, type AgentTool } from "@earendil-works/pi-agent-core";
import type { Message, Model, ModelThinkingLevel } from "@earendil-works/pi-ai";
import { Type } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/compat";
import type { Static } from "typebox";
import { hashId } from "../../ids.js";
import { replayTruncatedThinkingAsText } from "../replay-truncated-thinking.js";
import { logAgentStreamError } from "../stream-errors.js";
import { OBSERVER_AGENT_LOOP_MAX_TOKENS, boundedMaxTokens } from "../../model-budget.js";
import { OBSERVER_SYSTEM } from "./prompts.js";
import { nowTimestamp, truncateRecordContent } from "../../serialize.js";
import type { Observation, Relevance, Retention } from "../../session-ledger/index.js";
import { estimateStringTokens } from "../../tokens.js";
import type { LlmUsageInput } from "../../runtime.js";

interface RunObserverArgs {
	model: Model<any>;
	apiKey: string;
	headers?: Record<string, string>;
	priorSummaries?: string[];
	priorObservations: string[];
	chunk: string;
	allowedSourceEntryIds: string[];
	signal?: AbortSignal;
	agentLoop?: typeof agentLoop;
	maxTurns?: number;
	thinkingLevel?: ModelThinkingLevel;
	recordUsage?: (usage: LlmUsageInput) => void;
	onProgress?: () => void;
	onMessages?: (messages: readonly AgentMessage[]) => void;
}

const RelevanceSchema = Type.Union([
	Type.Literal("low"),
	Type.Literal("medium"),
	Type.Literal("high"),
	Type.Literal("critical"),
]);

const RetentionSchema = Type.Union([
	Type.Literal("ephemeral"),
	Type.Literal("contextual"),
	Type.Literal("durable"),
]);

export const OBSERVATION_TIMESTAMP_PATTERN = "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}$";

const RecordObservationsSchema = Type.Object({
	observations: Type.Array(
		Type.Object({
			timestamp: Type.String({
				pattern: OBSERVATION_TIMESTAMP_PATTERN,
				description: "Observation time in local 'YYYY-MM-DD HH:MM' format.",
			}),
			content: Type.String({
				minLength: 1,
				description: "Single-line plain prose. No markdown, no tags, no embedded timestamp.",
			}),
			relevance: RelevanceSchema,
			retention: Type.Optional(RetentionSchema),
			sourceEntryIds: Type.Array(
				Type.String({ minLength: 1 }),
				{
					minItems: 1,
					description:
						"Exact source entry ids from the chunk that directly support this observation. " +
						"Use only ids shown in '[Source entry id: ...]' labels; never invent ids.",
				},
			),
		}),
		{ description: "Batch of new observations. May be empty only if the tool is not called at all." },
	),
});

type RecordObservationsArgs = Static<typeof RecordObservationsSchema>;

export const OBSERVER_MAX_LENGTH_ATTEMPTS = 4;

/** A terminal provider/agent-loop failure that must not advance observation coverage. */
export class ObserverStreamError extends Error {
	readonly stopReason: string;
	constructor(stopReason: string, errorMessage?: string) {
		super(`observer stream ended with stopReason "${stopReason}"${errorMessage ? `: ${errorMessage}` : ""}`);
		this.name = "ObserverStreamError";
		this.stopReason = stopReason;
	}
}

function joinOrEmpty(items: string[]): string {
	return items.length ? items.join("\n") : "(none yet)";
}

export function normalizeSourceEntryIds(
	sourceEntryIds: readonly string[] | undefined,
	allowedSourceEntryIds: readonly string[],
): string[] | undefined {
	if (!sourceEntryIds || sourceEntryIds.length === 0) return undefined;
	const allowedOrder = new Map<string, number>();
	for (let i = 0; i < allowedSourceEntryIds.length; i++) allowedOrder.set(allowedSourceEntryIds[i], i);

	const seen = new Set<string>();
	for (const id of sourceEntryIds) {
		if (!allowedOrder.has(id)) return undefined;
		seen.add(id);
	}
	if (seen.size === 0) return undefined;
	return Array.from(seen).sort((a, b) => (allowedOrder.get(a) ?? 0) - (allowedOrder.get(b) ?? 0));
}

export async function runObserver(args: RunObserverArgs): Promise<Observation[] | undefined> {
	const { model, apiKey, headers, priorSummaries = [], priorObservations, chunk, allowedSourceEntryIds, signal } = args;
	const conversation = chunk.trim();
	if (!conversation) return undefined;

	const accumulated = new Map<string, Observation>();
	let rejectedTotal = 0;
	let doneCalled = false;

	const recordObservations: AgentTool<typeof RecordObservationsSchema> = {
		name: "record_observations",
		label: "Record observations",
		description:
			"Record a batch of new observations distilled from the conversation chunk. " +
			"Call this multiple times as you work through the chunk, then call done alone when coverage is complete.",
		parameters: RecordObservationsSchema,
		execute: async (_id, params: RecordObservationsArgs) => {
			let added = 0;
			let duplicates = 0;
			let rejected = 0;
			for (const obs of params.observations) {
				const sourceEntryIds = normalizeSourceEntryIds(obs.sourceEntryIds, allowedSourceEntryIds);
				if (!sourceEntryIds) {
					rejected++;
					continue;
				}
				const content = truncateRecordContent(obs.content);
				const id = hashId(content);
				if (accumulated.has(id)) {
					duplicates++;
					continue;
				}
				accumulated.set(id, {
					id,
					content,
					timestamp: obs.timestamp,
					relevance: obs.relevance as Relevance,
					retention: (obs.retention ?? "contextual") as Retention,
					sourceEntryIds,
					tokenCount: estimateStringTokens(content),
				});
				added++;
			}
			rejectedTotal += rejected;
			const rejectedPart = rejected > 0
				? ` ${rejected} observation${rejected === 1 ? "" : "s"} rejected for missing or invalid sourceEntryIds.`
				: "";
			const ack =
				`Recorded ${added} new observation${added === 1 ? "" : "s"} ` +
				(duplicates > 0 ? `(${duplicates} duplicate${duplicates === 1 ? "" : "s"} skipped).` : ".") +
				rejectedPart +
				` Total so far this run: ${accumulated.size}. ` +
				`Continue if the chunk still has uncovered content; otherwise call done alone.`;
			return { content: [{ type: "text", text: ack }], details: { added, duplicates, rejected, total: accumulated.size } };
		},
	};

	const doneTool: AgentTool<any> = {
		name: "done",
		label: "Done",
		description: "Confirm that the entire provided conversation chunk has been inspected and all useful new observations have been recorded. Call alone, including when there is nothing new to record.",
		parameters: Type.Object({}),
		execute: async () => {
			doneCalled = true;
			return { content: [{ type: "text", text: "Observer coverage confirmed." }], details: {}, terminate: true };
		},
	};

	const now = nowTimestamp();
	const userText = `Current local time: ${now}

CURRENT SUMMARIES:
${joinOrEmpty(priorSummaries)}

CURRENT OBSERVATIONS:
${joinOrEmpty(priorObservations)}

Compress the following new conversation chunk into observations by calling record_observations one or more times. Do not restate facts already present in current summaries or current observations. Prefer inline conversation timestamps when assigning times; fall back to the current local time above only if no message timestamp applies. When the chunk is fully covered, call done alone. If the chunk contains no useful new information, call done without calling record_observations.

NEW CONVERSATION CHUNK:
${conversation}

END NEW CONVERSATION CHUNK

IMPORTANT: Now call record_observations to record the useful new observations from this conversation chunk. When the chunk is covered, call done; if there are no useful observations, call done without recording any.`;

	const initialPrompt: Message = {
		role: "user",
		content: [{ type: "text", text: userText }],
		timestamp: Date.now(),
	};

	const reasoning = (model as { reasoning?: unknown }).reasoning;
	const thinkingLevel = args.thinkingLevel ?? "low";
	const effectiveMaxTurns = args.maxTurns && args.maxTurns > 0 ? args.maxTurns : undefined;
	let turnCount = 0;
	const baseConfig: AgentLoopConfig = {
		model,
		apiKey,
		headers,
		maxTokens: boundedMaxTokens(model, OBSERVER_AGENT_LOOP_MAX_TOKENS),
		convertToLlm: replayTruncatedThinkingAsText,
		toolExecution: "sequential",
		shouldStopAfterTurn: () => {
			turnCount++;
			return doneCalled || (effectiveMaxTurns !== undefined && turnCount >= effectiveMaxTurns);
		},
		...(reasoning && thinkingLevel !== "off" ? { reasoning: thinkingLevel } : {}),
	};

	const loop = args.agentLoop ?? agentLoop;
	const history: AgentMessage[] = [];
	let terminalFailure: { stopReason: string; errorMessage?: string } | undefined;
	let lengthAttempts = 0;
	const runInvocation = async (prompt: Message, afterLength = false): Promise<void> => {
		const context: AgentContext = {
			systemPrompt: OBSERVER_SYSTEM,
			messages: history.slice(),
			tools: [recordObservations as AgentTool<any>, doneTool],
		};
		// Publish a live launch-local transcript for /om:view observer. Keep an
		// invocation-local list because agentLoop owns its internal context copy.
		let liveMessages: AgentMessage[] = [...history, prompt];
		args.onMessages?.(liveMessages.slice());
		const invocationConfig: AgentLoopConfig = afterLength && reasoning
			? { ...baseConfig, reasoning: "minimal" }
			: baseConfig;
		const stream = loop([prompt], context, invocationConfig, signal, streamSimple);
		for await (const event of stream) {
			args.onProgress?.();
			logAgentStreamError("observer", event);
			const typedEvent = event as { type?: string; message?: AgentMessage & { role?: string; stopReason?: string; errorMessage?: string } };
			const message = typedEvent.message;
			if (message && message !== prompt && message.role !== "user") {
				if (typedEvent.type === "message_start") liveMessages.push(message);
				else if (typedEvent.type === "message_update" || typedEvent.type === "message_end") {
					const index = liveMessages.map((item) => item.role).lastIndexOf(message.role);
					if (index >= 0) liveMessages[index] = message;
					else liveMessages.push(message);
				}
				args.onMessages?.(liveMessages.slice());
			}
			if (message?.role === "assistant" && ["error", "aborted", "length"].includes(message.stopReason ?? "")) {
				terminalFailure = { stopReason: message.stopReason!, errorMessage: message.errorMessage };
			}
		}
		const result = await stream.result();
		if (!Array.isArray(result)) return;
		history.push(...result);
		liveMessages = history.slice();
		args.onMessages?.(liveMessages);
		for (const message of result) {
			if (message.role === "assistant" && ["error", "aborted", "length"].includes(message.stopReason ?? "")) {
				terminalFailure = { stopReason: message.stopReason, errorMessage: message.errorMessage };
			}
			if (args.recordUsage && message.role === "assistant" && message.usage) args.recordUsage(message.usage);
		}
	};

	await runInvocation(initialPrompt);
	if (terminalFailure?.stopReason === "length") lengthAttempts = 1;
	while (accumulated.size === 0 && terminalFailure?.stopReason === "length" && lengthAttempts < OBSERVER_MAX_LENGTH_ATTEMPTS) {
		// A provider can impose a lower output ceiling than the advertised model
		// maximum. agentLoop stops on `length` when no tool call was completed; it
		// does not automatically send a continuation request. Preserve the partial
		// response so the model can continue from work it already performed rather
		// than paying to reproduce it. Plaintext thinking is replayed as ordinary
		// assistant text at the LLM boundary because some provider templates strip
		// historical reasoning; encrypted thinking retains its opaque structure.
		// Then append a short tool-focused instruction and reduce reasoning to
		// minimal. The bounded attempt limit prevents pathological chunks from
		// consuming background tokens forever.
		terminalFailure = undefined;
		const retryPrompt: Message = {
			role: "user",
			content: [{ type: "text", text: "IMPORTANT: The previous response reached the provider output limit before recording anything. Continue from the work already above and call record_observations now instead of spending another response budget analyzing." }],
			timestamp: Date.now(),
		};
		await runInvocation(retryPrompt, true);
		// runInvocation mutates this through its async stream callbacks; TypeScript
		// cannot infer that mutation after the explicit reset above.
		const retryFailure = terminalFailure as { stopReason: string; errorMessage?: string } | undefined;
		if (retryFailure?.stopReason === "length") lengthAttempts++;
	}
	if (accumulated.size === 0 && !doneCalled && !terminalFailure && rejectedTotal === 0) {
		const reminder: Message = {
			role: "user",
			content: [{ type: "text", text: `You stopped without confirming coverage. Observations recorded so far: ${accumulated.size}. If the chunk is fully covered, call done now. Otherwise call record_observations for anything still missing, then call done.` }],
			timestamp: Date.now(),
		};
		await runInvocation(reminder);
	}

	// `done` is a behavioral aid and terminal shortcut, not a transaction gate.
	// Accepted observations commit even if it was omitted. A second prose-only
	// zero-observation stop is also a valid empty result after the reminder;
	// actual stream failures, truncation, and malformed records still throw.
	if (accumulated.size === 0 && terminalFailure) {
		const detail = terminalFailure.stopReason === "length" && lengthAttempts > 0
			? `provider reached the output limit ${lengthAttempts} times without recording an observation (effective max output request: ${baseConfig.maxTokens} tokens)`
			: terminalFailure.errorMessage;
		throw new ObserverStreamError(terminalFailure.stopReason, detail);
	}
	if (accumulated.size === 0 && rejectedTotal > 0) {
		throw new ObserverStreamError("invalid_observations", `${rejectedTotal} proposed observation${rejectedTotal === 1 ? " was" : "s were"} rejected`);
	}
	if (accumulated.size === 0) return undefined;
	return Array.from(accumulated.values());
}
