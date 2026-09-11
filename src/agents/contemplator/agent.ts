import { agentLoop, type AgentContext, type AgentLoopConfig, type AgentMessage, type AgentTool } from "@earendil-works/pi-agent-core";
import { Type, type Message, type Model } from "@earendil-works/pi-ai";
import type { Static } from "typebox";
import { streamSimple } from "@earendil-works/pi-ai/compat";
import { estimateTokens as estimateAgentMessageTokens, generateSummaryWithUsage } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Box, Text } from "@earendil-works/pi-tui";
import { agentActiveTimeMs, assistantOutputTokens, assistantToolCallCount, fullProjection, isReviewRequestEntry, isReviewResultEntry, OM_AGENT_ACTIVITY, OM_REVIEWER_MESSAGE, OM_REVIEWER_NOTICE, OM_REVIEWER_STATE, OM_REVIEW_REQUEST, OM_REVIEW_RESULT, rawTokensSinceObservationCoverage, recallMemorySources, type Entry, type ReviewResult, type StructuralReviewRequest } from "../../session-ledger/index.js";
import { hashId } from "../../ids.js";
import { createSearchMemoriesAgentTool } from "../../tools/search-memories.js";
import { createRecallAgentTool } from "../../tools/recall-observation.js";
import type { MemoryUpdateCtx, Runtime } from "../../runtime.js";
import { logAgentStreamError } from "../stream-errors.js";
import { debugLog, withDebugLogContext } from "../../debug-log.js";
import { boundedMaxTokens, AGENT_LOOP_MAX_TOKENS } from "../../model-budget.js";
import { forceRequiredToolPayload, requiredToolChoice } from "../../required-tool-choice.js";
import { memoryReferenceIds } from "../../memory-citations.js";
import { buildContemplatorSystemPrompt } from "./prompts.js";
import { runStructuralReview } from "../reviewer/agent.js";
import { createWorkerStallWatchdog } from "../../worker-watchdog.js";

const CONTEMPLATOR_HISTORY_FALLBACK_TRIGGER_TOKENS = 20_000;
const CONTEMPLATOR_HISTORY_MAX_TRIGGER_TOKENS = 40_000;
const CONTEMPLATOR_HISTORY_CONTEXT_FRACTION = 0.25;
const CONTEMPLATOR_HISTORY_KEEP_RECENT_TOKENS = 12_000;
const CONTEMPLATOR_HISTORY_SUMMARY_RESERVE_TOKENS = 16_000;
const CONTEMPLATOR_HISTORY_SUMMARY_INSTRUCTIONS = "This is an older prefix of a private contemplator transcript. Be concise. Preserve durable user intent, decisions, evidence, unresolved reasoning gaps, and review/probe outcomes. Newer transcript messages are retained verbatim after this checkpoint.";

function contemplatorMessageTokens(message: AgentMessage): number {
	try {
		return Math.max(1, estimateAgentMessageTokens(message));
	} catch {
		return Math.max(1, Math.ceil(JSON.stringify(message).length / 4));
	}
}

function contemplatorHistoryTokens(history: readonly AgentMessage[]): number {
	return history.reduce((total, message) => total + contemplatorMessageTokens(message), 0);
}

function contemplatorHistoryTriggerTokens(model: Model<any>): number {
	const contextWindow = Number(model.contextWindow);
	if (!Number.isFinite(contextWindow) || contextWindow <= 0) return CONTEMPLATOR_HISTORY_FALLBACK_TRIGGER_TOKENS;
	return Math.max(1, Math.min(CONTEMPLATOR_HISTORY_MAX_TRIGGER_TOKENS, Math.floor(contextWindow * CONTEMPLATOR_HISTORY_CONTEXT_FRACTION)));
}

/** Return a user-message boundary that leaves approximately targetTokens recent. */
function recentHistoryStart(history: readonly AgentMessage[], entryIds: readonly (string | undefined)[], targetTokens: number): number | undefined {
	let tokens = 0;
	let start = history.length;
	while (start > 0 && tokens < targetTokens) {
		start--;
		tokens += contemplatorMessageTokens(history[start]);
	}
	while (start > 0 && history[start].role !== "user") start--;
	// Retained messages are durable pointers, not copied payloads. If a legacy
	// snapshot supplied messages without entry ids, retain only the newer complete
	// updates whose original om.contemplator.message entries can be referenced.
	let lastMissingId = -1;
	for (let index = start; index < entryIds.length; index++) {
		if (entryIds[index] === undefined) lastMissingId = index;
	}
	if (lastMissingId >= 0) {
		start = lastMissingId + 1;
		while (start < history.length && history[start].role !== "user") start++;
	}
	return start > 0 && start < history.length ? start : undefined;
}

/** Pick a smaller complete oldest prefix after a length-truncated first attempt. */
function smallerPrefixEnd(history: readonly AgentMessage[], initialEnd: number): number | undefined {
	const target = Math.max(1, Math.floor(contemplatorHistoryTokens(history.slice(0, initialEnd)) / 2));
	let tokens = 0;
	for (let index = 0; index < initialEnd; index++) {
		tokens += contemplatorMessageTokens(history[index]);
		const next = index + 1;
		if (tokens >= target && next < initialEnd && history[next].role === "user") return next;
	}
	return undefined;
}

interface PendingUpdate {
	observations: string[];
	reviews: string[];
	mainAgentOutputTokens: number;
	mainAgentToolCalls: number;
	mainAgentActiveTimeMs: number;
}

type Intervention =
	| { kind: "probe"; question: string }
	| { kind: "review"; request: Omit<StructuralReviewRequest, "createdAt" | "requestedBy"> }
	| { kind: "none" };

type ReviewerSession = {
	scope: StructuralReviewRequest["scope"];
	history: AgentMessage[];
	/** Latest compact checkpoint; older transcript content stays in referenced append-only entries. */
	checkpointEntryId?: string;
	/** Reviewer message entries appended since checkpointEntryId. */
	messageEntryIds: string[];
	/** Message-entry ids already folded into `history` in this live session (restore re-walk guard). */
	foldedEntryIds: Set<string>;
};

type QueueStructuralReviewOptions = {
	ctx: MemoryUpdateCtx;
	requestArgs: Extract<Intervention, { kind: "review" }>["request"];
	branchEntries: Entry[];
	model: Model<any>;
	apiKey: string;
	headers: Record<string, string> | undefined;
	sessionGeneration: number;
};

type LaunchStructuralReviewOptions = {
	ctx: MemoryUpdateCtx;
	request: StructuralReviewRequest;
	model: Model<any>;
	apiKey: string;
	headers: Record<string, string> | undefined;
	sessionGeneration: number;
	key?: string;
	history?: AgentMessage[];
};

function mergeMemoryLines(existing: string[], incoming: string[]): string[] {
	const merged = [...existing];
	const seen = new Set(existing.map((line) => line.match(/^\[([^\]]+)\]/)?.[1] ?? line));
	for (const line of incoming) {
		const key = line.match(/^\[([^\]]+)\]/)?.[1] ?? line;
		if (seen.has(key)) continue;
		seen.add(key);
		merged.push(line);
	}
	return merged;
}

function reviewSummaryLine(review: ReviewResult): string {
	return review.outcome === "proposal"
		? `[${review.id}] ${review.scope} proposal: ${review.title} — ${review.summary}`
		: `[${review.id}] ${review.scope} review concluded with no proposal — ${review.reason}`;
}

function reviewRequestKey(request: RequestReviewArgs): string {
	return `${request.scope}:${hashId(`${request.evidence}\n${request.concern}`)}`;
}

function customMessageText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((block) => block && typeof block === "object" && "text" in block && typeof block.text === "string" ? block.text : "")
		.filter(Boolean)
		.join("\n");
}

const AGENT_TIME_BUCKET_MINUTES = 5;
export const CONTEMPLATOR_MAX_INVOCATIONS = 3;

function coarseAgentTime(durationMs: number): string {
	const totalMinutes = Math.floor(durationMs / 60_000);
	const bucketMinutes = Math.floor(totalMinutes / AGENT_TIME_BUCKET_MINUTES) * AGENT_TIME_BUCKET_MINUTES;
	if (bucketMinutes < AGENT_TIME_BUCKET_MINUTES) return `less than ${AGENT_TIME_BUCKET_MINUTES} minutes`;
	const hours = Math.floor(bucketMinutes / 60);
	const minutes = bucketMinutes % 60;
	if (hours === 0) return `about ${minutes} minutes`;
	const hourLabel = `${hours} hour${hours === 1 ? "" : "s"}`;
	return minutes === 0 ? `about ${hourLabel}` : `about ${hourLabel} ${minutes} minutes`;
}
const CONTEMPLATOR_MESSAGE = "om.contemplator.message";
const CONTEMPLATOR_STATE = "om.contemplator.state";
const CONTEMPLATOR_SUGGESTION = "om.contemplator.suggestion";
const REVIEW_PROPOSAL_MESSAGE = "om.review.proposal";
const SendProbeSchema = Type.Object({ question: Type.String({ minLength: 1, description: "One concise, memory-grounded probing question, optionally preceded by one short sentence of context. Cite relevant memory identifiers." }) });
const NoInterventionSchema = Type.Object({});
const ReviewScopeSchema = Type.Union([Type.Literal("workflow"), Type.Literal("software")]);
export const RequestReviewSchema = Type.Object({
	scope: ReviewScopeSchema,
	evidence: Type.String({ minLength: 1, description: "Memory-grounded evidence for the suspected recurring pattern." }),
	concern: Type.String({ minLength: 1, description: "Suspected structural concern stated as a possibility." }),
	review_focus: Type.String({ minLength: 1, description: "What the reviewer should determine without prescribing a solution." }),
	constraints: Type.Optional(Type.String({ minLength: 1, description: "Relevant user requirements, boundaries, or uncertainties." })),
});
type SendProbeArgs = Static<typeof SendProbeSchema>;
export type RequestReviewArgs = Static<typeof RequestReviewSchema>;

type InterventionWrite = { overwritten: boolean };
type ReviewWrite = InterventionWrite & { reviewRequestId: string };

function interventionResultText(options: {
	kind: "probe" | "review";
	memoryIds: string[];
	memoryExists: (id: string) => boolean;
	overwritten: boolean;
	queuedText: string;
}): string {
	const replacementTool = options.kind === "probe" ? "send_probe" : "request_review";
	const warnings = options.memoryIds
		.filter((id) => !options.memoryExists(id))
		.map((id) => `WARNING: memory ${id} not found; use search_memories and recall to find the correct memory, then call ${replacementTool} again to replace the ${options.kind} before it is sent.`);
	if (options.overwritten) warnings.push("WARNING: overwriting prior probe/review tool call; only one action may be taken per turn.");
	warnings.push(options.queuedText);
	return warnings.join("\n");
}

export function createSendProbeTool(
	onProbe: (question: string) => InterventionWrite,
	memoryExists: (id: string) => boolean = () => true,
): AgentTool<typeof SendProbeSchema> {
	return {
		name: "send_probe",
		label: "Send probe",
		description: "Send one concise, high-level probing question to the primary agent asynchronously. The message must contain one focused question, optionally preceded by one short sentence of context, and cite relevant memory identifiers. Do not use it for routine reminders, status updates, generic advice, direct task management, or a structural design deserving review. This is a terminal tool when all cited memory ids are valid; citation warnings leave the turn open so the action can be replaced. A later intervention call in the same turn replaces this one.",
		parameters: SendProbeSchema,
		execute: async (_toolCallId, params: SendProbeArgs) => {
			const question = params.question.trim();
			const write = onProbe(question);
			const memoryIds = memoryReferenceIds(question);
			debugLog("contemplator.tool_call", { tool: "send_probe", suggestionLength: question.length, memoryIds, overwritten: write.overwritten });
			return {
				content: [{ type: "text", text: interventionResultText({ kind: "probe", memoryIds, memoryExists, overwritten: write.overwritten, queuedText: "Probe will be delivered at the end of your turn." }) }],
				details: { queued: true, overwritten: write.overwritten, memoryIds },
			};
		},
	};
}

export function createNoInterventionTool(
	onNoIntervention: () => InterventionWrite,
): AgentTool<typeof NoInterventionSchema> {
	return {
		name: "no_intervention",
		label: "No intervention",
		description: "Terminally end this contemplator update without sending anything to the primary agent. This argument-free tool is the preferred default whenever no specific, grounded, materially useful intervention is clearly warranted or usefulness is uncertain. Never send a probe merely to avoid choosing no_intervention. A later final-action call in the same turn replaces an earlier warned action.",
		parameters: NoInterventionSchema,
		execute: async () => {
			const write = onNoIntervention();
			debugLog("contemplator.no_intervention", { overwritten: write.overwritten });
			const warning = write.overwritten ? "WARNING: overwriting prior probe/review/no_intervention tool call; only one final action may be taken per turn.\n" : "";
			return {
				content: [{ type: "text", text: `${warning}No intervention will be sent.` }],
				details: { selected: true, overwritten: write.overwritten },
			};
		},
	};
}

export function createRequestReviewTool(
	onReview: (request: RequestReviewArgs) => ReviewWrite,
	memoryExists: (id: string) => boolean = () => true,
): AgentTool<typeof RequestReviewSchema> {
	return {
		name: "request_review",
		label: "Request structural review",
		description: "Request a short-lived structural review grounded in cited memories. Use workflow for recurring problems in how work is performed and software for recurring problems in the product structure. Identify evidence, the suspected concern, review focus, and constraints without designing the solution. This is a terminal tool when all cited memory ids are valid; citation warnings leave the turn open so the action can be replaced. A later intervention call in the same turn replaces this one.",
		parameters: RequestReviewSchema,
		execute: async (_toolCallId, params: RequestReviewArgs) => {
			const request = { ...params, evidence: params.evidence.trim(), concern: params.concern.trim(), review_focus: params.review_focus.trim(), constraints: params.constraints?.trim() || undefined };
			const write = onReview(request);
			const memoryIds = memoryReferenceIds([request.evidence, request.concern, request.review_focus, request.constraints].filter((value): value is string => Boolean(value)).join("\n"));
			debugLog("contemplator.review_requested", { reviewRequestId: write.reviewRequestId, scope: request.scope, evidenceLength: request.evidence.length, concernLength: request.concern.length, memoryIds, overwritten: write.overwritten });
			const queuedText = `${request.scope === "workflow" ? "Workflow" : "Software"} review [${write.reviewRequestId}] will be started at the end of your turn.`;
			return {
				content: [{ type: "text", text: interventionResultText({ kind: "review", memoryIds, memoryExists, overwritten: write.overwritten, queuedText }) }],
				details: { queued: true, overwritten: write.overwritten, scope: request.scope, reviewRequestId: write.reviewRequestId, memoryIds },
			};
		},
	};
}

export class Contemplator {
	private history: AgentMessage[] = [];
	/** Ledger entry id for each private-history message; used by compact checkpoints to retain a suffix without copying it. */
	private historyEntryIds: Array<string | undefined> = [];
	private pending: PendingUpdate | undefined;
	private running = false;
	/** Invalidates stale/hard-timed-out flush finalizers across session changes. */
	private flushEpoch = 0;
	/** Bounds retries of one poisoned memory update so future updates can run. */
	private consecutiveFlushFailures = 0;
	private seenObservationIds = new Set<string>();
	private seenReviewIds = new Set<string>();
	private inFlightReviewKeys = new Set<string>();
	private inFlightReviewIds = new Set<string>();
	private resolvingReviewIds = new Set<string>();
	private resumedReviewIds = new Set<string>();
	private reviewerSessions = new Map<string, ReviewerSession>();
	private deliveredProbeIds = new Set<string>();
	/** Probe ids passed to pi.sendMessage by this live extension runtime. */
	private queuedProbeIds = new Set<string>();
	/** Probes whose provider-context delivery will establish the next response-spacing anchor. */
	private probeCooldownPendingIds = new Set<string>();
	private sessionGeneration = 0;
	private latestCtx: MemoryUpdateCtx | undefined;
	/** Completed primary-model responses since the current completion/probe-delivery spacing anchor. */
	private turnsSinceRun = 0;
	/** Used to avoid counting the final turn_end after its assistant message_end. */
	private assistantResponsesInCurrentTurn = 0;
	private restoredTipId: string | undefined;
	/** Start of the unpersisted portion of the current main-agent run. */
	private agentActiveSince: number | undefined;

	constructor(private readonly pi: ExtensionAPI, private readonly runtime: Runtime) {}

	register(): void {
		this.pi.registerMessageRenderer(CONTEMPLATOR_SUGGESTION, (message, _options, theme) => {
			const details = message.details as { question?: unknown } | undefined;
			const content = typeof details?.question === "string"
				? details.question
				: customMessageText(message.content)
					.replace(/^Background contemplator probe \(advisory\):\n?/, "")
					.replace(/\n\nReferenced memories can be reviewed using the recall tool\.\s*$/, "");
			const box = new Box(1, 1, (text) => theme.bg("customMessageBg", text));
			box.addChild(new Text(theme.fg("thinkingHigh", `${theme.bold("◆ CONTEMPLATOR PROBE")}\n${content}`), 0, 0));
			return box;
		});
		this.pi.registerMessageRenderer(REVIEW_PROPOSAL_MESSAGE, (message, _options, theme) => {
			const box = new Box(1, 1, (text) => theme.bg("customMessageBg", text));
			box.addChild(new Text(theme.fg("thinkingHigh", `${theme.bold("◆ CONTEMPLATOR REVIEW")}\n${customMessageText(message.content)}`), 0, 0));
			return box;
		});
		this.runtime.setMemoryUpdateListener((ctx) => this.withDebugContext(ctx, () => this.observeTurn(ctx)));
		this.pi.on("agent_start", () => {
			this.agentActiveSince = Date.now();
		});
		this.pi.on("agent_end", (_event: unknown, ctx: ExtensionContext) => {
			this.persistAgentActivity(ctx);
			this.agentActiveSince = undefined;
		});
		this.pi.on("session_start", (event: any, ctx: ExtensionContext) => {
			this.sessionGeneration++;
			this.flushEpoch++;
			this.running = false;
			this.consecutiveFlushFailures = 0;
			const generation = this.sessionGeneration;
			this.agentActiveSince = undefined;
			// AgentSession preserves its steering queue across extension reloads. An
			// undelivered tracking entry therefore still has a live queued message;
			// restoring it here would enqueue the same probe a second time.
			const reload = event?.reason === "reload";
			this.restore(ctx, true, reload, reload);
			// Reconstruct and schedule durable memory immediately. In particular, a
			// reload after a failed run must not silently mark its pending backlog seen.
			queueMicrotask(() => {
				if (generation === this.sessionGeneration) this.withDebugContext(ctx, () => this.observeTurn(ctx));
			});
		});
		this.pi.on("session_tree", (_event: any, ctx: ExtensionContext) => {
			this.sessionGeneration++;
			this.flushEpoch++;
			this.running = false;
			this.consecutiveFlushFailures = 0;
			this.agentActiveSince = undefined;
			// Pending steering messages remain queued while navigating the tree.
			this.restore(ctx, true, true);
		});
		this.pi.on("session_shutdown", () => {
			this.sessionGeneration++;
			this.flushEpoch++;
			this.running = false;
			this.consecutiveFlushFailures = 0;
			this.agentActiveSince = undefined;
			this.history = [];
			this.historyEntryIds = [];
			this.pending = undefined;
			this.seenObservationIds.clear();
			this.seenReviewIds.clear();
			this.inFlightReviewKeys.clear();
			this.inFlightReviewIds.clear();
			this.resolvingReviewIds.clear();
			this.resumedReviewIds.clear();
			this.reviewerSessions.clear();
			this.deliveredProbeIds.clear();
			this.queuedProbeIds.clear();
			this.probeCooldownPendingIds.clear();
			this.latestCtx = undefined;
			this.turnsSinceRun = 0;
			this.assistantResponsesInCurrentTurn = 0;
			this.restoredTipId = undefined;
			this.runtime.contemplatorState = {
				running: false,
				pendingObservations: 0,
				pendingReviews: 0,
				responsesSinceRun: 0,
				waitingFor: "idle",
			};
		});
		this.pi.on("session_compact", (_event: any, ctx: ExtensionContext) => {
			// In-flight invocation messages remain local to flush until its required
			// final action is selected, so history always contains only completed work.
			if (this.history.length > 0) {
				this.pi.appendEntry(CONTEMPLATOR_STATE, { version: 1, history: this.history });
				this.markTipPersisted(ctx);
				debugLog("contemplator.state_persisted", { historyMessageCount: this.history.length, running: this.running });
			}
			this.persistReviewerStates(ctx);
		});
		this.pi.on("message_end", (event: any, ctx: ExtensionContext) => {
			const message = event?.message;
			// A Pi turn can contain hours of assistant/tool/model rounds. Count each
			// completed primary-model response, not only the eventual turn_end, or the
			// contemplator can remain throttled forever during a long autonomous run.
			if (message?.role === "assistant") {
				this.persistAgentActivity(ctx);
				this.turnsSinceRun++;
				this.assistantResponsesInCurrentTurn++;
				this.withDebugContext(ctx, () => this.observeTurn(ctx));
			}
			if (message?.role !== "custom" || message.customType !== CONTEMPLATOR_SUGGESTION) return;
			if (typeof message.details?.probeId !== "string") return;
			// message_end means Pi has drained the steer into the conversation
			// stream. It is no longer protected by an in-memory queue, so a later
			// tree restore must be allowed to requeue it until context acknowledges it.
			this.queuedProbeIds.delete(message.details.probeId);
		});
		this.pi.on("tool_execution_end", (_event: unknown, ctx: ExtensionContext) => {
			// This records one wall-clock interval regardless of how many tools were
			// running concurrently; persistAgentActivity restarts the shared clock.
			this.persistAgentActivity(ctx);
		});
		this.pi.on("context", (event: any, ctx: ExtensionContext) => {
			const deliveredMessages = event.messages?.filter((message: any) => message?.role === "custom" && message.customType === CONTEMPLATOR_SUGGESTION && typeof message.details?.probeId === "string") ?? [];
			let cooldownAnchored = false;
			for (const delivered of deliveredMessages) {
				if (this.deliveredProbeIds.has(delivered.details.probeId)) continue;
				this.deliveredProbeIds.add(delivered.details.probeId);
				this.probeCooldownPendingIds.delete(delivered.details.probeId);
				cooldownAnchored = true;
				// Once Pi includes the probe in a provider context it is no longer in
				// either in-memory delivery queue. Keeping this id indefinitely caused
				// later tree restores to suppress a genuinely needed requeue.
				this.queuedProbeIds.delete(delivered.details.probeId);
				this.pi.appendEntry(CONTEMPLATOR_SUGGESTION, {
					version: 1,
					suggestion: typeof delivered.details.question === "string" ? delivered.details.question : String(delivered.content ?? ""),
					probeId: delivered.details.probeId,
					delivered: true,
				});
				this.markTipPersisted(ctx);
				debugLog("contemplator.suggestion_delivered", { probeId: delivered.details.probeId });
			}
			if (cooldownAnchored) {
				// Probe spacing begins only once Pi proves the probe reached an actual
				// provider context. Responses generated before this point do not count.
				this.turnsSinceRun = 0;
				this.withDebugContext(ctx, () => this.observeTurn(ctx));
			}
		});
		this.pi.on("turn_end", (_event: any, ctx: ExtensionContext) => {
			this.persistAgentActivity(ctx);
			// Normally message_end already counted the final assistant response. Keep a
			// one-response fallback for hosts/tests that emit turn_end without it.
			if (this.assistantResponsesInCurrentTurn === 0) this.turnsSinceRun++;
			this.assistantResponsesInCurrentTurn = 0;
			this.withDebugContext(ctx, () => this.observeTurn(ctx));
		});
	}

	private persistAgentActivity(ctx: MemoryUpdateCtx): void {
		const startedAt = this.agentActiveSince;
		if (startedAt === undefined) return;
		const endedAt = Date.now();
		this.agentActiveSince = endedAt;
		const durationMs = Math.max(0, endedAt - startedAt);
		if (durationMs === 0) return;
		this.pi.appendEntry(OM_AGENT_ACTIVITY, { version: 1, durationMs, endedAt });
		// Notify only after appendEntry so active-time schedulers always observe the
		// checkpoint, regardless of Pi's ordering between independent event handlers.
		this.runtime.notifyAgentActivity(ctx);
		debugLog("agent.activity_recorded", { durationMs });
	}

	private withDebugContext<T>(ctx: MemoryUpdateCtx, fn: () => T): T {
		this.runtime.ensureConfig(ctx.cwd);
		const sessionManager = ctx.sessionManager as { getSessionId?: () => string; getSessionFile?: () => string };
		return withDebugLogContext({
			enabled: this.runtime.config.debugLog === true,
			cwd: ctx.cwd,
			sessionId: sessionManager.getSessionId?.(),
			sessionFile: sessionManager.getSessionFile?.(),
		}, fn);
	}

	private publishState(
		waitingFor: typeof this.runtime.contemplatorState.waitingFor,
		overrides: Partial<typeof this.runtime.contemplatorState> = {},
	): void {
		this.runtime.contemplatorState = {
			...this.runtime.contemplatorState,
			running: this.running,
			pendingObservations: this.pending?.observations.length ?? 0,
			pendingReviews: this.pending?.reviews.length ?? 0,
			responsesSinceRun: this.turnsSinceRun,
			waitingFor,
			...overrides,
		};
	}

	private restore(ctx: MemoryUpdateCtx, resetTracking = false, retainQueuedIds = false, skipUndeliveredRestore = false): void {
		this.latestCtx = ctx;
		const entries = ctx.sessionManager.getBranch() as Entry[];
		const tipId = entries.at(-1)?.id;
		if (this.running && !resetTracking) return;
		if (tipId === this.restoredTipId && !resetTracking) return;
		this.history = [];
		this.historyEntryIds = [];
		const historyMessagesByEntryId = new Map<string, AgentMessage>();
		const checkpointCoveredObservationIds = new Set<string>();
		const checkpointCoveredReviewIds = new Set<string>();
		let resetProjection: ReturnType<typeof fullProjection> | undefined;
		if (resetTracking) {
			this.deliveredProbeIds.clear();
			this.probeCooldownPendingIds.clear();
			if (!retainQueuedIds) this.queuedProbeIds.clear();
			this.inFlightReviewIds.clear();
			this.resolvingReviewIds.clear();
			this.resumedReviewIds.clear();
			this.reviewerSessions.clear();
			resetProjection = fullProjection(entries);
			this.seenObservationIds.clear();
			this.seenReviewIds.clear();
			this.pending = undefined;
			this.turnsSinceRun = 0;
			this.assistantResponsesInCurrentTurn = 0;
			this.runtime.contemplatorState = {
				running: false,
				pendingObservations: 0,
				pendingReviews: 0,
				responsesSinceRun: 0,
				waitingFor: "idle",
			};
		}
		const undeliveredSuggestions = new Map<string, string>();
		for (const entry of entries) {
			if (entry.customType === CONTEMPLATOR_STATE && entry.data && typeof entry.data === "object") {
				const state = entry.data as { history?: unknown };
				if (Array.isArray(state.history)) {
					this.history = state.history.filter((message): message is AgentMessage => !!message && typeof message === "object");
					this.historyEntryIds = this.history.map(() => undefined);
				}
			}
			if (entry.customType === CONTEMPLATOR_MESSAGE && entry.data && typeof entry.data === "object") {
				const data = entry.data as { message?: unknown; compacted?: unknown; retainedMessageEntryIds?: unknown; coveredObservationIds?: unknown; coveredReviewIds?: unknown };
				const message = data.message;
				if (message && typeof message === "object") {
					const typedMessage = message as AgentMessage;
					historyMessagesByEntryId.set(entry.id, typedMessage);
					if (data.compacted === true) {
						const retainedIds = Array.isArray(data.retainedMessageEntryIds)
							? data.retainedMessageEntryIds.filter((id): id is string => typeof id === "string" && historyMessagesByEntryId.has(id))
							: [];
						this.history = [typedMessage, ...retainedIds.map((id) => historyMessagesByEntryId.get(id)!)];
						this.historyEntryIds = [entry.id, ...retainedIds];
						if (Array.isArray(data.coveredObservationIds)) for (const id of data.coveredObservationIds) if (typeof id === "string") checkpointCoveredObservationIds.add(id);
						if (Array.isArray(data.coveredReviewIds)) for (const id of data.coveredReviewIds) if (typeof id === "string") checkpointCoveredReviewIds.add(id);
					} else {
						this.history.push(typedMessage);
						this.historyEntryIds.push(entry.id);
					}
				}
			}
			if (entry.customType === OM_REVIEWER_STATE && entry.data && typeof entry.data === "object") {
				const state = entry.data as { version?: unknown; reviewRequestId?: unknown; scope?: unknown; history?: unknown; messageEntryIds?: unknown };
				if (typeof state.reviewRequestId === "string" && (state.scope === "workflow" || state.scope === "software")) {
					if (state.version === 1 && Array.isArray(state.history)) {
						// Backward compatibility for the old, expensive full-transcript snapshots.
						this.reviewerSessions.set(state.reviewRequestId, {
							scope: state.scope,
							history: state.history.filter((message): message is AgentMessage => !!message && typeof message === "object"),
							checkpointEntryId: entry.id,
							messageEntryIds: [],
							foldedEntryIds: new Set(),
						});
					} else if (state.version === 2 && Array.isArray(state.messageEntryIds) && state.messageEntryIds.every((id) => typeof id === "string")) {
						// V2 checkpoints contain references only. The referenced state/messages
						// have already been folded while walking this append-only branch.
						const session = this.reviewerSessions.get(state.reviewRequestId) ?? { scope: state.scope, history: [], messageEntryIds: [], foldedEntryIds: new Set() };
						session.checkpointEntryId = entry.id;
						session.messageEntryIds = [];
						this.reviewerSessions.set(state.reviewRequestId, session);
					}
				}
			}
			if (entry.customType === OM_REVIEWER_MESSAGE && entry.data && typeof entry.data === "object") {
				const data = entry.data as { reviewRequestId?: unknown; scope?: unknown; message?: unknown };
				if (typeof data.reviewRequestId === "string" && (data.scope === "workflow" || data.scope === "software") && data.message && typeof data.message === "object") {
					let session = this.reviewerSessions.get(data.reviewRequestId);
					if (!session) {
						session = { scope: data.scope, history: [], messageEntryIds: [], foldedEntryIds: new Set() };
						this.reviewerSessions.set(data.reviewRequestId, session);
					}
					// restore() re-walks the whole branch whenever the tip moves
					// (turn_end), so fold each message entry at most once per live session.
					if (!session.foldedEntryIds.has(entry.id)) {
						session.foldedEntryIds.add(entry.id);
						session.history.push(data.message as AgentMessage);
						session.messageEntryIds.push(entry.id);
					}
				}
			}
			if (entry.customType === CONTEMPLATOR_SUGGESTION && entry.data && typeof entry.data === "object") {
				const data = entry.data as { suggestion?: unknown; delivered?: unknown; probeId?: unknown };
				if (typeof data.probeId !== "string") continue;
				if (data.delivered === true) {
					this.deliveredProbeIds.add(data.probeId);
					undeliveredSuggestions.delete(data.probeId);
				} else if (typeof data.suggestion === "string") {
					undeliveredSuggestions.set(data.probeId, data.suggestion);
				}
			}
		}
		if (resetTracking && resetProjection) {
			// Successful update prompts and private-history compaction checkpoints are
			// the durable coverage record. Checkpoints carry ids from prompts replaced
			// by their summary; failed, unpersisted runs remain pending and retryable.
			const coveredIds = new Set<string>([...checkpointCoveredObservationIds, ...checkpointCoveredReviewIds]);
			for (const message of this.history) {
				if (message.role !== "user") continue;
				const text = customMessageText(message.content);
				if (!text.includes("NEW MEMORY UPDATE")) continue;
				for (const id of memoryReferenceIds(text)) coveredIds.add(id);
			}
			this.seenObservationIds = new Set(resetProjection.observations.filter((item) => coveredIds.has(item.id)).map((item) => item.id));
			this.seenReviewIds = new Set((resetProjection.reviews ?? []).filter((item) => coveredIds.has(item.id)).map((item) => item.id));
			const unprocessedObservations = resetProjection.observations.length - this.seenObservationIds.size;
			const unprocessedReviews = (resetProjection.reviews?.length ?? 0) - this.seenReviewIds.size;
			if (unprocessedReviews > 0 || unprocessedObservations >= this.runtime.config.contemplatorMinNewObservations) {
				this.turnsSinceRun = this.runtime.config.contemplatorMinTurns;
			}
		}
		this.restoredTipId = tipId;
		for (const [probeId, question] of undeliveredSuggestions) {
			// An undelivered durable probe remains the cooldown anchor even when Pi's
			// live queue survived an extension reload and must not be duplicated.
			this.probeCooldownPendingIds.add(probeId);
			// A durable custom_message proves only that Pi inserted the probe at some
			// point; it does not prove an in-memory queue still owns it, and compaction
			// may have removed it from active model context. Suppress requeue only for
			// ids this live extension instance still knows are queued.
			if (skipUndeliveredRestore || this.queuedProbeIds.has(probeId)) continue;
			this.queueProbe(ctx, question, "restore", probeId);
		}
		if (resetTracking) void this.resumePendingReviews(ctx);
	}

	private observeTurn(ctx: MemoryUpdateCtx): void {
		this.restore(ctx);
		this.runtime.ensureConfig(ctx.cwd);
		if (!this.runtime.config.contemplatorEnabled) {
			this.publishState("disabled");
			debugLog("contemplator.skipped", { reason: "disabled" });
			return;
		}
		if (this.runtime.config.passive) {
			this.publishState("passive");
			debugLog("contemplator.skipped", { reason: "passive" });
			return;
		}
		const branchEntries = ctx.sessionManager.getBranch() as Entry[];
		const observerBacklogTokens = rawTokensSinceObservationCoverage(branchEntries);
		const waitingForCapturedObserverBacklog = this.runtime.observerBacklogBlocking || (
			!this.runtime.consolidationInFlight &&
			observerBacklogTokens >= this.runtime.config.observeAfterTokens &&
			this.runtime.lastObserverError === undefined
		);
		if (waitingForCapturedObserverBacklog) {
			// A catch-up observer pipeline may append several partial batches while it
			// drains the finite source snapshot captured at launch. Do not feed those
			// fragments to the contemplator one at a time. Source appended concurrently
			// belongs to the next snapshot and must not extend this waiting period.
			this.publishState("observer");
			debugLog("contemplator.waiting", {
				reason: "observer_backlog",
				observerBacklogTokens,
				observerBacklogBlocking: this.runtime.observerBacklogBlocking,
				observeAfterTokens: this.runtime.config.observeAfterTokens,
			});
			return;
		}
		const projection = fullProjection(branchEntries);
		const observations = projection.observations.map((item) => `[${item.id}] ${item.content}`);
		const reviews = projection.reviews ?? [];
		const newObservationItems = projection.observations.filter((item) => !this.seenObservationIds.has(item.id));
		const newReviewItems = reviews.filter((item) => !this.seenReviewIds.has(item.id));
		const newObservations = newObservationItems.map((item) => `[${item.id}] ${item.content}`);
		const newReviews = newReviewItems.map(reviewSummaryLine);
		for (const item of newObservationItems) this.seenObservationIds.add(item.id);
		for (const item of newReviewItems) this.seenReviewIds.add(item.id);
		debugLog("contemplator.update", {
			observationCount: observations.length,
			newObservationCount: newObservations.length,
			newReviewCount: newReviews.length,
			turnsSinceRun: this.turnsSinceRun,
			pending: this.pending !== undefined,
			running: this.running,
		});
		if (newObservations.length > 0 || newReviews.length > 0) {
			this.pending = {
				observations: mergeMemoryLines(this.pending?.observations ?? [], newObservations),
				reviews: mergeMemoryLines(this.pending?.reviews ?? [], newReviews),
				mainAgentOutputTokens: assistantOutputTokens(branchEntries),
				mainAgentToolCalls: assistantToolCallCount(branchEntries),
				mainAgentActiveTimeMs: agentActiveTimeMs(branchEntries),
			};
		}
		if (!this.pending) {
			this.publishState(this.running ? "running" : this.probeCooldownPendingIds.size > 0 ? "probe" : "idle");
			return;
		}
		// Activity values are cumulative send-time snapshots, not values frozen when
		// the first memory entered a pending batch. This includes work performed
		// while that batch waits for its memory/turn thresholds.
		this.pending.mainAgentOutputTokens = assistantOutputTokens(branchEntries);
		this.pending.mainAgentToolCalls = assistantToolCallCount(branchEntries);
		this.pending.mainAgentActiveTimeMs = agentActiveTimeMs(branchEntries);
		const enoughMemories = this.pending.reviews.length > 0 || this.pending.observations.length >= this.runtime.config.contemplatorMinNewObservations;
		if (this.probeCooldownPendingIds.size > 0) {
			this.publishState("probe");
			debugLog("contemplator.waiting", { reason: "probe_delivery", pendingProbeCount: this.probeCooldownPendingIds.size });
			return;
		}
		if (!enoughMemories || this.turnsSinceRun < this.runtime.config.contemplatorMinTurns) {
			this.publishState(!enoughMemories ? "memories" : "responses");
			debugLog("contemplator.waiting", {
				enoughMemories,
				turnsSinceRun: this.turnsSinceRun,
				minTurns: this.runtime.config.contemplatorMinTurns,
				minNewObservations: this.runtime.config.contemplatorMinNewObservations,
			});
			return;
		}
		this.publishState(this.running ? "running" : "ready");
		debugLog("contemplator.triggered", {
			pendingObservationCount: this.pending.observations.length,
			pendingReviewCount: this.pending.reviews.length,
			turnsSinceRun: this.turnsSinceRun,
		});
		void this.flush(ctx);
	}

	private async flush(ctx: MemoryUpdateCtx): Promise<void> {
		if (this.running || !this.pending) {
			debugLog("contemplator.flush_skipped", { reason: this.running ? "already_running" : "no_pending_update" });
			return;
		}
		// Checkpoint a live main-agent interval before taking the activity snapshot.
		// persistAgentActivity restarts the running clock at Date.now(), so a later
		// turn/agent checkpoint records only the remainder.
		this.persistAgentActivity(ctx);
		const update = this.pending;
		const branchEntriesAtStart = ctx.sessionManager.getBranch() as Entry[];
		update.mainAgentOutputTokens = assistantOutputTokens(branchEntriesAtStart);
		update.mainAgentToolCalls = assistantToolCallCount(branchEntriesAtStart);
		update.mainAgentActiveTimeMs = agentActiveTimeMs(branchEntriesAtStart);
		this.pending = undefined;
		const turnsBeforeRun = this.turnsSinceRun;
		const sessionGeneration = this.sessionGeneration;
		const flushEpoch = ++this.flushEpoch;
		this.running = true;
		this.turnsSinceRun = 0;
		const startedAt = Date.now();
		let failed = false;
		let failureMessage: string | undefined;
		let workerNotified = false;
		let promptPersisted = false;
		let emittedProbeId: string | undefined;
		let workerWatchdog: ReturnType<typeof createWorkerStallWatchdog> | undefined;
		this.publishState("running", { lastStartedAt: startedAt, lastError: undefined });
		debugLog("contemplator.start", {
			newObservationCount: update.observations.length,
			newReviewCount: update.reviews.length,
			historyMessageCount: this.history.length,
		});
		try {
			const resolved = await this.runtime.resolveModel({
				model: ctx.model,
				modelRegistry: ctx.modelRegistry,
				hasUI: ctx.hasUI,
				ui: ctx.ui,
				configuredModel: this.runtime.config.contemplatorModel ?? null,
			});
			if (!resolved.ok) {
				failed = true;
				failureMessage = resolved.reason;
				debugLog("contemplator.model_unavailable", { reason: resolved.reason });
				if (sessionGeneration === this.sessionGeneration) {
					this.consecutiveFlushFailures++;
					if (this.consecutiveFlushFailures < 2) {
						const pending = this.pending as PendingUpdate | undefined;
						this.pending = {
							observations: mergeMemoryLines(pending?.observations ?? [], update.observations),
							reviews: mergeMemoryLines(pending?.reviews ?? [], update.reviews),
							mainAgentOutputTokens: update.mainAgentOutputTokens,
							mainAgentToolCalls: update.mainAgentToolCalls,
							mainAgentActiveTimeMs: update.mainAgentActiveTimeMs,
						};
					} else {
						debugLog("contemplator.poisoned_update_released", { reason: resolved.reason, observationCount: update.observations.length, reviewCount: update.reviews.length });
						this.consecutiveFlushFailures = 0;
					}
					this.turnsSinceRun = 0; // Back off until fresh primary responses arrive; never retry every checkpoint.
				}
				return;
			}
			if (sessionGeneration !== this.sessionGeneration) {
				debugLog("contemplator.flush_stale", { reason: "session_changed" });
				return;
			}
			const selectedModel = resolved.model as { provider?: unknown; id?: unknown; contextWindow?: unknown };
			debugLog("contemplator.model_resolved", {
				provider: selectedModel.provider,
				modelId: selectedModel.id,
				contextWindow: selectedModel.contextWindow,
			});
			workerWatchdog = createWorkerStallWatchdog("contemplator");
			if (this.runtime.config.showWorkerNotifications && ctx.hasUI) {
				ctx.ui?.notify("pi-contemplator: contemplator running", "info");
				workerNotified = true;
			}
			const reviewerEnabled = this.runtime.config.reviewerEnabled;
			const updateSections: string[] = [];
			if (update.observations.length > 0) updateSections.push(`OBSERVATIONS:\n${update.observations.join("\n")}`);
			if (update.reviews.length > 0) updateSections.push(`REVIEWS:\n${update.reviews.join("\n")}`);
			const updateBody = updateSections.length > 0 ? updateSections.join("\n\n") : "(no new memories)";
			const finalActionNames = reviewerEnabled
				? "send_probe, request_review, or no_intervention"
				: "send_probe or no_intervention";
			const interventionInstruction = reviewerEnabled
				? `You must end this update by calling exactly one final-action tool: ${finalActionNames}. The tool requirement is bookkeeping, not a reason to intervene. Prefer the argument-free no_intervention whenever no specific, grounded, materially useful intervention is clearly warranted or usefulness is uncertain. Use send_probe only for one unusually useful focused question, and request_review only when a deeper workflow or software review is justified. Never send a probe merely to satisfy the final-action requirement. If a tool warns about a bad memory citation, use search_memories and recall, then call a final-action tool again to replace it.`
				: `You must end this update by calling exactly one final-action tool: ${finalActionNames}. The tool requirement is bookkeeping, not a reason to intervene. Prefer the argument-free no_intervention whenever no specific, grounded, materially useful probe is clearly warranted or usefulness is uncertain. Use send_probe only for one unusually useful focused question. Never send a probe merely to satisfy the final-action requirement. If send_probe warns about a bad memory citation, use search_memories and recall, then call a final-action tool again to replace it.`;
			const prompt: Message = { role: "user", content: [{ type: "text", text: `NEW MEMORY UPDATE\n\n${updateBody}\n\nCUMULATIVE ACTIVITY: ${update.mainAgentOutputTokens} generated tokens; ${update.mainAgentToolCalls} tool calls; ${coarseAgentTime(update.mainAgentActiveTimeMs)} active.\n\nConsider these updates in the context of the accumulated memories. Prioritize reasoning gaps, contradictions, user-intent alignment, relevant overlooked alternatives, well-supported loops, and recurring structural patterns. ${interventionInstruction}` }], timestamp: Date.now() };
			let intervention: Intervention | undefined;
			let finalActionWarned = false;
			const branchEntries = ctx.sessionManager.getBranch() as Entry[];
			const getBranch = () => branchEntries;
			const searchMemoriesTool = createSearchMemoriesAgentTool(getBranch);
			const recallTool = createRecallAgentTool(getBranch);
			const memoryExists = (id: string) => {
				const exists = recallMemorySources(branchEntries, id).status === "found";
				if (!exists) finalActionWarned = true;
				return exists;
			};
			const sendProbe = createSendProbeTool((question) => {
				const overwritten = intervention !== undefined;
				finalActionWarned = false;
				intervention = { kind: "probe", question };
				return { overwritten };
			}, memoryExists);
			const noIntervention = createNoInterventionTool(() => {
				const overwritten = intervention !== undefined;
				finalActionWarned = false;
				intervention = { kind: "none" };
				return { overwritten };
			});
			const tools: AgentTool<any>[] = [searchMemoriesTool as AgentTool<any>, recallTool as AgentTool<any>, sendProbe as AgentTool<any>, noIntervention as AgentTool<any>];
			if (reviewerEnabled) {
				const requestReview = createRequestReviewTool((request) => {
					const overwritten = intervention !== undefined;
					finalActionWarned = false;
					const reviewRequestId = `review-${Date.now().toString(36)}-${Math.random().toString(16).slice(2, 8)}`;
					intervention = { kind: "review", request: {
						id: reviewRequestId,
						scope: request.scope,
						evidence: request.evidence,
						concern: request.concern,
						reviewFocus: request.review_focus,
						constraints: request.constraints,
					} };
					return { reviewRequestId, overwritten };
				}, memoryExists);
				tools.push(requestReview as AgentTool<any>);
			}
			const selectedThinkingLevel = this.runtime.config.contemplatorModel?.thinking ?? this.runtime.config.model?.thinking ?? "medium";
			const supportsReasoning = (resolved.model as { reasoning?: unknown }).reasoning === true;
			const config: AgentLoopConfig & { onPayload?: (payload: unknown) => unknown } = {
				model: resolved.model as Model<any>,
				apiKey: resolved.apiKey,
				headers: resolved.headers,
				maxTokens: boundedMaxTokens(resolved.model as Model<any>, AGENT_LOOP_MAX_TOKENS),
				convertToLlm: (messages) => messages as Message[],
				toolExecution: "sequential",
				// A clean final-action call is the end of the contemplator turn. Do not
				// spend another model request asking it to narrate after its decision.
				// Citation warnings leave the loop open so it can correct the action.
				shouldStopAfterTurn: () => intervention !== undefined && !finalActionWarned,
				...(supportsReasoning && selectedThinkingLevel !== "off" ? { reasoning: selectedThinkingLevel } : {}),
			};
			const runMessages: AgentMessage[] = [];
			let nextPrompt = prompt;
			for (let invocation = 1; invocation <= CONTEMPLATOR_MAX_INVOCATIONS && !intervention; invocation++) {
				runMessages.push(nextPrompt);
				const context: AgentContext = { systemPrompt: buildContemplatorSystemPrompt(reviewerEnabled), messages: [...this.history, ...runMessages.slice(0, -1)], tools };
				const api = (resolved.model as Model<any>).api;
				const invocationConfig: AgentLoopConfig & { onPayload?: (payload: unknown) => unknown } = invocation === 1 ? config : {
					...config,
					onPayload: (payload) => forceRequiredToolPayload(payload, api),
				};
				// SimpleStreamOptions 0.84.3 types provider-neutral choice as auto/none,
				// while individual provider APIs also support required/any. Preserve the
				// runtime hint and final-payload enforcement without weakening base types.
				if (invocation > 1) (invocationConfig as any).toolChoice = requiredToolChoice(api);
				workerWatchdog.progress();
				const stream = agentLoop([nextPrompt], context, invocationConfig, workerWatchdog.signal, streamSimple);
				const result = await workerWatchdog.race((async () => {
					for await (const event of stream) {
						workerWatchdog!.progress();
						logAgentStreamError("contemplator", event);
					}
					return stream.result();
				})());
				// agentLoop returns its input prompt as the first new message. We already
				// added nextPrompt above, so do not duplicate each update in the durable
				// contemplator history or in a subsequent retry's context.
				const returnedMessages = result[0] === nextPrompt ? result.slice(1) : result;
				runMessages.push(...returnedMessages);
				// The LLM call happened and was billed regardless of what we do next.
				for (const message of result) {
					if (message.role === "assistant" && message.usage) this.runtime.recordAgentUsage(message.usage);
				}
				const assistant = [...result].reverse().find((message) => message.role === "assistant");
				debugLog("contemplator.result", {
					invocation,
					messageCount: result.length,
					assistantFound: assistant !== undefined,
					assistantStopReason: assistant && "stopReason" in assistant ? assistant.stopReason : undefined,
					intervention: (intervention as Intervention | undefined)?.kind,
				});
				if (assistant && "stopReason" in assistant && (assistant.stopReason === "error" || assistant.stopReason === "aborted")) {
					const errorMessage = "errorMessage" in assistant && typeof assistant.errorMessage === "string"
						? assistant.errorMessage
						: `Contemplator model ${assistant.stopReason}`;
					throw new Error(errorMessage);
				}
				if (!intervention && invocation < CONTEMPLATOR_MAX_INVOCATIONS) {
					nextPrompt = { role: "user", content: [{ type: "text", text: `You stopped without selecting a final action. If stopping meant that no intervention was clearly warranted, call the argument-free no_intervention tool now; that is the preferred default, and no explanation is required. Do not invent or send a probe merely to satisfy the tool requirement. Use send_probe only for a specific, memory-grounded question that is materially likely to improve the primary agent's reasoning${reviewerEnabled ? ", and request_review only for a well-supported recurring structural concern" : ""}. Call one final-action tool now: ${finalActionNames}. search_memories and recall do not satisfy this requirement.` }], timestamp: Date.now() };
				}
			}
			if (!intervention) throw new Error(`Contemplator stopped ${CONTEMPLATOR_MAX_INVOCATIONS} times without calling a final-action tool`);
			if (sessionGeneration === this.sessionGeneration) {
				// Keep the durable contemplator history compact: prompts and assistant
				// decisions are sufficient to resume its reasoning. Tool-result bodies
				// are available within this run but are not copied into the ledger.
				for (const message of runMessages) {
					if (message.role !== "user" && message.role !== "assistant") continue;
					this.history.push(message);
					this.historyEntryIds.push(this.appendContemplatorHistoryEntry(ctx, { version: 1, message }));
					promptPersisted = true;
					this.markTipPersisted(ctx);
				}
			}
			if (intervention?.kind === "probe" && sessionGeneration === this.sessionGeneration) emittedProbeId = this.queueProbe(ctx, intervention.question, "send_probe");
			if (intervention?.kind === "review" && this.runtime.config.reviewerEnabled && sessionGeneration === this.sessionGeneration) {
				const reviewerModel = await this.runtime.resolveModel({
					model: ctx.model,
					modelRegistry: ctx.modelRegistry,
					hasUI: ctx.hasUI,
					ui: ctx.ui,
					configuredModel: this.runtime.config.reviewerModel ?? null,
				});
				if (!reviewerModel.ok) {
					debugLog("reviewer.model_unavailable", { reason: reviewerModel.reason });
				} else {
					this.queueStructuralReview({
						ctx,
						requestArgs: intervention.request,
						branchEntries,
						model: reviewerModel.model as Model<any>,
						apiKey: reviewerModel.apiKey,
						headers: reviewerModel.headers,
						sessionGeneration,
					});
				}
			}
			if (sessionGeneration === this.sessionGeneration) {
				workerWatchdog.progress();
				try {
					await workerWatchdog.race(this.compactHistory(ctx, resolved.model as Model<any>, resolved.apiKey, resolved.headers, sessionGeneration, flushEpoch, workerWatchdog.signal));
				} catch (compactionError) {
					// The intervention and its durable messages are already complete. Private
					// history maintenance must never relabel that successful work as failed.
					debugLog("contemplator.compaction_postponed", { reason: compactionError instanceof Error ? compactionError.message : String(compactionError) });
				}
			}
		} catch (error) {
			failed = true;
			failureMessage = error instanceof Error ? error.message : String(error);
			debugLog("contemplator.error", { errorMessage: failureMessage });
			if (sessionGeneration === this.sessionGeneration && !promptPersisted) {
				this.consecutiveFlushFailures++;
				if (this.consecutiveFlushFailures < 2) {
					const pending = this.pending as PendingUpdate | undefined;
					this.pending = {
						observations: mergeMemoryLines(pending?.observations ?? [], update.observations),
						reviews: mergeMemoryLines(pending?.reviews ?? [], update.reviews),
						mainAgentOutputTokens: update.mainAgentOutputTokens,
						mainAgentToolCalls: update.mainAgentToolCalls,
						mainAgentActiveTimeMs: update.mainAgentActiveTimeMs,
					};
				} else {
					debugLog("contemplator.poisoned_update_released", { reason: failureMessage, observationCount: update.observations.length, reviewCount: update.reviews.length });
					this.consecutiveFlushFailures = 0;
				}
				this.turnsSinceRun = 0; // Back off until fresh primary responses arrive; never retry every checkpoint.
			}
		} finally {
			workerWatchdog?.dispose();
			if (flushEpoch !== this.flushEpoch) return;
			this.running = false;
			if (!failed) this.consecutiveFlushFailures = 0;
			// Normal runs establish their spacing anchor at completion. A probe run
			// instead anchors at provider-context delivery: if delivery already occurred
			// during this run, retain responses counted since it; otherwise the pending
			// probe gate blocks launches until the context event resets the counter.
			if (emittedProbeId === undefined) this.turnsSinceRun = 0;
			const waitingForProbe = this.probeCooldownPendingIds.size > 0;
			const pendingHasEnoughMemories = this.pending !== undefined && (
				this.pending.reviews.length > 0 ||
				this.pending.observations.length >= this.runtime.config.contemplatorMinNewObservations
			);
			const waitingFor = waitingForProbe
				? "probe"
				: !this.pending
					? "idle"
				: !pendingHasEnoughMemories
					? "memories"
					: this.turnsSinceRun < this.runtime.config.contemplatorMinTurns
						? "responses"
						: "ready";
			this.publishState(waitingFor, {
				lastCompletedAt: Date.now(),
				lastError: failureMessage,
			});
			debugLog("contemplator.complete", {
				durationMs: Date.now() - startedAt,
				historyMessageCount: this.history.length,
				pendingUpdate: this.pending !== undefined,
			});
			if (workerNotified && sessionGeneration === this.sessionGeneration) {
				ctx.ui?.notify(
					failed ? `pi-contemplator: contemplator failed — ${failureMessage ?? "unknown error"}` : "pi-contemplator: contemplator completed",
					failed ? "warning" : "info",
				);
			}
			if (!failed && sessionGeneration === this.sessionGeneration && this.pending) this.observeTurn(ctx);
		}
	}

	private queueProbe(ctx: MemoryUpdateCtx, question: string, source: "send_probe" | "restore", existingProbeId?: string): string {
		const probeId = existingProbeId ?? `${Date.now().toString(36)}-${Math.random().toString(16).slice(2, 8)}`;
		// Persist intent before touching Pi's in-memory queue. A crash in between
		// leaves a recoverable pending probe rather than an invisible lost one.
		this.pi.appendEntry(CONTEMPLATOR_SUGGESTION, { version: 1, suggestion: question, delivered: false, source, probeId });
		this.markTipPersisted(ctx);
		// DELIVERY INVARIANT: probes must always use steer. The contemplator is
		// designed for agents that run for hours; a probe must be injected after
		// the current tool-call batch, before the very next model request. Never
		// change this to nextTurn: that can postpone delivery until a user prompt.
		// IMPORTANT: omit triggerTurn entirely. Pi 0.84+ interprets an explicit
		// triggerTurn:false as "do not queue while streaming" and inserts directly
		// into agent.state, outside the active run's context snapshot. Omitting it
		// still does not start a turn while idle, but allows steer to work in-run.
		// Whether Pi is currently running or idle, sendMessage owns this probe in an
		// in-memory steer queue until message_end drains it. Track both cases so an
		// unrelated observer update or compaction callback cannot restore and enqueue
		// a duplicate while the original idle steer is still pending.
		this.queuedProbeIds.add(probeId);
		this.probeCooldownPendingIds.add(probeId);
		this.pi.sendMessage({
			customType: CONTEMPLATOR_SUGGESTION,
			content: `Background contemplator probe (advisory):\n${question}\n\nReferenced memories can be reviewed using the recall tool.`,
			display: this.runtime.config.showContemplatorMessages,
			details: { version: 1, question, source, probeId },
		}, { deliverAs: "steer" });
		debugLog("contemplator.suggestion_queued", {
			probeId,
			suggestionLength: question.length,
			delivery: "pi.sendMessage",
			deliverAs: "steer",
			triggerTurn: "omitted",
			source,
		});
		return probeId;
	}

	private queueStructuralReview(options: QueueStructuralReviewOptions): void {
		const { ctx, requestArgs, branchEntries, model, apiKey, headers, sessionGeneration } = options;
		const requestForKey: RequestReviewArgs = { scope: requestArgs.scope, evidence: requestArgs.evidence, concern: requestArgs.concern, review_focus: requestArgs.reviewFocus, constraints: requestArgs.constraints };
		const key = reviewRequestKey(requestForKey);
		const duplicateRequest = branchEntries.some((entry) => isReviewRequestEntry(entry) && reviewRequestKey({ scope: entry.data.request.scope, evidence: entry.data.request.evidence, concern: entry.data.request.concern, review_focus: entry.data.request.reviewFocus, constraints: entry.data.request.constraints }) === key);
		if (this.inFlightReviewKeys.has(key) || duplicateRequest) {
			debugLog("contemplator.review_coalesced", { scope: requestArgs.scope, key });
			return;
		}
		const request: StructuralReviewRequest = { ...requestArgs, id: requestArgs.id, createdAt: Date.now(), requestedBy: "contemplator" };
		this.pi.appendEntry(OM_REVIEW_REQUEST, { version: 1, request });
		this.markTipPersisted(ctx);
		this.launchStructuralReview({ ctx, request, model, apiKey, headers, sessionGeneration, key });
	}

	private launchStructuralReview(options: LaunchStructuralReviewOptions): boolean {
		const { ctx, request, model, apiKey, headers, sessionGeneration, key } = options;
		if (this.runtime.reviewInFlight || this.inFlightReviewIds.has(request.id)) return false;
		// Do not spin a no-progress reviewer repeatedly in one live session. The
		// request stays pending and a later session/tree restoration resumes it.
		this.resumedReviewIds.add(request.id);
		this.inFlightReviewIds.add(request.id);
		if (key) this.inFlightReviewKeys.add(key);
		const session = this.reviewerSessions.get(request.id) ?? { scope: request.scope, history: options.history ?? [], messageEntryIds: [], foldedEntryIds: new Set() };
		this.reviewerSessions.set(request.id, session);
		const task = this.runtime.launchReviewTask(ctx, async () => {
			const watchdog = createWorkerStallWatchdog("structural reviewer");
			let acceptsMessages = true;
			try {
				debugLog("reviewer.started", { reviewRequestId: request.id, scope: request.scope, resumed: session.history.length > 0 });
				const result = await watchdog.race(runStructuralReview({
					request, model, apiKey, headers,
					signal: watchdog.signal,
					onProgress: watchdog.progress,
					getBranch: () => ctx.sessionManager.getBranch() as Entry[],
					recordUsage: (usage) => this.runtime.recordAgentUsage(usage),
					history: session.history,
					onMessages: (messages) => {
						watchdog.progress();
						if (!acceptsMessages || sessionGeneration !== this.sessionGeneration || !this.reviewIsPending(ctx, request.id)) return;
						for (const message of messages) {
							session.history.push(message);
							const entryId = this.appendEntryWithId(ctx, OM_REVIEWER_MESSAGE, { version: 1, reviewRequestId: request.id, scope: request.scope, message }, request.id);
							if (entryId) {
								session.foldedEntryIds.add(entryId);
								session.messageEntryIds.push(entryId);
							}
						}
					},
				}));
				if (sessionGeneration !== this.sessionGeneration) {
					debugLog("reviewer.failed", { reviewRequestId: request.id, reason: "session_changed" });
					return;
				}
				if (!this.reviewIsPending(ctx, request.id)) {
					debugLog("reviewer.failed", { reviewRequestId: request.id, reason: "request_no_longer_pending" });
					return;
				}
				if (!result) {
					debugLog("reviewer.incomplete", { reviewRequestId: request.id, reason: "no_terminal_tool_call" });
					return;
				}
				this.pi.appendEntry(OM_REVIEW_RESULT, { result });
				this.reviewerSessions.delete(request.id);
				this.markTipPersisted(ctx);
				debugLog(result.outcome === "proposal" ? "reviewer.proposal_created" : "reviewer.no_proposal", { reviewRequestId: request.id, reviewMemoryId: result.id, scope: result.scope });
				if (result.outcome === "proposal") {
					const notice = `BACKGROUND ${result.scope.toUpperCase()} REVIEW PROPOSAL [${result.id}]\n\nIMPORTANT: This is only an abbreviated notice, not the full proposal. When this proposal is relevant to the current work, call the recall tool with memory id [${result.id}] before deciding whether or how to act on it.\n\nSUMMARY:\n${result.summary}\n\nThis is advisory. After recalling memory [${result.id}], evaluate the full proposal against the actual environment and current work.`;
					// As with probes, triggerTurn must be omitted or Pi 0.84+ bypasses
					// the streaming steer queue and the active run never sees this message.
					this.pi.sendMessage({ customType: REVIEW_PROPOSAL_MESSAGE, content: notice, display: this.runtime.config.showContemplatorMessages, details: { version: 1, reviewRequestId: request.id, reviewMemoryId: result.id, scope: result.scope } }, { deliverAs: "steer" });
					this.pi.appendEntry(OM_REVIEWER_NOTICE, { version: 1, reviewRequestId: request.id, reviewMemoryId: result.id, scope: result.scope, content: notice });
					this.markTipPersisted(ctx);
					debugLog("reviewer.primary_notice_queued", { reviewRequestId: request.id, reviewMemoryId: result.id });
				}
				this.runtime.notifyMemoryUpdate(ctx);
			} finally {
				acceptsMessages = false;
				watchdog.dispose();
				this.inFlightReviewIds.delete(request.id);
				if (key) this.inFlightReviewKeys.delete(key);
			}
		});
		if (!task) {
			this.inFlightReviewIds.delete(request.id);
			this.resumedReviewIds.delete(request.id);
			if (key) this.inFlightReviewKeys.delete(key);
			return false;
		}
		// Runtime clears reviewInFlight in its own finally before this continuation,
		// so the next persisted pending request can start without overlap.
		void task.then(() => {
			const resumeCtx = this.latestCtx;
			if (resumeCtx) void this.resumePendingReviews(resumeCtx);
		});
		return true;
	}

	private reviewIsPending(ctx: MemoryUpdateCtx, reviewRequestId: string): boolean {
		const entries = ctx.sessionManager.getBranch() as Entry[];
		return entries.some((entry) => isReviewRequestEntry(entry) && entry.data.request.id === reviewRequestId)
			&& !entries.some((entry) => isReviewResultEntry(entry) && entry.data.result.reviewRequestId === reviewRequestId);
	}

	private persistReviewerStates(ctx: MemoryUpdateCtx): void {
		for (const [reviewRequestId, session] of this.reviewerSessions) {
			if (session.messageEntryIds.length === 0) continue;
			const checkpointEntryId = this.appendEntryWithId(ctx, OM_REVIEWER_STATE, {
				version: 2,
				reviewRequestId,
				scope: session.scope,
				previousStateEntryId: session.checkpointEntryId,
				messageEntryIds: session.messageEntryIds,
			}, reviewRequestId);
			if (checkpointEntryId) session.checkpointEntryId = checkpointEntryId;
			session.messageEntryIds = [];
		}
	}

	private async resumePendingReviews(ctx: MemoryUpdateCtx): Promise<void> {
		if (!this.runtime.config.reviewerEnabled || this.runtime.config.passive || this.runtime.reviewInFlight) return;
		const entries = ctx.sessionManager.getBranch() as Entry[];
		const completed = new Set(entries.filter(isReviewResultEntry).map((entry) => entry.data.result.reviewRequestId));
		const request = entries.filter(isReviewRequestEntry).map((entry) => entry.data.request).find((item) => !completed.has(item.id) && !this.resumedReviewIds.has(item.id) && !this.inFlightReviewIds.has(item.id) && !this.resolvingReviewIds.has(item.id));
		if (!request) return;
		this.resolvingReviewIds.add(request.id);
		const generation = this.sessionGeneration;
		try {
			const resolved = await this.runtime.resolveModel({ model: ctx.model, modelRegistry: ctx.modelRegistry, hasUI: ctx.hasUI, ui: ctx.ui, configuredModel: this.runtime.config.reviewerModel ?? null });
			if (!resolved.ok || generation !== this.sessionGeneration) {
				debugLog("reviewer.resume_skipped", { reviewRequestId: request.id, reason: resolved.ok ? "session_changed" : resolved.reason });
				return;
			}
			this.launchStructuralReview({ ctx, request, model: resolved.model as Model<any>, apiKey: resolved.apiKey, headers: resolved.headers, sessionGeneration: generation, history: this.reviewerSessions.get(request.id)?.history ?? [] });
		} finally {
			this.resolvingReviewIds.delete(request.id);
		}
	}

	/**
	 * Append one private-history entry and attribute its synchronous ledger id.
	 * Pi does not return the id from appendEntry(), so prefer the exact data
	 * object and accept a cloned payload only when exactly one matching entry was
	 * appended. Ambiguity returns undefined rather than pointing at another
	 * writer's message.
	 */
	private appendContemplatorHistoryEntry(ctx: MemoryUpdateCtx, data: Record<string, unknown>): string | undefined {
		const branch = ctx.sessionManager.getBranch() as readonly Entry[];
		const before = branch.length;
		this.pi.appendEntry(CONTEMPLATOR_MESSAGE, data);
		const candidates = (ctx.sessionManager.getBranch() as readonly Entry[])
			.slice(before)
			.filter((entry) => entry.customType === CONTEMPLATOR_MESSAGE);
		return candidates.find((entry) => entry.data === data)?.id
			?? (candidates.length === 1 ? candidates[0].id : undefined);
	}

	private markTipPersisted(ctx: MemoryUpdateCtx): string | undefined {
		this.restoredTipId = (ctx.sessionManager.getBranch() as Entry[]).at(-1)?.id;
		return this.restoredTipId;
	}

	/**
	 * Append a custom entry and return the id of the entry just written, or
	 * undefined when it cannot be attributed. pi.appendEntry() does not return
	 * the entry id, so the branch is diffed around the synchronous append and the
	 * candidate is matched by customType + reviewRequestId. This never credits a
	 * concurrently-appended foreign entry reached via the branch tail. A failed
	 * attribution is harmless: restore rebuilds transcripts from the durable
	 * om.reviewer.message entries themselves and only skips a checkpoint.
	 */
	private appendEntryWithId(ctx: MemoryUpdateCtx, customType: string, data: Record<string, unknown>, reviewRequestId: string): string | undefined {
		const branch = ctx.sessionManager.getBranch() as readonly Entry[];
		const before = branch.length;
		this.pi.appendEntry(customType, data);
		const added = (ctx.sessionManager.getBranch() as readonly Entry[]).slice(before);
		const own = added.find((entry) => entry.customType === customType && (entry.data as { reviewRequestId?: unknown } | undefined)?.reviewRequestId === reviewRequestId);
		return own?.id;
	}

	private async compactHistory(ctx: MemoryUpdateCtx, model: Model<any>, apiKey: string, headers: Record<string, string> | undefined, sessionGeneration: number, flushEpoch: number, signal?: AbortSignal): Promise<void> {
		const history = this.history.slice();
		const historyEntryIds = this.historyEntryIds.slice();
		const historyTokens = contemplatorHistoryTokens(history);
		const triggerTokens = contemplatorHistoryTriggerTokens(model);
		if (history.length < 12 || historyTokens < triggerTokens) return;

		const keepRecentTokens = Math.min(CONTEMPLATOR_HISTORY_KEEP_RECENT_TOKENS, Math.max(1, Math.floor(triggerTokens * 0.5)));
		const initialPrefixEnd = recentHistoryStart(history, historyEntryIds, keepRecentTokens);
		if (initialPrefixEnd === undefined) return;
		const previousMessageCount = history.length;
		debugLog("contemplator.compaction_start", {
			historyMessageCount: previousMessageCount,
			historyTokens,
			triggerTokens,
			keepRecentTokens,
			prefixMessageCount: initialPrefixEnd,
			retainedMessageCount: history.length - initialPrefixEnd,
		});

		const summarizePrefix = async (prefixEnd: number) => generateSummaryWithUsage(
			history.slice(0, prefixEnd),
			model,
			CONTEMPLATOR_HISTORY_SUMMARY_RESERVE_TOKENS,
			apiKey,
			headers,
			signal,
			CONTEMPLATOR_HISTORY_SUMMARY_INSTRUCTIONS,
		);

		let prefixEnd = initialPrefixEnd;
		let summaryWithUsage: Awaited<ReturnType<typeof generateSummaryWithUsage>>;
		try {
			summaryWithUsage = await summarizePrefix(prefixEnd);
		} catch (error) {
			const failure = error instanceof Error ? error.message : String(error);
			const fallbackEnd = /generation hit the token cap/i.test(failure)
				? smallerPrefixEnd(history, initialPrefixEnd)
				: undefined;
			if (fallbackEnd === undefined) {
				debugLog("contemplator.compaction_postponed", { reason: failure, historyTokens, prefixMessageCount: prefixEnd });
				return;
			}
			prefixEnd = fallbackEnd;
			debugLog("contemplator.compaction_retry_smaller_prefix", { reason: failure, prefixMessageCount: prefixEnd, retainedMessageCount: history.length - prefixEnd });
			try {
				summaryWithUsage = await summarizePrefix(prefixEnd);
			} catch (fallbackError) {
				debugLog("contemplator.compaction_postponed", {
					reason: fallbackError instanceof Error ? fallbackError.message : String(fallbackError),
					historyTokens,
					prefixMessageCount: prefixEnd,
				});
				return;
			}
		}

		this.runtime.recordAgentUsage(summaryWithUsage.usage);
		if (sessionGeneration !== this.sessionGeneration || flushEpoch !== this.flushEpoch) {
			debugLog("contemplator.compaction_stale", { reason: "session_or_branch_changed" });
			return;
		}
		const summaryModel = model as Model<any> & { api?: unknown; provider?: string; id?: string };
		const summaryUsage = summaryWithUsage.usage;
		const summaryMessage = {
			role: "assistant",
			content: [{ type: "text", text: `Previous contemplator context summary:\n${summaryWithUsage.text}` }],
			api: summaryModel.api,
			provider: summaryModel.provider ?? "unknown",
			model: summaryModel.id ?? "contemplator",
			usage: {
				input: summaryUsage.input,
				output: summaryUsage.output,
				cacheRead: summaryUsage.cacheRead,
				cacheWrite: summaryUsage.cacheWrite,
				totalTokens: summaryUsage.input + summaryUsage.output + summaryUsage.cacheRead + summaryUsage.cacheWrite,
				cost: summaryUsage.cost,
			},
			stopReason: "stop",
			timestamp: Date.now(),
		} as AgentMessage;
		// A flush currently owns private-history mutation while this awaits the
		// model, and session/tree movement invalidates the generation above. Still,
		// preserve any future append-only writer rather than replacing a stale
		// snapshot and silently dropping messages. Non-append mutation is unsafe to
		// merge, so postpone and let a later pass compact the live transcript.
		const liveHistory = this.history;
		const liveHistoryEntryIds = this.historyEntryIds;
		const snapshotIsLivePrefix = liveHistory.length >= history.length
			&& history.every((message, index) => liveHistory[index] === message && liveHistoryEntryIds[index] === historyEntryIds[index]);
		if (!snapshotIsLivePrefix) {
			debugLog("contemplator.compaction_postponed", { reason: "private history changed during compaction" });
			return;
		}
		const retainedMessages = liveHistory.slice(prefixEnd);
		const retainedMessageEntryIds = liveHistoryEntryIds.slice(prefixEnd).filter((id): id is string => typeof id === "string");
		if (retainedMessageEntryIds.length !== retainedMessages.length) {
			debugLog("contemplator.compaction_postponed", { reason: "retained history lacked durable entry ids", retainedMessageCount: retainedMessages.length, retainedReferenceCount: retainedMessageEntryIds.length });
			return;
		}
		// Record only coverage evidenced by the exact durable prompt prefix being
		// replaced. `seen*Ids` may also contain observations appended concurrently
		// while summary generation awaited; those remain pending and must not be
		// declared covered before their own prompt is persisted. Coverage is a delta
		// per checkpoint, and restore unions checkpoints, avoiding cumulative O(n²)
		// id duplication across a very long session.
		const prefixCoveredIds = new Set<string>();
		for (const message of history.slice(0, prefixEnd)) {
			if (message.role !== "user") continue;
			const text = customMessageText(message.content);
			if (!text.includes("NEW MEMORY UPDATE")) continue;
			for (const id of memoryReferenceIds(text)) prefixCoveredIds.add(id);
		}
		const currentProjection = fullProjection(ctx.sessionManager.getBranch() as Entry[]);
		const checkpoint = {
			version: 2,
			compacted: true,
			message: summaryMessage,
			retainedMessageEntryIds,
			coveredObservationIds: currentProjection.observations.filter((item) => prefixCoveredIds.has(item.id)).map((item) => item.id),
			coveredReviewIds: (currentProjection.reviews ?? []).filter((item) => prefixCoveredIds.has(item.id)).map((item) => item.id),
		};
		const checkpointEntryId = this.appendContemplatorHistoryEntry(ctx, checkpoint);
		this.history = [summaryMessage, ...retainedMessages];
		this.historyEntryIds = [checkpointEntryId, ...retainedMessageEntryIds];
		this.markTipPersisted(ctx);
		debugLog("contemplator.compaction_complete", {
			previousMessageCount,
			newMessageCount: this.history.length,
			prefixMessageCount: prefixEnd,
			retainedMessageCount: retainedMessages.length,
			summaryLength: summaryWithUsage.text.length,
		});
	}
}
