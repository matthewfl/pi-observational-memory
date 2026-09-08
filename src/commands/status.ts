import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { resolveCompactAfterTokens } from "../config.js";
import type { Runtime } from "../runtime.js";
import {
	agentActiveTimeMs,
	diffProjection,
	foldLedger,
	fullProjection,
	partitionMemoryPools,
	rawTokensSinceLastCompaction,
	rawTokensSinceObservationCoverage,
	visibleProjection,
	type Entry,
} from "../session-ledger/index.js";

const CONTEMPLATOR_SUGGESTION = "om.contemplator.suggestion";
const REVIEWER_NOTICE = "om.reviewer.notice";

function pct(current: number, total: number): number {
	return total > 0 ? Math.round((current / total) * 100) : 0;
}

function formatTokens(n: number): string {
	if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
	if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
	return String(n);
}

function formatDuration(durationMs: number): string {
	const totalSeconds = Math.floor(durationMs / 1_000);
	const hours = Math.floor(totalSeconds / 3_600);
	const minutes = Math.floor((totalSeconds % 3_600) / 60);
	const seconds = totalSeconds % 60;
	if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
	if (minutes > 0) return `${minutes}m ${seconds}s`;
	return `${seconds}s`;
}

function formatRunAge(timestamp: number): string {
	return `${new Date(timestamp).toISOString()} (${formatDuration(Math.max(0, Date.now() - timestamp))} ago)`;
}

function configuredModelLabel(model: { provider: string; id: string } | null): string {
	return model ? `${model.provider}/${model.id}` : "current session model";
}

function contemplatorWaitingLabel(waitingFor: Runtime["contemplatorState"]["waitingFor"]): string {
	switch (waitingFor) {
		case "observer": return "waiting for observer backlog";
		case "probe": return "waiting for queued probe delivery";
		case "memories": return "waiting for memory threshold";
		case "responses": return "waiting for response spacing";
		case "ready": return "ready to launch";
		case "running": return "running";
		case "disabled": return "disabled";
		case "passive": return "passive mode";
		default: return "idle; no pending memories";
	}
}

function truncateStatusText(value: string, limit = 1_000): string {
	return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
}

function addedSuffix(count: number): string | undefined {
	return count > 0 ? `+${count.toLocaleString()}` : undefined;
}

function removedSuffix(count: number): string | undefined {
	return count > 0 ? `-${count.toLocaleString()}` : undefined;
}

function appendSuffixes(line: string, suffixes: (string | undefined)[]): string {
	const rendered = suffixes.filter((suffix): suffix is string => suffix !== undefined);
	return rendered.length > 0 ? `${line} ${rendered.join(" ")}` : line;
}

export function registerStatusCommand(pi: ExtensionAPI, runtime: Runtime): void {
	pi.registerCommand("om:status", {
		description: "Show pi-contemplator status",
		handler: async (_args, ctx) => {
			runtime.ensureConfig(ctx.cwd);
			const entries = ctx.sessionManager.getBranch() as Entry[];
			const folded = foldLedger(entries);
			const visible = visibleProjection(entries);
			const full = fullProjection(entries);
			const drift = diffProjection(visible, full);

			const pools = partitionMemoryPools(folded.activeObservations, folded.activeSummaries, runtime.config.newMemoryPoolMaxTokens);
			const observationLine = appendSuffixes(
				`Observations: ${folded.observations.length} recorded / ${folded.activeObservations.length} active / ${visible.observations.length} visible`,
				[addedSuffix(drift.observationsOnlyInFull.length), removedSuffix(drift.observationsOnlyInVisible.length)],
			);
			const summaryLine = appendSuffixes(
				`Summaries:    ${folded.summaries.length} recorded / ${folded.activeSummaries.length} active / ${visible.summaries.length} visible`,
				[addedSuffix(drift.summariesOnlyInFull.length), removedSuffix(drift.summariesOnlyInVisible.length)],
			);
			const obsProgress = rawTokensSinceObservationCoverage(entries);
			const compactionProgress = rawTokensSinceLastCompaction(entries);
			const contextWindow = typeof ctx.model?.contextWindow === "number" ? ctx.model.contextWindow : undefined;
			const compactThreshold = resolveCompactAfterTokens(runtime.config, contextWindow);

			const passiveLines = runtime.config.passive === true
				? [
					"── Mode ──",
					"Passive: automatic memory workers and auto-compaction disabled; manual/Pi compaction, commands, and recall remain active",
					"",
				]
				: [];

			const summarizerTrigger = runtime.summarizerNextTriggerTokens ?? runtime.config.oldMemoryPoolTargetTokens;
			const summarizerSamplingTokens = runtime.config.summarizerSamplingThresholdTokens;
			const lines = [
				...passiveLines,
				"── Memory ──",
				observationLine,
				summaryLine,
				"",
				"── Activity ──",
				`Observer source backlog: ~${obsProgress.toLocaleString()} / ${runtime.config.observeAfterTokens.toLocaleString()} tokens (${pct(obsProgress, runtime.config.observeAfterTokens)}%)`,
				`Summarizer trigger:      old pool ~${pools.oldTokens.toLocaleString()} / ${summarizerTrigger.toLocaleString()} tokens (${pct(pools.oldTokens, summarizerTrigger)}%)`,
				`Automatic compaction source backlog: ~${compactionProgress.toLocaleString()} / ${compactThreshold.toLocaleString()} tokens (${pct(compactionProgress, compactThreshold)}%; injected memory excluded)`,
				`Active memory total:     ~${pools.totalTokens.toLocaleString()} tokens (observations + summaries; split below)`,
				`New memory pool:         ~${pools.newTokens.toLocaleString()} / ${runtime.config.newMemoryPoolMaxTokens.toLocaleString()} protection-budget tokens (${pct(pools.newTokens, runtime.config.newMemoryPoolMaxTokens)}%; newest memory always protected whole)`,
				`Old memory pool:         ~${pools.oldTokens.toLocaleString()} / ${runtime.config.oldMemoryPoolTargetTokens.toLocaleString()} advisory target tokens (${pct(pools.oldTokens, runtime.config.oldMemoryPoolTargetTokens)}%; observations + summaries)`,
				`Summarizer:              ${runtime.config.summarizerEnabled === false ? "disabled" : "enabled"}; retrigger after +${runtime.config.summarizerRetriggerTokens.toLocaleString()} old-pool tokens / sample above ~${summarizerSamplingTokens.toLocaleString()} tokens`,
				`Summarizer model:        ${configuredModelLabel(runtime.configuredMemoryWorkerModel("summarizer"))}`,
				`Observer model:          ${configuredModelLabel(runtime.configuredMemoryWorkerModel("observer"))}`,
				`Cumulative agent time:   ${formatDuration(agentActiveTimeMs(entries))}`,
				`Observe source during compaction: ${runtime.config.compactionObserverEnabled === false ? "disabled" : "enabled"}`,
				`Contemplator:             ${runtime.config.contemplatorEnabled ? "enabled" : "disabled"}`,
				`Contemplator trigger:     ${runtime.contemplatorState.pendingObservations} observations / ${runtime.contemplatorState.pendingReviews} reviews pending; ${runtime.contemplatorState.responsesSinceRun} / ${runtime.config.contemplatorMinTurns} primary responses; ${contemplatorWaitingLabel(runtime.contemplatorState.waitingFor)}`,
				`Contemplator model:      ${runtime.config.contemplatorModel ? `${runtime.config.contemplatorModel.provider}/${runtime.config.contemplatorModel.id}` : "current session model"}`,
				`Contemplator messages:   ${runtime.config.showContemplatorMessages ? "visible" : "hidden"}`,
				`Structural reviewer:     ${runtime.config.reviewerEnabled === false ? "disabled" : "enabled"}`,
				`Reviewer model:          ${runtime.config.reviewerModel ? `${runtime.config.reviewerModel.provider}/${runtime.config.reviewerModel.id}` : "current session model"}`,
			];

			if (runtime.agentUsage.runs > 0) {
				const u = runtime.agentUsage;
				lines.push(`Token usage:            ↑${formatTokens(u.input)} ↓${formatTokens(u.output)}${u.cacheRead ? ` R${formatTokens(u.cacheRead)}` : ""}${u.cacheWrite ? ` W${formatTokens(u.cacheWrite)}` : ""} $${u.cost.toFixed(3)} (${u.runs} call${u.runs === 1 ? "" : "s"})`);
			}

			lines.push("", "── Last worker runs ──");
			lines.push(`Observer chunk start:    ${runtime.lastObserverStartedAt === undefined ? "not run this launch" : formatRunAge(runtime.lastObserverStartedAt)}`);
			lines.push(`Observer chunk end:      ${runtime.lastObserverCompletedAt === undefined ? runtime.lastObserverRun?.status === "running" ? "running" : "not completed this launch" : formatRunAge(runtime.lastObserverCompletedAt)}`);
			lines.push(`Last summarizer start:   ${runtime.lastSummarizerStartedAt === undefined ? "not run this launch" : formatRunAge(runtime.lastSummarizerStartedAt)}`);
			lines.push(`Last summarizer end:     ${runtime.lastSummarizerCompletedAt === undefined ? "not completed this launch" : formatRunAge(runtime.lastSummarizerCompletedAt)}`);
			lines.push(`Last contemplator start: ${runtime.contemplatorState.lastStartedAt === undefined ? "not run this launch" : formatRunAge(runtime.contemplatorState.lastStartedAt)}`);
			lines.push(`Last contemplator end:   ${runtime.contemplatorState.lastCompletedAt === undefined ? "not completed this launch" : formatRunAge(runtime.contemplatorState.lastCompletedAt)}`);

			lines.push("", "── Interventions ──");
			// Probe stats come from the branch ledger (like /om:view contemplator):
			// deduped by probeId so restore re-queues don't inflate the count, and
			// entries without a probeId (sent before probe tracking existed) count
			// individually. Survives reloads, unlike an in-memory counter.
			const probeSuggestions: { suggestion: string }[] = [];
			const probeIndexByProbeId = new Map<string, number>();
			for (const entry of entries) {
				if (entry.customType !== CONTEMPLATOR_SUGGESTION) continue;
				const data = (entry.data ?? {}) as { suggestion?: unknown; probeId?: unknown };
				if (typeof data.suggestion !== "string") continue;
				if (typeof data.probeId !== "string") {
					probeSuggestions.push({ suggestion: data.suggestion });
					continue;
				}
				const existingIndex = probeIndexByProbeId.get(data.probeId);
				if (existingIndex === undefined) {
					probeIndexByProbeId.set(data.probeId, probeSuggestions.length);
					probeSuggestions.push({ suggestion: data.suggestion });
				} else {
					probeSuggestions[existingIndex] = { suggestion: data.suggestion };
				}
			}
			lines.push(`Probes sent:            ${probeSuggestions.length}`);
			if (probeSuggestions.length > 0) lines.push(`Last probe:             ${probeSuggestions[probeSuggestions.length - 1].suggestion}`);

			const reviews = full.reviews ?? [];
			lines.push(`Reviews completed:      ${reviews.length}`);
			const latestReview = reviews.at(-1);
			if (latestReview) {
				lines.push(`Last review:            [${latestReview.id}] ${latestReview.scope} ${latestReview.outcome}`);
				if (latestReview.outcome === "proposal") lines.push(`Last review summary:    ${truncateStatusText(latestReview.summary)}`);
				else lines.push(`Last review reason:     ${truncateStatusText(latestReview.reason)}`);
			}
			let latestNotice: string | undefined;
			for (const entry of entries) {
				if (entry.customType !== REVIEWER_NOTICE || !entry.data || typeof entry.data !== "object") continue;
				const content = (entry.data as { content?: unknown }).content;
				if (typeof content === "string") latestNotice = content;
			}
			if (latestNotice) lines.push(`Last reviewer notice:  ${truncateStatusText(latestNotice)}`);

			if (runtime.consolidationInFlight || runtime.summarizerInFlight || runtime.contemplatorState.running || runtime.compactInFlight || runtime.compactHookInFlight || runtime.reviewInFlight) {
				lines.push("", "── In flight ──");
				if (runtime.consolidationInFlight) lines.push("Observer: running");
				if (runtime.summarizerInFlight) lines.push("Summarizer: running");
				if (runtime.contemplatorState.running) lines.push("Contemplator: running");
				if (runtime.compactInFlight) lines.push("Automatic compaction: running");
				if (runtime.compactHookInFlight) lines.push("Compaction hook: running");
				if (runtime.reviewInFlight) lines.push("Structural review: running");
			}

			if (runtime.lastObserverError || runtime.lastSummarizerError || runtime.contemplatorState.lastError) {
				lines.push("", "── Last error ──");
				if (runtime.lastObserverError) lines.push(`Observer: ${runtime.lastObserverError}`);
				if (runtime.lastSummarizerError) lines.push(`Summarizer: ${runtime.lastSummarizerError}`);
				if (runtime.contemplatorState.lastError) lines.push(`Contemplator: ${runtime.contemplatorState.lastError}`);
			}

			ctx.ui.notify(lines.join("\n"), "info");
		},
	});
}
