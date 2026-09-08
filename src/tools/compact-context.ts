import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Runtime } from "../runtime.js";

export const COMPACT_CONTEXT_TOOL_NAME = "compact_context";
export const COMPACT_CONTEXT_DESCRIPTION =
	"Force manual compaction and resume with agent-authored next-step instructions. Use sparingly when substantial work remains but the remaining context is insufficient, or when accumulated context has become noisy or stale enough to impair focus and reliable reasoning.";

export type CompactContextDetails = {
	status: "scheduled" | "already_pending" | "in_progress";
};

export function createCompactContextTool(runtime: Runtime) {
	return defineTool({
		name: COMPACT_CONTEXT_TOOL_NAME,
		label: "Compact context",
		description: COMPACT_CONTEXT_DESCRIPTION,
		promptSnippet:
			"Force manual context compaction when substantial work remains but context is running out, or when context degradation is impairing focus",
		promptGuidelines: [
			"Use compact_context sparingly when either substantial additional work remains and there is not enough context left to complete it, or accumulated past context has become noisy, stale, or distracting enough that you are struggling to focus on the current task or reason about it reliably.",
			"Do not use compact_context routinely, for short tasks, or merely because the conversation is long; use it for genuine context-capacity pressure or context degradation that is interfering with the work.",
			"Call compact_context by itself and provide short_continuation_prompt with concrete instructions for the next agent step. The tool ends the current turn automatically; pi-contemplator will compact the context and resume with those instructions, so do not add a separate response after the tool call.",
		],
		parameters: Type.Object({
			short_continuation_prompt: Type.String({
				minLength: 1,
				pattern: "\\S",
				description: "Short, concrete instructions to your post-compaction self describing the next action and any critical immediate constraint. Do not summarize the whole conversation.",
			}),
		}),
		async execute(_toolCallId, params) {
			if (runtime.compactInFlight) {
				return {
					content: [{ type: "text" as const, text: "Context compaction is already in progress. This turn is ending automatically; wait for automatic resume." }],
					details: { status: "in_progress" } as CompactContextDetails,
					terminate: true,
				};
			}
			if (runtime.compactRequested) {
				return {
					content: [{ type: "text" as const, text: "Context compaction is already scheduled. This turn is ending automatically; wait for automatic resume." }],
					details: { status: "already_pending" } as CompactContextDetails,
					terminate: true,
				};
			}

			runtime.compactRequested = true;
			runtime.compactContinuationPrompt = params.short_continuation_prompt.trim();
			return {
				content: [{ type: "text" as const, text: "Context compaction scheduled. This turn is ending automatically, and the task will resume after compaction." }],
				details: { status: "scheduled" } as CompactContextDetails,
				terminate: true,
			};
		},
	});
}

export function registerCompactContextTool(pi: ExtensionAPI, runtime: Runtime): void {
	pi.registerTool(createCompactContextTool(runtime));
}
