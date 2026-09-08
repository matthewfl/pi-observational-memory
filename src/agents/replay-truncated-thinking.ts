import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Message } from "@earendil-works/pi-ai";

/**
 * Many provider chat templates discard historical reasoning blocks. Preserve
 * unfinished plaintext work after an output-length stop by replaying it as
 * ordinary assistant text. Redacted/encrypted blocks remain structured so
 * their opaque provider payload stays replayable. The original transcript is
 * never mutated; this transformation is only applied at the LLM boundary.
 */
export function replayTruncatedThinkingAsText(messages: readonly AgentMessage[]): Message[] {
	return messages.map((message) => {
		if (message.role !== "assistant" || message.stopReason !== "length" || !message.content.some((part) => part.type === "thinking" && !part.redacted)) return message as Message;
		return {
			...message,
			content: message.content.map((part) => part.type === "thinking" && !part.redacted
				? { type: "text" as const, text: `[Incomplete analysis from the preceding truncated response]\n${part.thinking}` }
				: part),
		} as Message;
	});
}
