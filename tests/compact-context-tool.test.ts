import { describe, expect, it, vi } from "vitest";

import {
	COMPACT_CONTEXT_DESCRIPTION,
	COMPACT_CONTEXT_TOOL_NAME,
	createCompactContextTool,
	registerCompactContextTool,
} from "../src/tools/compact-context.js";

function runtime(overrides: Record<string, unknown> = {}) {
	return {
		compactInFlight: false,
		compactRequested: false,
		...overrides,
	};
}

describe("compact_context tool", () => {
	it("registers the manual compaction tool with sparing-use guidance", () => {
		const pi = { registerTool: vi.fn() };
		const state = runtime();

		registerCompactContextTool(pi as any, state as any);

		const tool = pi.registerTool.mock.calls[0]?.[0];
		expect(tool.name).toBe(COMPACT_CONTEXT_TOOL_NAME);
		expect(tool.name).toBe("compact_context");
		expect(tool.description).toBe(COMPACT_CONTEXT_DESCRIPTION);
		expect(tool.description).not.toMatch(/observational|\bOM\b/i);
		expect(tool.promptGuidelines.join(" ")).toMatch(/sparingly/i);
		expect(tool.promptGuidelines.join(" ")).toMatch(/substantial additional work/i);
		expect(tool.promptGuidelines.join(" ")).toMatch(/not enough context left/i);
		expect(tool.promptGuidelines.join(" ")).toMatch(/struggling to focus/i);
		expect(tool.promptGuidelines.join(" ")).toMatch(/reason about it reliably/i);
		expect(tool.promptGuidelines.join(" ")).toContain("short_continuation_prompt");
		expect(tool.parameters.properties.short_continuation_prompt).toMatchObject({
			type: "string",
			minLength: 1,
		});
		expect(tool.parameters.properties.short_continuation_prompt.maxLength).toBeUndefined();
	});

	it("accepts a continuation prompt longer than the former 1,000-character limit", async () => {
		const state = runtime();
		const tool = createCompactContextTool(state as any);
		const continuation = `Continue with this retained plan: ${"x".repeat(2_000)}`;

		await tool.execute("tool-long", { short_continuation_prompt: continuation }, undefined as any, undefined as any, {} as any);

		expect(state.compactContinuationPrompt).toBe(continuation);
	});

	it("schedules compaction with a post-compaction continuation and terminates the current tool turn", async () => {
		const state = runtime();
		const tool = createCompactContextTool(state as any);

		const result = await tool.execute("tool-1", { short_continuation_prompt: "Run the focused regression test next." }, undefined as any, undefined as any, {} as any);

		expect(state.compactRequested).toBe(true);
		expect(state.compactContinuationPrompt).toBe("Run the focused regression test next.");
		expect(result).toMatchObject({
			details: { status: "scheduled" },
			terminate: true,
		});
	});

	it("does not enqueue duplicate compaction requests", async () => {
		const state = runtime({ compactRequested: true });
		const tool = createCompactContextTool(state as any);

		const result = await tool.execute("tool-1", { short_continuation_prompt: "Do not replace the pending request." }, undefined as any, undefined as any, {} as any);

		expect(state.compactRequested).toBe(true);
		expect(state.compactContinuationPrompt).toBeUndefined();
		expect(result).toMatchObject({
			details: { status: "already_pending" },
			terminate: true,
		});
	});

	it("reports in_progress and does not double-request when a compaction is already running", async () => {
		const state = runtime({ compactInFlight: true });
		const tool = createCompactContextTool(state as any);

		const result = await tool.execute("tool-1", { short_continuation_prompt: "Continue." }, undefined as any, undefined as any, {} as any);

		expect(state.compactRequested).toBe(false);
		expect(result).toMatchObject({
			details: { status: "in_progress" },
			terminate: true,
		});
	});
});
