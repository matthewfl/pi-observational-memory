import { describe, expect, it, vi } from "vitest";
import { renderObserver } from "../src/commands/observer-view.js";
import { registerViewCommand } from "../src/commands/view.js";

describe("renderObserver", () => {
	it("reports when no observer chunk has run this launch", () => {
		expect(renderObserver(undefined)).toContain("Observer has not run yet during this launch");
	});

	it("shows a live chunk with thinking and source/backlog diagnostics", () => {
		const output = renderObserver({
			startedAt: 1_000,
			status: "running",
			messages: [
				{ role: "user", content: [{ type: "text", text: "NEW CONVERSATION CHUNK" }] },
				{ role: "assistant", content: [{ type: "thinking", thinking: "Inspecting the current chunk" }] },
			],
			chunkTokens: 12_345,
			backlogTokens: 98_765,
			sourceEntryIds: ["source-a", "source-b"],
		}, 6_000);

		expect(output).toContain("OBSERVER · running");
		expect(output).toContain("Chunk ~12,345 tokens");
		expect(output).toContain("backlog at start ~98,765 tokens");
		expect(output).toContain("2 source entries");
		expect(output).toContain("running for 5s");
		expect(output).toContain("[thinking]\nInspecting the current chunk");
	});

	it("shows completion timing, summary, and failure details", () => {
		const output = renderObserver({
			startedAt: 1_000,
			completedAt: 4_500,
			status: "failed",
			messages: [],
			chunkTokens: 100,
			backlogTokens: 200,
			sourceEntryIds: ["source-a"],
			summary: "Chunk covered.",
			error: "provider reached the output limit 4 times",
		});

		expect(output).toContain("ended 1970-01-01T00:00:04.500Z after 3s");
		expect(output).toContain("Completion summary");
		expect(output).toContain("provider reached the output limit 4 times");
	});
	it("registers /om:view observer and renders the launch-local active run", async () => {
		let handler: ((args: unknown, ctx: any) => Promise<void>) | undefined;
		const pi = { registerCommand: (_name: string, command: any) => { handler = command.handler; } };
		const runtime = {
			ensureConfig: () => {},
			lastObserverRun: {
				startedAt: 1_000, status: "running", messages: [], chunkTokens: 100,
				backlogTokens: 200, sourceEntryIds: ["source-a"],
			},
		};
		const notify = vi.fn();
		registerViewCommand(pi as any, runtime as any, { copyToClipboard: async () => true });

		await handler?.("observer", {
			cwd: "/tmp", ui: { notify }, sessionManager: { getBranch: () => [] },
		});

		expect(notify).toHaveBeenCalledWith(expect.stringContaining("OBSERVER · running"), "info");
		expect(notify).toHaveBeenCalledWith(expect.stringContaining("Copied /om:view observer output"), "info");
	});

});
