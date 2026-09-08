import { describe, expect, it, vi } from "vitest";
import { registerStatusCommand } from "../src/commands/status.js";
import { Runtime } from "../src/runtime.js";

describe("/om:status memory accounting", () => {
	it("shows summaries and observations as one combined new/old memory pool", async () => {
		let handler: ((args: string, ctx: any) => Promise<void>) | undefined;
		const pi = {
			registerCommand: vi.fn((_name: string, definition: { handler: typeof handler }) => { handler = definition.handler; }),
		};
		const runtime = new Runtime();
		runtime.configLoaded = true;
		const notify = vi.fn();
		registerStatusCommand(pi as any, runtime);

		await handler?.("", {
			cwd: "/tmp/project",
			model: { contextWindow: 128_000 },
			sessionManager: { getBranch: () => [] },
			ui: { notify },
		});

		const output = notify.mock.calls[0]?.[0] as string;
		expect(output).toContain("Active memory total:");
		expect(output).toContain("Old memory pool:");
		expect(output).toContain("observations + summaries");
		expect(output).not.toContain("Summary pool:");
		expect(output).not.toContain("Visible observation pool:");
	});
});
