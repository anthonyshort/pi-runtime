import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExtensionAPI, ExtensionToolContext, ToolCallEvent } from "../../src/core/extensions/types.ts";
import { createHarness, type Harness } from "./harness.ts";

describe("AgentSession direct tools", () => {
	const harnesses: Harness[] = [];
	afterEach(() => {
		while (harnesses.length) harnesses.pop()?.cleanup();
	});

	async function setup(extension: (pi: ExtensionAPI) => void): Promise<Harness> {
		const harness = await createHarness({ initialActiveToolNames: [], extensionFactories: [extension] });
		harnesses.push(harness);
		await harness.session.bindExtensions({});
		harness.setResponses([
			() => {
				throw new Error("A direct tool must not call the parent model");
			},
		]);
		return harness;
	}

	it("runs validated nested tools and notifications without a parent model", async () => {
		const calls: ToolCallEvent[] = [];
		let notify: (() => void) | undefined;
		const harness = await setup((pi) => {
			pi.on("tool_call", (event) => {
				calls.push(event);
			});
			pi.registerTool({
				name: "save",
				label: "save",
				description: "Save a result",
				parameters: Type.Object({ text: Type.String() }),
				execute: async (id, args) => ({ content: [{ type: "text", text: args.text }], details: { id } }),
			});
			pi.registerTool({
				name: "entry",
				label: "entry",
				description: "Run and save",
				exposure: "codemode",
				parameters: Type.Object({}),
				execute: async (...[, , signal, update, ctx]) => {
					expect(ctx.signal).toBe(signal);
					notify = () =>
						pi.sendMessage(
							{ customType: "complete", content: "completed", display: false },
							{ triggerTurn: true },
						);
					notify();
					update?.({ content: [{ type: "text", text: "saving" }], details: {} });
					const saved = await ctx.executeTool("save", { text: "saved" });
					return saved.result;
				},
			});
		});
		const result = await harness.session.executeTool("entry", {});
		expect(harness.session.getActiveToolNames()).not.toContain("entry");
		expect(result).toMatchObject({ content: [{ type: "text", text: "saved" }], isError: false });
		expect(calls.map((call) => [call.toolName, call.origin])).toEqual([
			["entry", "host"],
			["save", "host"],
		]);
		expect(calls[1].parentToolCallId).toBe(calls[0].toolCallId);
		expect(calls[1].toolCallId).toBe(`${calls[0].toolCallId}/1`);
		expect(harness.eventsOfType("tool_execution_update")).toHaveLength(1);
		expect(harness.eventsOfType("agent_settled")).toHaveLength(0);
		notify?.();
		await new Promise<void>((resolve) => setImmediate(resolve));
		expect(harness.getPendingResponseCount()).toBe(1);
		expect(harness.session.messages.filter((message) => message.role === "assistant")).toEqual([]);
		expect(harness.session.messages.filter((message) => message.role === "custom")).toHaveLength(2);
		expect(harness.session.isIdle).toBe(true);
	});

	it("passes cancellation to permission hooks before a tool starts", async () => {
		let started: (() => void) | undefined;
		const permissionStarted = new Promise<void>((resolve) => {
			started = resolve;
		});
		const execute = vi.fn(async () => ({ content: [], details: {} }));
		const harness = await setup((pi) => {
			pi.registerTool({
				name: "protected",
				label: "protected",
				description: "Needs permission",
				parameters: Type.Object({}),
				execute,
			});
			pi.on("tool_call", async (event, ctx) => {
				expect(event.origin).toBe("host");
				expect(ctx.signal).toBeInstanceOf(AbortSignal);
				started?.();
				await new Promise<void>((resolve) =>
					ctx.signal?.addEventListener("abort", () => resolve(), { once: true }),
				);
			});
		});
		const operation = harness.session.executeTool("protected", {});
		await permissionStarted;
		await harness.session.abort();
		expect((await operation).isError).toBe(true);
		expect(execute).not.toHaveBeenCalled();
		expect(harness.session.isIdle).toBe(true);
	});

	it("rejects calls through a tool context retained after root completion", async () => {
		let retained: ExtensionToolContext | undefined;
		const child = vi.fn(async () => ({ content: [], details: {} }));
		const harness = await setup((pi) => {
			pi.registerTool({
				name: "child",
				label: "child",
				description: "Child",
				parameters: Type.Object({}),
				execute: child,
			});
			pi.registerTool({
				name: "root",
				label: "root",
				description: "Retain context",
				parameters: Type.Object({}),
				execute: async (...[, , , , ctx]) => {
					retained = ctx;
					return { content: [], details: {} };
				},
			});
		});
		harness.setResponses([fauxAssistantMessage("Earlier model turn")]);
		await harness.session.prompt("Earlier prompt");
		expect((await harness.session.executeTool("root", {})).isError).toBe(false);
		expect(retained?.signal?.aborted).toBe(true);
		if (!retained) throw new Error("Root context was not captured");
		expect((await retained.executeTool("child", {})).isError).toBe(true);
		expect((await retained.executeTool("child", {}, { signal: new AbortController().signal })).isError).toBe(true);
		expect(child).not.toHaveBeenCalled();
		expect(harness.session.isIdle).toBe(true);
	});

	it("validates arguments and runs permission and result hooks", async () => {
		const execute = vi.fn(async () => ({ content: [{ type: "text" as const, text: "wrote" }], details: {} }));
		const harness = await setup((pi) => {
			pi.registerTool({
				name: "write_result",
				label: "write_result",
				description: "Write",
				parameters: Type.Object({ allow: Type.Boolean() }),
				execute,
			});
			pi.on("tool_call", (event) =>
				"allow" in event.input && event.input.allow === false
					? { block: true, reason: "Permission denied" }
					: undefined,
			);
			pi.on("tool_result", () => ({ content: [{ type: "text", text: "checked" }], isError: true }));
		});
		expect((await harness.session.executeTool("write_result", { allow: "invalid" })).isError).toBe(true);
		const blocked = await harness.session.executeTool("write_result", { allow: false });
		expect(blocked).toMatchObject({ content: [{ type: "text", text: "Permission denied" }], isError: true });
		expect(execute).not.toHaveBeenCalled();
		expect(await harness.session.executeTool("write_result", { allow: true })).toMatchObject({
			content: [{ type: "text", text: "checked" }],
			isError: true,
		});
		expect(execute).toHaveBeenCalledTimes(1);
		expect((await harness.session.executeTool("missing", {})).isError).toBe(true);
	});

	it("waits for nested calls before returning the root receipt", async () => {
		let release: (() => void) | undefined;
		const child = new Promise<void>((resolve) => {
			release = resolve;
		});
		const harness = await setup((pi) => {
			pi.registerTool({
				name: "child",
				label: "child",
				description: "Finish later",
				parameters: Type.Object({}),
				execute: async () => {
					await child;
					return { content: [], details: {} };
				},
			});
			pi.registerTool({
				name: "root",
				label: "root",
				description: "Start child",
				parameters: Type.Object({}),
				execute: async (...[, , , , ctx]) => {
					void ctx.executeTool("child", {});
					return { content: [], details: {} };
				},
			});
		});
		const received = vi.fn();
		const root = harness.session.executeTool("root", {}).then(received);
		await new Promise<void>((resolve) => setImmediate(resolve));
		expect(received).not.toHaveBeenCalled();
		expect(harness.session.isIdle).toBe(false);
		release?.();
		await root;
		expect(harness.eventsOfType("tool_execution_end").map((event) => event.toolName)).toEqual(["child", "root"]);
		expect(harness.session.isIdle).toBe(true);
	});

	it("stays busy until cancellation cleanup finishes", async () => {
		let started: (() => void) | undefined;
		const running = new Promise<void>((resolve) => {
			started = resolve;
		});
		let release: (() => void) | undefined;
		const cleanup = new Promise<void>((resolve) => {
			release = resolve;
		});
		const harness = await setup((pi) => {
			pi.registerTool({
				name: "wait",
				label: "wait",
				description: "Wait for cancellation",
				parameters: Type.Object({}),
				execute: async (...[, , signal]) => {
					started?.();
					await new Promise<void>((resolve) => signal?.addEventListener("abort", () => resolve(), { once: true }));
					await cleanup;
					throw new Error("Cancelled");
				},
			});
		});
		const operation = harness.session.executeTool("wait", {});
		await running;
		expect(harness.session.isIdle).toBe(false);
		expect(harness.session.isStreaming).toBe(true);
		await expect(harness.session.executeTool("wait", {})).rejects.toThrow("Session is busy");
		await expect(harness.session.prompt("go")).rejects.toThrow("direct tool operation");
		const aborted = vi.fn();
		const abort = harness.session.abort().then(aborted);
		await new Promise<void>((resolve) => setImmediate(resolve));
		expect(aborted).not.toHaveBeenCalled();
		expect(harness.session.isIdle).toBe(false);
		release?.();
		expect((await operation).isError).toBe(true);
		await abort;
		expect(harness.session.isIdle).toBe(true);
		expect(harness.getPendingResponseCount()).toBe(1);
	});
});
