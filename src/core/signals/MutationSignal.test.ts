import { describe, it, expect, vi } from "vitest";
import { createMutationSignal } from "./MutationSignal";

describe("createMutationSignal", () => {
    it("starts at tick 0", () => {
        const host = document.createElement("div");
        document.body.appendChild(host);
        const sig = createMutationSignal({ host, debounceMs: 0 });
        sig.start();
        expect(sig.get()).toBe(0);
        sig.stop();
        document.body.removeChild(host);
    });

    it("increments on a mutation, debounced", async () => {
        vi.useFakeTimers();
        const host = document.createElement("div");
        document.body.appendChild(host);
        const sig = createMutationSignal({ host, debounceMs: 50 });
        sig.start();
        host.appendChild(document.createElement("span"));
        host.appendChild(document.createElement("span"));
        // Mutations have been queued; mutation observer fires async.
        await Promise.resolve();
        await vi.advanceTimersByTimeAsync(60);
        expect(sig.get()).toBe(1);
        sig.stop();
        document.body.removeChild(host);
        vi.useRealTimers();
    });

    it("notifies subscribers on tick", async () => {
        vi.useFakeTimers();
        const host = document.createElement("div");
        document.body.appendChild(host);
        const sig = createMutationSignal({ host, debounceMs: 20 });
        sig.start();
        const seen: number[] = [];
        const off = sig.subscribe((n) => seen.push(n));
        host.appendChild(document.createElement("span"));
        await Promise.resolve();
        await vi.advanceTimersByTimeAsync(30);
        off();
        sig.stop();
        document.body.removeChild(host);
        expect(seen).toEqual([1]);
        vi.useRealTimers();
    });
});
