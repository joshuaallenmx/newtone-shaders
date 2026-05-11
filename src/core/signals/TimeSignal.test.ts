import { describe, it, expect } from "vitest";
import { createTimeSignal } from "./TimeSignal";

describe("createTimeSignal", () => {
    it("starts at time=0 frame=0", () => {
        const t = createTimeSignal({ now: () => 1000 });
        expect(t.get()).toEqual({ time: 0, frame: 0 });
    });

    it("advances time and frame on tick", () => {
        let now = 1000;
        const t = createTimeSignal({ now: () => now });
        now = 1500;
        t.tick();
        expect(t.get().time).toBeCloseTo(0.5);
        expect(t.get().frame).toBe(1);
        now = 2500;
        t.tick();
        expect(t.get().time).toBeCloseTo(1.5);
        expect(t.get().frame).toBe(2);
    });

    it("notifies subscribers on tick", () => {
        const t = createTimeSignal({ now: () => 1000 });
        const seen: number[] = [];
        const off = t.subscribe((s) => seen.push(s.frame));
        t.tick();
        t.tick();
        off();
        t.tick();
        expect(seen).toEqual([1, 2]);
    });
});
