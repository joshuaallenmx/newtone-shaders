import { describe, it, expect } from "vitest";
import { sizeSpecToPixels } from "./Pipeline";

describe("sizeSpecToPixels", () => {
    const canvas = { w: 800, h: 600 };

    it("'full' returns the canvas size", () => {
        expect(sizeSpecToPixels("full", canvas)).toEqual([800, 600]);
    });

    it("'half' returns half the canvas size", () => {
        expect(sizeSpecToPixels("half", canvas)).toEqual([400, 300]);
    });

    it("an explicit tuple is returned as-is", () => {
        expect(sizeSpecToPixels([2, 2], canvas)).toEqual([2, 2]);
        expect(sizeSpecToPixels([128, 256], canvas)).toEqual([128, 256]);
    });

    it("a function spec is invoked with the canvas size", () => {
        expect(
            sizeSpecToPixels((s) => [s.w / 4, s.h / 4] as const, canvas),
        ).toEqual([200, 150]);
    });

    it("clamps degenerate dimensions to 2", () => {
        expect(sizeSpecToPixels("full", { w: 0, h: 0 })).toEqual([2, 2]);
        expect(sizeSpecToPixels([1, 1], canvas)).toEqual([2, 2]);
    });
});
