import { describe, it, expect } from "vitest";
import { fillEnvData } from "./env";

describe("fillEnvData", () => {
    it("paints a single solid color when given one stop", () => {
        const data = new Uint8Array(1 * 1 * 4);
        fillEnvData(data, 1, ["#ff0000"]);
        expect(Array.from(data)).toEqual([0xff, 0, 0, 0xff]);
    });
    it("interpolates vertically between two stops", () => {
        const size = 4;
        const data = new Uint8Array(size * size * 4);
        fillEnvData(data, size, ["#000000", "#ffffff"]);
        expect(data[0]).toBe(0); // top-left red
        expect(data[(size - 1) * size * 4]).toBe(255); // bottom-left red
    });
    it("is uniform across x for a given y", () => {
        const size = 4;
        const data = new Uint8Array(size * size * 4);
        fillEnvData(data, size, ["#112233", "#ffeedd"]);
        for (let y = 0; y < size; y++) {
            const rowStart = y * size * 4;
            const r = data[rowStart];
            for (let x = 1; x < size; x++) {
                expect(data[rowStart + x * 4]).toBe(r);
            }
        }
    });
    it("falls back to a default gradient when no stops are provided", () => {
        const data = new Uint8Array(1 * 1 * 4);
        fillEnvData(data, 1, []);
        expect(data[3]).toBe(255); // alpha is set
    });
});
