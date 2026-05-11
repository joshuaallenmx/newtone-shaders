import { describe, it, expect } from "vitest";
import { hexToRgbBytes, cssColorToRgbBytes, cssColorToLinearRgb } from "./css";

describe("hexToRgbBytes", () => {
    it("parses 6-digit hex", () => {
        expect(hexToRgbBytes("#ff66aa")).toEqual([0xff, 0x66, 0xaa]);
    });
    it("parses 3-digit shorthand", () => {
        expect(hexToRgbBytes("#f6a")).toEqual([0xff, 0x66, 0xaa]);
    });
    it("tolerates a missing hash", () => {
        expect(hexToRgbBytes("ff66aa")).toEqual([0xff, 0x66, 0xaa]);
    });
    it("falls back to white on garbage", () => {
        expect(hexToRgbBytes("xxx")).toEqual([255, 255, 255]);
    });
});

describe("cssColorToRgbBytes", () => {
    it("delegates hex to the fast path", () => {
        expect(cssColorToRgbBytes("#000000")).toEqual([0, 0, 0]);
    });
    it("parses rgb()", () => {
        expect(cssColorToRgbBytes("rgb(10, 20, 30)")).toEqual([10, 20, 30]);
    });
    it("parses named colors", () => {
        expect(cssColorToRgbBytes("white")).toEqual([255, 255, 255]);
    });
});

describe("cssColorToLinearRgb", () => {
    it("white maps to (1, 1, 1)", () => {
        const [r, g, b] = cssColorToLinearRgb("#ffffff");
        expect(r).toBeCloseTo(1);
        expect(g).toBeCloseTo(1);
        expect(b).toBeCloseTo(1);
    });
    it("black maps to (0, 0, 0)", () => {
        expect(cssColorToLinearRgb("#000000")).toEqual([0, 0, 0]);
    });
    it("mid-gray sRGB sits below 0.25 in linear", () => {
        const [r] = cssColorToLinearRgb("#808080");
        expect(r).toBeGreaterThan(0.2);
        expect(r).toBeLessThan(0.25);
    });
});
