import { describe, it, expect } from "vitest";
import { findAccumulatorPeaks } from "./peaks";

function makeBuffer(width: number, height: number): Uint8Array {
    return new Uint8Array(width * height * 4);
}

function setCell(
    buf: Uint8Array,
    width: number,
    x: number,
    y: number,
    value: number,
): void {
    const i = (y * width + x) * 4;
    buf[i] = value;
    buf[i + 1] = value;
    buf[i + 2] = value;
    buf[i + 3] = 255;
}

describe("findAccumulatorPeaks", () => {
    it("returns an empty list when nothing crosses the threshold", () => {
        const w = 8;
        const h = 8;
        const buf = makeBuffer(w, h);
        for (let i = 0; i < buf.length; i += 4) buf[i] = 50;
        const peaks = findAccumulatorPeaks({
            width: w,
            height: h,
            pixels: buf,
            threshold: 0.5,
            suppressionRadius: 1,
        });
        expect(peaks).toEqual([]);
    });

    it("extracts two distinct peaks sorted by score", () => {
        const w = 16;
        const h = 16;
        const buf = makeBuffer(w, h);
        // Strong peak at (3, 4)
        setCell(buf, w, 3, 4, 240);
        // Weaker peak at (12, 11)
        setCell(buf, w, 12, 11, 180);
        const peaks = findAccumulatorPeaks({
            width: w,
            height: h,
            pixels: buf,
            threshold: 0.4,
            suppressionRadius: 2,
        });
        expect(peaks.length).toBe(2);
        expect(peaks[0]).toMatchObject({ cx: 3, cy: 4 });
        expect(peaks[1]).toMatchObject({ cx: 12, cy: 11 });
        expect(peaks[0].score).toBeGreaterThan(peaks[1].score);
    });

    it("suppresses neighbors within the suppression radius", () => {
        const w = 16;
        const h = 16;
        const buf = makeBuffer(w, h);
        setCell(buf, w, 7, 7, 240);
        setCell(buf, w, 8, 7, 220); // neighbor — should be suppressed by NMS
        setCell(buf, w, 7, 8, 200);
        const peaks = findAccumulatorPeaks({
            width: w,
            height: h,
            pixels: buf,
            threshold: 0.4,
            suppressionRadius: 3,
        });
        expect(peaks.length).toBe(1);
        expect(peaks[0]).toMatchObject({ cx: 7, cy: 7 });
    });

    it("respects maxPeaks", () => {
        const w = 32;
        const h = 4;
        const buf = makeBuffer(w, h);
        for (let i = 0; i < 8; i++) {
            setCell(buf, w, i * 4, 1, 200 + i);
        }
        const peaks = findAccumulatorPeaks({
            width: w,
            height: h,
            pixels: buf,
            threshold: 0.4,
            suppressionRadius: 1,
            maxPeaks: 3,
        });
        expect(peaks.length).toBe(3);
    });

    it("flips Y when flipY is set", () => {
        const w = 8;
        const h = 8;
        const buf = makeBuffer(w, h);
        // Peak at row 1 (near the top of the WebGL buffer = bottom of the image).
        setCell(buf, w, 4, 1, 240);
        const peaks = findAccumulatorPeaks({
            width: w,
            height: h,
            pixels: buf,
            threshold: 0.4,
            suppressionRadius: 1,
            flipY: true,
        });
        expect(peaks).toHaveLength(1);
        // h - 1 - y = 8 - 1 - 1 = 6
        expect(peaks[0]).toMatchObject({ cx: 4, cy: 6 });
    });
});
