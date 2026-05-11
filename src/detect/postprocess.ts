import { NUDENET_CLASSES } from "./classes";
import type { DetectedRegion } from "./types";

interface Candidate {
    readonly box: [number, number, number, number];
    readonly score: number;
    readonly classId: number;
}

function iou(
    a: readonly [number, number, number, number],
    b: readonly [number, number, number, number],
): number {
    const [ax, ay, aw, ah] = a;
    const [bx, by, bw, bh] = b;
    const x1 = Math.max(ax, bx);
    const y1 = Math.max(ay, by);
    const x2 = Math.min(ax + aw, bx + bw);
    const y2 = Math.min(ay + ah, by + bh);
    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const inter = interW * interH;
    const union = aw * ah + bw * bh - inter;
    return union > 0 ? inter / union : 0;
}

function nms(
    candidates: ReadonlyArray<Candidate>,
    iouThreshold: number,
): Candidate[] {
    const sorted = [...candidates].sort((a, b) => b.score - a.score);
    const kept: Candidate[] = [];
    const suppressed = new Uint8Array(sorted.length);
    for (let i = 0; i < sorted.length; i++) {
        if (suppressed[i]) continue;
        kept.push(sorted[i]);
        for (let j = i + 1; j < sorted.length; j++) {
            if (suppressed[j]) continue;
            if (iou(sorted[i].box, sorted[j].box) > iouThreshold) {
                suppressed[j] = 1;
            }
        }
    }
    return kept;
}

export interface PostprocessOptions {
    readonly scoreThreshold: number;
    readonly iouThreshold: number;
    readonly classes?: ReadonlyArray<string>;
}

/**
 * Decode a NudeNet YOLOv8-style output tensor into source-pixel
 * `DetectedRegion`s.
 *
 * Output layout (matching `notAI-tech/NudeNet`'s 320n / 640m exports):
 *  - Shape `[1, 4 + numClasses, numBoxes]` — channels are `[cx, cy, w, h,
 *    cls0, cls1, …]`, no objectness score, coordinates are absolute pixels
 *    in the model's input frame (0..inputSize).
 *  - The image was placed top-left in a `maxSize × maxSize` square then
 *    resized to `inputSize`. Scaling back from input → source is therefore
 *    a single multiplication by `maxSize / inputSize`.
 */
export function postprocess(
    output: Float32Array,
    dims: ReadonlyArray<number>,
    geometry: {
        readonly inputSize: number;
        readonly origWidth: number;
        readonly origHeight: number;
        readonly maxSize: number;
    },
    options: PostprocessOptions,
): DetectedRegion[] {
    if (dims.length !== 3 || dims[0] !== 1) {
        throw new Error(
            `postprocess: unexpected output rank, got dims=[${dims.join(", ")}]`,
        );
    }
    const numChannels = dims[1];
    const numBoxes = dims[2];
    const numClasses = numChannels - 4;
    if (numClasses !== NUDENET_CLASSES.length) {
        throw new Error(
            `postprocess: expected ${NUDENET_CLASSES.length} classes, got ${numClasses}`,
        );
    }

    const allow = options.classes ? new Set(options.classes) : null;
    const { inputSize, origWidth, origHeight, maxSize } = geometry;
    const scale = maxSize / inputSize;

    const candidates: Candidate[] = [];
    for (let i = 0; i < numBoxes; i++) {
        let maxScore = -Infinity;
        let maxClass = -1;
        for (let c = 0; c < numClasses; c++) {
            const s = output[(4 + c) * numBoxes + i];
            if (s > maxScore) {
                maxScore = s;
                maxClass = c;
            }
        }
        if (maxScore < options.scoreThreshold) continue;
        if (allow && !allow.has(NUDENET_CLASSES[maxClass])) continue;

        const cx = output[i];
        const cy = output[numBoxes + i];
        const w = output[2 * numBoxes + i];
        const h = output[3 * numBoxes + i];

        let x = (cx - w / 2) * scale;
        let y = (cy - h / 2) * scale;
        let bw = w * scale;
        let bh = h * scale;

        x = Math.max(0, Math.min(x, origWidth));
        y = Math.max(0, Math.min(y, origHeight));
        bw = Math.min(bw, origWidth - x);
        bh = Math.min(bh, origHeight - y);
        if (bw <= 0 || bh <= 0) continue;

        candidates.push({
            box: [x, y, bw, bh],
            score: maxScore,
            classId: maxClass,
        });
    }

    return nms(candidates, options.iouThreshold).map((c) => ({
        class: NUDENET_CLASSES[c.classId],
        box: c.box,
        score: c.score,
    }));
}
