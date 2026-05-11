import type { BinaryMask } from "./types";

/**
 * Element-wise OR of any number of equally-sized binary masks. Inputs
 * with mismatched dimensions throw — masks are expected to already be
 * upsampled to source-image space (which is what SAM's
 * `post_process_masks` does).
 */
export function composeMasks(
    masks: ReadonlyArray<BinaryMask>,
    width: number,
    height: number,
): BinaryMask {
    const data = new Uint8Array(width * height);
    if (masks.length === 0) {
        return { data, width, height };
    }
    for (const m of masks) {
        if (m.width !== width || m.height !== height) {
            throw new Error(
                `composeMasks: mask dimensions ${m.width}x${m.height} do not match target ${width}x${height}`,
            );
        }
        const src = m.data;
        for (let i = 0; i < data.length; i++) {
            if (src[i]) data[i] = 255;
        }
    }
    return { data, width, height };
}

/**
 * Element-wise AND of two equally-sized binary masks. Used to constrain
 * a SAM mask to a skin classifier's output (or any other gating mask).
 */
export function intersectMasks(a: BinaryMask, b: BinaryMask): BinaryMask {
    if (a.width !== b.width || a.height !== b.height) {
        throw new Error(
            `intersectMasks: dimensions ${a.width}x${a.height} vs ${b.width}x${b.height}`,
        );
    }
    const data = new Uint8Array(a.width * a.height);
    const aD = a.data;
    const bD = b.data;
    for (let i = 0; i < data.length; i++) {
        if (aD[i] && bD[i]) data[i] = 255;
    }
    return { data, width: a.width, height: a.height };
}

/**
 * Element-wise `A AND NOT B`. Used to remove regions matched by an
 * exclusion mask (e.g. parse-classified accessories, background, face)
 * from an inclusion mask.
 */
export function subtractMasks(a: BinaryMask, b: BinaryMask): BinaryMask {
    if (a.width !== b.width || a.height !== b.height) {
        throw new Error(
            `subtractMasks: dimensions ${a.width}x${a.height} vs ${b.width}x${b.height}`,
        );
    }
    const data = new Uint8Array(a.width * a.height);
    const aD = a.data;
    const bD = b.data;
    for (let i = 0; i < data.length; i++) {
        if (aD[i] && !bD[i]) data[i] = 255;
    }
    return { data, width: a.width, height: a.height };
}
