import type { ClassificationProbabilities, NsfwClassifier } from "../classify";
import type { DetectedRegion, NsfwDetector } from "../detect";
import type { HumanParser } from "../parse";
import type { ImageSegmenter, SegmentSource } from "../segment";
import type { SkinMaskParams } from "../shaders/skin-mask";

/**
 * Source-pixel binary mask — 0 outside, 255 inside. Both
 * `pipeline.combinedMask` and per-region `mask` properties share this
 * shape so consumers can blit either into a `THREE.DataTexture` (raw
 * `data`) or into a `THREE.CanvasTexture` (via `maskToCanvas`).
 */
export interface BinaryMask {
    readonly data: Uint8Array;
    readonly width: number;
    readonly height: number;
}

export interface PipelineRegion extends DetectedRegion {
    /** Source-pixel binary mask for this region (NOT cropped to box). */
    readonly mask: BinaryMask;
    /** SAM 2 mask quality score (IoU prediction). */
    readonly maskScore: number;
}

export interface PipelineResult {
    /**
     * Classification probabilities, or `null` when the pipeline was run
     * without a classifier.
     */
    readonly classification: ClassificationProbabilities | null;
    readonly regions: ReadonlyArray<PipelineRegion>;
    /**
     * Union of all per-region masks in source-image dimensions. Empty
     * (all-zero) when no regions were detected.
     */
    readonly combinedMask: BinaryMask;
    readonly width: number;
    readonly height: number;
}

/**
 * The image type accepted by every tier. We intersect `ClassifyInput`,
 * `DetectInput`, and `SegmentSource` to the types each tier reliably
 * supports.
 */
export type PipelineInput = HTMLImageElement | HTMLCanvasElement;

export interface PipelineDeps {
    readonly detector: NsfwDetector;
    readonly segmenter: ImageSegmenter;
    readonly classifier?: NsfwClassifier;
}

export interface RunPipelineOptions {
    /** Forwarded to the detector. */
    readonly detectScoreThreshold?: number;
    /** Forwarded to the detector. */
    readonly detectIouThreshold?: number;
    /** Optional NudeNet class allowlist. */
    readonly classes?: ReadonlyArray<string>;
    /**
     * Constrain each SAM mask to skin pixels via a YCbCr classifier.
     * `false`/undefined disables; `true` uses default params; pass a
     * `SkinMaskParams` object to tune the YCbCr windows.
     */
    readonly skinMask?: boolean | SkinMaskParams;
    /** Skin-score threshold when `skinMask` is enabled. @default 0.5 */
    readonly skinThreshold?: number;
    /**
     * Subtract pixels classified by the human-parser as one of the listed
     * classes from each SAM mask. Used as a negative filter against
     * accessories / background / face / hair where the SegFormer-B2 model
     * is highly confident. The caller owns the parser's lifecycle.
     */
    readonly parseExclusion?: {
        readonly parser: HumanParser;
        readonly classes: ReadonlyArray<number>;
    };
    /**
     * Receives per-stage progress events so callers can drive a status
     * indicator without instrumenting individual tiers.
     */
    readonly onStage?: (event: PipelineStage) => void;
}

export type PipelineStage =
    | { readonly kind: "classifying" }
    | { readonly kind: "detecting" }
    | { readonly kind: "parsing" }
    | { readonly kind: "skin-classifying" }
    | { readonly kind: "encoding" }
    | {
          readonly kind: "segmenting";
          readonly index: number;
          readonly total: number;
          readonly region: DetectedRegion;
      }
    | { readonly kind: "composing" }
    | { readonly kind: "done" };

// Re-export for callers building on top of the pipeline.
export type { SegmentSource };
