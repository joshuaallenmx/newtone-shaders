import { parseToMask } from "../parse";
import { classifySkin } from "../skin";
import { createDefaultSkinMaskParams } from "../shaders/skin-mask";
import { composeMasks, intersectMasks, subtractMasks } from "./compose";
import type {
    PipelineDeps,
    PipelineInput,
    PipelineRegion,
    PipelineResult,
    RunPipelineOptions,
} from "./types";
import type { BinaryMask } from "./types";

function imageDimensions(image: PipelineInput): {
    width: number;
    height: number;
} {
    if (image instanceof HTMLImageElement) {
        return {
            width: image.naturalWidth || image.width,
            height: image.naturalHeight || image.height,
        };
    }
    return { width: image.width, height: image.height };
}

/**
 * Run all three tiers in sequence and return per-region masks plus the
 * combined union mask in source-image space. The detector's bounding
 * boxes are used as box prompts for SAM 2 — there is no manual prompt
 * input.
 *
 * Stages, in order:
 *   1. (optional) classify — if a classifier is provided.
 *   2. detect — NudeNet bounding boxes.
 *   3. encode — SAM 2 vision encoder runs once.
 *   4. segment — SAM 2 mask decoder runs once per region.
 *   5. compose — element-wise union into `combinedMask`.
 *
 * If detection returns no regions, encoder/decoder are skipped and an
 * empty mask of the right size is returned.
 */
export async function runPipeline(
    image: PipelineInput,
    deps: PipelineDeps,
    options: RunPipelineOptions = {},
): Promise<PipelineResult> {
    const onStage = options.onStage;
    const { width, height } = imageDimensions(image);

    let classification = null;
    if (deps.classifier) {
        onStage?.({ kind: "classifying" });
        classification = await deps.classifier.classify(image);
    }

    onStage?.({ kind: "detecting" });
    const detected = await deps.detector.detect(image, {
        scoreThreshold: options.detectScoreThreshold,
        iouThreshold: options.detectIouThreshold,
        classes: options.classes,
    });

    if (detected.length === 0) {
        onStage?.({ kind: "done" });
        return {
            classification,
            regions: [],
            combinedMask: {
                data: new Uint8Array(width * height),
                width,
                height,
            },
            width,
            height,
        };
    }

    let parseExcludeMask: BinaryMask | null = null;
    if (options.parseExclusion) {
        onStage?.({ kind: "parsing" });
        const parseResult = await options.parseExclusion.parser.parse(image);
        parseExcludeMask = parseToMask(
            parseResult,
            options.parseExclusion.classes,
        );
    }

    let skinMap: BinaryMask | null = null;
    if (options.skinMask) {
        onStage?.({ kind: "skin-classifying" });
        const params =
            options.skinMask === true
                ? createDefaultSkinMaskParams()
                : options.skinMask;
        skinMap = classifySkin(image, {
            params,
            threshold: options.skinThreshold,
        });
    }

    onStage?.({ kind: "encoding" });
    await deps.segmenter.setImage(image);

    const regions: PipelineRegion[] = [];
    for (let i = 0; i < detected.length; i++) {
        const region = detected[i];
        onStage?.({
            kind: "segmenting",
            index: i,
            total: detected.length,
            region,
        });
        const mask = await deps.segmenter.segmentBox(region.box);
        let regionMask: BinaryMask = {
            data: mask.data,
            width: mask.width,
            height: mask.height,
        };
        if (skinMap) {
            regionMask = intersectMasks(regionMask, skinMap);
        }
        if (parseExcludeMask) {
            regionMask = subtractMasks(regionMask, parseExcludeMask);
        }
        regions.push({
            ...region,
            mask: regionMask,
            maskScore: mask.score,
        });
    }

    onStage?.({ kind: "composing" });
    const combinedMask = composeMasks(
        regions.map((r) => r.mask),
        width,
        height,
    );

    onStage?.({ kind: "done" });
    return { classification, regions, combinedMask, width, height };
}
