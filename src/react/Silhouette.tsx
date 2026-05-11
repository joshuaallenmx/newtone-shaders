import {
    forwardRef,
    useEffect,
    useImperativeHandle,
    useMemo,
    useRef,
    type ReactNode,
} from "react";
import { ShaderCanvas } from "./ShaderCanvas";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";
import { useMediaSource, type MediaKind } from "./useMediaSource";
import { Pipeline } from "../core/pipeline";
import {
    sampleCornerColorFromTexture,
    type SampleCornerColorOptions,
} from "../core/textures";
import type { RGB } from "../core/cluster";
import {
    createSilhouettePipelineConfig,
    createDefaultSilhouetteParams,
    type SilhouetteMode,
    type SilhouetteParams,
} from "../shaders/silhouette";

export interface SilhouetteHandle {
    /**
     * Auto-pick the reference background color by averaging the four corners
     * of the current frame. Returns the picked color (0..255 byte RGB) or
     * `null` if the source isn't sampleable yet (e.g. video metadata still
     * loading).
     */
    sampleBackground(opts?: SampleCornerColorOptions): RGB | null;
}

export interface SilhouetteProps {
    readonly children?: ReactNode;
    /** URL of an image or video. */
    readonly src: string;
    /** Force the loader. Auto-detected from the URL extension by default. */
    readonly kind?: MediaKind;
    /** Reference background color (byte RGB, 0..255). @default [255, 255, 255] */
    readonly referenceColor?: RGB;
    /**
     * Pre-blur radius in source pixels. Larger → only major silhouettes
     * survive (small surface detail is averaged out). Set to 0 to disable.
     * @default 4
     */
    readonly smoothRadius?: number;
    /** Distance threshold (0..1, RGB-Euclidean). @default 0.18 */
    readonly threshold?: number;
    /** Smoothstep half-width on the threshold. @default 0.04 */
    readonly feather?: number;
    /** Half-width of the threshold sweep used in `stable` mode. @default 0.1 */
    readonly thresholdSpread?: number;
    /** Gradient sample radius in source pixels — controls outline thickness. @default 1.5 */
    readonly outlineThickness?: number;
    /** Output composition. @default "outline" */
    readonly mode?: SilhouetteMode;
    /** Background dim factor in overlay mode. @default 0.15 */
    readonly offMix?: number;
    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
}

const DEFAULT_REFERENCE: RGB = [255, 255, 255];

export const Silhouette = forwardRef<SilhouetteHandle, SilhouetteProps>(
    function Silhouette(
        {
            children,
            src,
            kind,
            referenceColor = DEFAULT_REFERENCE,
            smoothRadius = 4,
            threshold = 0.18,
            feather = 0.04,
            thresholdSpread = 0.1,
            outlineThickness = 1.5,
            mode = "outline",
            offMix = 0.15,
            fillSection = true,
        },
        ref,
    ) {
        const { getTexture } = useMediaSource(src, kind);
        const paramsRef = useRef<SilhouetteParams | null>(null);
        const getTextureRef = useRef(getTexture);
        getTextureRef.current = getTexture;

        const refKey = useMemo(
            () => referenceColor.join(","),
            [referenceColor],
        );

        const setup: ShaderSetup = ({ renderer }): ShaderSetupResult => {
            const live = createDefaultSilhouetteParams();
            live.referenceColor.set(
                referenceColor[0] / 255,
                referenceColor[1] / 255,
                referenceColor[2] / 255,
            );
            live.smoothRadius = smoothRadius;
            live.threshold = threshold;
            live.feather = feather;
            live.thresholdSpread = thresholdSpread;
            live.outlineThickness = outlineThickness;
            live.mode = mode;
            live.offMix = offMix;
            paramsRef.current = live;

            const { config } = createSilhouettePipelineConfig({
                source: () => getTextureRef.current(),
                params: live,
            });
            return { pipeline: new Pipeline(renderer, config) };
        };

        useEffect(() => {
            const live = paramsRef.current;
            if (!live) return;
            live.referenceColor.set(
                referenceColor[0] / 255,
                referenceColor[1] / 255,
                referenceColor[2] / 255,
            );
            live.smoothRadius = smoothRadius;
            live.threshold = threshold;
            live.feather = feather;
            live.thresholdSpread = thresholdSpread;
            live.outlineThickness = outlineThickness;
            live.mode = mode;
            live.offMix = offMix;
            // eslint-disable-next-line react-hooks/exhaustive-deps
        }, [
            refKey,
            smoothRadius,
            threshold,
            feather,
            thresholdSpread,
            outlineThickness,
            mode,
            offMix,
        ]);

        useImperativeHandle(
            ref,
            () => ({
                sampleBackground(opts?: SampleCornerColorOptions) {
                    const tex = getTextureRef.current();
                    return sampleCornerColorFromTexture(tex, opts);
                },
            }),
            [],
        );

        return (
            <ShaderCanvas setup={setup} fillSection={fillSection}>
                {children}
            </ShaderCanvas>
        );
    },
);

Silhouette.displayName = "Silhouette";
