import {
    forwardRef,
    useEffect,
    useImperativeHandle,
    useRef,
    type ReactNode,
} from "react";
import { ShaderCanvas } from "./ShaderCanvas";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";
import { useMediaSource, type MediaKind } from "./useMediaSource";
import { Pipeline } from "../core/pipeline";
import {
    extractPaletteFromTexture,
    type ExtractPaletteOptions,
} from "../core/textures";
import type { RGB } from "../core/cluster";
import {
    createPaletteMaskPipelineConfig,
    createDefaultPaletteMaskParams,
    setPaletteFromRgb,
    type PaletteMaskMode,
    type PaletteMaskParams,
} from "../shaders/palette-mask";

export interface PaletteMaskHandle {
    /**
     * Sample the current frame and return a fresh palette via k-means.
     * Returns `null` if the source isn't sampleable yet (e.g. video metadata
     * still loading).
     */
    samplePalette(opts?: ExtractPaletteOptions): RGB[] | null;
}

export interface PaletteMaskProps {
    readonly children?: ReactNode;
    /** URL of an image or video. */
    readonly src: string;
    /** Force the loader. Auto-detected from the URL extension by default. */
    readonly kind?: MediaKind;
    /** Palette colors (byte RGB, [0..255]). @default [] */
    readonly palette?: ReadonlyArray<RGB>;
    /** Per-slot enabled flags. @default all true */
    readonly enabled?: ReadonlyArray<boolean>;
    /** Output mode. @default "posterize" */
    readonly mode?: PaletteMaskMode;
    /** Brightness multiplier on disabled entries in posterize/overlay. @default 0.15 */
    readonly offMix?: number;
    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
}

const EMPTY_PALETTE: ReadonlyArray<RGB> = [];

export const PaletteMask = forwardRef<PaletteMaskHandle, PaletteMaskProps>(
    function PaletteMask(
        {
            children,
            src,
            kind,
            palette = EMPTY_PALETTE,
            enabled,
            mode = "posterize",
            offMix = 0.15,
            fillSection = true,
        },
        ref,
    ) {
        const { getTexture } = useMediaSource(src, kind);
        const paramsRef = useRef<PaletteMaskParams | null>(null);
        const getTextureRef = useRef(getTexture);
        getTextureRef.current = getTexture;

        const setup: ShaderSetup = ({ renderer }): ShaderSetupResult => {
            const live = createDefaultPaletteMaskParams();
            setPaletteFromRgb(live, palette, enabled);
            live.mode = mode;
            live.offMix = offMix;
            paramsRef.current = live;

            const { config } = createPaletteMaskPipelineConfig({
                source: () => getTextureRef.current(),
                params: live,
            });
            return { pipeline: new Pipeline(renderer, config) };
        };

        useEffect(() => {
            const live = paramsRef.current;
            if (!live) return;
            setPaletteFromRgb(live, palette, enabled);
            live.mode = mode;
            live.offMix = offMix;
        }, [palette, enabled, mode, offMix]);

        useImperativeHandle(
            ref,
            () => ({
                samplePalette(opts?: ExtractPaletteOptions) {
                    const tex = getTextureRef.current();
                    return extractPaletteFromTexture(tex, opts);
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
