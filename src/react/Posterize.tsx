import { useEffect, useRef, type ReactNode } from "react";
import { ShaderCanvas } from "./ShaderCanvas";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";
import { useMediaSource, type MediaKind } from "./useMediaSource";
import { Pipeline } from "../core/pipeline";
import {
    createPosterizePipelineConfig,
    createDefaultPosterizeParams,
    type PosterizeParams,
} from "../shaders/posterize";

export interface PosterizeProps {
    readonly children?: ReactNode;
    /** URL of an image or video. */
    readonly src: string;
    /** Force the loader. Auto-detected from the URL extension by default. */
    readonly kind?: MediaKind;
    /**
     * Levels per RGB channel. Total possible output colors equals `levels³`
     * (4 → 64, 6 → 216, 8 → 512). Quantization happens in sRGB (display)
     * space, so bands are perceptually evenly spaced. @default 4
     */
    readonly levels?: number;
    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
}

export function Posterize({
    children,
    src,
    kind,
    levels = 4,
    fillSection = true,
}: PosterizeProps) {
    const { getTexture } = useMediaSource(src, kind);
    const paramsRef = useRef<PosterizeParams | null>(null);

    const setup: ShaderSetup = ({ renderer }): ShaderSetupResult => {
        const live = createDefaultPosterizeParams();
        live.levels = levels;
        paramsRef.current = live;

        const { config } = createPosterizePipelineConfig({
            source: getTexture,
            params: live,
        });
        return { pipeline: new Pipeline(renderer, config) };
    };

    useEffect(() => {
        const live = paramsRef.current;
        if (!live) return;
        live.levels = levels;
    }, [levels]);

    return (
        <ShaderCanvas setup={setup} fillSection={fillSection}>
            {children}
        </ShaderCanvas>
    );
}
