import { useEffect, useRef, type ReactNode } from "react";
import { ShaderCanvas } from "./ShaderCanvas";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";
import { useMediaSource, type MediaKind } from "./useMediaSource";
import { Pipeline } from "../core/pipeline";
import {
    createEdgeDetectPipelineConfig,
    createDefaultEdgeDetectParams,
    type EdgeDetectParams,
} from "../shaders/edge-detect";

export interface EdgeDetectProps {
    readonly children?: ReactNode;
    /** URL of an image or video to feed into the edge detector. */
    readonly src: string;
    /** Force the loader. Auto-detected from the URL extension by default. */
    readonly kind?: MediaKind;
    /** Linear multiplier on Sobel gradient magnitude. @default 1.0 */
    readonly edgeStrength?: number;
    /** Smoothstep low-edge — gradients below this are clipped. @default 0.0 */
    readonly edgeThreshold?: number;
    /** Smoothstep high-edge — gradients above this are full white. @default 1.0 */
    readonly edgeKnee?: number;
    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
}

export function EdgeDetect({
    children,
    src,
    kind,
    edgeStrength = 1,
    edgeThreshold = 0,
    edgeKnee = 1,
    fillSection = true,
}: EdgeDetectProps) {
    const { getTexture } = useMediaSource(src, kind);
    const paramsRef = useRef<EdgeDetectParams | null>(null);

    const setup: ShaderSetup = ({ renderer }): ShaderSetupResult => {
        const params = createDefaultEdgeDetectParams();
        params.edgeStrength = edgeStrength;
        params.edgeThreshold = edgeThreshold;
        params.edgeKnee = edgeKnee;
        paramsRef.current = params;

        const { config } = createEdgeDetectPipelineConfig({
            source: getTexture,
            params,
        });
        return { pipeline: new Pipeline(renderer, config) };
    };

    useEffect(() => {
        const p = paramsRef.current;
        if (!p) return;
        p.edgeStrength = edgeStrength;
        p.edgeThreshold = edgeThreshold;
        p.edgeKnee = edgeKnee;
    }, [edgeStrength, edgeThreshold, edgeKnee]);

    return (
        <ShaderCanvas setup={setup} fillSection={fillSection}>
            {children}
        </ShaderCanvas>
    );
}
