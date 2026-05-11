import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { ShaderCanvas } from "./ShaderCanvas";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";
import { useMediaSource, type MediaKind } from "./useMediaSource";
import { Pipeline } from "../core/pipeline";
import {
    createLowPolyPipelineConfig,
    createDefaultLowPolyParams,
    type LowPolyParams,
} from "../shaders/low-poly";

export interface LowPolyProps {
    readonly children?: ReactNode;
    /** URL of an image or video. */
    readonly src: string;
    /** Force the loader. Auto-detected from the URL extension by default. */
    readonly kind?: MediaKind;
    /** Partial overrides; merged on top of `createDefaultLowPolyParams()`. */
    readonly params?: Partial<LowPolyParams>;
    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
}

export function LowPoly({
    children,
    src,
    kind,
    params,
    fillSection = true,
}: LowPolyProps) {
    const { getTexture } = useMediaSource(src, kind);
    const paramsRef = useRef<LowPolyParams | null>(null);
    const paramsKey = useMemo(() => JSON.stringify(params ?? null), [params]);

    const setup: ShaderSetup = ({ renderer }): ShaderSetupResult => {
        const live = createDefaultLowPolyParams();
        if (params) Object.assign(live, params);
        paramsRef.current = live;

        const { config } = createLowPolyPipelineConfig({
            source: getTexture,
            params: live,
        });
        return { pipeline: new Pipeline(renderer, config) };
    };

    useEffect(() => {
        const live = paramsRef.current;
        if (!live) return;
        const fresh = createDefaultLowPolyParams();
        if (params) Object.assign(fresh, params);
        Object.assign(live, fresh);
    }, [paramsKey, params]);

    return (
        <ShaderCanvas setup={setup} fillSection={fillSection}>
            {children}
        </ShaderCanvas>
    );
}
