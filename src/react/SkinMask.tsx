import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { ShaderCanvas } from "./ShaderCanvas";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";
import { useMediaSource, type MediaKind } from "./useMediaSource";
import { Pipeline } from "../core/pipeline";
import {
    createSkinMaskPipelineConfig,
    createDefaultSkinMaskParams,
    type SkinMaskParams,
} from "../shaders/skin-mask";

export interface SkinMaskProps {
    readonly children?: ReactNode;
    /** URL of an image or video to feed into the mask. */
    readonly src: string;
    /** Force the loader. Auto-detected from the URL extension by default. */
    readonly kind?: MediaKind;
    /** Partial threshold overrides; merged on top of `createDefaultSkinMaskParams()`. */
    readonly params?: Partial<SkinMaskParams>;
    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
}

export function SkinMask({
    children,
    src,
    kind,
    params,
    fillSection = true,
}: SkinMaskProps) {
    const { getTexture } = useMediaSource(src, kind);
    const paramsRef = useRef<SkinMaskParams | null>(null);
    const paramsKey = useMemo(() => JSON.stringify(params ?? null), [params]);

    const setup: ShaderSetup = ({ renderer }): ShaderSetupResult => {
        const live = createDefaultSkinMaskParams();
        if (params) Object.assign(live, params);
        paramsRef.current = live;

        const { config } = createSkinMaskPipelineConfig({
            source: getTexture,
            params: live,
        });
        return { pipeline: new Pipeline(renderer, config) };
    };

    useEffect(() => {
        const live = paramsRef.current;
        if (!live) return;
        const fresh = createDefaultSkinMaskParams();
        if (params) Object.assign(fresh, params);
        Object.assign(live, fresh);
    }, [paramsKey, params]);

    return (
        <ShaderCanvas setup={setup} fillSection={fillSection}>
            {children}
        </ShaderCanvas>
    );
}
