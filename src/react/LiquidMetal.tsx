import { useEffect, useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";
import { ShaderCanvas } from "./ShaderCanvas";
import { useColorResolver } from "./ColorResolverProvider";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";
import {
    createPointerSignal,
    createMutationSignal,
    createScrollSignal,
    createTimeSignal,
    type PointerState,
    type ScrollState,
    type Signal,
} from "../core/signals";
import {
    makeNoiseTexture,
    makeEnvTexture,
    updateEnvTexture,
    createSvgMaskBuilder,
    loadImageTexture,
    type SvgMaskBuilder,
} from "../core/textures";
import { setVec3FromColor } from "../core/color";
import { Pipeline } from "../core/pipeline";
import {
    createLiquidMetalPipelineConfig,
    createDefaultColorUniforms,
    createDefaultParams,
    type LiquidMetalColorUniforms,
    type LiquidMetalParams,
} from "../shaders/liquid-metal";

/**
 * Anywhere a color is accepted, you can pass a CSS color (`"#ff66aa"`,
 * `"oklch(...)"`, `"rgb(...)"`) or a token ref like `"$text"` resolved by an
 * outer `<ColorResolverProvider>`.
 */
type ColorInput = string;

export interface LiquidMetalProps {
    readonly children?: ReactNode;
    /** Final tint multiplied over the metal. @default "#ffffff" */
    readonly tint?: ColorInput;
    /** Color the metal blends toward at its highlights. @default "#ffffff" */
    readonly baseColor?: ColorInput;
    /** 0..1 — how much sim pattern shows through. @default 0.35 */
    readonly contrast?: number;
    /** 0..1 — 0 desaturates fully toward luminance. @default 1 */
    readonly saturation?: number;
    /** Vertical gradient stops (top → bottom). @default ["#595294", "#e6de8c"] */
    readonly envColors?: readonly ColorInput[];
    /** URL of an image to use for reflection instead of the gradient. */
    readonly envImage?: string;
    /** Partial fluid-feel overrides, merged on top of `createDefaultParams()`. */
    readonly params?: Partial<LiquidMetalParams>;
    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
    /**
     * Width in CSS pixels of the implicit barrier band painted around the
     * canvas edge. Set to `0` to let fluid flow up to the edge unconstrained.
     * @default 8
     */
    readonly edgeBarrier?: number;
}

const DEFAULT_TINT = "#ffffff";
const DEFAULT_BASE_COLOR = "#ffffff";
const DEFAULT_CONTRAST = 0.35;
const DEFAULT_SATURATION = 1;
const DEFAULT_ENV_COLORS: readonly string[] = ["#595294", "#e6de8c"];

export function LiquidMetal({
    children,
    tint = DEFAULT_TINT,
    baseColor = DEFAULT_BASE_COLOR,
    contrast = DEFAULT_CONTRAST,
    saturation = DEFAULT_SATURATION,
    envColors = DEFAULT_ENV_COLORS,
    envImage,
    params,
    fillSection = true,
    edgeBarrier = 8,
}: LiquidMetalProps) {
    const resolver = useColorResolver();
    const resolvedTint = resolver(tint);
    const resolvedBaseColor = resolver(baseColor);
    const resolvedEnvColors = useMemo(
        () => envColors.map(resolver),
        [envColors, resolver],
    );
    const envKey = resolvedEnvColors.join("|");
    const paramsKey = useMemo(
        () => JSON.stringify(params ?? null),
        [params],
    );

    const colorUniformsRef = useRef<LiquidMetalColorUniforms | null>(null);
    const paramsRef = useRef<LiquidMetalParams | null>(null);
    const envTexRef = useRef<THREE.DataTexture | null>(null);
    const envImageTexRef = useRef<THREE.Texture | null>(null);
    const maskBuilderRef = useRef<SvgMaskBuilder | null>(null);
    const edgeBarrierRef = useRef(edgeBarrier);
    edgeBarrierRef.current = edgeBarrier;

    const setup: ShaderSetup = ({ canvasHost, contentHost, renderer }): ShaderSetupResult => {
        const colorUniforms = createDefaultColorUniforms();
        colorUniformsRef.current = colorUniforms;
        setVec3FromColor(colorUniforms.tint, resolvedTint);
        setVec3FromColor(colorUniforms.baseColor, resolvedBaseColor);
        colorUniforms.contrast = contrast;
        colorUniforms.saturation = saturation;

        const fluidParams = createDefaultParams();
        paramsRef.current = fluidParams;
        if (params) Object.assign(fluidParams, params);

        const noise = makeNoiseTexture();
        const env = makeEnvTexture(resolvedEnvColors);
        envTexRef.current = env;

        const maskBuilder = createSvgMaskBuilder({
            host: contentHost,
            canvasHost,
            dpr: () => renderer.getPixelRatio(),
            edgeBand: () => edgeBarrierRef.current,
        });
        maskBuilderRef.current = maskBuilder;

        const pointer = createPointerSignal({ host: canvasHost });
        const scroll = createScrollSignal();
        const time = createTimeSignal();
        const mutation = createMutationSignal({ host: contentHost });

        const offMutation = mutation.subscribe(() => {
            void maskBuilder.rebuild();
        });
        const retryTimers = [50, 250, 1000].map((ms) =>
            window.setTimeout(() => {
                void maskBuilder.rebuild();
            }, ms),
        );

        const { config } = createLiquidMetalPipelineConfig({
            signals: { pointer, scroll },
            textures: {
                noise,
                env: () => envImageTexRef.current ?? env,
                mask: maskBuilder.texture,
            },
            colorUniforms,
            params: fluidParams,
        });
        const pipeline = new Pipeline(renderer, config);

        const signals: ReadonlyArray<Signal<unknown>> = [
            pointer as Signal<PointerState> as Signal<unknown>,
            scroll as Signal<ScrollState> as Signal<unknown>,
            mutation as Signal<number> as Signal<unknown>,
        ];

        return {
            pipeline,
            signals,
            time,
            onResize: () => {
                void maskBuilder.rebuild();
            },
            dispose: () => {
                offMutation();
                retryTimers.forEach(clearTimeout);
                noise.dispose();
                env.dispose();
                envImageTexRef.current?.dispose();
                envImageTexRef.current = null;
                maskBuilder.dispose();
            },
        };
    };

    // Sync color props after setup has populated the live uniforms.
    useEffect(() => {
        const u = colorUniformsRef.current;
        if (!u) return;
        setVec3FromColor(u.tint, resolvedTint);
        setVec3FromColor(u.baseColor, resolvedBaseColor);
        u.contrast = contrast;
        u.saturation = saturation;
    }, [resolvedTint, resolvedBaseColor, contrast, saturation]);

    // Sync fluid params when they change.
    useEffect(() => {
        const p = paramsRef.current;
        if (!p) return;
        const fresh = createDefaultParams();
        if (params) Object.assign(fresh, params);
        Object.assign(p, fresh);
    }, [paramsKey, params]);

    // Rebuild the gradient when stops change.
    useEffect(() => {
        const tex = envTexRef.current;
        if (!tex) return;
        updateEnvTexture(tex, resolvedEnvColors);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [envKey]);

    // Rebuild the mask when the edge barrier changes (the builder reads it
    // through a getter, so we just trigger a rebuild).
    useEffect(() => {
        const mb = maskBuilderRef.current;
        if (!mb) return;
        void mb.rebuild();
    }, [edgeBarrier]);

    // Load / swap the env image when the URL changes.
    useEffect(() => {
        let cancelled = false;
        if (!envImage) {
            envImageTexRef.current?.dispose();
            envImageTexRef.current = null;
            return;
        }
        loadImageTexture(envImage)
            .then((tex) => {
                if (cancelled) {
                    tex.dispose();
                    return;
                }
                const previous = envImageTexRef.current;
                envImageTexRef.current = tex;
                previous?.dispose();
            })
            .catch(() => {
                // Stay on the gradient if the image fails to load.
            });
        return () => {
            cancelled = true;
        };
    }, [envImage]);

    // Push the canvas past the wrap by exactly the barrier width so the wall
    // sits in the clipped overshoot area — fluid is contained, but the wall
    // itself never paints inside the visible viewport.
    return (
        <ShaderCanvas
            setup={setup}
            fillSection={fillSection}
            canvasOverflow={edgeBarrier}
        >
            {children}
        </ShaderCanvas>
    );
}
