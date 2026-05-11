import { useEffect, useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";
import { ShaderCanvas } from "./ShaderCanvas";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";
import { useMediaSource, type MediaKind } from "./useMediaSource";
import { Pipeline } from "../core/pipeline";
import { findAccumulatorPeaks, type RGB } from "../core/cluster";
import {
    createCircleDetectPipelineConfig,
    createDefaultCircleDetectParams,
    CIRCLE_HOUGH_PASS_ID,
    type CircleDetectParams,
    type DetectedCircle,
} from "../shaders/circle-detect";

/** Subset of params accepted via the `params` prop. */
export type CircleDetectParamsInput = Partial<CircleDetectParams>;

export interface CircleDetectProps {
    readonly children?: ReactNode;
    /** URL of an image or video. */
    readonly src: string;
    /** Force the loader. Auto-detected from the URL extension by default. */
    readonly kind?: MediaKind;
    /** Detection params (radius, samples, threshold, output mode, etc.). */
    readonly params?: CircleDetectParamsInput;

    /** Stroke color for the 2D overlay (byte RGB). @default [255, 80, 120] */
    readonly strokeColor?: RGB;
    /** Stroke width in CSS pixels. @default 2 */
    readonly strokeWidth?: number;
    /** Stroke alpha. @default 1 */
    readonly strokeOpacity?: number;
    /** Optional fill (byte RGB) — pass `null` for stroke only. @default null */
    readonly fillColor?: RGB | null;
    /** Fill alpha. @default 0.15 */
    readonly fillOpacity?: number;

    /** Maximum circles drawn per frame. @default 32 */
    readonly maxCircles?: number;
    /**
     * Non-max-suppression radius in accumulator pixels. Defaults to
     * `params.radius * 0.5` so peaks closer than the circle's radius can't
     * both fire.
     */
    readonly suppressionRadius?: number;
    /**
     * Read back + redraw every Nth frame. `1` = every frame (smoothest);
     * raise for cheaper video. @default 1
     */
    readonly readbackEvery?: number;
    /**
     * Notified after every readback with the latest detected circles in
     * CSS-pixel space. Use for labels, exports, or custom overlays.
     */
    readonly onCirclesDetected?: (circles: ReadonlyArray<DetectedCircle>) => void;

    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
}

const DEFAULT_STROKE: RGB = [255, 80, 120];

function rgbaCss(rgb: RGB, alpha: number): string {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

interface OverlayBuffers {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    accBuf: Uint8Array;
    accW: number;
    accH: number;
}

export function CircleDetect({
    children,
    src,
    kind,
    params,
    strokeColor = DEFAULT_STROKE,
    strokeWidth = 2,
    strokeOpacity = 1,
    fillColor = null,
    fillOpacity = 0.15,
    maxCircles = 32,
    suppressionRadius,
    readbackEvery = 1,
    onCirclesDetected,
    fillSection = true,
}: CircleDetectProps) {
    const { getTexture } = useMediaSource(src, kind);
    const paramsRef = useRef<CircleDetectParams | null>(null);
    const paramsKey = useMemo(() => JSON.stringify(params ?? null), [params]);

    // Style props captured in a single ref so the rAF loop reads the latest
    // values without re-running setup on every change.
    const styleRef = useRef({
        strokeColor,
        strokeWidth,
        strokeOpacity,
        fillColor,
        fillOpacity,
        maxCircles,
        suppressionRadius,
        readbackEvery,
        onCirclesDetected,
    });
    styleRef.current = {
        strokeColor,
        strokeWidth,
        strokeOpacity,
        fillColor,
        fillOpacity,
        maxCircles,
        suppressionRadius,
        readbackEvery,
        onCirclesDetected,
    };

    const setup: ShaderSetup = ({
        renderer,
        overlayHost,
    }): ShaderSetupResult => {
        const live = createDefaultCircleDetectParams();
        if (params) Object.assign(live, params);
        paramsRef.current = live;

        const { config } = createCircleDetectPipelineConfig({
            source: getTexture,
            params: live,
        });
        const pipeline = new Pipeline(renderer, config);

        let buffers: OverlayBuffers | null = null;
        if (overlayHost) {
            const canvas = document.createElement("canvas");
            canvas.style.position = "absolute";
            canvas.style.inset = "0";
            canvas.style.width = "100%";
            canvas.style.height = "100%";
            canvas.style.pointerEvents = "none";
            const ctx = canvas.getContext("2d");
            if (ctx) {
                overlayHost.appendChild(canvas);
                buffers = {
                    canvas,
                    ctx,
                    accBuf: new Uint8Array(0),
                    accW: 0,
                    accH: 0,
                };
            }
        }

        const ensureAccBuffer = (w: number, h: number) => {
            if (!buffers) return;
            if (buffers.accW === w && buffers.accH === h) return;
            buffers.accBuf = new Uint8Array(w * h * 4);
            buffers.accW = w;
            buffers.accH = h;
        };

        const reusableCircles: DetectedCircle[] = [];
        const tmpScratchVec = new THREE.Vector2();

        return {
            pipeline,
            onResize: (renderW, renderH) => {
                if (!buffers) return;
                const dpr = renderer.getPixelRatio();
                const cssW = renderW / dpr;
                const cssH = renderH / dpr;
                buffers.canvas.width = renderW;
                buffers.canvas.height = renderH;
                buffers.canvas.style.width = `${cssW}px`;
                buffers.canvas.style.height = `${cssH}px`;
                buffers.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            },
            onFrame: ({ frame }) => {
                if (!buffers) return;
                const style = styleRef.current;
                const every = Math.max(1, Math.round(style.readbackEvery));
                if (frame % every !== 0) return;

                const liveParams = paramsRef.current;
                if (!liveParams) return;

                const target = pipeline.getPassTarget(CIRCLE_HOUGH_PASS_ID);
                if (!target) return;
                const accW = target.width;
                const accH = target.height;
                if (accW <= 0 || accH <= 0) return;
                ensureAccBuffer(accW, accH);

                renderer.readRenderTargetPixels(
                    target,
                    0,
                    0,
                    accW,
                    accH,
                    buffers.accBuf,
                );

                const suppression =
                    style.suppressionRadius ??
                    Math.max(2, liveParams.radius * 0.5);
                const peaks = findAccumulatorPeaks({
                    width: accW,
                    height: accH,
                    pixels: buffers.accBuf,
                    threshold: liveParams.minScore,
                    suppressionRadius: suppression,
                    maxPeaks: style.maxCircles,
                    flipY: true,
                });

                // Map accumulator-pixel space → CSS pixels.
                renderer.getDrawingBufferSize(tmpScratchVec);
                const dpr = renderer.getPixelRatio();
                const renderW = tmpScratchVec.x;
                const renderH = tmpScratchVec.y;
                const scaleX = renderW / accW / dpr;
                const scaleY = renderH / accH / dpr;
                // Accumulator was rendered at half canvas-render res, so
                // params.radius (canvas-render-pixels) → CSS via /dpr.
                const rCss = liveParams.radius / dpr;

                reusableCircles.length = 0;
                for (const peak of peaks) {
                    reusableCircles.push({
                        cx: peak.cx * scaleX,
                        cy: peak.cy * scaleY,
                        r: rCss,
                        score: peak.score,
                    });
                }

                const ctx = buffers.ctx;
                const cssW = renderW / dpr;
                const cssH = renderH / dpr;
                ctx.clearRect(0, 0, cssW, cssH);

                if (style.fillColor) {
                    ctx.fillStyle = rgbaCss(
                        style.fillColor,
                        style.fillOpacity,
                    );
                }
                ctx.strokeStyle = rgbaCss(
                    style.strokeColor,
                    style.strokeOpacity,
                );
                ctx.lineWidth = style.strokeWidth;

                for (const c of reusableCircles) {
                    ctx.beginPath();
                    ctx.arc(c.cx, c.cy, c.r, 0, Math.PI * 2);
                    if (style.fillColor) ctx.fill();
                    ctx.stroke();
                }

                style.onCirclesDetected?.(reusableCircles);
            },
            dispose: () => {
                if (buffers) {
                    buffers.canvas.remove();
                    buffers = null;
                }
            },
        };
    };

    useEffect(() => {
        const live = paramsRef.current;
        if (!live) return;
        const fresh = createDefaultCircleDetectParams();
        if (params) Object.assign(fresh, params);
        Object.assign(live, fresh);
    }, [paramsKey, params]);

    return (
        <ShaderCanvas setup={setup} fillSection={fillSection} overlay>
            {children}
        </ShaderCanvas>
    );
}
