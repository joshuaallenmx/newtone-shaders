import { useEffect, useRef, type RefObject } from "react";
import * as THREE from "three";
import type { Pipeline } from "../core/pipeline";
import type { Signal, TimeSignalControls } from "../core/signals";
import { applyBleedFit, type BleedFitController } from "../core/dom/bleed";

export interface ShaderSetupEnv {
    /** Outermost shader-canvas div — bleed-fit attaches here. */
    readonly wrap: HTMLDivElement;
    /** Where the WebGL canvas mounts. */
    readonly canvasHost: HTMLDivElement;
    /** Where children render (above the canvas). */
    readonly contentHost: HTMLDivElement;
    /**
     * Optional 2D overlay surface for shaders that draw on top of the WebGL
     * canvas (e.g. CPU-readback vector strokes). Present only when the
     * `<ShaderCanvas overlay>` prop is set; otherwise `null`.
     */
    readonly overlayHost: HTMLDivElement | null;
    /** Shared renderer. */
    readonly renderer: THREE.WebGLRenderer;
}

export interface ShaderFrameContext {
    readonly renderer: THREE.WebGLRenderer;
    readonly time: number;
    readonly frame: number;
}

export interface ShaderSetupResult {
    readonly pipeline: Pipeline;
    /** Signals to start before the rAF loop and stop on unmount. */
    readonly signals?: ReadonlyArray<Signal<unknown>>;
    /** Time signal — `tick()` is called once per render frame. */
    readonly time?: TimeSignalControls;
    /** Called on canvas resize with the new render-pixel dimensions. */
    readonly onResize?: (width: number, height: number) => void;
    /** Called once per frame after `pipeline.render(...)` returns. */
    readonly onFrame?: (ctx: ShaderFrameContext) => void;
    /** Custom cleanup beyond pipeline + signals. */
    readonly dispose?: () => void;
}

export type ShaderSetup = (env: ShaderSetupEnv) => ShaderSetupResult;

export interface UseShaderPipelineOptions {
    readonly wrapRef: RefObject<HTMLDivElement>;
    readonly canvasHostRef: RefObject<HTMLDivElement>;
    readonly contentHostRef: RefObject<HTMLDivElement>;
    /** Optional ref to an overlay host (a div sibling above the WebGL canvas). */
    readonly overlayHostRef?: RefObject<HTMLDivElement>;
    readonly setup: ShaderSetup;
    /** Apply the bleed-fit layout trick to `wrap`. @default true */
    readonly fillSection?: boolean;
}

/**
 * Mounts a WebGLRenderer into `canvasHost`, runs the user's setup once,
 * starts signals, drives a rAF loop calling `pipeline.render(...)` per frame,
 * and tears everything down on unmount. The setup function captures the
 * initial prop values; subsequent prop updates should mutate live objects
 * (e.g. uniform refs) shared with the setup closure.
 */
export function useShaderPipeline(opts: UseShaderPipelineOptions): void {
    const setupRef = useRef(opts.setup);
    setupRef.current = opts.setup;

    useEffect(() => {
        const wrap = opts.wrapRef.current;
        const canvasHost = opts.canvasHostRef.current;
        const contentHost = opts.contentHostRef.current;
        if (!wrap || !canvasHost || !contentHost) return;

        const renderer = new THREE.WebGLRenderer({
            antialias: false,
            alpha: true,
            premultipliedAlpha: false,
        });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        const canvasEl = renderer.domElement;
        canvasEl.style.display = "block";
        canvasEl.style.width = "100%";
        canvasEl.style.height = "100%";
        canvasHost.appendChild(canvasEl);

        let bleed: BleedFitController | null = null;
        let bleedObserver: ResizeObserver | null = null;
        if (opts.fillSection !== false) {
            bleed = applyBleedFit(wrap);
            const parent = wrap.parentElement;
            if (parent) {
                bleedObserver = new ResizeObserver(() => bleed?.update());
                bleedObserver.observe(parent);
            }
        }

        const setup = setupRef.current({
            wrap,
            canvasHost,
            contentHost,
            overlayHost: opts.overlayHostRef?.current ?? null,
            renderer,
        });

        for (const sig of setup.signals ?? []) sig.start();
        setup.time?.start();

        const sizeVec = new THREE.Vector2();
        const resize = () => {
            const w = canvasHost.clientWidth;
            const h = canvasHost.clientHeight;
            if (w === 0 || h === 0) return;
            renderer.setSize(w, h, false);
            const dpr = renderer.getPixelRatio();
            const W = Math.max(2, Math.round(w * dpr));
            const H = Math.max(2, Math.round(h * dpr));
            setup.pipeline.resize(W, H);
            setup.onResize?.(W, H);
        };
        const ro = new ResizeObserver(resize);
        ro.observe(canvasHost);
        resize();

        let raf = 0;
        const tick = () => {
            raf = requestAnimationFrame(tick);
            setup.time?.tick();
            const t = setup.time?.get().time ?? 0;
            const f = setup.time?.get().frame ?? 0;
            renderer.getDrawingBufferSize(sizeVec);
            setup.pipeline.render({
                time: t,
                frame: f,
                canvas: { w: sizeVec.x, h: sizeVec.y },
            });
            setup.onFrame?.({ renderer, time: t, frame: f });
        };
        tick();

        return () => {
            cancelAnimationFrame(raf);
            ro.disconnect();
            bleedObserver?.disconnect();
            bleed?.dispose();
            for (const sig of setup.signals ?? []) sig.stop();
            setup.time?.stop();
            setup.dispose?.();
            setup.pipeline.dispose();
            renderer.dispose();
            canvasEl.remove();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
}
