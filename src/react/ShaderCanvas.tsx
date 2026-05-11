import { useRef, type CSSProperties, type ReactNode } from "react";
import { useShaderPipeline, type ShaderSetup } from "./useShaderPipeline";

const WRAP_STYLE: CSSProperties = {
    position: "relative",
    width: "100%",
    height: "100%",
    flex: "1 1 0",
    alignSelf: "stretch",
    minHeight: 0,
    isolation: "isolate",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
};

const CANVAS_HOST_BASE: CSSProperties = {
    position: "absolute",
    zIndex: 0,
    pointerEvents: "none",
};

const CONTENT_HOST_STYLE: CSSProperties = {
    position: "relative",
    zIndex: 1,
    width: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    pointerEvents: "none",
};

const OVERLAY_HOST_BASE: CSSProperties = {
    position: "absolute",
    zIndex: 2,
    pointerEvents: "none",
};

export interface ShaderCanvasProps {
    /** Called once on mount with the host elements + renderer. */
    readonly setup: ShaderSetup;
    /** Render above the canvas with `z-index: 1`. */
    readonly children?: ReactNode;
    /** Apply bleed-fit so the canvas covers the parent section. @default true */
    readonly fillSection?: boolean;
    /**
     * CSS pixels the canvas extends past the visible wrap on each side. The
     * wrap has `overflow: hidden`, so this region is clipped from view —
     * useful for putting a barrier mask off-screen so fluid is contained
     * without a visible wall. @default 0
     */
    readonly canvasOverflow?: number;
    /**
     * Mount a 2D-canvas overlay host above the WebGL canvas for shaders that
     * draw vector strokes / labels via `setup.overlayHost`. @default false
     */
    readonly overlay?: boolean;
}

/**
 * Generic shader-canvas primitive. Mounts a `WebGLRenderer` into a host div,
 * runs a rAF loop driving `Pipeline.render`, and renders children layered
 * above the canvas. Use this for any shader; the LiquidMetal component is a
 * thin wrapper around it.
 */
export function ShaderCanvas({
    setup,
    children,
    fillSection = true,
    canvasOverflow = 0,
    overlay = false,
}: ShaderCanvasProps) {
    const wrapRef = useRef<HTMLDivElement>(null);
    const canvasHostRef = useRef<HTMLDivElement>(null);
    const contentHostRef = useRef<HTMLDivElement>(null);
    const overlayHostRef = useRef<HTMLDivElement>(null);
    useShaderPipeline({
        wrapRef,
        canvasHostRef,
        contentHostRef,
        overlayHostRef: overlay ? overlayHostRef : undefined,
        setup,
        fillSection,
    });
    const canvasHostStyle: CSSProperties = {
        ...CANVAS_HOST_BASE,
        inset: -canvasOverflow,
    };
    const overlayHostStyle: CSSProperties = {
        ...OVERLAY_HOST_BASE,
        inset: -canvasOverflow,
    };
    return (
        <div ref={wrapRef} style={WRAP_STYLE}>
            <div ref={canvasHostRef} style={canvasHostStyle} />
            {overlay ? <div ref={overlayHostRef} style={overlayHostStyle} /> : null}
            <div ref={contentHostRef} style={CONTENT_HOST_STYLE}>
                {children}
            </div>
        </div>
    );
}
