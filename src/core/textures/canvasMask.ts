import * as THREE from "three";

export interface SvgMaskBuilderOptions {
    /** Element whose descendants are rasterized into the mask. */
    readonly host: HTMLElement;
    /** Reference frame for relative positioning (typically the canvas host). */
    readonly canvasHost: HTMLElement;
    /** CSS selector for shapes to rasterize. @default "svg" */
    readonly selector?: string;
    /** Provider for the current device pixel ratio. */
    readonly dpr: () => number;
    /**
     * Width (in CSS pixels) of a white edge band painted at the mask border.
     * Pass a getter to have it re-read on every rebuild. @default 0
     */
    readonly edgeBand?: number | (() => number);
}

export interface SvgMaskBuilder {
    readonly canvas: HTMLCanvasElement;
    readonly texture: THREE.CanvasTexture;
    /** Rebuild the mask from the current DOM. SVG → Image → drawImage. */
    rebuild(): Promise<void>;
    /** Resize the underlying canvas. */
    setSize(width: number, height: number): void;
    dispose(): void;
}

/**
 * Walks the host's SVG (or any selector) descendants, rasterizes each into an
 * offscreen canvas at its bounding-rect position, and exposes a CanvasTexture.
 * Used by shaders that need a barrier mask matching DOM-rendered shapes.
 */
export function createSvgMaskBuilder(opts: SvgMaskBuilderOptions): SvgMaskBuilder {
    const selector = opts.selector ?? "svg";
    const readEdgeBand =
        typeof opts.edgeBand === "function"
            ? opts.edgeBand
            : () => (opts.edgeBand as number | undefined) ?? 0;
    const canvas = document.createElement("canvas");
    canvas.width = 2;
    canvas.height = 2;
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    let token = 0;

    const setSize = (w: number, h: number) => {
        const W = Math.max(2, Math.round(w));
        const H = Math.max(2, Math.round(h));
        if (canvas.width !== W) canvas.width = W;
        if (canvas.height !== H) canvas.height = H;
    };

    const rebuild = async () => {
        const myToken = ++token;
        const dpr = opts.dpr();
        setSize(opts.canvasHost.clientWidth * dpr, opts.canvasHost.clientHeight * dpr);
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const edgeBand = readEdgeBand();
        if (edgeBand > 0) {
            const e = Math.max(2, Math.round(edgeBand * dpr));
            ctx.fillStyle = "#fff";
            ctx.fillRect(0, 0, canvas.width, e);
            ctx.fillRect(0, canvas.height - e, canvas.width, e);
            ctx.fillRect(0, 0, e, canvas.height);
            ctx.fillRect(canvas.width - e, 0, e, canvas.height);
        }

        const hostRect = opts.canvasHost.getBoundingClientRect();
        const elements = Array.from(opts.host.querySelectorAll(selector));
        await Promise.all(
            elements.map(
                (el) =>
                    new Promise<void>((resolve) => {
                        const r = el.getBoundingClientRect();
                        const x = (r.left - hostRect.left) * dpr;
                        const y = (r.top - hostRect.top) * dpr;
                        const w = r.width * dpr;
                        const h = r.height * dpr;
                        if (w <= 0 || h <= 0) return resolve();
                        if (el instanceof SVGElement) {
                            const clone = el.cloneNode(true) as SVGElement;
                            clone.setAttribute("style", "color:#ffffff");
                            clone.setAttribute("width", String(w));
                            clone.setAttribute("height", String(h));
                            const xml = new XMLSerializer().serializeToString(clone);
                            const blob = new Blob([xml], {
                                type: "image/svg+xml;charset=utf-8",
                            });
                            const url = URL.createObjectURL(blob);
                            const img = new Image();
                            img.onload = () => {
                                if (myToken === token) ctx.drawImage(img, x, y, w, h);
                                URL.revokeObjectURL(url);
                                resolve();
                            };
                            img.onerror = () => {
                                if (myToken === token) {
                                    ctx.fillStyle = "#fff";
                                    ctx.fillRect(x, y, w, h);
                                }
                                URL.revokeObjectURL(url);
                                resolve();
                            };
                            img.src = url;
                        } else {
                            if (myToken === token) {
                                ctx.fillStyle = "#fff";
                                ctx.fillRect(x, y, w, h);
                            }
                            resolve();
                        }
                    }),
            ),
        );
        if (myToken === token) texture.needsUpdate = true;
    };

    return {
        canvas,
        texture,
        rebuild,
        setSize,
        dispose() {
            texture.dispose();
        },
    };
}
