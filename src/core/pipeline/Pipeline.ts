import * as THREE from "three";
import { Pass, type PassResources } from "./Pass";
import { PingPong } from "./PingPong";
import { makeRenderTarget } from "./RenderTargetPool";
import { getSharedQuad } from "./Quad";
import type {
    FrameContext,
    PassConfig,
    PipelineConfig,
    PipelineRenderInput,
    SizeSpec,
} from "./types";

interface PassEntry {
    readonly pass: Pass;
    readonly config: PassConfig;
    /** PingPong for ping-pong targets, RT for fixed, null for screen. */
    readonly target: PingPong | THREE.WebGLRenderTarget | null;
}

/**
 * An ordered list of passes that share a renderer. The pipeline owns its
 * render targets, resolves intra-pipeline texture refs to the correct
 * ping-pong read side, and runs all passes in declaration order each frame.
 */
export class Pipeline {
    readonly config: PipelineConfig;
    private readonly renderer: THREE.WebGLRenderer;
    private readonly passes = new Map<string, PassEntry>();
    private size = { w: 2, h: 2 };

    constructor(renderer: THREE.WebGLRenderer, config: PipelineConfig) {
        this.renderer = renderer;
        this.config = config;
        for (const passConfig of config.passes) {
            const pass = new Pass(passConfig);
            const target = createTarget(passConfig.target, this.size);
            this.passes.set(passConfig.id, { pass, config: passConfig, target });
        }
    }

    resize(width: number, height: number): void {
        this.size = { w: Math.max(2, width), h: Math.max(2, height) };
        for (const entry of this.passes.values()) {
            if (entry.target instanceof PingPong) {
                if (entry.config.target.kind === "pingpong") {
                    const [w, h] = sizeSpecToPixels(
                        entry.config.target.size,
                        this.size,
                    );
                    entry.target.setSize(w, h);
                }
            }
            // Fixed targets and screen don't resize.
        }
    }

    render(input: PipelineRenderInput): void {
        const quad = getSharedQuad(this.renderer);
        const resources: PassResources = {
            resolveTextureRef: (passId) => {
                const entry = this.passes.get(passId);
                if (!entry) return null;
                if (entry.target instanceof PingPong) return entry.target.read().texture;
                if (entry.target instanceof THREE.WebGLRenderTarget)
                    return entry.target.texture;
                return null;
            },
        };

        for (const passConfig of this.config.passes) {
            const entry = this.passes.get(passConfig.id);
            if (!entry) continue;
            const target = passTargetSize(entry, input.canvas);
            const ctx: FrameContext = {
                time: input.time,
                frame: input.frame,
                canvas: input.canvas,
                target,
            };
            entry.pass.sync(ctx, resources);
            quad.mesh.material = entry.pass.material;
            if (entry.target === null) {
                this.renderer.setRenderTarget(null);
            } else if (entry.target instanceof PingPong) {
                this.renderer.setRenderTarget(entry.target.write());
            } else {
                this.renderer.setRenderTarget(entry.target);
            }
            this.renderer.render(quad.scene, quad.camera);
            if (entry.target instanceof PingPong) entry.target.swap();
        }
    }

    /** Sample a named output (e.g. for chaining or external use). */
    getOutput(name: string): THREE.Texture | null {
        const passId = this.config.outputs?.[name];
        if (!passId) return null;
        const entry = this.passes.get(passId);
        if (!entry) return null;
        if (entry.target instanceof PingPong) return entry.target.read().texture;
        if (entry.target instanceof THREE.WebGLRenderTarget)
            return entry.target.texture;
        return null;
    }

    /**
     * Return the render target most recently written by a pass, suitable for
     * `renderer.readRenderTargetPixels(...)`. Returns null for screen passes
     * (no readable target) and unknown pass ids.
     */
    getPassTarget(passId: string): THREE.WebGLRenderTarget | null {
        const entry = this.passes.get(passId);
        if (!entry) return null;
        if (entry.target instanceof PingPong) return entry.target.read();
        if (entry.target instanceof THREE.WebGLRenderTarget) return entry.target;
        return null;
    }

    dispose(): void {
        for (const entry of this.passes.values()) {
            entry.pass.dispose();
            if (entry.target instanceof PingPong) entry.target.dispose();
            else if (entry.target instanceof THREE.WebGLRenderTarget)
                entry.target.dispose();
        }
        this.passes.clear();
    }
}

function passTargetSize(
    entry: PassEntry,
    canvas: { w: number; h: number },
): { w: number; h: number } {
    if (entry.target instanceof PingPong) {
        return { w: entry.target.width, h: entry.target.height };
    }
    if (entry.target instanceof THREE.WebGLRenderTarget) {
        return { w: entry.target.width, h: entry.target.height };
    }
    return { w: canvas.w, h: canvas.h };
}

function createTarget(
    spec: PassConfig["target"],
    size: { w: number; h: number },
): PassEntry["target"] {
    if (spec.kind === "screen") return null;
    if (spec.kind === "fixed") {
        return makeRenderTarget(spec.size[0], spec.size[1], {
            format: spec.format,
            type: spec.type,
        });
    }
    const [w, h] = sizeSpecToPixels(spec.size, size);
    return new PingPong(w, h, { format: spec.format, type: spec.type });
}

export function sizeSpecToPixels(
    spec: SizeSpec,
    canvas: { w: number; h: number },
): [number, number] {
    if (spec === "full") return [Math.max(2, canvas.w), Math.max(2, canvas.h)];
    if (spec === "half")
        return [
            Math.max(2, Math.floor(canvas.w / 2)),
            Math.max(2, Math.floor(canvas.h / 2)),
        ];
    if (typeof spec === "function") {
        const out = spec(canvas);
        return [Math.max(2, out[0]), Math.max(2, out[1])];
    }
    return [Math.max(2, spec[0]), Math.max(2, spec[1])];
}
