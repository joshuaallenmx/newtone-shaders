// Pipeline: a topo-sorted DAG of render passes that all write to a
// shared internal buffer at the Global Input's aspect ratio.
//
// One canvas, one GL context, one rAF loop. Every node FBO is sized to
// `bufferW × bufferH` — the working buffer — whose aspect matches the
// Global Input source so ingest never distorts. The canvas backing
// buffer matches `bufferW × bufferH` 1:1; the terminal blit copies the
// terminal node's outputTex straight to it.

import type {
    ChainSpec,
    CompiledProgramRef,
    GpuPassSpec,
    PointerFrame,
    ProducerSpec,
    RenderOverrideContext,
    RenderOverrideSpec,
} from "../shaders";

/** Sentinel `inputNodeId` indicating an optional input slot left unwired.
 *  Render nodes bind the placeholder texture for these slots. */
export const MISSING_INPUT_NODE_ID = "__missing__";

/** Longest-side cap for per-node thumbnails. The actual `previewW ×
 *  previewH` matches the working buffer's aspect (so thumbnails are a
 *  faithful miniature of what each shader writes), with the longest
 *  side held at this value. 128² = 64 KiB, sub-millisecond readPixels
 *  on integrated GPUs. */
export const PREVIEW_LONG = 128;

/** Per-node capture size for NSFW classification and similar JS-side
 *  consumers that need real pixel data. 256² is enough for NSFWJS
 *  (which wants ≥224²) without wasting VRAM. */
export const CAPTURE_W = 256;
export const CAPTURE_H = 256;
import {
    bindQuadAttribute,
    createColorTexture,
    createProgram,
    createQuadVbo,
    DEFAULT_VERT_SRC,
    resizeColorTexture,
} from "./glUtil";

// ─── Plan ────────────────────────────────────────────────────────────────────

interface SourcePlanNode {
    readonly kind: "source";
    readonly nodeId: string;
    readonly src: string;
}

interface ProducerPlanNode {
    readonly kind: "producer";
    readonly nodeId: string;
    /** ChainSpec.nodeId for params lookup. */
    readonly originalNodeId: string;
    readonly entryId: string;
    readonly spec: ProducerSpec;
    readonly inputNodeIds: readonly string[];
}

interface GpuPlanNode {
    readonly kind: "gpu";
    readonly nodeId: string;
    readonly originalNodeId: string;
    readonly programKey: string;
    readonly spec: GpuPassSpec;
    readonly inputNodeIds: readonly string[];
}

interface OverridePlanNode {
    readonly kind: "override";
    readonly nodeId: string;
    readonly originalNodeId: string;
    readonly entryId: string;
    readonly spec: RenderOverrideSpec;
    readonly inputNodeIds: readonly string[];
}

export type PlanNode =
    | SourcePlanNode
    | ProducerPlanNode
    | GpuPlanNode
    | OverridePlanNode;

export interface CompiledPlan {
    readonly nodes: readonly PlanNode[];
    /** The node whose `outputTex` the terminal blit samples. */
    readonly terminalNodeId: string;
}

/**
 * Walk the ChainSpec post-order and emit a topo-sorted list of nodes plus
 * the terminal node id. Same `nodeId` reached via two routes collapses to
 * a single PlanNode.
 *
 * Source/producer nodes have natural dims (set when their content lands).
 * Gpu nodes have no natural size — they always render at project size.
 */
export function flattenChain(chain: ChainSpec): CompiledPlan {
    const nodes: PlanNode[] = [];
    const seen = new Set<string>();

    const visit = (c: ChainSpec): string => {
        // Optional-input absence sentinel: produced by the editor's
        // compiler when a declared `optional` input slot has no edge.
        // No PlanNode emitted; the consumer slot binds the placeholder.
        if (c.entry === null && c.src === null && c.nodeId === null) {
            return MISSING_INPUT_NODE_ID;
        }
        // Source node — leaf upload.
        if (c.entry === null) {
            if (!c.src) throw new Error("pipeline: source missing url");
            const id = c.nodeId ?? `__src:${c.src}`;
            if (!seen.has(id)) {
                seen.add(id);
                nodes.push({ kind: "source", nodeId: id, src: c.src });
            }
            return id;
        }

        const entry = c.entry;
        const declaredInputs = entry.inputs;
        const isVariadic = !!entry.variadic;
        const isMultiInput =
            isVariadic ||
            (!!declaredInputs && declaredInputs.length > 1);
        const isZeroInput =
            !isVariadic &&
            !!declaredInputs &&
            declaredInputs.length === 0;

        let inputNodeIds: string[] = [];
        if (isMultiInput) {
            for (const upstream of c.inputs) {
                inputNodeIds.push(visit(upstream));
            }
        } else if (isZeroInput) {
            // Source-style shader (Gradient). No upstream inputs.
        } else if (c.inputs.length > 0) {
            inputNodeIds.push(visit(c.inputs[0]!));
        }

        const baseId = c.nodeId ?? `${entry.id}-anon`;

        // Producer + gpu hybrid (Depth): emit two nodes — producer feeds gpu.
        // Producer-only entry (Normals): emit producer only; downstream
        // consumers sample its `outputTex` directly.
        if (entry.producer) {
            const producerId = entry.gpu ? `${baseId}:producer` : baseId;
            if (!seen.has(producerId)) {
                seen.add(producerId);
                nodes.push({
                    kind: "producer",
                    nodeId: producerId,
                    originalNodeId: c.nodeId ?? "",
                    entryId: entry.id,
                    spec: entry.producer,
                    inputNodeIds,
                });
            }
            inputNodeIds = [producerId];
        }

        if (entry.gpu) {
            if (!seen.has(baseId)) {
                seen.add(baseId);
                nodes.push({
                    kind: "gpu",
                    nodeId: baseId,
                    originalNodeId: c.nodeId ?? "",
                    programKey: entry.id,
                    spec: entry.gpu,
                    inputNodeIds,
                });
            }
            return baseId;
        }

        if (entry.renderOverride) {
            if (!seen.has(baseId)) {
                seen.add(baseId);
                nodes.push({
                    kind: "override",
                    nodeId: baseId,
                    originalNodeId: c.nodeId ?? "",
                    entryId: entry.id,
                    spec: entry.renderOverride,
                    inputNodeIds,
                });
            }
            return baseId;
        }

        if (entry.producer) {
            return baseId;
        }

        throw new Error(
            `pipeline: shader ${entry.id} has none of gpu / producer / renderOverride`,
        );
    };

    const terminalNodeId = visit(chain);
    return { nodes, terminalNodeId };
}

// ─── Built-in fit shader ─────────────────────────────────────────────────────
//
// Samples a texture and writes to the bound framebuffer with aspect-fit
// (centered letterbox / pillarbox). Used for:
//   - source ingest: aspect-fit the natural-dim image into a project-size FBO
//   - producer ingest: same, for model output textures
//   - terminal blit: copy the terminal node's outputTex (already project size)
//     to the canvas. When sizes match the math reduces to identity.
// One built-in shader, three uses.

const FIT_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform vec2 uOutputSize;
uniform int uMode;  // 0 = fit (letterbox), 1 = cover (crop to fill)
void main() {
    vec2 srcSize = vec2(textureSize(uSrc, 0));
    float outAspect = uOutputSize.x / max(1.0, uOutputSize.y);
    float srcAspect = srcSize.x / max(1.0, srcSize.y);
    vec2 uv = vUv;
    if (uMode == 1) {
        // Cover: shrink the sample window so the source crops to fill.
        if (srcAspect > outAspect) {
            float scale = outAspect / srcAspect;
            uv.x = (vUv.x - 0.5) * scale + 0.5;
        } else if (srcAspect < outAspect) {
            float scale = srcAspect / outAspect;
            uv.y = (vUv.y - 0.5) * scale + 0.5;
        }
        outColor = texture(uSrc, uv);
    } else {
        // Fit: stretch the sample window so the source letterboxes.
        if (srcAspect > outAspect) {
            float scale = outAspect / srcAspect;
            uv.y = (vUv.y - 0.5) / scale + 0.5;
        } else if (srcAspect < outAspect) {
            float scale = srcAspect / outAspect;
            uv.x = (vUv.x - 0.5) / scale + 0.5;
        }
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            // Transparent letterbox: lets stacked layers show through
            // their neighbours' bands. The terminal canvas display still
            // looks dark because the host div's CSS background paints
            // through the alpha=0 region.
            outColor = vec4(0.0, 0.0, 0.0, 0.0);
        } else {
            outColor = texture(uSrc, uv);
        }
    }
}
`;
const FIT_KEY = "__fit";

export type FitMode = "fit" | "cover";

// ─── Per-node runtime state ──────────────────────────────────────────────────

interface SourceNodeState {
    readonly kind: "source";
    /** Texture holding the loaded image at natural dims. */
    readonly rawTex: WebGLTexture;
    /** Project-size FBO + texture for downstream sampling. */
    fbo: WebGLFramebuffer | null;
    outputTex: WebGLTexture | null;
    rawWidth: number;
    rawHeight: number;
    loadedFor: string | null;
    pendingFor: string | null;
}

interface ProducerNodeState {
    readonly kind: "producer";
    readonly rawTex: WebGLTexture;
    fbo: WebGLFramebuffer | null;
    outputTex: WebGLTexture | null;
    rawWidth: number;
    rawHeight: number;
    ready: boolean;
    lastInputKey: string | null;
    abort: AbortController | null;
    lastProgress: unknown;
    lastError: string | null;
}

interface GpuNodeState {
    readonly kind: "gpu";
    readonly programKey: string;
    fbo: WebGLFramebuffer | null;
    outputTex: WebGLTexture | null;
}

interface OverrideNodeState {
    readonly kind: "override";
    fbo: WebGLFramebuffer | null;
    outputTex: WebGLTexture | null;
    /** Programs the override compiled via `RenderOverrideContext.compileProgram`,
     *  freed when the node is disposed. */
    readonly programs: Map<string, CompiledProgramRef>;
}

type NodeState =
    | SourceNodeState
    | ProducerNodeState
    | GpuNodeState
    | OverrideNodeState;

interface CompiledProgram {
    readonly program: WebGLProgram;
    readonly samplerLocs: readonly WebGLUniformLocation[];
    readonly uniformLocs: ReadonlyMap<string, WebGLUniformLocation>;
    refCount: number;
}

// ─── Pipeline class ──────────────────────────────────────────────────────────

export class Pipeline {
    private readonly canvas: HTMLCanvasElement;
    private readonly gl: WebGL2RenderingContext;
    private readonly vbo: WebGLBuffer;
    private readonly nodes = new Map<string, NodeState>();
    private readonly planByNode = new Map<string, PlanNode>();
    private order: string[] = [];
    private terminalNodeId: string | null = null;
    private readonly programs = new Map<string, CompiledProgram>();
    private readonly fitProgram: CompiledProgram;
    /** 1×1 opaque-black texture bound in place of unwired optional inputs.
     *  Shaders should gate any use on `frame.inputsPresent[i]` rather than
     *  rely on the placeholder's specific value. */
    private readonly missingInputTex: WebGLTexture;
    /** Persistent low-res FBO used by `snapshotNode` to downscale any
     *  node's outputTex into a small texture before reading pixels back
     *  for per-node thumbnails. Sized to `previewW × previewH` (matches
     *  buffer aspect, longest side = PREVIEW_LONG) and resized in
     *  `setOutput` whenever the buffer aspect changes. */
    private previewFbo: WebGLFramebuffer | null = null;
    private previewTex: WebGLTexture | null = null;
    private previewBuffer: Uint8Array;
    private previewW = PREVIEW_LONG;
    private previewH = PREVIEW_LONG;
    /** Persistent FBO used by `captureNodeImageData` for JS consumers
     *  that need real pixel data (NSFW classification, etc.). Sized
     *  once at CAPTURE_W × CAPTURE_H. */
    private captureFbo: WebGLFramebuffer | null = null;
    private captureTex: WebGLTexture | null = null;
    private readonly captureBuffer: Uint8ClampedArray;
    /** Internal working-buffer dimensions. Every node FBO is sized to
     *  these and the canvas backing buffer matches them 1:1 — aspect =
     *  Global Input source's natural aspect. When no Global Input is
     *  wired the editor falls back to a 1:1 buffer so the chain still
     *  has a sensible shape to render into. */
    private bufferW = 1;
    private bufferH = 1;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const gl = canvas.getContext("webgl2", {
            antialias: false,
            premultipliedAlpha: false,
            preserveDrawingBuffer: false,
        });
        if (!gl) throw new Error("WebGL2 not supported");
        this.gl = gl;
        this.vbo = createQuadVbo(gl);
        this.fitProgram = this.compileBuiltin(
            FIT_KEY,
            FIT_FRAG_SRC,
            ["uSrc"],
            ["uOutputSize", "uMode"],
        );
        this.missingInputTex = createColorTexture(gl);
        gl.bindTexture(gl.TEXTURE_2D, this.missingInputTex);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            1,
            1,
            0,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            new Uint8Array([0, 0, 0, 255]),
        );

        // Preview FBO + readback buffer for snapshotNode. Sized to the
        // working buffer's aspect; resized in setOutput when that
        // changes. Reused for every per-node thumbnail.
        this.previewTex = createColorTexture(gl);
        resizeColorTexture(gl, this.previewTex, this.previewW, this.previewH);
        this.previewFbo = gl.createFramebuffer();
        if (!this.previewFbo) throw new Error("createFramebuffer failed");
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.previewFbo);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D,
            this.previewTex,
            0,
        );

        // Capture FBO + buffer for JS-side classifiers etc.
        this.captureTex = createColorTexture(gl);
        resizeColorTexture(gl, this.captureTex, CAPTURE_W, CAPTURE_H);
        this.captureFbo = gl.createFramebuffer();
        if (!this.captureFbo) throw new Error("createFramebuffer failed");
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.captureFbo);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D,
            this.captureTex,
            0,
        );
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        this.previewBuffer = new Uint8Array(this.previewW * this.previewH * 4);
        this.captureBuffer = new Uint8ClampedArray(CAPTURE_W * CAPTURE_H * 4);
    }

    /** Configure the pipeline's working buffer size. `bufferW × bufferH`
     *  drive the canvas backing buffer and every per-node FBO. Their
     *  aspect = Global Input source's natural aspect; the long side =
     *  Output's chosen resolution. Called whenever Output's resolution
     *  or the Global Input's source aspect change. */
    setOutput(opts: {
        readonly bufferW: number;
        readonly bufferH: number;
    }): void {
        const gl = this.gl;
        const newBufW = Math.max(1, Math.floor(opts.bufferW));
        const newBufH = Math.max(1, Math.floor(opts.bufferH));
        if (newBufW === this.bufferW && newBufH === this.bufferH) return;
        this.bufferW = newBufW;
        this.bufferH = newBufH;
        for (const state of this.nodes.values()) {
            if (state.outputTex) {
                resizeColorTexture(gl, state.outputTex, newBufW, newBufH);
            }
        }
        if (this.canvas.width !== newBufW) this.canvas.width = newBufW;
        if (this.canvas.height !== newBufH) this.canvas.height = newBufH;
        // Recompute preview dimensions to match buffer aspect with
        // longest side held at PREVIEW_LONG. Resize the persistent
        // FBO/texture/readback buffer.
        const aspect = newBufW / newBufH;
        let pw: number;
        let ph: number;
        if (aspect >= 1) {
            pw = PREVIEW_LONG;
            ph = Math.max(1, Math.round(PREVIEW_LONG / aspect));
        } else {
            ph = PREVIEW_LONG;
            pw = Math.max(1, Math.round(PREVIEW_LONG * aspect));
        }
        if (pw !== this.previewW || ph !== this.previewH) {
            this.previewW = pw;
            this.previewH = ph;
            if (this.previewTex) {
                resizeColorTexture(gl, this.previewTex, pw, ph);
            }
            this.previewBuffer = new Uint8Array(pw * ph * 4);
        }
    }

    /** Current per-node thumbnail dimensions (matches buffer aspect).
     *  Editor uses these to size the canvas elements registered with
     *  `useNodeSnapshot`. */
    getPreviewSize(): { readonly w: number; readonly h: number } {
        return { w: this.previewW, h: this.previewH };
    }

    /** Replace the topology. Programs/textures for surviving nodes are
     *  preserved; departed nodes are released. */
    rebuild(plan: CompiledPlan): void {
        const newIds = new Set(plan.nodes.map((n) => n.nodeId));
        for (const [id, state] of [...this.nodes.entries()]) {
            if (!newIds.has(id)) {
                this.disposeNode(state, this.planByNode.get(id));
                this.nodes.delete(id);
            }
        }
        this.order = plan.nodes.map((n) => n.nodeId);
        this.terminalNodeId = plan.terminalNodeId;
        this.planByNode.clear();
        for (const node of plan.nodes) {
            this.planByNode.set(node.nodeId, node);
            if (!this.nodes.has(node.nodeId)) {
                this.nodes.set(node.nodeId, this.allocateNode(node));
            }
        }
    }

    renderFrame(
        tNow: number,
        pointer: PointerFrame,
        paramsByNode: Record<string, unknown>,
    ): void {
        if (!this.terminalNodeId) return;
        const gl = this.gl;

        // Canvas backing buffer is the working buffer 1:1, so the
        // pointer's canvas UV already matches the buffer UV every shader
        // samples in. No remap needed.
        const bufferPointer = pointer;

        // Phase A: kick async loads.
        for (const id of this.order) {
            const node = this.planByNode.get(id);
            const state = this.nodes.get(id);
            if (!node || !state) continue;
            if (node.kind === "source" && state.kind === "source") {
                this.kickSourceLoad(state, node.src);
            } else if (
                node.kind === "producer" &&
                state.kind === "producer"
            ) {
                const params = paramsByNode[node.originalNodeId];
                this.kickProducer(state, node, params);
            }
        }

        // Phase B: render every node into its outputTex FBO at project size.
        for (const id of this.order) {
            const node = this.planByNode.get(id);
            const state = this.nodes.get(id);
            if (!node || !state) continue;
            this.renderNode(node, state, paramsByNode, tNow, bufferPointer);
        }

        // Final blit: copy the terminal node's outputTex 1:1 to the
        // canvas (canvas backing buffer matches `bufferW × bufferH`).
        // The crop overlay in PreviewPanel describes the export window;
        // the live preview shows the full buffer so users can see what
        // every shader is operating on.
        const terminalState = this.nodes.get(this.terminalNodeId);
        if (!terminalState || !terminalState.outputTex) return;
        if (
            terminalState.kind === "source" &&
            terminalState.loadedFor === null
        )
            return;
        if (
            terminalState.kind === "producer" &&
            !terminalState.ready
        )
            return;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.bufferW, this.bufferH);
        this.runFitPass(
            terminalState.outputTex,
            this.bufferW,
            this.bufferH,
        );
    }

    /** Snapshot of in-flight producer state for the loading HUD. */
    getStatus(): {
        readonly producersInFlight: number;
        readonly latestProgress: unknown;
        readonly error: string | null;
    } {
        let inFlight = 0;
        let latest: unknown = null;
        let firstError: string | null = null;
        for (const state of this.nodes.values()) {
            if (state.kind !== "producer") continue;
            if (state.lastInputKey != null && !state.ready) inFlight++;
            if (state.lastProgress != null) latest = state.lastProgress;
            if (state.lastError && !firstError)
                firstError = state.lastError;
        }
        return {
            producersInFlight: inFlight,
            latestProgress: latest,
            error: firstError,
        };
    }

    /** Downscale a node's outputTex into the persistent preview FBO,
     *  read pixels back, and paint into a 2D canvas context. Returns
     *  `false` (no-op) when the node hasn't rendered yet — caller
     *  should re-queue and try again on the next frame.
     *
     *  The preview FBO matches the working buffer's aspect (longest
     *  side = `PREVIEW_LONG`), so the fit pass is identity in both
     *  axes — the thumbnail is a faithful miniature of what the shader
     *  writes, with no cropping or letterboxing. The target canvas
     *  context is presumed sized to the same `previewW × previewH`
     *  (editor sizes its thumbnail canvases via `getPreviewSize`).
     *  Drain the dirty queue right after `renderFrame` so the readback
     *  fetches the freshest possible state. */
    snapshotNode(target: CanvasRenderingContext2D, nodeId: string): boolean {
        if (!this.previewFbo || !this.previewTex) return false;
        const state = this.nodes.get(nodeId);
        if (!state || !state.outputTex) return false;
        if (state.kind === "source" && state.loadedFor === null) return false;
        if (state.kind === "producer" && !state.ready) return false;

        const gl = this.gl;
        const pw = this.previewW;
        const ph = this.previewH;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.previewFbo);
        gl.viewport(0, 0, pw, ph);
        this.runFitPass(state.outputTex, pw, ph, "fit");
        gl.readPixels(
            0,
            0,
            pw,
            ph,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            this.previewBuffer,
        );
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        // readPixels returns rows bottom-up (origin lower-left); ImageData
        // is top-down. Flip rows during the copy so the thumbnail matches
        // what the terminal blit shows on the main canvas.
        const flipped = new Uint8ClampedArray(pw * ph * 4);
        const rowBytes = pw * 4;
        for (let y = 0; y < ph; y++) {
            const srcOffset = (ph - 1 - y) * rowBytes;
            const dstOffset = y * rowBytes;
            flipped.set(
                this.previewBuffer.subarray(srcOffset, srcOffset + rowBytes),
                dstOffset,
            );
        }
        const image = new ImageData(flipped, pw, ph);
        target.putImageData(image, 0, 0);
        return true;
    }

    /** Read a node's outputTex into an `ImageData` at CAPTURE_W × CAPTURE_H.
     *  Default mode is `cover` (fills the square, crops mismatched aspect)
     *  — best for whole-image classifiers. Use `fit` when downstream
     *  consumers need to map output coordinates back to the canvas
     *  display: fit produces letterboxed pixels matching what the
     *  terminal blit shows, so a detector's bounding box at (x, y) in the
     *  capture lines up with the canvas pixel at the same normalized
     *  position. Each call allocates a fresh Uint8ClampedArray to avoid
     *  callers racing on a shared buffer. Returns `null` when the node
     *  hasn't rendered yet. */
    captureNodeImageData(
        nodeId: string,
        mode: FitMode = "cover",
    ): ImageData | null {
        if (!this.captureFbo || !this.captureTex) return null;
        const state = this.nodes.get(nodeId);
        if (!state || !state.outputTex) return null;
        if (state.kind === "source" && state.loadedFor === null) return null;
        if (state.kind === "producer" && !state.ready) return null;

        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.captureFbo);
        gl.viewport(0, 0, CAPTURE_W, CAPTURE_H);
        this.runFitPass(state.outputTex, CAPTURE_W, CAPTURE_H, mode);
        gl.readPixels(
            0,
            0,
            CAPTURE_W,
            CAPTURE_H,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            this.captureBuffer,
        );
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        // readPixels is bottom-up; ImageData is top-down. Allocate a
        // fresh buffer per call so the returned ImageData stays valid
        // even after subsequent captures.
        const flipped = new Uint8ClampedArray(CAPTURE_W * CAPTURE_H * 4);
        const rowBytes = CAPTURE_W * 4;
        for (let y = 0; y < CAPTURE_H; y++) {
            const srcOffset = (CAPTURE_H - 1 - y) * rowBytes;
            const dstOffset = y * rowBytes;
            flipped.set(
                this.captureBuffer.subarray(srcOffset, srcOffset + rowBytes),
                dstOffset,
            );
        }
        return new ImageData(flipped, CAPTURE_W, CAPTURE_H);
    }

    dispose(): void {
        for (const [id, state] of this.nodes) {
            this.disposeNode(state, this.planByNode.get(id));
        }
        this.nodes.clear();
        for (const compiled of this.programs.values()) {
            this.gl.deleteProgram(compiled.program);
        }
        this.programs.clear();
        this.gl.deleteBuffer(this.vbo);
        this.gl.deleteTexture(this.missingInputTex);
        if (this.previewFbo) {
            this.gl.deleteFramebuffer(this.previewFbo);
            this.previewFbo = null;
        }
        if (this.previewTex) {
            this.gl.deleteTexture(this.previewTex);
            this.previewTex = null;
        }
        if (this.captureFbo) {
            this.gl.deleteFramebuffer(this.captureFbo);
            this.captureFbo = null;
        }
        if (this.captureTex) {
            this.gl.deleteTexture(this.captureTex);
            this.captureTex = null;
        }
        // Don't call WEBGL_lose_context — React StrictMode double-invokes
        // the mount effect and a force-lost context can't be re-created on
        // the same canvas.
    }

    // ─── Per-frame node rendering ───────────────────────────────────────

    private renderNode(
        node: PlanNode,
        state: NodeState,
        paramsByNode: Record<string, unknown>,
        tNow: number,
        pointer: PointerFrame,
    ): void {
        const gl = this.gl;
        if (node.kind === "source" && state.kind === "source") {
            if (state.loadedFor === null) return;
            this.ensureFbo(state);
            gl.bindFramebuffer(gl.FRAMEBUFFER, state.fbo!);
            gl.viewport(0, 0, this.bufferW, this.bufferH);
            this.runFitPass(state.rawTex, this.bufferW, this.bufferH);
            return;
        }
        if (node.kind === "producer" && state.kind === "producer") {
            if (!state.ready) return;
            this.ensureFbo(state);
            gl.bindFramebuffer(gl.FRAMEBUFFER, state.fbo!);
            gl.viewport(0, 0, this.bufferW, this.bufferH);
            this.runFitPass(state.rawTex, this.bufferW, this.bufferH);
            return;
        }
        if (node.kind === "gpu" && state.kind === "gpu") {
            for (const inputId of node.inputNodeIds) {
                if (inputId === MISSING_INPUT_NODE_ID) continue;
                const inputState = this.nodes.get(inputId);
                if (!inputState || !inputState.outputTex) return;
                if (
                    inputState.kind === "source" &&
                    inputState.loadedFor === null
                )
                    return;
                if (
                    inputState.kind === "producer" &&
                    !inputState.ready
                )
                    return;
            }
            // Phase B': any input slots the shader opted to capture as
            // ImageData are read back BEFORE we bind this node's FBO,
            // so the readback's intermediate FBO/viewport changes don't
            // pollute our render state.
            const capturedInputs: (ImageData | null)[] = [];
            const captureSlots = node.spec.capturedInputSlots;
            if (captureSlots && captureSlots.length > 0) {
                const maxSlot = Math.max(...captureSlots);
                for (let s = 0; s <= maxSlot; s++) capturedInputs[s] = null;
                for (const slot of captureSlots) {
                    const inputId = node.inputNodeIds[slot];
                    if (!inputId || inputId === MISSING_INPUT_NODE_ID) {
                        capturedInputs[slot] = null;
                    } else {
                        capturedInputs[slot] = this.captureNodeImageData(
                            inputId,
                            "fit",
                        );
                    }
                }
            }
            this.ensureFbo(state);
            gl.bindFramebuffer(gl.FRAMEBUFFER, state.fbo!);
            gl.viewport(0, 0, this.bufferW, this.bufferH);
            const compiled = this.programs.get(state.programKey);
            if (!compiled) return;
            gl.useProgram(compiled.program);
            bindQuadAttribute(gl, compiled.program, this.vbo);
            // Bind one texture per declared sampler slot. For variadic
            // shaders the sampler array is longer than `inputNodeIds`;
            // we bind `missingInputTex` to the unused tail so unset
            // samplers don't alias whatever happens to be at unit 0.
            const sampleCount = node.spec.samplers.length;
            const inputsPresent: boolean[] = [];
            for (let s = 0; s < sampleCount; s++) {
                let tex: WebGLTexture;
                let present = false;
                if (s < node.inputNodeIds.length) {
                    const inputId = node.inputNodeIds[s]!;
                    if (inputId === MISSING_INPUT_NODE_ID) {
                        tex = this.missingInputTex;
                    } else {
                        const t = this.nodes.get(inputId)?.outputTex;
                        if (!t) return;
                        tex = t;
                        present = true;
                    }
                } else {
                    tex = this.missingInputTex;
                }
                inputsPresent.push(present);
                gl.activeTexture(gl.TEXTURE0 + s);
                gl.bindTexture(gl.TEXTURE_2D, tex);
                gl.uniform1i(compiled.samplerLocs[s]!, s);
            }
            const params = paramsByNode[node.originalNodeId];
            node.spec.setUniforms(gl, compiled.uniformLocs, params, {
                tNow,
                pointer,
                inputsPresent,
                nodeId: node.originalNodeId,
                capturedInputs,
            });
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
        if (node.kind === "override" && state.kind === "override") {
            for (const inputId of node.inputNodeIds) {
                if (inputId === MISSING_INPUT_NODE_ID) continue;
                const inputState = this.nodes.get(inputId);
                if (!inputState || !inputState.outputTex) return;
                if (
                    inputState.kind === "source" &&
                    inputState.loadedFor === null
                )
                    return;
                if (
                    inputState.kind === "producer" &&
                    !inputState.ready
                )
                    return;
            }
            this.ensureFbo(state);
            // Resolve inputs to bare WebGLTexture handles. The override
            // gets full GL access and can bind these to whichever
            // texture units it likes; we don't pre-bind sampler slots
            // for it.
            const inputTextures: (WebGLTexture | null)[] = [];
            const inputsPresent: boolean[] = [];
            for (const inputId of node.inputNodeIds) {
                if (inputId === MISSING_INPUT_NODE_ID) {
                    inputTextures.push(null);
                    inputsPresent.push(false);
                } else {
                    const t = this.nodes.get(inputId)?.outputTex ?? null;
                    inputTextures.push(t);
                    inputsPresent.push(t != null);
                }
            }
            const params = paramsByNode[node.originalNodeId];
            const ctx: RenderOverrideContext = {
                gl,
                bufferW: this.bufferW,
                bufferH: this.bufferH,
                inputTextures,
                inputsPresent,
                outputFbo: state.fbo!,
                outputTex: state.outputTex!,
                quadVbo: this.vbo,
                defaultVertSrc: DEFAULT_VERT_SRC,
                compileProgram: (key, fragSrc, samplers, uniforms, vertSrc) =>
                    this.acquireOverrideProgram(
                        state,
                        key,
                        fragSrc,
                        samplers,
                        uniforms,
                        vertSrc,
                    ),
                bindQuadAttribute: (program) =>
                    bindQuadAttribute(gl, program, this.vbo),
                tNow,
                pointer,
                nodeId: node.originalNodeId,
                params,
            };
            node.spec.render(ctx);
        }
    }

    /** Compile-and-cache a program inside an override node's private
     *  cache. Subsequent calls with the same key reuse the cached
     *  ref. The cache is freed when the node is disposed. */
    private acquireOverrideProgram(
        state: OverrideNodeState,
        key: string,
        fragSrc: string,
        samplers: readonly string[],
        uniforms: readonly string[],
        vertSrc?: string,
    ): CompiledProgramRef {
        const cached = state.programs.get(key);
        if (cached) return cached;
        const gl = this.gl;
        const program = createProgram(
            gl,
            vertSrc ?? DEFAULT_VERT_SRC,
            fragSrc,
            key,
        );
        const samplerLocs: WebGLUniformLocation[] = [];
        for (const name of samplers) {
            const loc = gl.getUniformLocation(program, name);
            if (!loc) {
                throw new Error(
                    `[${key}] sampler "${name}" not found ` +
                        `(stripped or misspelled)`,
                );
            }
            samplerLocs.push(loc);
        }
        const uniformLocs = new Map<string, WebGLUniformLocation>();
        for (const name of uniforms) {
            const loc = gl.getUniformLocation(program, name);
            if (!loc) {
                throw new Error(
                    `[${key}] uniform "${name}" not found ` +
                        `(stripped or misspelled)`,
                );
            }
            uniformLocs.set(name, loc);
        }
        const ref: CompiledProgramRef = {
            program,
            samplerLocs,
            uniformLocs,
        };
        state.programs.set(key, ref);
        return ref;
    }

    private runFitPass(
        srcTex: WebGLTexture,
        outW: number,
        outH: number,
        mode: FitMode = "fit",
    ): void {
        const gl = this.gl;
        gl.useProgram(this.fitProgram.program);
        bindQuadAttribute(gl, this.fitProgram.program, this.vbo);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, srcTex);
        gl.uniform1i(this.fitProgram.samplerLocs[0]!, 0);
        gl.uniform2f(
            this.fitProgram.uniformLocs.get("uOutputSize")!,
            outW,
            outH,
        );
        gl.uniform1i(
            this.fitProgram.uniformLocs.get("uMode")!,
            mode === "cover" ? 1 : 0,
        );
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    private ensureFbo(
        state:
            | SourceNodeState
            | ProducerNodeState
            | GpuNodeState
            | OverrideNodeState,
    ): void {
        const gl = this.gl;
        if (!state.outputTex) {
            state.outputTex = createColorTexture(gl);
            resizeColorTexture(
                gl,
                state.outputTex,
                this.bufferW,
                this.bufferH,
            );
        }
        if (!state.fbo) {
            state.fbo = gl.createFramebuffer();
            if (!state.fbo) throw new Error("createFramebuffer failed");
            gl.bindFramebuffer(gl.FRAMEBUFFER, state.fbo);
            gl.framebufferTexture2D(
                gl.FRAMEBUFFER,
                gl.COLOR_ATTACHMENT0,
                gl.TEXTURE_2D,
                state.outputTex,
                0,
            );
        }
    }

    // ─── Allocation / disposal ──────────────────────────────────────────

    private allocateNode(node: PlanNode): NodeState {
        const gl = this.gl;
        if (node.kind === "source") {
            const rawTex = createColorTexture(gl);
            // 1×1 placeholder so the fit pass never samples undefined data
            // before the real image lands.
            gl.bindTexture(gl.TEXTURE_2D, rawTex);
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                1,
                1,
                0,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                new Uint8Array([0, 0, 0, 255]),
            );
            return {
                kind: "source",
                rawTex,
                fbo: null,
                outputTex: null,
                rawWidth: 0,
                rawHeight: 0,
                loadedFor: null,
                pendingFor: null,
            };
        }
        if (node.kind === "producer") {
            const rawTex = createColorTexture(gl);
            gl.bindTexture(gl.TEXTURE_2D, rawTex);
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                1,
                1,
                0,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                new Uint8Array([0, 0, 0, 255]),
            );
            return {
                kind: "producer",
                rawTex,
                fbo: null,
                outputTex: null,
                rawWidth: 0,
                rawHeight: 0,
                ready: false,
                lastInputKey: null,
                abort: null,
                lastProgress: null,
                lastError: null,
            };
        }
        if (node.kind === "override") {
            return {
                kind: "override",
                fbo: null,
                outputTex: null,
                programs: new Map(),
            };
        }
        const compiled = this.acquireProgram(node.programKey, node.spec);
        compiled.refCount++;
        return {
            kind: "gpu",
            programKey: node.programKey,
            fbo: null,
            outputTex: null,
        };
    }

    private disposeNode(state: NodeState, plan?: PlanNode): void {
        const gl = this.gl;
        if (state.kind === "source") {
            gl.deleteTexture(state.rawTex);
        } else if (state.kind === "producer") {
            state.abort?.abort();
            gl.deleteTexture(state.rawTex);
        } else if (state.kind === "override") {
            // Free the override's per-node program cache.
            for (const compiled of state.programs.values()) {
                gl.deleteProgram(compiled.program);
            }
            state.programs.clear();
            // Hand off to the override for any extra resources (private
            // FBOs / textures / VBOs) it allocated outside the cache.
            if (plan && plan.kind === "override") {
                plan.spec.dispose?.(gl, plan.originalNodeId);
            }
        } else {
            const compiled = this.programs.get(state.programKey);
            if (compiled) {
                compiled.refCount--;
                if (compiled.refCount <= 0) {
                    gl.deleteProgram(compiled.program);
                    this.programs.delete(state.programKey);
                }
            }
        }
        if (state.fbo) gl.deleteFramebuffer(state.fbo);
        if (state.outputTex) gl.deleteTexture(state.outputTex);
    }

    // ─── Program cache ──────────────────────────────────────────────────

    private compileBuiltin(
        key: string,
        fragSrc: string,
        samplers: readonly string[],
        uniforms: readonly string[],
    ): CompiledProgram {
        const gl = this.gl;
        const program = createProgram(gl, DEFAULT_VERT_SRC, fragSrc, key);
        const samplerLocs: WebGLUniformLocation[] = [];
        for (const name of samplers) {
            const loc = gl.getUniformLocation(program, name);
            if (!loc) throw new Error(`built-in ${key}: missing ${name}`);
            samplerLocs.push(loc);
        }
        const uniformLocs = new Map<string, WebGLUniformLocation>();
        for (const name of uniforms) {
            const loc = gl.getUniformLocation(program, name);
            if (!loc) throw new Error(`built-in ${key}: missing ${name}`);
            uniformLocs.set(name, loc);
        }
        const compiled: CompiledProgram = {
            program,
            samplerLocs,
            uniformLocs,
            refCount: 1, // pinned
        };
        this.programs.set(key, compiled);
        return compiled;
    }

    private acquireProgram(
        key: string,
        spec: GpuPassSpec,
    ): CompiledProgram {
        const cached = this.programs.get(key);
        if (cached) return cached;
        const gl = this.gl;
        const program = createProgram(
            gl,
            spec.vertSrc ?? DEFAULT_VERT_SRC,
            spec.fragSrc,
            key,
        );
        const samplerLocs: WebGLUniformLocation[] = [];
        for (const name of spec.samplers) {
            const loc = gl.getUniformLocation(program, name);
            if (!loc) {
                throw new Error(
                    `[${key}] sampler "${name}" not found ` +
                        `(stripped or misspelled)`,
                );
            }
            samplerLocs.push(loc);
        }
        const uniformLocs = new Map<string, WebGLUniformLocation>();
        for (const name of spec.uniforms) {
            const loc = gl.getUniformLocation(program, name);
            if (!loc) {
                throw new Error(
                    `[${key}] uniform "${name}" not found ` +
                        `(stripped or misspelled)`,
                );
            }
            uniformLocs.set(name, loc);
        }
        const compiled: CompiledProgram = {
            program,
            samplerLocs,
            uniformLocs,
            refCount: 0,
        };
        this.programs.set(key, compiled);
        return compiled;
    }

    // ─── Source loading ─────────────────────────────────────────────────

    private kickSourceLoad(
        state: SourceNodeState,
        src: string,
    ): void {
        if (state.loadedFor === src || state.pendingFor === src) return;
        state.pendingFor = src;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            if (state.pendingFor !== src) return;
            const gl = this.gl;
            gl.bindTexture(gl.TEXTURE_2D, state.rawTex);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                img,
            );
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
            state.loadedFor = src;
            state.pendingFor = null;
            state.rawWidth = img.naturalWidth;
            state.rawHeight = img.naturalHeight;
        };
        img.onerror = () => {
            if (state.pendingFor !== src) return;
            state.pendingFor = null;
            console.error(`[pipeline] source load failed: ${src}`);
        };
        img.src = src;
    }

    // ─── Producer scheduling ────────────────────────────────────────────

    private kickProducer(
        state: ProducerNodeState,
        node: ProducerPlanNode,
        params: unknown,
    ): void {
        const upstream = node.inputNodeIds.map((id) => ({
            src: this.getSrcForNode(id),
        }));
        const key = node.spec.inputKey(params, upstream);
        if (key === state.lastInputKey) return;
        state.abort?.abort();
        state.lastInputKey = key;
        state.ready = false;
        state.lastProgress = null;
        state.lastError = null;
        state.abort = new AbortController();
        const localAbort = state.abort;
        node.spec
            .run(params, {
                upstream,
                signal: localAbort.signal,
                onProgress: (info) => {
                    if (localAbort.signal.aborted) return;
                    state.lastProgress = info;
                },
            })
            .then((upload) => {
                if (localAbort.signal.aborted) return;
                this.uploadProducerTexture(state, upload);
                state.ready = true;
                state.rawWidth = upload.width;
                state.rawHeight = upload.height;
            })
            .catch((err) => {
                if (localAbort.signal.aborted) return;
                state.lastError =
                    err instanceof Error ? err.message : String(err);
                console.error(
                    `[pipeline] producer ${node.entryId} failed:`,
                    err,
                );
            });
    }

    private uploadProducerTexture(
        state: ProducerNodeState,
        upload: import("../shaders").TextureUpload,
    ): void {
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, state.rawTex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        const src = upload.source;
        if (src.kind === "image") {
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                src.image,
            );
        } else if (src.kind === "canvas") {
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                src.canvas,
            );
        } else {
            // RawImage — expand single channel into RGBA grayscale so the
            // fit shader's `texture(uSrc, uv).rgb` works uniformly.
            const planeSize = src.image.width * src.image.height;
            const ch = src.image.channels;
            const bytes = new Uint8Array(planeSize * 4);
            const data = src.image.data;
            for (let i = 0; i < planeSize; i++) {
                const v = data[i * ch]!;
                bytes[i * 4] = v;
                bytes[i * 4 + 1] = v;
                bytes[i * 4 + 2] = v;
                bytes[i * 4 + 3] = 255;
            }
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                src.image.width,
                src.image.height,
                0,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                bytes,
            );
        }
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    }

    /** Helper for producer.run upstream resolution. */
    private getSrcForNode(nodeId: string): string | null {
        const node = this.planByNode.get(nodeId);
        if (node?.kind === "source") return node.src;
        return null;
    }
}
