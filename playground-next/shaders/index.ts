import type { ComponentType } from "react";
import { topologyEntry } from "./Topology";
import { mergeEntry } from "./Merge";
import { depthEntry } from "./Depth";
import { normalsEntry } from "./Normals";
import { bumpNormalsEntry } from "./BumpNormals";
import { sunlightEntry } from "./Sunlight";
import { illuminationEntry } from "./Illumination";
import { gradientEntry } from "./Gradient";
import { colorEntry } from "./Color";
import { depthRampEntry } from "./DepthRamp";
import { compositeShaderEntry } from "./Composite";
import { colorGradeEntry } from "./ColorGrade";
import { particlesEntry } from "./Particles";
import { maskMergeEntry } from "./MaskMerge";
import { circleGridEntry } from "./CircleGrid";
import { blackHoleEntry } from "./BlackHole";
import { nsfwCompareEntry } from "./NsfwCompare";
import { layersEntry } from "./Layers";
import { gravityMapEntry } from "./GravityMap";
import { marblesEntry } from "./Marbles";
import { swarmEntry } from "./Swarm";
import { contourFlowEntry } from "./ContourFlow";
import { samEntry } from "./Sam";
import { describeMaskEntry } from "./DescribeMask";
import { nsfwDetectEntry } from "./NsfwDetect";

export type ViewMode = "fit" | "actual";

export interface ShaderControlsProps {
    readonly params: unknown;
    readonly onChange: (next: unknown) => void;
    /** Editor workspace state — only populated when the host editor
     *  surfaces it (variadic shaders need to look up upstream node
     *  labels and the wiring to compute their layer list). Optional so
     *  most Controls don't have to care. */
    readonly nodes?: readonly EditorNodeLike[];
    readonly edges?: readonly EditorEdgeLike[];
    /** Selected node's id — variadic Controls need it to know which
     *  edges are "incoming" to them. */
    readonly nodeId?: string;
}

/** Minimal subset of @xyflow/react's Node/Edge that variadic Controls
 *  consume — kept structural so playground-next doesn't have to import
 *  React Flow types. */
export interface EditorNodeLike {
    readonly id: string;
    readonly type?: string;
    readonly data?: unknown;
}

export interface EditorEdgeLike {
    readonly id: string;
    readonly source: string;
    readonly target: string;
    readonly sourceHandle?: string | null;
    readonly targetHandle?: string | null;
}

export interface InputSpec {
    readonly id: string;
    readonly label: string;
    /** When true, compileChain accepts a missing edge for this input and the
     *  pipeline binds a 1×1 placeholder texture in its place. The shader's
     *  `setUniforms` learns about the absence via `frame.inputsPresent[i]`. */
    readonly optional?: boolean;
}

/**
 * Recursive description of a chain produced by the editor's compiler. Sole
 * input to the rendering pipeline.
 *
 * - Source asset: `entry === null`, `src` holds the URL, `inputs === []`.
 * - Zero-input shader (e.g. Gradient): `entry` set, `inputs === []`,
 *   `src === null`.
 * - Single-input shader: `entry` set, `inputs[0]` is the upstream chain,
 *   `src` carries the leaf source URL (informational; the pipeline samples
 *   `inputs[0]`'s output texture, not the URL).
 * - Multi-input shader: `entry` set, `inputs[i]` is the chain feeding
 *   handle `entry.inputs[i].id`.
 */
export interface ChainSpec {
    readonly entry: ShaderEntry | null;
    readonly params: unknown;
    readonly src: string | null;
    readonly inputs: readonly ChainSpec[];
    readonly nodeId: string | null;
}

/**
 * Shader entry — pure data describing a node type. The pipeline DAG runtime
 * is the only renderer; entries declare a `gpu` (fragment-shader pass) or a
 * `producer` (async CPU/model stage) or both. Display is automatic — the
 * pipeline's terminal blit reads the terminal node's output texture and
 * aspect-fits it to the canvas.
 */
export interface ShaderEntry {
    readonly id: string;
    readonly name: string;
    readonly defaultParams: unknown;
    readonly Controls?: ComponentType<ShaderControlsProps>;
    /** Length 0 = source-style shader (zero inputs). Length 1 (or omitted) =
     *  single-input shader. Length > 1 = multi-input. Ignored when
     *  `variadic` is true — variadic shaders accept N edges into a
     *  single handle with id `"in"` and resolve their inputs from
     *  `params.order` instead. */
    readonly inputs?: readonly InputSpec[];
    /** Variadic shaders accept any number of incoming edges into one
     *  handle. The compiler builds `ChainSpec.inputs` from `params.order`
     *  (an array of upstream nodeIds), and the GLSL pass should declare
     *  enough samplers to cover the cap (16 in practice). */
    readonly variadic?: boolean;
    /** Pipeline-native fragment-shader pass. */
    readonly gpu?: GpuPassSpec;
    /** Async producer for CPU/model stages. */
    readonly producer?: ProducerSpec;
    /** Stateful / multi-pass render path. Mutually exclusive with
     *  `gpu` / `producer`. */
    readonly renderOverride?: RenderOverrideSpec;
}

/** Frame-time information passed into uniform writers. */
export interface FrameContext {
    /** Seconds since the pipeline mounted. Use this for animation phase
     *  rather than an internal accumulator — keeps animation stateless and
     *  immune to play/pause jumps. */
    readonly tNow: number;
    /** Pointer position in normalized canvas UV (0..1, bottom-up to match
     *  `vUv`). `active` is false when the pointer is outside the canvas;
     *  `uv` is the last-seen position in that case. */
    readonly pointer: PointerFrame;
    /** Parallel to the entry's declared `inputs`. False at slot `i` means
     *  the input is missing and the pipeline bound a 1×1 placeholder; the
     *  shader should gate any sampler reads on this. */
    readonly inputsPresent: readonly boolean[];
    /** Editor-side stable identifier of the node this pass belongs to.
     *  Stateful shaders (e.g. particle simulations) key their per-frame
     *  state by this id. Empty string when the chain wasn't built from
     *  an editor (test harnesses, etc.). */
    readonly nodeId: string;
    /** Pre-captured `ImageData` for each input slot the shader's
     *  `capturedInputSlots` opted into. Indexed parallel to the entry's
     *  `inputs`; entries not opted in (or whose source isn't ready) are
     *  `null`. The pipeline reads back the upstream node's outputTex
     *  before binding this pass's FBO, so `setUniforms` can do
     *  JS-side computation over the pixel data without disturbing GL
     *  state. */
    readonly capturedInputs: ReadonlyArray<ImageData | null>;
}

export interface PointerFrame {
    readonly uv: readonly [number, number];
    readonly active: boolean;
}

/** Writes uniforms for a pass. The pipeline binds the program, locates
 *  uniforms (provided in `locs`), then calls this once per frame. */
export type UniformWriter = (
    gl: WebGL2RenderingContext,
    locs: ReadonlyMap<string, WebGLUniformLocation>,
    params: unknown,
    frame: FrameContext,
) => void;

/** Compiled-program handle returned by `RenderOverrideContext.compileProgram`.
 *  Locations are resolved up-front and cached for the lifetime of the
 *  override's plan node. */
export interface CompiledProgramRef {
    readonly program: WebGLProgram;
    readonly samplerLocs: readonly WebGLUniformLocation[];
    readonly uniformLocs: ReadonlyMap<string, WebGLUniformLocation>;
}

/** Context passed to a `RenderOverrideSpec.render` call. The override has
 *  full GL access for the duration of the call and is responsible for
 *  ending with the desired result drawn into `outputFbo` at the
 *  square-size viewport — that's what downstream readers and the
 *  terminal blit consume. State (private FBOs, ping-pong, etc.) lives
 *  in the override's own module-scoped store, keyed by `nodeId`; the
 *  `dispose` hook on the spec frees it when the node is removed. */
export interface RenderOverrideContext {
    readonly gl: WebGL2RenderingContext;
    /** Internal working-buffer dimensions (aspect = Global Input's). */
    readonly bufferW: number;
    readonly bufferH: number;
    /** Upstream output textures, indexed parallel to `entry.inputs`.
     *  `null` for slots whose source is missing or not yet ready —
     *  `inputsPresent[slot]` mirrors this. */
    readonly inputTextures: ReadonlyArray<WebGLTexture | null>;
    readonly inputsPresent: readonly boolean[];
    /** The Pipeline-allocated output framebuffer + texture. The
     *  override's last draw must write into `outputFbo` at the
     *  full square viewport. */
    readonly outputFbo: WebGLFramebuffer;
    readonly outputTex: WebGLTexture;
    /** Fullscreen-quad VBO using the standard `aPos vec2` layout — pair
     *  with `defaultVertSrc` (or compatible) and `bindQuadAttribute`. */
    readonly quadVbo: WebGLBuffer;
    readonly defaultVertSrc: string;
    /** Compile and cache a program by stable string key. Subsequent
     *  calls with the same key return the cached `CompiledProgramRef`.
     *  Programs are freed automatically when the node is disposed. */
    readonly compileProgram: (
        key: string,
        fragSrc: string,
        samplers: readonly string[],
        uniforms: readonly string[],
        vertSrc?: string,
    ) => CompiledProgramRef;
    /** Bind the standard quad VBO to a program's `aPos` attribute. */
    readonly bindQuadAttribute: (program: WebGLProgram) => void;
    readonly tNow: number;
    readonly pointer: PointerFrame;
    readonly nodeId: string;
    readonly params: unknown;
}

/** Escape hatch for stateful / multi-pass shaders (particle systems,
 *  iterative solvers). The pipeline allocates the standard `outputTex`
 *  + `outputFbo` for the node, then hands the full GL context to
 *  `render` instead of running a single fragment-shader pass. */
export interface RenderOverrideSpec {
    readonly render: (ctx: RenderOverrideContext) => void;
    /** Cleanup hook for any GL resources the override allocated outside
     *  the auto-managed program cache (extra FBOs, textures, VBOs).
     *  Called on node removal. */
    readonly dispose?: (gl: WebGL2RenderingContext, nodeId: string) => void;
}

export interface GpuPassSpec {
    /** GLSL ES 3.00 fragment shader source. Must declare `in vec2 vUv` and
     *  `out vec4 outColor`. Renders at project resolution against a
     *  full-screen quad with the default vertex shader. */
    readonly fragSrc: string;
    /** Optional vertex shader override — almost never needed. The default
     *  emits `vUv` running bottom-up (matches `UNPACK_FLIP_Y_WEBGL` uploads
     *  so sampling at `vUv = (0,0)` reads the source's bottom-left). */
    readonly vertSrc?: string;
    /** Sampler uniform names, indexed parallel to the entry's `inputs`. */
    readonly samplers: readonly string[];
    /** Non-sampler uniform names the pipeline pre-locates. */
    readonly uniforms: readonly string[];
    /** Writes per-frame uniform values. */
    readonly setUniforms: UniformWriter;
    /** Optional list of input slots whose upstream `outputTex` should be
     *  read back into a JS-accessible `ImageData` before `setUniforms`
     *  runs. The captured data lands in `frame.capturedInputs[slot]`.
     *  Used by stateful shaders (particle physics, etc.) that integrate
     *  in JS over upstream pixel data. Each entry costs one synchronous
     *  GPU readback per frame, so opt in narrowly. */
    readonly capturedInputSlots?: readonly number[];
}

export interface RawImageLike {
    readonly data: Uint8Array;
    readonly width: number;
    readonly height: number;
    readonly channels: number;
}

export interface TextureUpload {
    readonly source:
        | { kind: "rawimage"; image: RawImageLike }
        | { kind: "image"; image: HTMLImageElement }
        | { kind: "canvas"; canvas: HTMLCanvasElement };
    readonly width: number;
    readonly height: number;
}

export interface ProducerContext {
    /** Source URLs for upstream chains, indexed parallel to the shader's
     *  `inputs`. Producers typically only need `src`. */
    readonly upstream: readonly { readonly src: string | null }[];
    /** Aborts when the producer's inputs change or the pipeline disposes. */
    readonly signal: AbortSignal;
    /** Optional progress callback (loading model weights, etc.). */
    readonly onProgress?: (info: unknown) => void;
}

export interface ProducerSpec {
    /** Memoization key built from params + upstream URLs. The producer
     *  re-runs only when this string changes. */
    readonly inputKey: (
        params: unknown,
        upstream: readonly { readonly src: string | null }[],
    ) => string;
    readonly run: (
        params: unknown,
        ctx: ProducerContext,
    ) => Promise<TextureUpload>;
}

export { ChainRenderer } from "./ChainRenderer";

export const SHADERS: readonly ShaderEntry[] = [
    topologyEntry,
    mergeEntry,
    depthEntry,
    normalsEntry,
    bumpNormalsEntry,
    sunlightEntry,
    illuminationEntry,
    gradientEntry,
    colorEntry,
    depthRampEntry,
    compositeShaderEntry,
    colorGradeEntry,
    particlesEntry,
    maskMergeEntry,
    circleGridEntry,
    blackHoleEntry,
    nsfwCompareEntry,
    layersEntry,
    gravityMapEntry,
    marblesEntry,
    swarmEntry,
    contourFlowEntry,
    samEntry,
    describeMaskEntry,
    nsfwDetectEntry,
];
