import type * as THREE from "three";

export type UniformValue =
    | number
    | boolean
    | THREE.Vector2
    | THREE.Vector3
    | THREE.Vector4
    | THREE.Color
    | THREE.Matrix3
    | THREE.Matrix4
    | THREE.Texture
    | Float32Array
    | Int32Array
    | readonly number[]
    | readonly THREE.Vector2[]
    | readonly THREE.Vector3[]
    | readonly THREE.Vector4[]
    | null;

/** How to size a render target relative to the canvas. */
export type SizeSpec =
    | "full"
    | "half"
    | readonly [number, number]
    | ((size: { w: number; h: number }) => readonly [number, number]);

/**
 * A texture input. `pass` references another pass in the same pipeline (the
 * pipeline resolves to its READ target each frame); `output` references a
 * named output of another pipeline (resolved by a graph layer); `asset` is a
 * pre-built texture (noise, env gradient, mask, image).
 */
export type TextureRef =
    | { kind: "pass"; passId: string }
    | { kind: "output"; pipelineId: string; name: string }
    | { kind: "asset"; texture: THREE.Texture };

/** Where a uniform's value comes from on each frame. */
export type UniformProvider<TSignal = unknown> =
    | { kind: "static"; value: UniformValue }
    | {
          kind: "signal";
          signal: { get: () => TSignal };
          project?: (state: TSignal) => UniformValue;
      }
    | { kind: "texture"; ref: TextureRef }
    | { kind: "computed"; fn: (ctx: FrameContext) => UniformValue };

/** Per-frame context passed to computed uniform providers. */
export interface FrameContext {
    readonly time: number;
    readonly frame: number;
    /** Final canvas size in render pixels. */
    readonly canvas: { w: number; h: number };
    /** This pass's render-target size in render pixels. */
    readonly target: { w: number; h: number };
}

/** What the runner passes to `Pipeline.render` — `target` is filled in per-pass. */
export type PipelineRenderInput = Omit<FrameContext, "target">;

export interface PingPongTarget {
    readonly kind: "pingpong";
    readonly size: SizeSpec;
    readonly format?: THREE.PixelFormat;
    readonly type?: THREE.TextureDataType;
}

export interface FixedTarget {
    readonly kind: "fixed";
    readonly size: readonly [number, number];
    readonly format?: THREE.PixelFormat;
    readonly type?: THREE.TextureDataType;
}

export interface ScreenTarget {
    readonly kind: "screen";
}

export type PassTarget = PingPongTarget | FixedTarget | ScreenTarget;

export interface PassConfig {
    readonly id: string;
    /**
     * GLSL fragment source. If it doesn't already contain `void main(`, the
     * Pass wraps it via `composeFragment` (treats it as Shadertoy-style
     * `mainImage(out vec4, in vec2)` body).
     */
    readonly fragment: string;
    readonly uniforms: Readonly<Record<string, UniformProvider>>;
    readonly target: PassTarget;
}

export interface PipelineConfig {
    readonly id: string;
    readonly passes: readonly PassConfig[];
    /** Map of named outputs to pass ids, for chaining and external sampling. */
    readonly outputs?: Readonly<Record<string, string>>;
}
