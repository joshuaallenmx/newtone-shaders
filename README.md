# @newtonedev/shaders

GPU shader pipeline for React + Three.js. Designed around composable shader
nodes, signal-driven uniforms, and reusable texture / DOM utilities — write a
new shader by declaring a fragment, picking signals, and slotting it into a
`Pipeline`.

## Install

```sh
npm install @newtonedev/shaders three react react-dom
```

`three`, `react`, and `react-dom` are peer dependencies.

## What's in the box

- **`Pipeline` / `Pass`** — ordered fragment passes with named outputs and
  intra-pipeline texture refs. Single-pass shaders and multi-pass simulations
  use the same primitives.
- **Signals** (`PointerSignal`, `ScrollSignal`, `SizeSignal`, `MutationSignal`,
  `TimeSignal`) — DOM-side input sources with a uniform pull/subscribe
  interface. Plug them into uniforms via `{ kind: "signal", ... }`.
- **Textures** — noise, env-gradient, image loader, SVG/DOM mask builder.
- **Color** — `hexToRgbBytes`, `cssColorToRgbBytes` (handles `oklch()`,
  `rgb()`, named colors via the browser's parser), `cssColorToLinearRgb`,
  `setVec3FromColor`. Plus a `ColorResolver` injection point so the package
  doesn't depend on any specific token system.
- **DOM** — `applyBleedFit` to make a canvas cover its parent section past
  padding.
- **Shaders** — `LiquidMetal` (Florian Berger's CFD adapted as a 3-pass
  pipeline) as the reference implementation.
- **React layer** — `<ShaderCanvas>`, `<LiquidMetal>`, `useShaderPipeline`,
  `<ColorResolverProvider>`.

## Standard uniforms

The `SHADERTOY_PRELUDE` declares the inputs every shader can rely on. Bind
only the ones you use; declarations are free.

| Uniform | Type | Source |
|---|---|---|
| `iTime` | `float` | `TimeSignal` (seconds) |
| `iFrame` | `int` | `TimeSignal` (counter) |
| `iResolution` | `vec3` | this pass's render target |
| `iMouse` | `vec4` | `PointerSignal.position` (Shadertoy compat) |
| `iPointer` | `vec4` | `PointerSignal.position` |
| `iPointerDelta` | `vec2` | `PointerSignal.delta` |
| `iPointerVelocity` | `vec2` | `PointerSignal.velocity` (px/sec, EMA) |
| `iPointerSpeed` | `float` | `PointerSignal.speed` |
| `iScroll` | `vec2` | `ScrollSignal.position` |
| `iScrollVelocity` | `vec2` | `ScrollSignal.velocity` (px/sec, EMA) |
| `iChannel0..3` | `sampler2D` | textures (pass output, asset, image) |

## Quick start — using LiquidMetal

```tsx
import { LiquidMetal, ColorResolverProvider } from "@newtonedev/shaders";

export function Hero() {
    return (
        <ColorResolverProvider resolver={(s) => s /* or your token resolver */}>
            <LiquidMetal
                tint="#ffffff"
                envColors={["#1a0a3a", "#ff8866", "#ffe9c0"]}
                params={{ pointerForce: 0.0008, viscosity: 0.05 }}
            >
                <h1>Headline</h1>
            </LiquidMetal>
        </ColorResolverProvider>
    );
}
```

## Authoring a new shader

A shader is just a `PipelineConfig` — one or more passes that read named
inputs and write to a render target. The simplest case:

```tsx
import {
    Pipeline,
    composeFragment,
    createPointerSignal,
    createTimeSignal,
    ShaderCanvas,
    type ShaderSetup,
} from "@newtonedev/shaders";
import * as THREE from "three";

const fragment = /* glsl */ `
void mainImage(out vec4 c, in vec2 p) {
    vec2 uv = p / iResolution.xy;
    float d = length(uv - iPointer.xy / iResolution.xy);
    c = vec4(vec3(smoothstep(0.2, 0.0, d) * iPointerSpeed * 0.001), 1.0);
}
`;

const setup: ShaderSetup = ({ canvasHost, renderer }) => {
    const pointer = createPointerSignal({ host: canvasHost });
    const time = createTimeSignal();
    const iResolution = new THREE.Vector3();

    const pipeline = new Pipeline(renderer, {
        id: "sparkle",
        passes: [{
            id: "main",
            target: { kind: "screen" },
            fragment: composeFragment(fragment),
            uniforms: {
                iTime: { kind: "computed", fn: (ctx) => ctx.time },
                iResolution: {
                    kind: "computed",
                    fn: (ctx) => {
                        iResolution.set(ctx.target.w, ctx.target.h, 1);
                        return iResolution;
                    },
                },
                iPointer: {
                    kind: "signal",
                    signal: pointer,
                    project: (s) => (s as { position: THREE.Vector4 }).position,
                },
                iPointerSpeed: {
                    kind: "signal",
                    signal: pointer,
                    project: (s) => (s as { speed: number }).speed,
                },
            },
        }],
    });

    return { pipeline, signals: [pointer], time };
};

export function Sparkle({ children }) {
    return <ShaderCanvas setup={setup}>{children}</ShaderCanvas>;
}
```

For multi-pass shaders (e.g. a sim → final-shade pipeline), declare the
upstream pass with a `pingpong` target and reference it in the downstream
pass's uniforms:

```ts
{
    id: "sim",
    target: { kind: "pingpong", size: "full" },
    uniforms: {
        iChannel0: { kind: "texture", ref: { kind: "pass", passId: "sim" } },
        // ...
    },
},
{
    id: "image",
    target: { kind: "screen" },
    uniforms: {
        iChannel0: { kind: "texture", ref: { kind: "pass", passId: "sim" } },
        // ...
    },
},
```

The pipeline resolves `pass` refs to the **read** side of the upstream
ping-pong each frame — no manual `aIdx` / `bIdx` bookkeeping.

## Shader chaining

Pipelines expose named outputs:

```ts
{ id: "metal", passes: [...], outputs: { final: "image" } }
```

Future versions will add a `PipelineGraph` that lets a downstream pipeline
sample `{ kind: "output", pipelineId: "metal", name: "final" }`. For now,
single pipelines are the unit of composition.

## Color resolution

Anywhere a color is accepted, you can pass any CSS color (hex, `rgb()`,
`oklch()`, named) — or a token reference like `"$text"` that resolves through
an injected `ColorResolver`:

```tsx
import { ColorResolverProvider } from "@newtonedev/shaders";

<ColorResolverProvider resolver={(s) => myTokenLookup(s)}>
    <LiquidMetal tint="$text" envColors={["$fill.deep", "$fill"]} />
</ColorResolverProvider>
```

The resolver is a single function `(input: string) => string`. Bridge it to
whatever token system you use — `@newtonedev/interface`, CSS variables, or
a config object.

## License

MIT, except where individual GLSL files note otherwise (`src/shaders/liquid-metal/glsl.ts`
carries flockaroo's CC BY-NC-SA 3.0 header until the rewrite of `BUFFER_A`
supersedes the original solver).
