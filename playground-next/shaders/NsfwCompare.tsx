import type { GpuPassSpec, ShaderEntry } from ".";

// NSFW Compare — measures the original ("before") and the processed
// ("after") feed against the NSFWJS classifier and overlays the
// comparison on top of the canvas. The shader itself is a pass-through
// of "after" — all the work happens on the JS side, in the editor's
// `NsfwStatusOverlay` component, which detects this entry id at the
// chain root and renders the text overlay. We still declare both inputs
// here so the pipeline plan includes them as PlanNodes, which gives the
// overlay something to point `captureNodeImageData` at.

// uBeforeRetention is a uniform always set to 0. Multiplying the
// `uBefore` sample by it keeps the symbol "live" so the GLSL compiler
// can't strip the sampler — Pipeline.acquireProgram requires every
// declared sampler to resolve to a real location. The `before`
// texture is consumed by the JS overlay (via captureNodeImageData),
// not by this fragment pass, but we still need the sampler binding.
const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uBefore;
uniform sampler2D uAfter;
uniform float uBeforeRetention;
void main() {
    vec4 a = texture(uAfter, vUv);
    vec4 b = texture(uBefore, vUv);
    outColor = a + b * uBeforeRetention;
}
`;

const nsfwCompareGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uBefore", "uAfter"],
    uniforms: ["uBeforeRetention"],
    setUniforms: (gl, locs) => {
        gl.uniform1f(locs.get("uBeforeRetention")!, 0);
    },
};

/** Stable id used by the editor to detect this terminal and render the
 *  classification overlay. */
export const NSFW_COMPARE_ENTRY_ID = "nsfwCompare";

export const nsfwCompareEntry: ShaderEntry = {
    id: NSFW_COMPARE_ENTRY_ID,
    name: "NSFW Compare (before/after)",
    defaultParams: {},
    inputs: [
        { id: "before", label: "before · source" },
        { id: "after", label: "after · processed" },
    ],
    gpu: nsfwCompareGpuSpec,
};
