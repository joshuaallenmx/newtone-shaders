/**
 * Shared vertex shader for all fullscreen-quad passes — emits clip-space
 * coordinates so `gl_FragCoord` covers the full target in pixel units.
 */
export const VERTEX = /* glsl */ `
void main() {
    gl_Position = vec4(position, 1.0);
}
`;

/**
 * The standard set of uniforms every shader in the framework can rely on.
 * Declarations cost nothing if the shader doesn't read them; bindings are
 * opt-in (only uniforms in a `Pass`'s `uniforms` map get updated each frame).
 *
 * Naming follows Shadertoy conventions where applicable; new framework-level
 * inputs use the `i*` prefix too.
 */
export const SHADERTOY_PRELUDE = /* glsl */ `
precision highp float;
precision highp int;

uniform float iTime;
uniform int iFrame;
uniform vec3 iResolution;
uniform vec4 iMouse;

uniform vec4 iPointer;
uniform vec2 iPointerDelta;
uniform vec2 iPointerVelocity;
uniform float iPointerSpeed;

uniform vec2 iScroll;
uniform vec2 iScrollVelocity;

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;

out vec4 outColor;
`;

/**
 * Wrap a Shadertoy-style `mainImage(out vec4 fragColor, in vec2 fragCoord)`
 * body with the prelude and a `main()` that calls it. Use this for any
 * fragment that follows the Shadertoy conventions; pass raw GLSL to `Pass`
 * directly if your shader has its own `void main(...)`.
 */
export function composeFragment(
    userCode: string,
    prelude = SHADERTOY_PRELUDE,
): string {
    return `${prelude}
${userCode}

void main() {
    mainImage(outColor, gl_FragCoord.xy);
}
`;
}
