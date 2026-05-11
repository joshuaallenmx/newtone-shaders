// Shared WebGL2 helpers used by the Pipeline executor and any shader entry
// that declares a GpuPassSpec. The same compileShader / createProgram pair
// was previously inlined in Gradient, Merge, Composite, DepthRamp; they all
// import from here now (or used to before the deletion pass).

export const DEFAULT_VERT_SRC = `#version 300 es
in vec2 aPosition;
out vec2 vUv;
void main() {
    // vUv runs bottom-up; pair with UNPACK_FLIP_Y_WEBGL on uploads so the
    // sampled image matches its on-screen orientation. For purely procedural
    // passes (e.g. Gradient) the orientation is set by the pass's own math.
    vUv = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

export const QUAD_VERTICES = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

export function compileShader(
    gl: WebGL2RenderingContext,
    type: number,
    src: string,
    label?: string,
): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) throw new Error("createShader failed");
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        const kind = type === gl.VERTEX_SHADER ? "vertex" : "fragment";
        const tag = label ? ` [${label}]` : "";
        // Some drivers return null/empty for the info log; dump the source
        // numbered so the failing line is visible from the console.
        const numbered = src
            .split("\n")
            .map((line, i) => `${String(i + 1).padStart(3, " ")}: ${line}`)
            .join("\n");
        console.error(
            `[gl] ${kind} shader compile failed${tag}\n` +
                `info: ${info ?? "(empty)"}\n` +
                `source:\n${numbered}`,
        );
        gl.deleteShader(shader);
        throw new Error(
            `${kind} shader compile failed${tag}: ${info ?? "(no info log — see console for source)"}`,
        );
    }
    return shader;
}

export function createProgram(
    gl: WebGL2RenderingContext,
    vertSrc: string,
    fragSrc: string,
    label?: string,
): WebGLProgram {
    const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc, label);
    const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc, label);
    const program = gl.createProgram();
    if (!program) throw new Error("createProgram failed");
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        throw new Error(
            `program link failed${label ? ` [${label}]` : ""}: ${info ?? "(no info log)"}`,
        );
    }
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    return program;
}

export function createQuadVbo(gl: WebGL2RenderingContext): WebGLBuffer {
    const vbo = gl.createBuffer();
    if (!vbo) throw new Error("createBuffer failed");
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD_VERTICES, gl.STATIC_DRAW);
    return vbo;
}

export function bindQuadAttribute(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    vbo: WebGLBuffer,
): void {
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    const aPos = gl.getAttribLocation(program, "aPosition");
    if (aPos < 0) return; // program may not declare aPosition; fine
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
}

export interface ResizableTexture {
    texture: WebGLTexture;
    width: number;
    height: number;
}

export function createColorTexture(
    gl: WebGL2RenderingContext,
): WebGLTexture {
    const tex = gl.createTexture();
    if (!tex) throw new Error("createTexture failed");
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
}

export function resizeColorTexture(
    gl: WebGL2RenderingContext,
    tex: WebGLTexture,
    width: number,
    height: number,
): void {
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA8,
        width,
        height,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
    );
}
