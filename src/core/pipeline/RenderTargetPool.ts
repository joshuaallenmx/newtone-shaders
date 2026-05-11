import * as THREE from "three";

export interface RenderTargetOptions {
    readonly format?: THREE.PixelFormat;
    readonly type?: THREE.TextureDataType;
    readonly minFilter?: THREE.MagnificationTextureFilter;
    readonly magFilter?: THREE.MagnificationTextureFilter;
    readonly wrapS?: THREE.Wrapping;
    readonly wrapT?: THREE.Wrapping;
}

const DEFAULTS: Required<RenderTargetOptions> = {
    format: THREE.RGBAFormat,
    type: THREE.HalfFloatType,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
};

/**
 * Half-float RGBA render target with sane simulation defaults: linear
 * filtering, clamp-to-edge wrap, no depth/stencil. Override per-shader when
 * the sim writes outside [0, 1] or needs full-float precision.
 */
export function makeRenderTarget(
    width: number,
    height: number,
    options: RenderTargetOptions = {},
): THREE.WebGLRenderTarget {
    const opts = { ...DEFAULTS, ...options };
    return new THREE.WebGLRenderTarget(Math.max(2, width), Math.max(2, height), {
        format: opts.format,
        type: opts.type,
        minFilter: opts.minFilter,
        magFilter: opts.magFilter,
        wrapS: opts.wrapS,
        wrapT: opts.wrapT,
        depthBuffer: false,
        stencilBuffer: false,
    });
}
