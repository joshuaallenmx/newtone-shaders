import * as THREE from "three";

export interface ImageTextureOptions {
    /** Color space the file is encoded in. @default `THREE.SRGBColorSpace` */
    readonly colorSpace?: typeof THREE.SRGBColorSpace | typeof THREE.LinearSRGBColorSpace;
    /** Wrapping mode for both axes. @default `THREE.ClampToEdgeWrapping` */
    readonly wrap?: THREE.Wrapping;
    /** `crossOrigin` attribute on the underlying `<img>`. @default "anonymous" */
    readonly crossOrigin?: string;
}

/**
 * Load an image into a `THREE.Texture`. Resolves once the image has loaded.
 * Suitable for env reflection inputs and any other static image asset.
 */
export function loadImageTexture(
    src: string,
    opts: ImageTextureOptions = {},
): Promise<THREE.Texture> {
    return new Promise((resolve, reject) => {
        const loader = new THREE.TextureLoader();
        loader.crossOrigin = opts.crossOrigin ?? "anonymous";
        loader.load(
            src,
            (tex) => {
                tex.colorSpace = opts.colorSpace ?? THREE.SRGBColorSpace;
                const wrap = opts.wrap ?? THREE.ClampToEdgeWrapping;
                tex.wrapS = wrap;
                tex.wrapT = wrap;
                tex.minFilter = THREE.LinearFilter;
                tex.magFilter = THREE.LinearFilter;
                tex.needsUpdate = true;
                resolve(tex);
            },
            undefined,
            (err) => reject(err),
        );
    });
}
