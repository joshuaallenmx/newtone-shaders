import * as THREE from "three";

/**
 * RGBA noise texture for hash / dither / surface-perturbation lookups. Wraps
 * with `RepeatWrapping` so a small texture tiles seamlessly under any UV.
 */
export function makeNoiseTexture(size = 256): THREE.DataTexture {
    const data = new Uint8Array(size * size * 4);
    for (let i = 0; i < data.length; i++) {
        data[i] = Math.floor(Math.random() * 256);
    }
    const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
}
