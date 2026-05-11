import * as THREE from "three";

/**
 * 1×1 transparent texture for binding to declared-but-unused `iChannelN`
 * slots. The GPU still needs a valid sampler target even if the shader never
 * samples it.
 */
export function makeStubTexture(): THREE.DataTexture {
    const data = new Uint8Array([0, 0, 0, 0]);
    const tex = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat);
    tex.needsUpdate = true;
    return tex;
}
