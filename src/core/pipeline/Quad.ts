import * as THREE from "three";

/**
 * Fullscreen-quad scaffolding shared across all passes that draw to a single
 * renderer. Reusing one geometry/scene/camera per renderer avoids per-pass
 * allocation overhead.
 */
export interface SharedQuad {
    readonly geometry: THREE.PlaneGeometry;
    readonly camera: THREE.OrthographicCamera;
    readonly mesh: THREE.Mesh;
    readonly scene: THREE.Scene;
    /** Release the cached resources; subsequent `getSharedQuad` rebuilds them. */
    dispose(): void;
}

const cache = new WeakMap<THREE.WebGLRenderer, SharedQuad>();

export function getSharedQuad(renderer: THREE.WebGLRenderer): SharedQuad {
    const cached = cache.get(renderer);
    if (cached) return cached;
    const geometry = new THREE.PlaneGeometry(2, 2);
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const mesh = new THREE.Mesh(geometry);
    const scene = new THREE.Scene();
    scene.add(mesh);
    const entry: SharedQuad = {
        geometry,
        camera,
        mesh,
        scene,
        dispose() {
            geometry.dispose();
            scene.remove(mesh);
            cache.delete(renderer);
        },
    };
    cache.set(renderer, entry);
    return entry;
}
