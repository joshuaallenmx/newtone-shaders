import * as THREE from "three";
import type { FrameContext, PassConfig, UniformProvider, UniformValue } from "./types";
import { VERTEX, composeFragment } from "./prelude";

/** External resolvers the Pass needs each frame to fill texture refs. */
export interface PassResources {
    readonly resolveTextureRef: (passId: string) => THREE.Texture | null;
    readonly resolveOutputRef?: (
        pipelineId: string,
        outputName: string,
    ) => THREE.Texture | null;
}

/**
 * A single fragment-shader pass over a fullscreen quad. Holds the compiled
 * `ShaderMaterial`, a snapshot of its uniform providers, and knows how to
 * resync each uniform per frame.
 */
export class Pass {
    readonly id: string;
    readonly config: PassConfig;
    readonly material: THREE.ShaderMaterial;

    constructor(config: PassConfig, vertex = VERTEX) {
        this.id = config.id;
        this.config = config;
        const fragment = config.fragment.includes("void main(")
            ? config.fragment
            : composeFragment(config.fragment);
        const uniforms: Record<string, THREE.IUniform<unknown>> = {};
        for (const [name, provider] of Object.entries(config.uniforms)) {
            uniforms[name] = { value: defaultForProvider(provider) };
        }
        this.material = new THREE.ShaderMaterial({
            glslVersion: THREE.GLSL3,
            vertexShader: vertex,
            fragmentShader: fragment,
            uniforms,
        });
    }

    sync(ctx: FrameContext, resources: PassResources): void {
        const u = this.material.uniforms;
        for (const [name, provider] of Object.entries(this.config.uniforms)) {
            const slot = u[name];
            if (!slot) continue;
            slot.value = resolveProvider(provider, ctx, resources, slot.value);
        }
    }

    dispose(): void {
        this.material.dispose();
    }
}

function defaultForProvider(p: UniformProvider): unknown {
    switch (p.kind) {
        case "static":
            return p.value;
        case "texture":
            if (p.ref.kind === "asset") return p.ref.texture;
            return null;
        default:
            return null;
    }
}

function resolveProvider(
    p: UniformProvider,
    ctx: FrameContext,
    resources: PassResources,
    current: unknown,
): unknown {
    switch (p.kind) {
        case "static":
            return p.value;
        case "computed":
            return p.fn(ctx);
        case "signal": {
            const v = p.signal.get();
            return p.project ? p.project(v as never) : (v as UniformValue);
        }
        case "texture": {
            switch (p.ref.kind) {
                case "asset":
                    return p.ref.texture;
                case "pass":
                    return resources.resolveTextureRef(p.ref.passId);
                case "output":
                    return resources.resolveOutputRef
                        ? resources.resolveOutputRef(p.ref.pipelineId, p.ref.name)
                        : current;
            }
        }
    }
}
