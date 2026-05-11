import type * as THREE from "three";
import { makeRenderTarget, type RenderTargetOptions } from "./RenderTargetPool";

/**
 * Two render targets that swap on every render: the previous frame is the
 * READ target, the new frame writes to the WRITE target, then `swap()`
 * promotes write → read.
 */
export class PingPong {
    private a: THREE.WebGLRenderTarget;
    private b: THREE.WebGLRenderTarget;
    private currentReadIsA = true;

    constructor(width: number, height: number, options?: RenderTargetOptions) {
        this.a = makeRenderTarget(width, height, options);
        this.b = makeRenderTarget(width, height, options);
    }

    /** The texture other passes should sample this frame. */
    read(): THREE.WebGLRenderTarget {
        return this.currentReadIsA ? this.a : this.b;
    }

    /** The target this frame's pass should render into. */
    write(): THREE.WebGLRenderTarget {
        return this.currentReadIsA ? this.b : this.a;
    }

    /** Promote the just-written target to be the next frame's read source. */
    swap(): void {
        this.currentReadIsA = !this.currentReadIsA;
    }

    setSize(width: number, height: number): void {
        const w = Math.max(2, width);
        const h = Math.max(2, height);
        this.a.setSize(w, h);
        this.b.setSize(w, h);
    }

    get width(): number {
        return this.a.width;
    }

    get height(): number {
        return this.a.height;
    }

    dispose(): void {
        this.a.dispose();
        this.b.dispose();
    }
}
