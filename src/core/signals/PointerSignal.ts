import * as THREE from "three";
import type { Signal } from "./Signal";

export interface PointerState {
    /** xy = current GL pixels (DPI-scaled, Y from bottom); zw = previous-frame xy. */
    position: THREE.Vector4;
    /** Per-frame raw delta in GL pixels. */
    delta: THREE.Vector2;
    /** EMA-smoothed dx/dt, dy/dt in GL pixels/sec. */
    velocity: THREE.Vector2;
    /** ||velocity||, useful as a single `float` uniform. */
    speed: number;
    /** Pointer is currently inside the host element. */
    active: boolean;
}

export interface PointerSignalOptions {
    readonly host: HTMLElement;
    /** Returns the device pixel ratio. @default () => devicePixelRatio */
    readonly dpr?: () => number;
    /** EMA time constant for velocity smoothing, in seconds. @default 0.08 */
    readonly velocitySmoothing?: number;
    /** Listen on `window` so off-canvas drag still updates. @default true */
    readonly windowMove?: boolean;
}

let counter = 0;

export function createPointerSignal(
    opts: PointerSignalOptions,
): Signal<PointerState> {
    const id = `pointer-${++counter}`;
    const dpr = opts.dpr ?? (() => window.devicePixelRatio || 1);
    const tau = opts.velocitySmoothing ?? 0.08;
    const useWindow = opts.windowMove ?? true;

    const state: PointerState = {
        position: new THREE.Vector4(0, 0, 0, 0),
        delta: new THREE.Vector2(0, 0),
        velocity: new THREE.Vector2(0, 0),
        speed: 0,
        active: false,
    };
    const listeners = new Set<(s: PointerState) => void>();
    let lastMoveTs = 0;
    let started = false;

    const onMove = (e: PointerEvent) => {
        const rect = opts.host.getBoundingClientRect();
        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const px = dpr();
        const x = cssX * px;
        // GL Y is bottom-up; DOM Y is top-down.
        const y = (rect.height - cssY) * px;
        const prevX = state.position.x;
        const prevY = state.position.y;
        state.position.set(x, y, prevX, prevY);
        state.delta.set(x - prevX, y - prevY);
        const now = performance.now() / 1000;
        const dt = lastMoveTs > 0 ? Math.max(1 / 240, now - lastMoveTs) : 1 / 60;
        lastMoveTs = now;
        const vx = state.delta.x / dt;
        const vy = state.delta.y / dt;
        const alpha = 1 - Math.exp(-dt / tau);
        state.velocity.set(
            state.velocity.x + (vx - state.velocity.x) * alpha,
            state.velocity.y + (vy - state.velocity.y) * alpha,
        );
        state.speed = state.velocity.length();
        state.active =
            cssX >= 0 && cssY >= 0 && cssX <= rect.width && cssY <= rect.height;
        for (const fn of listeners) fn(state);
    };

    const onLeave = () => {
        state.active = false;
        for (const fn of listeners) fn(state);
    };

    return {
        id,
        get: () => state,
        subscribe(fn) {
            listeners.add(fn);
            return () => {
                listeners.delete(fn);
            };
        },
        start() {
            if (started) return;
            started = true;
            const target: EventTarget = useWindow ? window : opts.host;
            target.addEventListener("pointermove", onMove as EventListener);
            opts.host.addEventListener("pointerleave", onLeave);
        },
        stop() {
            if (!started) return;
            started = false;
            const target: EventTarget = useWindow ? window : opts.host;
            target.removeEventListener("pointermove", onMove as EventListener);
            opts.host.removeEventListener("pointerleave", onLeave);
        },
    };
}
