import * as THREE from "three";
import type { Signal } from "./Signal";

export interface ScrollState {
    /** Current scroll position. */
    position: THREE.Vector2;
    /** EMA-smoothed scroll velocity (px/sec). */
    velocity: THREE.Vector2;
    /** ||velocity||. */
    speed: number;
}

export interface ScrollSignalOptions {
    /** What to listen on. @default `window` */
    readonly target?: HTMLElement | Window;
    /** EMA time constant for velocity smoothing, in seconds. @default 0.12 */
    readonly velocitySmoothing?: number;
}

let counter = 0;

/**
 * Tracks scroll position and derived velocity. Defaults to `window` scrolling;
 * pass an element to track its `scrollLeft`/`scrollTop` instead.
 *
 * Velocity is sampled on every `scroll` event and smoothed with an EMA. When
 * scrolling stops, velocity stays at its last value (the consumer can apply
 * its own decay if needed — same trade-off as `PointerSignal`).
 */
export function createScrollSignal(
    opts: ScrollSignalOptions = {},
): Signal<ScrollState> {
    const id = `scroll-${++counter}`;
    const target = opts.target ?? (typeof window !== "undefined" ? window : null);
    const tau = opts.velocitySmoothing ?? 0.12;

    const state: ScrollState = {
        position: new THREE.Vector2(0, 0),
        velocity: new THREE.Vector2(0, 0),
        speed: 0,
    };
    const listeners = new Set<(s: ScrollState) => void>();
    let lastTs = 0;
    let started = false;

    const readPosition = (): [number, number] => {
        if (target instanceof Window) return [target.scrollX, target.scrollY];
        if (target) return [target.scrollLeft, target.scrollTop];
        return [0, 0];
    };

    const onScroll = () => {
        const [x, y] = readPosition();
        const prevX = state.position.x;
        const prevY = state.position.y;
        state.position.set(x, y);
        const now = performance.now() / 1000;
        const dt = lastTs > 0 ? Math.max(1 / 240, now - lastTs) : 1 / 60;
        lastTs = now;
        const vx = (x - prevX) / dt;
        const vy = (y - prevY) / dt;
        const alpha = 1 - Math.exp(-dt / tau);
        state.velocity.set(
            state.velocity.x + (vx - state.velocity.x) * alpha,
            state.velocity.y + (vy - state.velocity.y) * alpha,
        );
        state.speed = state.velocity.length();
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
            if (started || !target) return;
            started = true;
            target.addEventListener("scroll", onScroll, { passive: true });
            const [x, y] = readPosition();
            state.position.set(x, y);
        },
        stop() {
            if (!started || !target) return;
            started = false;
            target.removeEventListener("scroll", onScroll);
        },
    };
}
