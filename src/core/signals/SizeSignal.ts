import type { Signal } from "./Signal";

export interface SizeState {
    width: number;
    height: number;
}

export interface SizeSignalOptions {
    readonly host: HTMLElement;
}

let counter = 0;

/**
 * Tracks an element's content-box size via `ResizeObserver`. Reads `clientWidth`
 * / `clientHeight` (CSS pixels) — multiply by your render DPR before sizing
 * GPU resources.
 */
export function createSizeSignal(opts: SizeSignalOptions): Signal<SizeState> {
    const id = `size-${++counter}`;
    const state: SizeState = { width: 0, height: 0 };
    const listeners = new Set<(s: SizeState) => void>();
    let observer: ResizeObserver | null = null;

    const update = () => {
        const w = opts.host.clientWidth;
        const h = opts.host.clientHeight;
        if (w === state.width && h === state.height) return;
        state.width = w;
        state.height = h;
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
            if (observer) return;
            observer = new ResizeObserver(update);
            observer.observe(opts.host);
            update();
        },
        stop() {
            observer?.disconnect();
            observer = null;
        },
    };
}
