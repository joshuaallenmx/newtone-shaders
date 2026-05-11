import type { Signal } from "./Signal";

export interface TimeState {
    /** Seconds since the signal was created. */
    time: number;
    /** Frame counter; advances on each `tick()`. */
    frame: number;
}

export interface TimeSignalControls extends Signal<TimeState> {
    /** Advance the signal — called once per render frame by the pipeline runner. */
    tick(): void;
}

export interface TimeSignalOptions {
    /** Override `performance.now()` (mostly for tests). */
    readonly now?: () => number;
}

let counter = 0;

/**
 * `iTime` / `iFrame` source. Driven manually by a render loop calling
 * `tick()` rather than its own `requestAnimationFrame` — the pipeline owns
 * the loop and tells the signal to advance.
 */
export function createTimeSignal(
    opts: TimeSignalOptions = {},
): TimeSignalControls {
    const id = `time-${++counter}`;
    const now = opts.now ?? (() => performance.now());
    const start = now();
    const state: TimeState = { time: 0, frame: 0 };
    const listeners = new Set<(s: TimeState) => void>();

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
            // No DOM listeners; the runner advances via `tick()`.
        },
        stop() {},
        tick() {
            state.time = (now() - start) / 1000;
            state.frame++;
            for (const fn of listeners) fn(state);
        },
    };
}
