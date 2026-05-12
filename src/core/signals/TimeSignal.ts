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
    /**
     * Re-anchor the internal start timestamp so the next `tick()` continues
     * from the current `state.time` rather than jumping forward by the
     * wall-clock duration of the pause. Call this when resuming a paused
     * render loop (e.g. after the canvas scrolls back into view).
     */
    resume(): void;
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
    let start = now();
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
        resume() {
            start = now() - state.time * 1000;
        },
    };
}
