import type { Signal } from "./Signal";

export interface MutationSignalOptions {
    readonly host: HTMLElement;
    readonly init?: MutationObserverInit;
    /** Trailing-edge debounce window in ms. @default 80 */
    readonly debounceMs?: number;
}

const DEFAULT_INIT: MutationObserverInit = {
    childList: true,
    subtree: true,
    characterData: true,
    attributes: true,
};

let counter = 0;

/**
 * Increments a tick counter when the host's subtree mutates. Debounced so a
 * burst of mutations (e.g. a React re-render landing) produces one tick.
 */
export function createMutationSignal(
    opts: MutationSignalOptions,
): Signal<number> {
    const id = `mutation-${++counter}`;
    const init = opts.init ?? DEFAULT_INIT;
    const debounce = opts.debounceMs ?? 80;
    let tick = 0;
    const listeners = new Set<(n: number) => void>();
    let observer: MutationObserver | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const fire = () => {
        timer = null;
        tick++;
        for (const fn of listeners) fn(tick);
    };

    const onMutation = () => {
        if (timer !== null) clearTimeout(timer);
        timer = setTimeout(fire, debounce);
    };

    return {
        id,
        get: () => tick,
        subscribe(fn) {
            listeners.add(fn);
            return () => {
                listeners.delete(fn);
            };
        },
        start() {
            if (observer) return;
            observer = new MutationObserver(onMutation);
            observer.observe(opts.host, init);
        },
        stop() {
            observer?.disconnect();
            observer = null;
            if (timer !== null) {
                clearTimeout(timer);
                timer = null;
            }
        },
    };
}
