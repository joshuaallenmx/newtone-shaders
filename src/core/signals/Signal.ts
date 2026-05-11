/**
 * A small input source: DOM event listener, observer, or computed value with
 * a uniform pull/subscribe interface. Pipelines call `get()` per frame to
 * snapshot the current value into uniforms; React adapters can subscribe.
 *
 * Signals own their listeners. `start()` attaches them; `stop()` detaches.
 * Implementations are expected to be safe to start/stop repeatedly.
 */
export interface Signal<T> {
    readonly id: string;
    get(): T;
    subscribe(listener: (value: T) => void): () => void;
    start(): void;
    stop(): void;
}
