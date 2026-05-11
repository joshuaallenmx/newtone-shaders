// Asset dimension loader. The pipeline knows source dimensions only
// after textures finish loading inside its own GL context, but the
// editor needs aspect-ratio info synchronously to size the working
// buffer (e.g. for the Global Input → buffer-aspect derivation). This
// is a tiny independent JS-side cache that loads images via the regular
// browser image pipeline and notifies subscribers when dims arrive.

interface AssetDims {
    readonly w: number;
    readonly h: number;
}

const dimsByUrl = new Map<string, AssetDims>();
const pendingByUrl = new Map<string, Promise<AssetDims>>();
const subscribers = new Set<() => void>();

/** Synchronous lookup. Returns `null` if the URL hasn't been loaded. */
export function getAssetDims(url: string | null | undefined): AssetDims | null {
    if (!url) return null;
    return dimsByUrl.get(url) ?? null;
}

/** Trigger an async load (idempotent — concurrent calls share one
 *  Image()). Returns a promise that resolves when dims are known. */
export function loadAssetDims(url: string): Promise<AssetDims> {
    const cached = dimsByUrl.get(url);
    if (cached) return Promise.resolve(cached);
    const inflight = pendingByUrl.get(url);
    if (inflight) return inflight;
    const promise = new Promise<AssetDims>((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            const dims: AssetDims = {
                w: img.naturalWidth,
                h: img.naturalHeight,
            };
            dimsByUrl.set(url, dims);
            pendingByUrl.delete(url);
            for (const sub of subscribers) sub();
            resolve(dims);
        };
        img.onerror = () => {
            pendingByUrl.delete(url);
            reject(new Error(`asset image failed to load: ${url}`));
        };
        img.src = url;
    });
    pendingByUrl.set(url, promise);
    return promise;
}

/** Subscribe to "any dims arrived" notifications. Used by editor hooks
 *  to re-render when a previously-unknown URL's dims become available. */
export function subscribeAssetDims(listener: () => void): () => void {
    subscribers.add(listener);
    return () => {
        subscribers.delete(listener);
    };
}
