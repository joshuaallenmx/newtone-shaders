import { useEffect, useRef } from "react";
import * as THREE from "three";
import {
    loadImageTexture,
    loadVideoTexture,
    makeStubTexture,
    type VideoTextureHandle,
} from "../core/textures";

const VIDEO_EXT_RE = /\.(mp4|webm|mov|m4v|ogg|ogv|mkv)(\?|#|$)/i;

export type MediaKind = "image" | "video";

interface MediaHandle {
    readonly texture: THREE.Texture;
    dispose(): void;
}

function detectKind(src: string): MediaKind {
    return VIDEO_EXT_RE.test(src) ? "video" : "image";
}

async function loadMedia(src: string, kind: MediaKind): Promise<MediaHandle> {
    if (kind === "video") {
        const handle: VideoTextureHandle = await loadVideoTexture(src);
        return { texture: handle.texture, dispose: () => handle.dispose() };
    }
    const tex = await loadImageTexture(src);
    return { texture: tex, dispose: () => tex.dispose() };
}

export interface UseMediaSourceResult {
    /** Stable getter — returns the current texture or a 1×1 stub fallback. */
    readonly getTexture: () => THREE.Texture;
}

/**
 * Load an image or video URL into a stable texture handle. The returned
 * `getTexture` is suitable for `kind: "computed"` uniform providers — the
 * underlying `THREE.Texture` swaps in once the source loads, returning a
 * 1×1 black stub until then. Auto-detects video vs. image from the URL
 * extension; pass `kind` to override.
 */
export function useMediaSource(
    src: string,
    kind?: MediaKind,
): UseMediaSourceResult {
    const stubRef = useRef<THREE.DataTexture | null>(null);
    if (!stubRef.current) stubRef.current = makeStubTexture();
    const sourceRef = useRef<THREE.Texture | null>(null);
    const getTextureRef = useRef(
        () => sourceRef.current ?? (stubRef.current as THREE.Texture),
    );

    useEffect(() => {
        let cancelled = false;
        let handle: MediaHandle | null = null;
        const resolvedKind = kind ?? detectKind(src);
        loadMedia(src, resolvedKind)
            .then((h) => {
                if (cancelled) {
                    h.dispose();
                    return;
                }
                handle = h;
                sourceRef.current = h.texture;
            })
            .catch(() => {
                // Stay on the stub if the source fails to load.
            });
        return () => {
            cancelled = true;
            if (sourceRef.current === handle?.texture) {
                sourceRef.current = null;
            }
            handle?.dispose();
        };
    }, [src, kind]);

    useEffect(() => {
        return () => {
            stubRef.current?.dispose();
            stubRef.current = null;
        };
    }, []);

    return { getTexture: getTextureRef.current };
}
