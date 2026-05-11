import * as THREE from "three";

export interface VideoTextureOptions {
    /** Color space the file is encoded in. @default `THREE.SRGBColorSpace` */
    readonly colorSpace?:
        | typeof THREE.SRGBColorSpace
        | typeof THREE.LinearSRGBColorSpace;
    /** Wrapping mode for both axes. @default `THREE.ClampToEdgeWrapping` */
    readonly wrap?: THREE.Wrapping;
    /** `crossOrigin` attribute on the underlying `<video>`. @default "anonymous" */
    readonly crossOrigin?: string;
    /** Loop playback. @default true */
    readonly loop?: boolean;
    /** Mute audio (required for unattended autoplay in most browsers). @default true */
    readonly muted?: boolean;
    /** Play inline on iOS. @default true */
    readonly playsInline?: boolean;
    /** Start playback as soon as data is available. @default true */
    readonly autoplay?: boolean;
}

export interface VideoTextureHandle {
    readonly texture: THREE.VideoTexture;
    readonly video: HTMLVideoElement;
    /** Stop playback and release GPU + DOM resources. */
    dispose(): void;
}

/**
 * Load a video into a `THREE.VideoTexture`. Resolves once the first frame is
 * decoded so `textureSize()` is non-zero. The returned `dispose()` releases
 * the texture, pauses the video, and detaches its source.
 *
 * The `<video>` element is created in-memory and never attached to the DOM —
 * its frames are pushed straight to the GPU by `THREE.VideoTexture`.
 */
export function loadVideoTexture(
    src: string,
    opts: VideoTextureOptions = {},
): Promise<VideoTextureHandle> {
    return new Promise((resolve, reject) => {
        const video = document.createElement("video");
        video.crossOrigin = opts.crossOrigin ?? "anonymous";
        video.loop = opts.loop ?? true;
        video.muted = opts.muted ?? true;
        video.playsInline = opts.playsInline ?? true;
        video.preload = "auto";

        const cleanupListeners = () => {
            video.removeEventListener("loadeddata", onReady);
            video.removeEventListener("error", onError);
        };

        const onReady = () => {
            cleanupListeners();
            const tex = new THREE.VideoTexture(video);
            tex.colorSpace = opts.colorSpace ?? THREE.SRGBColorSpace;
            const wrap = opts.wrap ?? THREE.ClampToEdgeWrapping;
            tex.wrapS = wrap;
            tex.wrapT = wrap;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.generateMipmaps = false;

            if (opts.autoplay ?? true) {
                void video.play().catch(() => {
                    // Some browsers block autoplay even when muted; the
                    // texture is still valid and will update once playback
                    // is started by the caller.
                });
            }

            resolve({
                texture: tex,
                video,
                dispose() {
                    tex.dispose();
                    video.pause();
                    video.removeAttribute("src");
                    video.load();
                },
            });
        };

        const onError = () => {
            cleanupListeners();
            reject(new Error(`Failed to load video: ${src}`));
        };

        video.addEventListener("loadeddata", onReady);
        video.addEventListener("error", onError);
        video.src = src;
        video.load();
    });
}
