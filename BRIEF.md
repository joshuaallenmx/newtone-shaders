# Detection + Segmentation Pipeline — Brief

> Merged into `@newtonedev/shaders` (option 3 of the original three locations).
> The pipeline modules live under `src/classify`, `src/detect`, `src/segment`,
> `src/pipeline` and feed mask textures back into the existing shader runtime.

## Goal

Artistic censorship tool for a creative shader-based image/video processing app.
Instead of blocking NSFW content, **detect** it and apply artistic effects to
the offending regions — pixel-precise segmentation masks fed into custom GPU
shaders that posterize, low-poly, or otherwise stylize just those regions.

This work produces the **detection + segmentation pipeline**. It integrates
with the existing shader package (`@newtonedev/shaders`) — same repo, internal
modules consumed by the existing shader components.

## Three-tier architecture

| Tier | Purpose | Library | Cost |
|------|---------|---------|------|
| 1. Classify | Trigger — decide if any NSFW handling is needed at all | NSFWJS | Cheap |
| 2. Detect | Macro localization — bounding boxes per body-part class | NudeNet (ONNX in browser) | Medium |
| 3. Segment | Pixel-precise masks per region — the artistic substrate | SAM 2 via transformers.js (or MobileSAM) | Heavy |

The detection boxes are used as prompts for SAM — SAM 2 takes a box and
returns a tight mask within it. Don't run SAM on the whole image blind.

## Stack

- React 18 + Vite + TypeScript (strict).
- `nsfwjs` + `@tensorflow/tfjs` for classification.
- `onnxruntime-web` for NudeNet (the ONNX-exported detector from
  https://github.com/notAI-tech/NudeNet).
- `@huggingface/transformers` (transformers.js) for SAM 2 — model
  `facebook/sam2-hiera-tiny`. Use WebGPU backend if available, fall back to WASM.
- A small playground UI for testing: drop-in image, see classification result,
  bounding boxes drawn on a 2D canvas overlay, then segmentation masks
  overlaid in a translucent color.

## Repo layout (within `@newtonedev/shaders`)

```
src/
  classify/       — NSFWJS wrapper, returns classification probabilities
  detect/         — NudeNet ONNX inference, returns DetectedRegion[]
  segment/        — SAM 2 wrapper. Encoder runs once per image; decoder runs per prompt box
  pipeline/       — orchestrates the three tiers
  react/          — components: <ContentPipeline>, <DebugOverlay>, hooks (alongside existing shader components)
playground/
  App.tsx, vite.config.ts, assets/   (existing — adds a "classify" mode + later "detect"/"segment")
```

## Output contract (the integration boundary)

The end product is a mask texture (or set of masks per class) sized to match
the source image, white inside detected regions and black outside. This mask
is consumed by GPU shaders that blend effect ↔ source by mask value. So the
segmentation step exposes the mask as either:

- An `OffscreenCanvas` / `HTMLCanvasElement` (for `THREE.CanvasTexture` upload), or
- A raw `Uint8Array` of pixel data with width/height (for `THREE.DataTexture`).

Type sketch:

```ts
interface DetectedRegion {
  class: string;          // e.g. "FEMALE_BREAST_EXPOSED"
  box: [number, number, number, number];  // [x, y, w, h] in source pixels
  score: number;          // 0..1
  mask?: ImageBitmap;     // pixel-precise mask from SAM, same size as box
}

interface PipelineResult {
  classification: { porn: number; sexy: number; hentai: number; neutral: number; drawing: number };
  regions: DetectedRegion[];
  combinedMask: HTMLCanvasElement;  // union of all region masks, source-image-sized
}
```

## Build incrementally — milestones

1. **NSFWJS classification standalone.** Drop-in image, see probabilities. ~1 hour.
2. **NudeNet detection standalone.** Same image, see bounding boxes drawn on a 2D canvas overlay. ~half a day (mostly figuring out ONNX inference + class mapping).
3. **SAM 2 segmentation with hardcoded prompt box.** Pick a box manually, run SAM, render the mask as a translucent color overlay. ~half a day (transformers.js docs + WebGPU setup).
4. **Compose:** NudeNet boxes feed SAM as prompts. Output the combined mask. ~1–2 hours once steps 2 + 3 work.
5. React component wrapping the whole thing with reactive props (image src, classification threshold, class allowlist, mask render mode).

## Constraints

- TypeScript strict, no `any`.
- Browser-only (no server-side step). All inference happens client-side.
- Static images only for v1. Video can come later.
- Don't write generic mask blurring / shader code — that lives alongside in
  the existing shader components. The pipeline's job ends at producing the mask.

## First task (tier 1)

- Add `src/classify/` (NSFWJS wrapper) and re-export from `src/index.ts`.
- Install `nsfwjs` and `@tensorflow/tfjs`.
- Add a "classify" mode to the existing playground: load image, show
  probabilities. Smallest end-to-end loop.
- After confirmation, move to tier 2 (detection), then tier 3 (segmentation).

Each tier has enough complexity to get its own focused session.

## Integration with existing shaders

When the pipeline produces masks, every existing shader takes ~5 lines to
accept a mask texture and blend effect ↔ source by it. We do that integration
once tier 3 is producing masks.
