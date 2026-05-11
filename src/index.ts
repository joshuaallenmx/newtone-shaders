// Public surface for `@newtonedev/shaders`.
//
// We intentionally do NOT re-export the ML-using subtrees from the package
// root: classifier/segmenter/detector/parser modules pull `nsfwjs`,
// `onnxruntime-web`, `@huggingface/transformers`, and `@tensorflow/tfjs`
// as runtime imports — bundlers can't tree-shake namespace imports of
// packages whose `sideEffects` flag isn't set, so consumers that only need
// the GPU shaders would otherwise inherit hundreds of MB of model code
// (and a Node-only `Buffer` reference inside `nsfwjs`).
//
// The ML modules still live in this package and can be imported by deeper
// path (e.g. `@newtonedev/shaders/src/skin`) when needed; promote them to
// dedicated subpath exports when there's a real consumer.
export * from "./core";
export * from "./pipeline/compose";
export type {
    BinaryMask,
    PipelineDeps,
    PipelineInput,
    PipelineRegion,
    PipelineResult,
    PipelineStage,
    RunPipelineOptions,
} from "./pipeline/types";
export { boxCenter } from "./focus/box-center";
export type { FocusPoint } from "./focus/types";
export * from "./shaders/liquid-metal";
export * from "./shaders/edge-detect";
export * from "./shaders/skin-mask";
export * from "./shaders/palette-mask";
export * from "./shaders/hair-detect";
export * from "./shaders/low-poly";
export * from "./shaders/silhouette";
export * from "./shaders/circle-detect";
export * from "./shaders/posterize";
export * from "./react";
