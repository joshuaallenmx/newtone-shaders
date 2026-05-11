import * as ort from "onnxruntime-web";
import { preprocess } from "./preprocess";
import { postprocess } from "./postprocess";
import type {
    DetectInput,
    DetectOptions,
    DetectedRegion,
    LoadDetectorOptions,
    NsfwDetector,
} from "./types";

const DEFAULT_INPUT_SIZE = 320;
const DEFAULT_PROVIDERS: ReadonlyArray<"wasm"> = ["wasm"];
const DEFAULT_SCORE = 0.2;
const DEFAULT_IOU = 0.45;

let wasmConfigured = false;

function ensureWasmPaths(override: string | undefined): void {
    if (override) {
        ort.env.wasm.wasmPaths = override;
        wasmConfigured = true;
        return;
    }
    if (wasmConfigured || ort.env.wasm.wasmPaths) return;
    // Pin to the installed package version so the CDN bundle matches the JS
    // glue. Bumping `onnxruntime-web` requires updating this constant too.
    ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/";
    wasmConfigured = true;
}

export async function loadDetector(
    options: LoadDetectorOptions,
): Promise<NsfwDetector> {
    ensureWasmPaths(options.wasmPaths);

    const inputSize = options.inputSize ?? DEFAULT_INPUT_SIZE;
    const providers = options.executionProviders ?? DEFAULT_PROVIDERS;

    const session = await ort.InferenceSession.create(options.modelUrl, {
        executionProviders: providers as string[],
    });
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    let disposed = false;

    return {
        async detect(
            input: DetectInput,
            detectOptions: DetectOptions = {},
        ): Promise<DetectedRegion[]> {
            if (disposed) throw new Error("detector disposed");
            const pre = preprocess(input, inputSize);
            const tensor = new ort.Tensor("float32", pre.tensor, [
                1,
                3,
                inputSize,
                inputSize,
            ]);
            const outputs = await session.run({ [inputName]: tensor });
            const out = outputs[outputName];
            if (!out) {
                throw new Error(
                    `detector: missing output "${outputName}" in session result`,
                );
            }
            const data = out.data as Float32Array;
            return postprocess(
                data,
                out.dims,
                {
                    inputSize,
                    origWidth: pre.origWidth,
                    origHeight: pre.origHeight,
                    maxSize: pre.maxSize,
                },
                {
                    scoreThreshold:
                        detectOptions.scoreThreshold ?? DEFAULT_SCORE,
                    iouThreshold: detectOptions.iouThreshold ?? DEFAULT_IOU,
                    classes: detectOptions.classes,
                },
            );
        },
        dispose() {
            disposed = true;
            void session.release();
        },
    };
}
