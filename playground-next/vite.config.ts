import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            "@newtonedev/shaders": path.resolve(__dirname, "../src/index.ts"),
        },
    },
    server: {
        // Assets live in ../playground/assets — let Vite serve outside root.
        fs: { strict: false },
        // Vite only watches files inside its root by default. Without this,
        // adding/removing files in ../playground/assets won't invalidate
        // the App.tsx module (which globs that dir at compile time), and
        // a page refresh just re-runs the stale cached result.
        watch: {
            ignored: ["!**/playground/assets/**"],
        },
    },
    // These packages load wasm/onnx glue dynamically; pre-bundling breaks
    // their runtime detection of which binary to fetch.
    optimizeDeps: {
        exclude: ["onnxruntime-web", "@huggingface/transformers"],
    },
});
