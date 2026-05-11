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
        // Loop video assets fine without HEAD/Range gymnastics.
        fs: { strict: false },
    },
    // Both packages pull WASM/ORT glue dynamically; pre-bundling breaks
    // the runtime detection of which binary to fetch.
    optimizeDeps: {
        exclude: ["onnxruntime-web", "@huggingface/transformers"],
    },
});
