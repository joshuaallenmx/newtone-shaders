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
    optimizeDeps: {
        exclude: ["onnxruntime-web", "@huggingface/transformers"],
    },
});
