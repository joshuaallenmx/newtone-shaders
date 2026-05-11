import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { saveLibraryPlugin } from "./saveLibraryPlugin";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const ASSETS_DIR = path.resolve(__dirname, "../playground/assets");

/** Chokidar only watches inside the project root by default, and the
 *  shared playground/assets folder lives outside this package. Without
 *  this, files added to that directory don't invalidate the
 *  `import.meta.glob` result in `assets.ts` — page refreshes serve the
 *  cached module and the new files don't appear. We add the directory
 *  to the watcher and force a full reload on add/unlink so the glob
 *  re-evaluates on the next request. */
function watchAssetsPlugin(): Plugin {
    return {
        name: "watch-external-assets",
        configureServer(server) {
            server.watcher.add(ASSETS_DIR);
            const onChange = (file: string) => {
                if (!file.startsWith(ASSETS_DIR)) return;
                server.ws.send({ type: "full-reload" });
            };
            server.watcher.on("add", onChange);
            server.watcher.on("unlink", onChange);
        },
    };
}

export default defineConfig({
    plugins: [react(), watchAssetsPlugin(), saveLibraryPlugin()],
    resolve: {
        alias: {
            "@newtonedev/shaders": path.resolve(__dirname, "../src/index.ts"),
        },
    },
    // Reuse playground/public so the NudeNet ONNX file is served at
    // /nudenet/320n.onnx without copying it into a second location.
    publicDir: path.resolve(__dirname, "../playground/public"),
    server: {
        // Assets live in ../playground/assets — let Vite serve outside root.
        fs: { strict: false },
        // Vite only watches files inside its root by default. Without this,
        // adding/removing files in ../playground/assets won't invalidate
        // the asset glob, and a page refresh just re-runs the stale cached
        // result.
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
