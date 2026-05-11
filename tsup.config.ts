import { defineConfig } from "tsup";

export default defineConfig({
    entry: ["src/index.ts"],
    format: ["cjs", "esm"],
    dts: false,
    clean: true,
    sourcemap: true,
    splitting: false,
    treeshake: true,
    external: ["react", "react-dom", "three"],
    outExtension({ format }) {
        return {
            js: format === "cjs" ? ".cjs" : ".js",
        };
    },
    esbuildOptions(options) {
        options.jsx = "automatic";
    },
});
