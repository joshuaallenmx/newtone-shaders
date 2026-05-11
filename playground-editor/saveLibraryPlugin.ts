import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

// Vite dev plugin: persist the editor's project library to a single
// JSON file in the repo, alongside (and as a backup for) the
// localStorage copy the editor already keeps. Without this, refreshing
// in private mode / cleared cookies / a different browser would lose
// every saved project — the localStorage key is the only source of
// truth in the browser.
//
// Dev-only: the middleware is mounted in `configureServer`, which Vite
// only calls in `vite dev`. A production build doesn't include this
// plugin's runtime side; the editor falls back to localStorage there.
//
// Endpoints:
//   GET  /api/library  → returns the file's JSON contents (404 if
//                        the file doesn't exist yet).
//   POST /api/library  → body is the library JSON; written to disk
//                        atomically (write to .tmp, rename).
//
// File location: `playground-editor/projects.json`. Not gitignored —
// commit it if you want versioned history of your project library,
// or add it to .gitignore for a private dev cache.

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LIBRARY_PATH = path.resolve(__dirname, "projects.json");
const LIBRARY_TMP_PATH = `${LIBRARY_PATH}.tmp`;
const MAX_BODY_BYTES = 32 * 1024 * 1024; // 32MB safety cap

export function saveLibraryPlugin(): Plugin {
    return {
        name: "save-library",
        configureServer(server) {
            server.middlewares.use(
                "/api/library",
                async (req, res, next) => {
                    if (req.method === "GET") {
                        try {
                            const content = await fs.readFile(
                                LIBRARY_PATH,
                                "utf-8",
                            );
                            res.setHeader(
                                "Content-Type",
                                "application/json",
                            );
                            res.setHeader("Cache-Control", "no-store");
                            res.statusCode = 200;
                            res.end(content);
                        } catch (err) {
                            const code = (
                                err as NodeJS.ErrnoException
                            ).code;
                            if (code === "ENOENT") {
                                res.statusCode = 404;
                                res.end("not-found");
                            } else {
                                res.statusCode = 500;
                                res.end(String(err));
                            }
                        }
                        return;
                    }
                    if (req.method === "POST" || req.method === "PUT") {
                        let received = 0;
                        const chunks: Buffer[] = [];
                        let aborted = false;
                        req.on("data", (chunk: Buffer) => {
                            received += chunk.length;
                            if (received > MAX_BODY_BYTES) {
                                aborted = true;
                                res.statusCode = 413;
                                res.end("payload too large");
                                req.destroy();
                                return;
                            }
                            chunks.push(chunk);
                        });
                        req.on("end", async () => {
                            if (aborted) return;
                            const body = Buffer.concat(chunks).toString(
                                "utf-8",
                            );
                            try {
                                // Validate it's at least JSON-shaped before
                                // overwriting; better to reject garbage than
                                // corrupt the on-disk copy.
                                JSON.parse(body);
                            } catch {
                                res.statusCode = 400;
                                res.end("invalid json");
                                return;
                            }
                            try {
                                await fs.writeFile(
                                    LIBRARY_TMP_PATH,
                                    body,
                                    "utf-8",
                                );
                                await fs.rename(
                                    LIBRARY_TMP_PATH,
                                    LIBRARY_PATH,
                                );
                                res.statusCode = 204;
                                res.end();
                            } catch (err) {
                                res.statusCode = 500;
                                res.end(String(err));
                            }
                        });
                        req.on("error", () => {
                            if (!aborted) {
                                res.statusCode = 500;
                                res.end("request error");
                            }
                        });
                        return;
                    }
                    next();
                },
            );
        },
    };
}
