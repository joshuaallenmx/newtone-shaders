import type { Edge, Node } from "@xyflow/react";

export interface SerializedState {
    readonly version: 1;
    readonly nodes: readonly SerializedNode[];
    readonly edges: readonly Edge[];
    readonly paramsByNode: Record<string, unknown>;
}

export interface SerializedNode {
    readonly id: string;
    readonly type: string;
    readonly position: { x: number; y: number };
    readonly data: unknown;
}

export interface AppState {
    readonly nodes: readonly Node[];
    readonly edges: readonly Edge[];
    readonly paramsByNode: Record<string, unknown>;
}

export interface Project {
    readonly id: string;
    name: string;
    readonly createdAt: number;
    updatedAt: number;
    state: SerializedState;
}

export interface ProjectLibrary {
    readonly version: 1;
    projects: Project[];
    activeId: string;
}

const LIBRARY_KEY = "newtone-editor-library-v1";
const LEGACY_STATE_KEY = "newtone-editor-state-v1";

export function serialize(state: AppState): SerializedState {
    return {
        version: 1,
        nodes: state.nodes.map((n) => ({
            id: n.id,
            type: n.type ?? "default",
            position: n.position,
            data: n.data,
        })),
        edges: state.edges.map((e) => ({ ...e })),
        paramsByNode: { ...state.paramsByNode },
    };
}

export function deserialize(raw: unknown): RestoredProjectState {
    if (!raw || typeof raw !== "object") {
        throw new Error("invalid file: not a JSON object");
    }
    const obj = raw as Partial<SerializedState>;
    if (obj.version !== 1) {
        throw new Error(`unsupported version: ${obj.version}`);
    }
    if (!Array.isArray(obj.nodes) || !Array.isArray(obj.edges)) {
        throw new Error("invalid file: nodes/edges missing");
    }

    let nodes: Node[] = obj.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: (n.data ?? {}) as Record<string, unknown>,
    }));

    // Migrate legacy `slot` nodes → `globalInput`. The Global Input is a
    // singleton (one per chain dictates the working aspect), so we
    // promote the *first* passthrough we encounter and drop the rest.
    // Edges into dropped nodes are also removed so the graph doesn't
    // dangle. Console-warn for visibility — the typical case is a
    // single Slot, which converts cleanly.
    let edges: Edge[] = obj.edges.map((e) => ({ ...e }));
    const passthroughs = nodes.filter(
        (n) => n.type === "slot" || n.type === "globalInput",
    );
    if (passthroughs.length > 0) {
        const keep = passthroughs[0]!;
        const dropIds = new Set(
            passthroughs.slice(1).map((n) => n.id),
        );
        if (passthroughs.length > 1) {
            console.warn(
                `[editor] migrating saved graph: keeping ${keep.id} as ` +
                    `Global Input, dropping ${passthroughs.length - 1} ` +
                    `extra slot/globalInput node(s).`,
            );
        }
        nodes = nodes
            .filter((n) => !dropIds.has(n.id))
            .map((n) =>
                n.id === keep.id ? { ...n, type: "globalInput" } : n,
            );
        if (dropIds.size > 0) {
            edges = edges.filter(
                (e) => !dropIds.has(e.source) && !dropIds.has(e.target),
            );
        }
    }

    const paramsByNode: Record<string, unknown> =
        obj.paramsByNode && typeof obj.paramsByNode === "object"
            ? { ...obj.paramsByNode }
            : {};

    let maxSuffix = 0;
    for (const n of nodes) {
        const m = /-(\d+)$/.exec(n.id);
        if (m) maxSuffix = Math.max(maxSuffix, Number(m[1]));
    }

    return { nodes, edges, paramsByNode, maxNumericIdSuffix: maxSuffix };
}

export interface RestoredProjectState {
    readonly nodes: Node[];
    readonly edges: Edge[];
    readonly paramsByNode: Record<string, unknown>;
    readonly maxNumericIdSuffix: number;
}

function newProjectId(): string {
    return `p-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function makeProject(
    name: string,
    state: SerializedState,
): Project {
    const now = Date.now();
    return {
        id: newProjectId(),
        name,
        createdAt: now,
        updatedAt: now,
        state,
    };
}

export function makeEmptyState(): SerializedState {
    return {
        version: 1,
        nodes: [],
        edges: [],
        paramsByNode: {},
    };
}

export function loadProjectLibrary(): ProjectLibrary | null {
    if (typeof localStorage === "undefined") return null;
    try {
        const raw = localStorage.getItem(LIBRARY_KEY);
        if (raw) {
            const parsed = JSON.parse(raw) as ProjectLibrary;
            if (
                parsed?.version === 1 &&
                Array.isArray(parsed.projects) &&
                typeof parsed.activeId === "string" &&
                parsed.projects.some((p) => p.id === parsed.activeId)
            ) {
                return parsed;
            }
        }
        const legacy = localStorage.getItem(LEGACY_STATE_KEY);
        if (legacy) {
            const state = JSON.parse(legacy) as SerializedState;
            if (state?.version === 1) {
                const project = makeProject("Untitled", state);
                const library: ProjectLibrary = {
                    version: 1,
                    projects: [project],
                    activeId: project.id,
                };
                saveProjectLibrary(library);
                localStorage.removeItem(LEGACY_STATE_KEY);
                return library;
            }
        }
    } catch {
        // fall through
    }
    return null;
}

export function saveProjectLibrary(library: ProjectLibrary): void {
    if (typeof localStorage !== "undefined") {
        try {
            localStorage.setItem(LIBRARY_KEY, JSON.stringify(library));
        } catch {
            // quota exceeded / private mode — silently ignore
        }
    }
    // Mirror to disk via the dev-mode vite plugin. Fire-and-forget:
    // the localStorage save is the source of truth in the browser,
    // disk is a backup that survives cleared cookies / private mode /
    // a different browser. In production the endpoint just doesn't
    // exist and the fetch silently fails — localStorage carries on.
    void saveProjectLibraryToDisk(library);
}

const DISK_API_URL = "/api/library";

/** Async POST of the entire library to the dev-mode vite endpoint.
 *  Resolves on success, swallows network / 4xx / 5xx so a missing
 *  dev plugin (production build, alternative dev server) doesn't
 *  break the editor's localStorage save path. */
export async function saveProjectLibraryToDisk(
    library: ProjectLibrary,
): Promise<void> {
    if (typeof fetch === "undefined") return;
    try {
        await fetch(DISK_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(library),
            keepalive: true,
        });
    } catch {
        // dev plugin unavailable — that's fine
    }
}

/** Async fetch of the on-disk library. Returns null when the file
 *  doesn't exist, the dev endpoint isn't mounted, or the body fails
 *  shape validation. Used by the editor's bootstrap to prefer the
 *  on-disk copy over localStorage when both are present (so a fresh
 *  browser / cleared cookies still loads your project history). */
export async function loadProjectLibraryFromDisk(): Promise<ProjectLibrary | null> {
    if (typeof fetch === "undefined") return null;
    try {
        const res = await fetch(DISK_API_URL, { cache: "no-store" });
        if (!res.ok) return null;
        const parsed = (await res.json()) as ProjectLibrary;
        if (
            parsed?.version === 1 &&
            Array.isArray(parsed.projects) &&
            typeof parsed.activeId === "string" &&
            parsed.projects.some((p) => p.id === parsed.activeId)
        ) {
            return parsed;
        }
    } catch {
        // network error / not in dev mode
    }
    return null;
}
