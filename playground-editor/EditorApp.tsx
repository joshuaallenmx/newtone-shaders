import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ReactNode,
} from "react";
import {
    addEdge,
    applyEdgeChanges,
    applyNodeChanges,
    type Connection,
    type Edge,
    type EdgeChange,
    type Node,
    type NodeChange,
    type XYPosition,
} from "@xyflow/react";
import { ASSETS } from "./assets";
import { SHADERS } from "./shaders";
import { EditorShell } from "./EditorShell";
import { LibraryPanel } from "./panels/LibraryPanel";
import { PreviewPanel } from "./panels/PreviewPanel";
import { InspectorPanel } from "./panels/InspectorPanel";
import { type DragPayload } from "./panels/NodeLibraryPanel";
import { NodeGraphPanel } from "./panels/NodeGraphPanel";
import { Topbar, type SaveStatus } from "./Topbar";
import {
    deserialize,
    loadProjectLibrary,
    loadProjectLibraryFromDisk,
    makeProject,
    saveProjectLibrary,
    serialize,
    type ProjectLibrary,
    type SerializedState,
} from "./serialize";
import { PipelineHandleProvider } from "./PipelineHandleContext";

const SAVE_DEBOUNCE_MS = 400;

function defaultInitialState(): SerializedState {
    return serialize({
        nodes: [
            {
                id: "source-1",
                type: "source",
                position: { x: 80, y: 120 },
                data: { assetName: ASSETS[0]?.name },
            },
            {
                id: "output-1",
                type: "output",
                position: { x: 480, y: 120 },
                data: {},
            },
        ],
        edges: [],
        paramsByNode: {},
    });
}

function latestUpdatedAt(library: ProjectLibrary): number {
    let max = 0;
    for (const p of library.projects) {
        if (p.updatedAt > max) max = p.updatedAt;
    }
    return max;
}

function bootstrapLibrary(): ProjectLibrary {
    const existing = loadProjectLibrary();
    if (existing) return existing;
    const project = makeProject("Untitled", defaultInitialState());
    const library: ProjectLibrary = {
        version: 1,
        projects: [project],
        activeId: project.id,
    };
    saveProjectLibrary(library);
    return library;
}

interface WorkspaceContextValue {
    readonly nodes: Node[];
    readonly edges: Edge[];
    readonly paramsByNode: Record<string, unknown>;
    readonly onNodesChange: (changes: NodeChange[]) => void;
    readonly onEdgesChange: (changes: EdgeChange[]) => void;
    readonly onConnect: (connection: Connection) => void;
    readonly onAddNode: (payload: DragPayload, position: XYPosition) => void;
    readonly onParamsChange: (nodeId: string, params: unknown) => void;
}

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

function useWorkspace(): WorkspaceContextValue {
    const ctx = useContext(WorkspaceContext);
    if (!ctx) throw new Error("WorkspaceContext missing");
    return ctx;
}

interface WorkspaceProviderProps {
    readonly initialState: SerializedState;
    readonly onChange: (state: SerializedState) => void;
    readonly onSaved: (at: number) => void;
    readonly children: ReactNode;
}

function WorkspaceProvider({
    initialState,
    onChange,
    onSaved,
    children,
}: WorkspaceProviderProps) {
    const restored = useMemo(() => deserialize(initialState), [initialState]);

    const [nodes, setNodes] = useState<Node[]>(restored.nodes);
    const [edges, setEdges] = useState<Edge[]>(restored.edges);
    const [paramsByNode, setParamsByNode] = useState<Record<string, unknown>>(
        restored.paramsByNode,
    );
    // Refs let stable callbacks (onConnect) read the latest nodes
    // without re-creating themselves on every nodes change.
    const nodesRef = useRef(nodes);
    nodesRef.current = nodes;
    const counterRef = useRef(Math.max(1, restored.maxNumericIdSuffix));
    const nextId = useCallback(
        (kind: string) => `${kind}-${++counterRef.current}`,
        [],
    );

    const isFirstRender = useRef(true);
    useEffect(() => {
        if (isFirstRender.current) {
            isFirstRender.current = false;
            return;
        }
        const state = serialize({ nodes, edges, paramsByNode });
        onChange(state);
        const id = window.setTimeout(() => {
            onSaved(Date.now());
        }, SAVE_DEBOUNCE_MS);
        return () => window.clearTimeout(id);
    }, [nodes, edges, paramsByNode, onChange, onSaved]);

    const onNodesChange = useCallback((changes: NodeChange[]) => {
        setNodes((ns) => applyNodeChanges(changes, ns));
        const removed = changes.filter((c) => c.type === "remove");
        if (removed.length) {
            setParamsByNode((prev) => {
                const next = { ...prev };
                for (const c of removed) {
                    if (c.type === "remove") delete next[c.id];
                }
                return next;
            });
        }
    }, []);

    const onEdgesChange = useCallback((changes: EdgeChange[]) => {
        setEdges((es) => applyEdgeChanges(changes, es));
    }, []);

    const onConnect = useCallback((c: Connection) => {
        // Variadic shaders (Layers) accept many edges into a single
        // handle. For those targets we dedup by (target, source) so the
        // user can't wire the same upstream twice but can stack
        // anything else; for normal targets we keep the existing
        // (target, targetHandle) dedup so a fresh wire replaces the
        // previous one in that slot.
        const target = nodesRef.current.find((n) => n.id === c.target);
        const shaderId =
            target?.type === "shader"
                ? (target.data as { shaderId?: string } | undefined)?.shaderId
                : undefined;
        const isVariadic = !!SHADERS.find(
            (s) => s.id === shaderId && s.variadic,
        );
        setEdges((es) =>
            addEdge(
                c,
                es.filter((e) =>
                    isVariadic
                        ? !(
                            e.target === c.target &&
                            e.source === c.source
                        )
                        : !(
                            e.target === c.target &&
                            (e.targetHandle ?? null) ===
                                (c.targetHandle ?? null)
                        ),
                ),
            ),
        );
    }, []);

    const onAddNode = useCallback(
        (payload: DragPayload, position: XYPosition) => {
            if (payload.kind === "source-asset") {
                setNodes((ns) => [
                    ...ns,
                    {
                        id: nextId("source"),
                        type: "source",
                        position,
                        data: { assetName: payload.assetName },
                    },
                ]);
                return;
            }
            if (payload.kind === "output") {
                setNodes((ns) => [
                    ...ns,
                    { id: nextId("output"), type: "output", position, data: {} },
                ]);
                return;
            }
            if (payload.kind === "globalInput") {
                setNodes((ns) => {
                    // Singleton: refuse to add a second Global Input.
                    // The library item stays draggable but the drop is a
                    // no-op when one already exists. Easier to spot than
                    // a silent failure — at least a console warning.
                    if (ns.some((n) => n.type === "globalInput")) {
                        console.warn(
                            "[editor] Global Input already exists; only one is allowed.",
                        );
                        return ns;
                    }
                    return [
                        ...ns,
                        {
                            id: nextId("globalInput"),
                            type: "globalInput",
                            position,
                            data: {},
                        },
                    ];
                });
                return;
            }
            const entry = SHADERS.find((s) => s.id === payload.shaderId);
            if (!entry) return;
            const id = nextId("shader");
            setNodes((ns) => [
                ...ns,
                {
                    id,
                    type: "shader",
                    position,
                    data: { shaderId: entry.id },
                },
            ]);
            setParamsByNode((prev) => ({ ...prev, [id]: entry.defaultParams }));
        },
        [nextId],
    );

    const onParamsChange = useCallback((nodeId: string, params: unknown) => {
        setParamsByNode((prev) => ({ ...prev, [nodeId]: params }));
    }, []);

    const value = useMemo<WorkspaceContextValue>(
        () => ({
            nodes,
            edges,
            paramsByNode,
            onNodesChange,
            onEdgesChange,
            onConnect,
            onAddNode,
            onParamsChange,
        }),
        [
            nodes,
            edges,
            paramsByNode,
            onNodesChange,
            onEdgesChange,
            onConnect,
            onAddNode,
            onParamsChange,
        ],
    );

    return (
        <WorkspaceContext.Provider value={value}>
            {children}
        </WorkspaceContext.Provider>
    );
}

function WorkspacePreviewSlot() {
    const { nodes, edges, paramsByNode } = useWorkspace();
    return (
        <PreviewPanel
            nodes={nodes}
            edges={edges}
            paramsByNode={paramsByNode}
        />
    );
}

function WorkspaceInspectorSlot() {
    const { nodes, edges, paramsByNode, onParamsChange } = useWorkspace();
    return (
        <InspectorPanel
            nodes={nodes}
            edges={edges}
            paramsByNode={paramsByNode}
            onParamsChange={onParamsChange}
        />
    );
}

function WorkspaceGraphSlot() {
    const {
        nodes,
        edges,
        onNodesChange,
        onEdgesChange,
        onConnect,
        onAddNode,
    } = useWorkspace();
    return (
        <NodeGraphPanel
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onAddNode={onAddNode}
        />
    );
}

export function EditorApp() {
    const [library, setLibrary] = useState<ProjectLibrary>(bootstrapLibrary);
    const [status, setStatus] = useState<SaveStatus>("saved");
    const [lastSavedAt, setLastSavedAt] = useState<number | null>(() => {
        const active = library.projects.find((p) => p.id === library.activeId);
        return active?.updatedAt ?? null;
    });

    // On dev startup: try to read the on-disk copy and prefer whichever
    // (disk vs. localStorage) was updated most recently. If there is
    // no disk file yet, push the current localStorage state up so the
    // file gets created — that's the entry point that prevents losing
    // work to a cleared cookies / private mode reload.
    useEffect(() => {
        let cancelled = false;
        void (async () => {
            const disk = await loadProjectLibraryFromDisk();
            if (cancelled) return;
            if (disk) {
                const diskLatest = latestUpdatedAt(disk);
                const localLatest = latestUpdatedAt(library);
                if (diskLatest > localLatest) {
                    setLibrary(disk);
                    saveProjectLibrary(disk); // re-mirror to localStorage
                } else if (diskLatest < localLatest) {
                    saveProjectLibrary(library); // local newer → push to disk
                }
                // equal → nothing to do
            } else {
                // No disk file yet — write the current state up so the
                // dev plugin's projects.json starts populated.
                saveProjectLibrary(library);
            }
        })();
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const activeProject =
        library.projects.find((p) => p.id === library.activeId) ??
        library.projects[0];

    const updateLibrary = useCallback(
        (updater: (prev: ProjectLibrary) => ProjectLibrary) => {
            setLibrary((prev) => {
                const next = updater(prev);
                saveProjectLibrary(next);
                return next;
            });
        },
        [],
    );

    const handleSelect = useCallback(
        (id: string) => {
            const next = library.projects.find((p) => p.id === id);
            updateLibrary((prev) => ({ ...prev, activeId: id }));
            setLastSavedAt(next?.updatedAt ?? null);
            setStatus("saved");
        },
        [library.projects, updateLibrary],
    );

    const handleRename = useCallback(
        (id: string, name: string) => {
            updateLibrary((prev) => ({
                ...prev,
                projects: prev.projects.map((p) =>
                    p.id === id ? { ...p, name, updatedAt: Date.now() } : p,
                ),
            }));
        },
        [updateLibrary],
    );

    const handleCreate = useCallback(() => {
        const project = makeProject("Untitled", defaultInitialState());
        updateLibrary((prev) => ({
            ...prev,
            projects: [...prev.projects, project],
            activeId: project.id,
        }));
        setLastSavedAt(project.updatedAt);
        setStatus("saved");
    }, [updateLibrary]);

    const handleDuplicate = useCallback(
        (id: string) => {
            const source = library.projects.find((p) => p.id === id);
            if (!source) return;
            const copy = makeProject(`${source.name} copy`, source.state);
            updateLibrary((prev) => ({
                ...prev,
                projects: [...prev.projects, copy],
                activeId: copy.id,
            }));
            setLastSavedAt(copy.updatedAt);
            setStatus("saved");
        },
        [library.projects, updateLibrary],
    );

    const handleDelete = useCallback(
        (id: string) => {
            updateLibrary((prev) => {
                if (prev.projects.length <= 1) return prev;
                const remaining = prev.projects.filter((p) => p.id !== id);
                const nextActive =
                    prev.activeId === id ? remaining[0].id : prev.activeId;
                return {
                    ...prev,
                    projects: remaining,
                    activeId: nextActive,
                };
            });
        },
        [updateLibrary],
    );

    const handleExport = useCallback(
        (id: string) => {
            const project = library.projects.find((p) => p.id === id);
            if (!project) return;
            const blob = new Blob([JSON.stringify(project.state, null, 2)], {
                type: "application/json",
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            const safeName = project.name
                .replace(/[^a-z0-9_-]+/gi, "-")
                .replace(/^-+|-+$/g, "");
            const stamp = new Date().toISOString().replace(/[:.]/g, "-");
            a.href = url;
            a.download = `${safeName || "project"}-${stamp}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        },
        [library.projects],
    );

    const handleImport = useCallback(
        async (file: File) => {
            try {
                const text = await file.text();
                const parsed = JSON.parse(text) as SerializedState;
                deserialize(parsed); // validates shape
                const baseName = file.name.replace(/\.json$/i, "");
                const project = makeProject(baseName || "Imported", parsed);
                updateLibrary((prev) => ({
                    ...prev,
                    projects: [...prev.projects, project],
                    activeId: project.id,
                }));
                setLastSavedAt(project.updatedAt);
                setStatus("saved");
            } catch (err) {
                alert(
                    `Failed to import project: ${err instanceof Error ? err.message : String(err)}`,
                );
            }
        },
        [updateLibrary],
    );

    const handleWorkspaceChange = useCallback(
        (state: SerializedState) => {
            setStatus("unsaved");
            updateLibrary((prev) => ({
                ...prev,
                projects: prev.projects.map((p) =>
                    p.id === prev.activeId
                        ? { ...p, state, updatedAt: Date.now() }
                        : p,
                ),
            }));
        },
        [updateLibrary],
    );

    const handleWorkspaceSaved = useCallback((at: number) => {
        setStatus("saved");
        setLastSavedAt(at);
    }, []);

    return (
        <PipelineHandleProvider>
            <WorkspaceProvider
                key={activeProject.id}
                initialState={activeProject.state}
                onChange={handleWorkspaceChange}
                onSaved={handleWorkspaceSaved}
            >
                <EditorShell
                    topbar={
                        <Topbar
                            projects={library.projects}
                            activeId={library.activeId}
                            status={status}
                            lastSavedAt={lastSavedAt}
                            onSelect={handleSelect}
                            onRename={handleRename}
                            onCreate={handleCreate}
                            onDuplicate={handleDuplicate}
                            onDelete={handleDelete}
                            onExport={handleExport}
                            onImport={handleImport}
                        />
                    }
                    preview={<WorkspacePreviewSlot />}
                    inspector={<WorkspaceInspectorSlot />}
                    library={<LibraryPanel />}
                    nodeGraph={<WorkspaceGraphSlot />}
                />
            </WorkspaceProvider>
        </PipelineHandleProvider>
    );
}
