import {
    useCallback,
    type DragEvent,
    type MouseEvent as ReactMouseEvent,
} from "react";
import {
    Background,
    Controls,
    MiniMap,
    ReactFlow,
    ReactFlowProvider,
    useReactFlow,
    type Connection,
    type Edge,
    type EdgeChange,
    type Node,
    type NodeChange,
    type XYPosition,
} from "@xyflow/react";
import { NODE_TYPES } from "../nodes/nodeTypes";
import { DRAG_MIME, type DragPayload } from "./NodeLibraryPanel";
import {
    PANEL_BODY_STYLE,
    PANEL_HEADER_STYLE,
    PANEL_STYLE,
} from "../styles";

interface NodeGraphPanelProps {
    readonly nodes: Node[];
    readonly edges: Edge[];
    readonly onNodesChange: (changes: NodeChange[]) => void;
    readonly onEdgesChange: (changes: EdgeChange[]) => void;
    readonly onConnect: (connection: Connection) => void;
    readonly onAddNode: (payload: DragPayload, position: XYPosition) => void;
}

export function NodeGraphPanel(props: NodeGraphPanelProps) {
    // `position: relative` on the body anchors React Flow's measured
    // container so its ResizeObserver always sees pixel dimensions,
    // even on the first StrictMode mount where flexbox sizing hasn't
    // settled yet (otherwise React Flow logs the "needs width/height"
    // warning every reload).
    return (
        <div style={PANEL_STYLE}>
            <div style={PANEL_HEADER_STYLE}>Node Graph</div>
            <div
                style={{ ...PANEL_BODY_STYLE, padding: 0, position: "relative" }}
            >
                <ReactFlowProvider>
                    <Inner {...props} />
                </ReactFlowProvider>
            </div>
        </div>
    );
}

function Inner({
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    onAddNode,
}: NodeGraphPanelProps) {
    const rf = useReactFlow();

    const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
    }, []);

    const onDrop = useCallback(
        (e: DragEvent<HTMLDivElement>) => {
            e.preventDefault();
            const raw = e.dataTransfer.getData(DRAG_MIME);
            if (!raw) return;
            let payload: DragPayload;
            try {
                payload = JSON.parse(raw) as DragPayload;
            } catch {
                return;
            }
            const position = rf.screenToFlowPosition({
                x: e.clientX,
                y: e.clientY,
            });
            onAddNode(payload, position);
        },
        [rf, onAddNode],
    );

    const onEdgeDoubleClick = useCallback(
        (_e: ReactMouseEvent, edge: Edge) => {
            onEdgesChange([{ type: "remove", id: edge.id }]);
        },
        [onEdgesChange],
    );

    return (
        <div
            style={{ position: "absolute", inset: 0 }}
            onDragOver={onDragOver}
            onDrop={onDrop}
        >
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onEdgeDoubleClick={onEdgeDoubleClick}
                nodeTypes={NODE_TYPES}
                fitView
                colorMode="dark"
                proOptions={{ hideAttribution: true }}
                minZoom={0.05}
                maxZoom={4}
            >
                <Background />
                <Controls />
                <MiniMap pannable zoomable />
            </ReactFlow>
        </div>
    );
}
