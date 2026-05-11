import { useMemo, useRef, type CSSProperties } from "react";
import {
    Handle,
    Position,
    useEdges,
    useNodes,
    type NodeProps,
} from "@xyflow/react";
import { resolveSlotUpstream } from "../compileChain";
import { useNodeSnapshot, usePreviewSize } from "../PipelineHandleContext";
import {
    NODE_FRAME_SELECTED_STYLE,
    NODE_FRAME_STYLE,
    NODE_HEADER_STYLE,
} from "../styles";

const INPUT_LIST_STYLE: CSSProperties = {
    padding: "6px 0",
    color: "#bdbdbd",
    fontSize: 11,
    display: "flex",
    flexDirection: "column",
};

const INPUT_ROW_STYLE: CSSProperties = {
    position: "relative",
    paddingLeft: 14,
    paddingRight: 10,
    paddingTop: 4,
    paddingBottom: 4,
    minHeight: 18,
    display: "flex",
    alignItems: "center",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
};

const PREVIEW_WRAP_STYLE: CSSProperties = {
    position: "relative",
    background: "#0a0a0a",
    borderTop: "1px solid #2a2a2a",
    borderBottom: "1px solid #2a2a2a",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
};

const PREVIEW_CANVAS_STYLE: CSSProperties = {
    display: "block",
    imageRendering: "pixelated",
};

export function GlobalInputNode({ id, selected }: NodeProps) {
    // The Global Input is a passthrough that fans out to every consumer
    // and dictates the pipeline's working aspect ratio. It collapses out
    // of the render plan (no outputTex of its own), so we snapshot
    // whatever it's wired to — the node feels like a peephole into the
    // active source.
    const nodes = useNodes();
    const edges = useEdges();
    const upstreamId = useMemo(
        () => resolveSlotUpstream(id, nodes, edges),
        [id, nodes, edges],
    );
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { w: previewW, h: previewH } = usePreviewSize();
    useNodeSnapshot(upstreamId, canvasRef);

    return (
        <div style={selected ? NODE_FRAME_SELECTED_STYLE : NODE_FRAME_STYLE}>
            <div style={NODE_HEADER_STYLE}>Global Input</div>
            <div style={PREVIEW_WRAP_STYLE}>
                <canvas
                    ref={canvasRef}
                    width={previewW}
                    height={previewH}
                    style={{
                        ...PREVIEW_CANVAS_STYLE,
                        width: previewW,
                        height: previewH,
                    }}
                />
                <Handle type="source" position={Position.Right} />
            </div>
            <div style={INPUT_LIST_STYLE}>
                <div style={INPUT_ROW_STYLE}>
                    <Handle type="target" position={Position.Left} />
                    source
                </div>
            </div>
        </div>
    );
}

// Legacy alias so existing imports keep resolving while we migrate.
export const SlotNode = GlobalInputNode;
