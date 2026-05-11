import { useRef, type CSSProperties } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { SHADERS } from "../shaders";
import { useNodeSnapshot, usePreviewSize } from "../PipelineHandleContext";
import {
    NODE_FRAME_SELECTED_STYLE,
    NODE_FRAME_STYLE,
    NODE_HEADER_STYLE,
} from "../styles";

export type ShaderNodeData = { shaderId: string };
export type ShaderNodeType = Node<ShaderNodeData, "shader">;

const INPUT_LIST_STYLE: CSSProperties = {
    padding: "6px 0",
    color: "#bdbdbd",
    fontSize: 11,
    display: "flex",
    flexDirection: "column",
};

// Each input row is `position: relative` so the React Flow Handle nested
// inside it positions itself against the row, not the whole node frame.
// `top: 50%` then lands the handle on the label's vertical centre.
// Labels truncate with ellipsis so a verbose label doesn't push the
// node wider/taller than its peers — the full text is exposed via
// the row's `title` attribute on hover.
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

// Preview wrapper is the source-handle anchor — `position: relative` so
// the Handle's `top: 50%` resolves to the centre of the thumbnail.
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

export function ShaderNode({ id, data, selected }: NodeProps<ShaderNodeType>) {
    const entry = SHADERS.find((s) => s.id === data.shaderId);
    const inputs = entry?.inputs ?? [{ id: "in", label: "in" }];
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { w: previewW, h: previewH } = usePreviewSize();
    useNodeSnapshot(id, canvasRef);

    return (
        <div style={selected ? NODE_FRAME_SELECTED_STYLE : NODE_FRAME_STYLE}>
            <div style={NODE_HEADER_STYLE}>
                {entry ? entry.name : `Unknown: ${data.shaderId}`}
            </div>
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
                {inputs.map((input) => (
                    <div
                        key={input.id}
                        style={INPUT_ROW_STYLE}
                        title={input.label ?? input.id}
                    >
                        <Handle
                            id={input.id}
                            type="target"
                            position={Position.Left}
                        />
                        {input.label}
                    </div>
                ))}
            </div>
        </div>
    );
}
