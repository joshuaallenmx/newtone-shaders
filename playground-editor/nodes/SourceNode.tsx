import type { CSSProperties } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { findAssetByName } from "../assets";
import {
    NODE_FRAME_SELECTED_STYLE,
    NODE_FRAME_STYLE,
    NODE_HEADER_STYLE,
} from "../styles";

export type SourceNodeData = { assetName?: string };
export type SourceNodeType = Node<SourceNodeData, "source">;

const THUMB_WRAP_STYLE: CSSProperties = {
    width: 140,
    height: 80,
    background: "#0a0a0a",
    borderTop: "1px solid #2a2a2a",
    borderBottom: "1px solid #2a2a2a",
    overflow: "hidden",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
};

const THUMB_IMG_STYLE: CSSProperties = {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block",
    pointerEvents: "none",
};

const FILENAME_STYLE: CSSProperties = {
    padding: "4px 10px 6px",
    color: "#888",
    fontSize: 10,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
};

const EMPTY_THUMB_STYLE: CSSProperties = {
    color: "#555",
    fontSize: 10,
};

export function SourceNode({ data, selected }: NodeProps<SourceNodeType>) {
    const asset = findAssetByName(data.assetName);
    return (
        <div style={selected ? NODE_FRAME_SELECTED_STYLE : NODE_FRAME_STYLE}>
            <div style={NODE_HEADER_STYLE}>Source</div>
            <div style={THUMB_WRAP_STYLE}>
                {asset ? (
                    <img
                        src={asset.url}
                        style={THUMB_IMG_STYLE}
                        alt=""
                        draggable={false}
                    />
                ) : (
                    <div style={EMPTY_THUMB_STYLE}>drag a file here</div>
                )}
            </div>
            <div style={FILENAME_STYLE} title={asset?.name ?? ""}>
                {asset?.name ?? "(no file)"}
            </div>
            <Handle type="source" position={Position.Right} />
        </div>
    );
}
