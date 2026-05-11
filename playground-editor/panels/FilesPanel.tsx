import type { CSSProperties, DragEvent } from "react";
import { ASSETS } from "../assets";
import {
    EMPTY_STYLE,
    PANEL_BODY_STYLE,
    PANEL_HEADER_STYLE,
    PANEL_STYLE,
} from "../styles";
import { DRAG_MIME, type DragPayload } from "./NodeLibraryPanel";

const GRID_STYLE: CSSProperties = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))",
    gap: 6,
    padding: 0,
};

const TILE_STYLE: CSSProperties = {
    background: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: 4,
    padding: 4,
    display: "flex",
    flexDirection: "column",
    gap: 4,
    cursor: "grab",
    overflow: "hidden",
    userSelect: "none",
};

const THUMB_STYLE: CSSProperties = {
    width: "100%",
    aspectRatio: "1 / 1",
    objectFit: "cover",
    background: "#0a0a0a",
    borderRadius: 2,
    display: "block",
    pointerEvents: "none",
};

const NAME_STYLE: CSSProperties = {
    fontSize: 10,
    color: "#bdbdbd",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
};

function startDrag(e: DragEvent<HTMLDivElement>, payload: DragPayload) {
    e.dataTransfer.setData(DRAG_MIME, JSON.stringify(payload));
    e.dataTransfer.effectAllowed = "move";
}

/** Content-only — no PANEL_STYLE/HEADER wrapper. Used by both the
 *  standalone FilesPanel and the tabbed LibraryPanel. */
export function FilesPanelBody() {
    if (ASSETS.length === 0) {
        return (
            <div style={EMPTY_STYLE}>
                No assets in ../playground/assets/
            </div>
        );
    }
    return (
        <div style={GRID_STYLE}>
            {ASSETS.map((a) => (
                <div
                    key={a.url}
                    style={TILE_STYLE}
                    draggable
                    onDragStart={(e) =>
                        startDrag(e, {
                            kind: "source-asset",
                            assetName: a.name,
                        })
                    }
                    title={a.name}
                >
                    <img
                        src={a.url}
                        style={THUMB_STYLE}
                        alt=""
                        draggable={false}
                    />
                    <div style={NAME_STYLE}>{a.name}</div>
                </div>
            ))}
        </div>
    );
}

export function FilesPanel() {
    return (
        <div style={PANEL_STYLE}>
            <div style={PANEL_HEADER_STYLE}>Files ({ASSETS.length})</div>
            <div style={{ ...PANEL_BODY_STYLE, padding: 8 }}>
                <FilesPanelBody />
            </div>
        </div>
    );
}

export const FILE_COUNT = ASSETS.length;
