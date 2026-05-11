import type { CSSProperties, DragEvent } from "react";
import { SHADERS } from "../shaders";
import {
    LIST_ITEM_STYLE,
    PANEL_BODY_STYLE,
    PANEL_HEADER_STYLE,
    PANEL_STYLE,
} from "../styles";

export const DRAG_MIME = "application/x-newtone-node";

export type DragPayload =
    | { kind: "output" }
    | { kind: "globalInput" }
    | { kind: "shader"; shaderId: string }
    | { kind: "source-asset"; assetName: string };

const SECTION_HEADER_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 10,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    marginTop: 4,
};

function startDrag(e: DragEvent<HTMLDivElement>, payload: DragPayload) {
    e.dataTransfer.setData(DRAG_MIME, JSON.stringify(payload));
    e.dataTransfer.effectAllowed = "move";
}

/** Content-only — no PANEL_STYLE/HEADER wrapper. Used by both the
 *  standalone NodeLibraryPanel and the tabbed LibraryPanel. */
export function NodeLibraryPanelBody() {
    return (
        <>
            <div style={SECTION_HEADER_STYLE}>I/O</div>
            <div
                style={LIST_ITEM_STYLE}
                draggable
                onDragStart={(e) => startDrag(e, { kind: "output" })}
            >
                Output
            </div>
            <div
                style={LIST_ITEM_STYLE}
                draggable
                onDragStart={(e) => startDrag(e, { kind: "globalInput" })}
                title="The single primary input — sets the working aspect ratio that every downstream node adopts"
            >
                Global Input
            </div>
            <div style={SECTION_HEADER_STYLE}>Shaders</div>
            {SHADERS.map((s) => (
                <div
                    key={s.id}
                    style={LIST_ITEM_STYLE}
                    draggable
                    onDragStart={(e) =>
                        startDrag(e, { kind: "shader", shaderId: s.id })
                    }
                    title={s.id}
                >
                    {s.name}
                </div>
            ))}
        </>
    );
}

export function NodeLibraryPanel() {
    return (
        <div style={PANEL_STYLE}>
            <div style={PANEL_HEADER_STYLE}>Node Library</div>
            <div style={PANEL_BODY_STYLE}>
                <NodeLibraryPanelBody />
            </div>
        </div>
    );
}
