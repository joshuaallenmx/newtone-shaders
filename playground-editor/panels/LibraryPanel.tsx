import { useState, type CSSProperties } from "react";
import { ASSETS } from "../assets";
import { SHADERS } from "../shaders";
import {
    PANEL_BODY_STYLE,
    PANEL_HEADER_STYLE,
    PANEL_STYLE,
} from "../styles";
import { FilesPanelBody } from "./FilesPanel";
import { NodeLibraryPanelBody } from "./NodeLibraryPanel";

type Tab = "files" | "nodes";

const HEADER_STYLE: CSSProperties = {
    ...PANEL_HEADER_STYLE,
    padding: 0,
    display: "flex",
    alignItems: "stretch",
};

const TAB_BUTTON_BASE: CSSProperties = {
    flex: 1,
    background: "transparent",
    color: "#888",
    border: "none",
    borderBottom: "1px solid transparent",
    padding: "8px 10px",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    cursor: "pointer",
    fontFamily: "inherit",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
};

const TAB_BUTTON_ACTIVE: CSSProperties = {
    ...TAB_BUTTON_BASE,
    color: "#f0f0f0",
    borderBottom: "1px solid #4a90e2",
    background: "rgba(74, 144, 226, 0.06)",
};

const COUNT_BADGE_STYLE: CSSProperties = {
    color: "#666",
    fontSize: 10,
    fontVariantNumeric: "tabular-nums",
    background: "rgba(255, 255, 255, 0.05)",
    border: "1px solid #2a2a2a",
    borderRadius: 8,
    padding: "0 6px",
    lineHeight: "16px",
    minWidth: 16,
    textAlign: "center",
};

export function LibraryPanel() {
    const [tab, setTab] = useState<Tab>("files");

    return (
        <div style={PANEL_STYLE}>
            <div style={HEADER_STYLE} role="tablist">
                <button
                    type="button"
                    role="tab"
                    aria-selected={tab === "files"}
                    style={tab === "files" ? TAB_BUTTON_ACTIVE : TAB_BUTTON_BASE}
                    onClick={() => setTab("files")}
                >
                    Files
                    <span style={COUNT_BADGE_STYLE}>{ASSETS.length}</span>
                </button>
                <button
                    type="button"
                    role="tab"
                    aria-selected={tab === "nodes"}
                    style={tab === "nodes" ? TAB_BUTTON_ACTIVE : TAB_BUTTON_BASE}
                    onClick={() => setTab("nodes")}
                >
                    Nodes
                    <span style={COUNT_BADGE_STYLE}>
                        {SHADERS.length + 2 /* Output + Slot */}
                    </span>
                </button>
            </div>
            <div
                style={
                    tab === "files"
                        ? { ...PANEL_BODY_STYLE, padding: 8 }
                        : PANEL_BODY_STYLE
                }
            >
                {tab === "files" ? <FilesPanelBody /> : <NodeLibraryPanelBody />}
            </div>
        </div>
    );
}
