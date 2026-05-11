import type { CSSProperties } from "react";

export const PANEL_STYLE: CSSProperties = {
    width: "100%",
    height: "100%",
    background: "rgba(15,15,15,0.85)",
    border: "1px solid #2a2a2a",
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    fontSize: 12,
    lineHeight: 1.4,
    overflow: "hidden",
};

export const PANEL_HEADER_STYLE: CSSProperties = {
    padding: "6px 10px",
    borderBottom: "1px solid #2a2a2a",
    color: "#888",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    flexShrink: 0,
};

export const PANEL_BODY_STYLE: CSSProperties = {
    flex: 1,
    minHeight: 0,
    overflow: "auto",
    padding: 8,
    display: "flex",
    flexDirection: "column",
    gap: 6,
};

export const BUTTON_STYLE: CSSProperties = {
    background: "#1a1a1a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 10px",
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "inherit",
};

export const SELECT_STYLE: CSSProperties = {
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 6px",
    fontSize: 12,
    fontFamily: "inherit",
    width: "100%",
};

export const LIST_ITEM_STYLE: CSSProperties = {
    background: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: 4,
    padding: "6px 8px",
    cursor: "pointer",
    color: "#f0f0f0",
    userSelect: "none",
};

export const LIST_ITEM_ACTIVE_STYLE: CSSProperties = {
    ...LIST_ITEM_STYLE,
    border: "1px solid #4a90e2",
    background: "#1f2a3a",
};

export const RESIZE_HANDLE_H_STYLE: CSSProperties = {
    width: 4,
    background: "#0a0a0a",
    cursor: "col-resize",
};

export const RESIZE_HANDLE_V_STYLE: CSSProperties = {
    height: 4,
    background: "#0a0a0a",
    cursor: "row-resize",
};

export const EMPTY_STYLE: CSSProperties = {
    width: "100%",
    height: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#666",
    fontSize: 12,
    padding: 12,
    textAlign: "center",
};

export const NODE_FRAME_STYLE: CSSProperties = {
    background: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: 6,
    color: "#f0f0f0",
    fontSize: 12,
    fontFamily: "inherit",
    width: 160,
};

export const NODE_FRAME_SELECTED_STYLE: CSSProperties = {
    ...NODE_FRAME_STYLE,
    border: "1px solid #4a90e2",
};

export const NODE_HEADER_STYLE: CSSProperties = {
    padding: "6px 10px",
    borderBottom: "1px solid #2a2a2a",
    fontWeight: 500,
};

export const NODE_BODY_STYLE: CSSProperties = {
    padding: "6px 10px",
    color: "#888",
    fontSize: 11,
};
