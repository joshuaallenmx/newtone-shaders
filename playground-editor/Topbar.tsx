import { useEffect, useRef, useState, type CSSProperties } from "react";
import type { Project } from "./serialize";

export type SaveStatus = "saved" | "unsaved" | "saving";

interface TopbarProps {
    readonly projects: readonly Project[];
    readonly activeId: string;
    readonly status: SaveStatus;
    readonly lastSavedAt: number | null;
    readonly onSelect: (id: string) => void;
    readonly onRename: (id: string, name: string) => void;
    readonly onCreate: () => void;
    readonly onDuplicate: (id: string) => void;
    readonly onDelete: (id: string) => void;
    readonly onImport: (file: File) => void;
    readonly onExport: (id: string) => void;
}

const BAR_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "0 12px",
    height: 36,
    flexShrink: 0,
    background: "#0d0d0d",
    borderBottom: "1px solid #2a2a2a",
    color: "#f0f0f0",
    fontSize: 12,
    fontFamily: "inherit",
};

const BRAND_STYLE: CSSProperties = {
    fontWeight: 600,
    color: "#888",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    fontSize: 11,
};

const NAME_INPUT_STYLE: CSSProperties = {
    background: "transparent",
    color: "#f0f0f0",
    border: "1px solid transparent",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 13,
    fontFamily: "inherit",
    minWidth: 120,
    outline: "none",
};

const NAME_INPUT_EDITING_STYLE: CSSProperties = {
    ...NAME_INPUT_STYLE,
    border: "1px solid #4a90e2",
    background: "#0a0a0a",
};

const SELECT_STYLE: CSSProperties = {
    background: "#1a1a1a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 12,
    fontFamily: "inherit",
};

const BUTTON_STYLE: CSSProperties = {
    background: "#1a1a1a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "3px 8px",
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "inherit",
};

const STATUS_DOT_BASE: CSSProperties = {
    width: 8,
    height: 8,
    borderRadius: "50%",
    flexShrink: 0,
};

const STATUS_COLOR: Record<SaveStatus, string> = {
    saved: "#3ecf8e",
    unsaved: "#e2a84a",
    saving: "#4a90e2",
};

const STATUS_LABEL: Record<SaveStatus, string> = {
    saved: "Saved",
    unsaved: "Unsaved",
    saving: "Saving…",
};

function formatRelative(ts: number, now: number): string {
    const seconds = Math.max(0, Math.floor((now - ts) / 1000));
    if (seconds < 5) return "just now";
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return new Date(ts).toLocaleString();
}

export function Topbar({
    projects,
    activeId,
    status,
    lastSavedAt,
    onSelect,
    onRename,
    onCreate,
    onDuplicate,
    onDelete,
    onImport,
    onExport,
}: TopbarProps) {
    const active = projects.find((p) => p.id === activeId);
    const [editing, setEditing] = useState(false);
    const [draftName, setDraftName] = useState(active?.name ?? "");
    const inputRef = useRef<HTMLInputElement | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [now, setNow] = useState(() => Date.now());

    useEffect(() => {
        if (status !== "saved" || lastSavedAt === null) return;
        const id = window.setInterval(() => setNow(Date.now()), 5000);
        return () => window.clearInterval(id);
    }, [status, lastSavedAt]);

    useEffect(() => {
        if (!editing) setDraftName(active?.name ?? "");
    }, [active?.name, editing]);

    const commitRename = () => {
        if (!active) return;
        const next = draftName.trim();
        setEditing(false);
        if (next && next !== active.name) onRename(active.id, next);
        else setDraftName(active.name);
    };

    return (
        <div style={BAR_STYLE}>
            <span style={BRAND_STYLE}>Newtone</span>

            <input
                ref={inputRef}
                style={editing ? NAME_INPUT_EDITING_STYLE : NAME_INPUT_STYLE}
                value={draftName}
                readOnly={!editing}
                onChange={(e) => setDraftName(e.target.value)}
                onClick={() => {
                    if (!editing) {
                        setEditing(true);
                        requestAnimationFrame(() => inputRef.current?.select());
                    }
                }}
                onBlur={commitRename}
                onKeyDown={(e) => {
                    if (e.key === "Enter") {
                        e.preventDefault();
                        commitRename();
                        inputRef.current?.blur();
                    } else if (e.key === "Escape") {
                        setEditing(false);
                        setDraftName(active?.name ?? "");
                        inputRef.current?.blur();
                    }
                }}
                title="Click to rename"
            />

            <select
                style={SELECT_STYLE}
                value={activeId}
                onChange={(e) => onSelect(e.target.value)}
                title="Switch project"
            >
                {projects.map((p) => (
                    <option key={p.id} value={p.id}>
                        {p.name}
                    </option>
                ))}
            </select>

            <button
                type="button"
                style={BUTTON_STYLE}
                onClick={onCreate}
                title="New project"
            >
                + New
            </button>
            <button
                type="button"
                style={BUTTON_STYLE}
                onClick={() => active && onDuplicate(active.id)}
                disabled={!active}
                title="Duplicate current project"
            >
                Duplicate
            </button>
            <button
                type="button"
                style={BUTTON_STYLE}
                onClick={() => fileInputRef.current?.click()}
                title="Import a project from a JSON file"
            >
                Import
            </button>
            <button
                type="button"
                style={BUTTON_STYLE}
                onClick={() => active && onExport(active.id)}
                disabled={!active}
                title="Export current project to a JSON file"
            >
                Export
            </button>
            <button
                type="button"
                style={BUTTON_STYLE}
                onClick={() => {
                    if (!active) return;
                    if (projects.length <= 1) {
                        alert("Cannot delete the only project.");
                        return;
                    }
                    if (confirm(`Delete project "${active.name}"?`)) {
                        onDelete(active.id);
                    }
                }}
                disabled={!active || projects.length <= 1}
                title="Delete current project"
            >
                Delete
            </button>

            <input
                ref={fileInputRef}
                type="file"
                accept="application/json,.json"
                style={{ display: "none" }}
                onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) onImport(file);
                    e.target.value = "";
                }}
            />

            <div style={{ flex: 1 }} />

            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    color: "#888",
                }}
                title={
                    lastSavedAt
                        ? `Last saved ${new Date(lastSavedAt).toLocaleString()}`
                        : "Not yet saved"
                }
            >
                <span
                    style={{
                        ...STATUS_DOT_BASE,
                        background: STATUS_COLOR[status],
                    }}
                />
                <span>
                    {STATUS_LABEL[status]}
                    {status === "saved" && lastSavedAt
                        ? ` · ${formatRelative(lastSavedAt, now)}`
                        : ""}
                </span>
            </div>
        </div>
    );
}
