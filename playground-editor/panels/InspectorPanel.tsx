import type { CSSProperties } from "react";
import type { Edge, Node } from "@xyflow/react";
import { SHADERS } from "../shaders";
import type { ShaderNodeData } from "../nodes/ShaderNode";
import {
    OUTPUT_RESOLUTION_PRESETS,
    readOutputParams,
    type OutputParams,
} from "../nodes/OutputNode";
import {
    EMPTY_STYLE,
    PANEL_BODY_STYLE,
    PANEL_HEADER_STYLE,
    PANEL_STYLE,
} from "../styles";

interface InspectorPanelProps {
    readonly nodes: readonly Node[];
    readonly edges: readonly Edge[];
    readonly paramsByNode: Record<string, unknown>;
    readonly onParamsChange: (nodeId: string, params: unknown) => void;
}

export function InspectorPanel({
    nodes,
    edges,
    paramsByNode,
    onParamsChange,
}: InspectorPanelProps) {
    const selected = nodes.find((n) => n.selected);

    return (
        <div style={PANEL_STYLE}>
            <div style={PANEL_HEADER_STYLE}>Inspector</div>
            <div style={PANEL_BODY_STYLE}>
                {!selected ? (
                    <div style={EMPTY_STYLE}>Select a node to edit</div>
                ) : selected.type === "source" ? (
                    <div style={EMPTY_STYLE}>
                        Source — uses the selected file from the Files panel.
                    </div>
                ) : selected.type === "output" ? (
                    <OutputInspector
                        params={readOutputParams(paramsByNode[selected.id])}
                        onChange={(p) => onParamsChange(selected.id, p)}
                    />
                ) : (
                    <ShaderInspector
                        node={selected}
                        params={
                            paramsByNode[selected.id] ??
                            findEntry(selected)?.defaultParams
                        }
                        nodes={nodes}
                        edges={edges}
                        onChange={(p) => onParamsChange(selected.id, p)}
                    />
                )}
            </div>
        </div>
    );
}

function findEntry(node: Node) {
    const data = node.data as ShaderNodeData | undefined;
    if (!data) return undefined;
    return SHADERS.find((s) => s.id === data.shaderId);
}

interface ShaderInspectorProps {
    readonly node: Node;
    readonly params: unknown;
    readonly nodes: readonly Node[];
    readonly edges: readonly Edge[];
    readonly onChange: (next: unknown) => void;
}

interface OutputInspectorProps {
    readonly params: OutputParams;
    readonly onChange: (next: OutputParams) => void;
}

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
    marginBottom: 4,
};

const NUMBER_INPUT_STYLE: CSSProperties = {
    flex: 1,
    minWidth: 0,
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "3px 6px",
    fontSize: 12,
    fontFamily: "inherit",
    fontVariantNumeric: "tabular-nums",
};

const PRESET_BTN_STYLE: CSSProperties = {
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "3px 8px",
    fontSize: 11,
    fontFamily: "inherit",
    cursor: "pointer",
};

const SECTION_LABEL_STYLE: CSSProperties = {
    color: "#aaa",
    fontSize: 11,
    fontWeight: 500,
    marginTop: 12,
    marginBottom: 6,
    textTransform: "uppercase",
    letterSpacing: 0.5,
};

function OutputInspector({ params, onChange }: OutputInspectorProps) {
    const setResolution = (next: number) => {
        onChange({ resolution: clampDim(next) });
    };

    return (
        <>
            <div style={{ color: "#bdbdbd", fontWeight: 500, marginBottom: 4 }}>
                Output
            </div>

            <div style={SECTION_LABEL_STYLE}>resolution (longest side)</div>
            <div style={ROW_STYLE}>
                <input
                    type="number"
                    min={1}
                    max={8192}
                    step={1}
                    value={params.resolution}
                    onChange={(e) => setResolution(Number(e.target.value))}
                    style={NUMBER_INPUT_STYLE}
                />
            </div>
            <div style={{ ...ROW_STYLE, gap: 4 }}>
                {OUTPUT_RESOLUTION_PRESETS.map((p) => (
                    <button
                        key={p}
                        type="button"
                        onClick={() => setResolution(p)}
                        style={PRESET_BTN_STYLE}
                        title={`Set resolution to ${p}`}
                    >
                        {p}
                    </button>
                ))}
            </div>

            <div
                style={{
                    color: "#666",
                    fontSize: 11,
                    marginTop: 12,
                    lineHeight: 1.5,
                }}
            >
                The working buffer matches the Global Input source's aspect.
                Resolution is the longest side in pixels; the other side is
                derived from the source.
            </div>
        </>
    );
}

function clampDim(v: number): number {
    if (!Number.isFinite(v)) return 1;
    return Math.max(1, Math.min(8192, Math.round(v)));
}

function ShaderInspector({
    node,
    params,
    nodes,
    edges,
    onChange,
}: ShaderInspectorProps) {
    const entry = findEntry(node);
    if (!entry) {
        const data = node.data as ShaderNodeData | undefined;
        return (
            <div style={EMPTY_STYLE}>
                Unknown shader: {data?.shaderId ?? "(none)"}
            </div>
        );
    }
    return (
        <>
            <div style={{ color: "#bdbdbd", fontWeight: 500, marginBottom: 4 }}>
                {entry.name}
            </div>
            {entry.Controls ? (
                <entry.Controls
                    params={params}
                    onChange={onChange}
                    nodes={nodes}
                    edges={edges}
                    nodeId={node.id}
                />
            ) : (
                <div style={EMPTY_STYLE}>No editable parameters</div>
            )}
        </>
    );
}
