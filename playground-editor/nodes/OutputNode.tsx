import { Handle, Position, type NodeProps } from "@xyflow/react";
import {
    NODE_BODY_STYLE,
    NODE_FRAME_SELECTED_STYLE,
    NODE_FRAME_STYLE,
    NODE_HEADER_STYLE,
} from "../styles";

/** The Output node only chooses a resolution (longest-side pixels).
 *  The working buffer's aspect comes from the Global Input's source —
 *  Output never imposes its own aspect or crops the result. */
export interface OutputParams {
    /** Longest-side pixel size of the working buffer. The other side
     *  is derived from the Global Input source's aspect. */
    readonly resolution: number;
}

export const DEFAULT_OUTPUT_PARAMS: OutputParams = {
    resolution: 1024,
};

export const OUTPUT_RESOLUTION_PRESETS: ReadonlyArray<number> = [
    512, 1024, 2048, 4096,
];

/** Read an OutputParams from the store, filling in defaults for missing
 *  / malformed fields. Tolerates the legacy `{ aspect, width, height,
 *  crop }` shape so older saved graphs keep loading: collapses to
 *  `resolution = max(width, height)`. */
export function readOutputParams(raw: unknown): OutputParams {
    if (!raw || typeof raw !== "object") return DEFAULT_OUTPUT_PARAMS;
    const r = raw as Partial<OutputParams> & {
        readonly width?: number;
        readonly height?: number;
    };
    if (typeof r.resolution === "number" && Number.isFinite(r.resolution)) {
        return { resolution: clampDim(r.resolution) };
    }
    if (typeof r.width === "number" || typeof r.height === "number") {
        const w = typeof r.width === "number" ? r.width : DEFAULT_OUTPUT_PARAMS.resolution;
        const h = typeof r.height === "number" ? r.height : DEFAULT_OUTPUT_PARAMS.resolution;
        return { resolution: clampDim(Math.max(w, h)) };
    }
    return DEFAULT_OUTPUT_PARAMS;
}

function clampDim(v: unknown): number {
    if (typeof v !== "number" || !Number.isFinite(v)) {
        return DEFAULT_OUTPUT_PARAMS.resolution;
    }
    return Math.max(1, Math.min(8192, Math.round(v)));
}

export function OutputNode({ selected }: NodeProps) {
    return (
        <div style={selected ? NODE_FRAME_SELECTED_STYLE : NODE_FRAME_STYLE}>
            <div style={NODE_HEADER_STYLE}>Output</div>
            <div style={NODE_BODY_STYLE}>preview sink</div>
            <Handle type="target" position={Position.Left} />
        </div>
    );
}
