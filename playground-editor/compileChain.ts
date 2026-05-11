// Chain compiler.
//
// Produces a `ChainSpec` tree the pipeline renderer consumes. Each shader
// becomes its own pass; the pipeline is the only render path and
// `flattenChain` handles arbitrary DAGs with multi-input nodes at any
// depth. Zero-input shaders are treated as sources: they render standalone
// with no upstream and can feed either the Output or any input slot.

import type { Edge, Node } from "@xyflow/react";
import { findAssetByName } from "./assets";
import type { ChainSpec, ShaderEntry } from "./shaders";
import { readLayers } from "./shaders";
import type { ShaderNodeData } from "./nodes/ShaderNode";
import type { SourceNodeData } from "./nodes/SourceNode";

const DEFAULT_INPUT_HANDLE_ID = "in";

export function compileChain(
    nodes: readonly Node[],
    edges: readonly Edge[],
    paramsByNode: Record<string, unknown>,
    shaders: readonly ShaderEntry[],
): ChainSpec | null {
    const outputs = nodes.filter((n) => n.type === "output");
    if (outputs.length !== 1) return null;
    const output = outputs[0]!;

    const incomingEdge = edges.find((e) => e.target === output.id);
    if (!incomingEdge) return null;

    return resolveAt(
        incomingEdge.source,
        nodes,
        edges,
        paramsByNode,
        shaders,
        new Set(),
    );
}

function resolveAt(
    nodeId: string,
    nodes: readonly Node[],
    edges: readonly Edge[],
    paramsByNode: Record<string, unknown>,
    shaders: readonly ShaderEntry[],
    visited: ReadonlySet<string>,
): ChainSpec | null {
    if (visited.has(nodeId)) return null; // cycle
    const newVisited = new Set(visited).add(nodeId);

    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return null;

    if (node.type === "globalInput" || node.type === "slot") {
        // Global Input (formerly Slot) is a passthrough: forward to
        // whatever's wired into its single input. One Global Input can
        // fan out to many downstream consumers — swap the source once,
        // every consumer follows. The "slot" alias keeps legacy graphs
        // (saved before the rename) compiling.
        const upstreamEdge = findEdgeForHandle(
            edges,
            node.id,
            DEFAULT_INPUT_HANDLE_ID,
        );
        if (!upstreamEdge) return null;
        return resolveAt(
            upstreamEdge.source,
            nodes,
            edges,
            paramsByNode,
            shaders,
            newVisited,
        );
    }

    if (node.type === "source") {
        const data = node.data as SourceNodeData | undefined;
        const asset = findAssetByName(data?.assetName);
        if (!asset) return null;
        return {
            entry: null,
            params: undefined,
            src: asset.url,
            inputs: [],
            nodeId: node.id,
        };
    }

    if (node.type === "shader") {
        const shaderData = node.data as ShaderNodeData | undefined;
        const entry = shaders.find((s) => s.id === shaderData?.shaderId);
        if (!entry) return null;
        const inputSpecs = entry.inputs ?? [
            { id: DEFAULT_INPUT_HANDLE_ID, label: "in" },
        ];
        const params = paramsByNode[node.id] ?? entry.defaultParams;

        if (entry.variadic) {
            // Variadic: gather every incoming edge and order them by
            // `params.layers` (an array of {src, opacity, blend}). The
            // list is self-healed — entries without a matching edge are
            // dropped, and any edges not yet in the list get appended
            // with default opacity 1 / blend "normal". Legacy graphs that
            // saved `params.order: string[]` flow through the same
            // helper.
            const incoming = edges.filter((e) => e.target === node.id);
            const incomingSources = new Set(incoming.map((e) => e.source));
            const stored = readLayers(params);
            const seenSrc = new Set<string>();
            const alignedLayers: {
                src: string;
                opacity: number;
                blend: import("./shaders").LayerBlend;
            }[] = [];
            for (const layer of stored) {
                if (incomingSources.has(layer.src) && !seenSrc.has(layer.src)) {
                    alignedLayers.push({
                        src: layer.src,
                        opacity: layer.opacity,
                        blend: layer.blend,
                    });
                    seenSrc.add(layer.src);
                }
            }
            for (const e of incoming) {
                if (!seenSrc.has(e.source)) {
                    alignedLayers.push({
                        src: e.source,
                        opacity: 1,
                        blend: "normal",
                    });
                    seenSrc.add(e.source);
                }
            }

            const inputs: ChainSpec[] = [];
            for (const layer of alignedLayers) {
                const child = resolveAt(
                    layer.src,
                    nodes,
                    edges,
                    paramsByNode,
                    shaders,
                    newVisited,
                );
                if (!child) return null;
                inputs.push(child);
            }
            // Emit a compiled params object where `layers[i]` aligns 1:1
            // with `inputs[i]`. The runtime's setUniforms reads opacity
            // and blend off this list directly. The persisted user-edit
            // copy (with potentially stale entries for since-disconnected
            // sources) stays in `paramsByNode` for the inspector.
            const compiledParams = { layers: alignedLayers };
            // Empty (no wires) → still render as black; the GPU pass
            // returns `vec4(0)` when uLayerCount is 0.
            return {
                entry,
                params: compiledParams,
                src: null,
                inputs,
                nodeId: node.id,
            };
        }

        if (inputSpecs.length === 0) {
            // Source shader — renders standalone, no upstream needed.
            return {
                entry,
                params,
                src: null,
                inputs: [],
                nodeId: node.id,
            };
        }

        const inputs: ChainSpec[] = [];
        for (const inputSpec of inputSpecs) {
            const childEdge = findEdgeForHandle(
                edges,
                node.id,
                inputSpec.id,
            );
            if (!childEdge) {
                if (inputSpec.optional) {
                    // Sentinel: pipeline binds a 1×1 placeholder for this
                    // slot and tells the shader's setUniforms via
                    // `frame.inputsPresent[i] === false`.
                    inputs.push({
                        entry: null,
                        params: undefined,
                        src: null,
                        inputs: [],
                        nodeId: null,
                    });
                    continue;
                }
                return null;
            }
            const child = resolveAt(
                childEdge.source,
                nodes,
                edges,
                paramsByNode,
                shaders,
                newVisited,
            );
            if (!child) return null;
            inputs.push(child);
        }
        return {
            entry,
            params,
            src: null,
            inputs,
            nodeId: node.id,
        };
    }

    return null;
}

function findEdgeForHandle(
    edges: readonly Edge[],
    targetNodeId: string,
    handleId: string,
): Edge | undefined {
    return edges.find(
        (e) =>
            e.target === targetNodeId &&
            (e.targetHandle === handleId ||
                (e.targetHandle == null &&
                    handleId === DEFAULT_INPUT_HANDLE_ID)),
    );
}

/** Walk a Global Input (or legacy Slot) node's incoming edge
 *  transitively and return the first non-passthrough upstream nodeId,
 *  or null when the path ends at a missing/unwired source. Used by the
 *  editor's per-node thumbnails: passthroughs collapse out of the
 *  pipeline plan, so to paint a Global Input's preview we point its
 *  `<canvas>` at the resolved upstream's FBO instead. Cycle-safe via a
 *  visited set, mirroring `resolveAt`. */
export function resolveSlotUpstream(
    slotId: string,
    nodes: readonly Node[],
    edges: readonly Edge[],
): string | null {
    const visited = new Set<string>();
    let current = slotId;
    while (true) {
        if (visited.has(current)) return null;
        visited.add(current);
        const edge = findEdgeForHandle(edges, current, DEFAULT_INPUT_HANDLE_ID);
        if (!edge) return null;
        const upstream = nodes.find((n) => n.id === edge.source);
        if (!upstream) return null;
        if (upstream.type === "globalInput" || upstream.type === "slot") {
            current = upstream.id;
            continue;
        }
        return upstream.id;
    }
}
