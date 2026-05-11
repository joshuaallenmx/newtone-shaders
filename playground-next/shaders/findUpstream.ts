import type { ShaderControlsProps } from ".";

/** Walk back from `nodeId` along its `"in"` edge, hopping through any
 *  Global Input / Slot passthroughs, to the first upstream node that
 *  has a real render plan entry. That's the one we can snapshot for a
 *  PreviewPad backdrop — Global Input nodes collapse out of the
 *  flattened plan and don't own an outputTex, so calling
 *  `pipeline.snapshotNode` with their id would always return false.
 *
 *  Returns null when the chain is unwired or hits a cycle. */
export function findUpstreamId(
    nodes: ShaderControlsProps["nodes"],
    edges: ShaderControlsProps["edges"],
    nodeId: string | undefined,
): string | null {
    if (!nodes || !edges || !nodeId) return null;
    const visited = new Set<string>();
    let current: string = nodeId;
    while (!visited.has(current)) {
        visited.add(current);
        let nextEdge: NonNullable<typeof edges>[number] | null = null;
        for (const e of edges) {
            if (e.target !== current) continue;
            if (e.targetHandle != null && e.targetHandle !== "in") continue;
            nextEdge = e;
            break;
        }
        if (!nextEdge) return null;
        const upstream = nodes.find((n) => n.id === nextEdge!.source);
        if (!upstream) return null;
        if (upstream.type === "globalInput" || upstream.type === "slot") {
            current = upstream.id;
            continue;
        }
        return upstream.id;
    }
    return null;
}
