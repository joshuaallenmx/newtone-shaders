import type { ChainSpec, ViewMode } from ".";
import type { Pipeline } from "../pipeline/Pipeline";
import { PipelineRenderer } from "./PipelineRenderer";

interface ChainRendererProps {
    readonly chain: ChainSpec;
    readonly viewMode: ViewMode;
    readonly bufferW: number;
    readonly bufferH: number;
    readonly onPipelineChange?: (pipeline: Pipeline | null) => void;
    readonly onPostRender?: () => void;
}

/**
 * Sole renderer for compiled chains. The pipeline is the only render path —
 * legacy per-component rendering is gone. Every shader entry must declare
 * `gpu` and/or `producer`; if any node lacks both, `flattenChain` will
 * throw and `PipelineRenderer` will surface the error in its rebuild guard.
 */
export function ChainRenderer({
    chain,
    viewMode,
    bufferW,
    bufferH,
    onPipelineChange,
    onPostRender,
}: ChainRendererProps) {
    return (
        <PipelineRenderer
            chain={chain}
            viewMode={viewMode}
            bufferW={bufferW}
            bufferH={bufferH}
            onPipelineChange={onPipelineChange}
            onPostRender={onPostRender}
        />
    );
}
