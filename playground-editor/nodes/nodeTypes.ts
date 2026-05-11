import type { NodeTypes } from "@xyflow/react";
import { SourceNode } from "./SourceNode";
import { ShaderNode } from "./ShaderNode";
import { OutputNode } from "./OutputNode";
import { GlobalInputNode } from "./SlotNode";

export const NODE_TYPES: NodeTypes = {
    source: SourceNode,
    shader: ShaderNode,
    output: OutputNode,
    globalInput: GlobalInputNode,
    // Legacy alias: graphs saved before the rename use `slot`. The
    // deserialize step also rewrites them, but keeping this entry means
    // anything still referencing the old type renders correctly until
    // the rewrite happens.
    slot: GlobalInputNode,
};

export type EditorNodeKind = "source" | "shader" | "output" | "globalInput";
