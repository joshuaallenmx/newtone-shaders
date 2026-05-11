import type { ReactNode } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { RESIZE_HANDLE_H_STYLE, RESIZE_HANDLE_V_STYLE } from "./styles";

interface EditorShellProps {
    readonly topbar: ReactNode;
    readonly preview: ReactNode;
    readonly inspector: ReactNode;
    /** Combined Files + Node Library panel (tabs). */
    readonly library: ReactNode;
    readonly nodeGraph: ReactNode;
}

export function EditorShell({
    topbar,
    preview,
    inspector,
    library,
    nodeGraph,
}: EditorShellProps) {
    return (
        <div
            style={{
                width: "100vw",
                height: "100vh",
                background: "#0a0a0a",
                display: "flex",
                flexDirection: "column",
            }}
        >
            {topbar}
            <PanelGroup direction="horizontal" autoSaveId="editor-outer">
                <Panel defaultSize={80} minSize={50}>
                    <PanelGroup direction="vertical" autoSaveId="editor-main">
                        <Panel defaultSize={55} minSize={20}>
                            {preview}
                        </Panel>
                        <PanelResizeHandle style={RESIZE_HANDLE_V_STYLE} />
                        <Panel minSize={20}>
                            <PanelGroup
                                direction="horizontal"
                                autoSaveId="editor-bottom-row"
                            >
                                <Panel defaultSize={25} minSize={12}>
                                    {library}
                                </Panel>
                                <PanelResizeHandle
                                    style={RESIZE_HANDLE_H_STYLE}
                                />
                                <Panel minSize={20}>{nodeGraph}</Panel>
                            </PanelGroup>
                        </Panel>
                    </PanelGroup>
                </Panel>
                <PanelResizeHandle style={RESIZE_HANDLE_H_STYLE} />
                <Panel defaultSize={20} minSize={12}>
                    {inspector}
                </Panel>
            </PanelGroup>
        </div>
    );
}
