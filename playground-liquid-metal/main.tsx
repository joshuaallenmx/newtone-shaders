import { createRoot } from "react-dom/client";
import { App } from "./App";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

// StrictMode disabled while debugging the pointer-binding issue.
// StrictMode mounts components twice in dev, which can interact badly
// with the useShaderPipeline lifecycle (renderer/pipeline disposed on
// the first mount's cleanup). Re-enable once the bug is fixed.
createRoot(root).render(<App />);
