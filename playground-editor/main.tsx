import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@xyflow/react/dist/style.css";
import { EditorApp } from "./EditorApp";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

createRoot(root).render(
    <StrictMode>
        <EditorApp />
    </StrictMode>,
);
