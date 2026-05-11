import {
    useCallback,
    useEffect,
    useState,
    type CSSProperties,
} from "react";
import { SHADERS, type ViewMode } from "./shaders";

const ASSET_MODULES = import.meta.glob<string>(
    "../playground/assets/*.{png,jpg,jpeg,gif,webp,avif,svg}",
    { query: "?url", import: "default", eager: true },
);
const ASSETS = Object.entries(ASSET_MODULES)
    .map(([key, url]) => ({ name: key.replace(/^.*\//, ""), url }))
    .sort((a, b) => a.name.localeCompare(b.name));

const STAGE_FIT_STYLE: CSSProperties = {
    width: "100vw",
    height: "100vh",
    position: "relative",
    overflow: "hidden",
};

const STAGE_ACTUAL_STYLE: CSSProperties = {
    width: "100vw",
    height: "100vh",
    position: "relative",
    overflow: "auto",
};

const IMAGE_FIT_STYLE: CSSProperties = {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    display: "block",
};

const IMAGE_ACTUAL_STYLE: CSSProperties = {
    display: "block",
};

const PANEL_STYLE: CSSProperties = {
    position: "fixed",
    top: 12,
    left: 12,
    zIndex: 10,
    padding: "10px 12px",
    background: "rgba(15,15,15,0.85)",
    border: "1px solid #2a2a2a",
    borderRadius: 8,
    backdropFilter: "blur(6px)",
    fontSize: 12,
    lineHeight: 1.4,
    width: 240,
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    gap: 8,
};

const NAV_ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 6,
};

const BUTTON_STYLE: CSSProperties = {
    background: "#1a1a1a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 10px",
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "inherit",
};

const COUNTER_STYLE: CSSProperties = {
    flex: 1,
    textAlign: "center",
    color: "#bdbdbd",
    fontVariantNumeric: "tabular-nums",
};

const FILENAME_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
};

const SELECT_STYLE: CSSProperties = {
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 6px",
    fontSize: 12,
    fontFamily: "inherit",
    width: "100%",
};

const EMPTY_STYLE: CSSProperties = {
    width: "100%",
    height: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#666",
    fontSize: 13,
};

export function App() {
    const [imageIndex, setImageIndex] = useState(0);
    const [shaderId, setShaderId] = useState<string>("none");
    const [viewMode, setViewMode] = useState<ViewMode>("fit");
    const [paramsByShader, setParamsByShader] = useState<
        Record<string, unknown>
    >(() => {
        const init: Record<string, unknown> = {};
        for (const s of SHADERS) init[s.id] = s.defaultParams;
        return init;
    });

    const next = useCallback(() => {
        setImageIndex((i) => (i + 1) % ASSETS.length);
    }, []);
    const prev = useCallback(() => {
        setImageIndex((i) => (i - 1 + ASSETS.length) % ASSETS.length);
    }, []);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "ArrowRight") next();
            else if (e.key === "ArrowLeft") prev();
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [next, prev]);

    if (ASSETS.length === 0) {
        return <div style={EMPTY_STYLE}>No assets found in ../playground/assets/</div>;
    }

    const asset = ASSETS[imageIndex]!;
    const shader = SHADERS.find((s) => s.id === shaderId);
    // Fall back to defaultParams when state was initialized before this
    // shader entry existed (e.g., HMR after adding a new shader).
    const shaderParams = shader
        ? (paramsByShader[shader.id] ?? shader.defaultParams)
        : undefined;
    const stageStyle =
        viewMode === "fit" ? STAGE_FIT_STYLE : STAGE_ACTUAL_STYLE;
    const imgStyle =
        viewMode === "fit" ? IMAGE_FIT_STYLE : IMAGE_ACTUAL_STYLE;

    return (
        <div style={stageStyle}>
            {shader ? (
                <shader.Component
                    src={asset.url}
                    params={shaderParams}
                    viewMode={viewMode}
                />
            ) : (
                <img src={asset.url} alt={asset.name} style={imgStyle} />
            )}
            <div style={PANEL_STYLE}>
                <div style={NAV_ROW_STYLE}>
                    <button type="button" onClick={prev} style={BUTTON_STYLE}>
                        ←
                    </button>
                    <span style={COUNTER_STYLE}>
                        {imageIndex + 1} / {ASSETS.length}
                    </span>
                    <button type="button" onClick={next} style={BUTTON_STYLE}>
                        →
                    </button>
                </div>
                <div style={FILENAME_STYLE} title={asset.name}>
                    {asset.name}
                </div>
                <button
                    type="button"
                    style={BUTTON_STYLE}
                    onClick={() =>
                        setViewMode((m) => (m === "fit" ? "actual" : "fit"))
                    }
                    title="toggle zoom"
                >
                    {viewMode === "fit" ? "1:1" : "fit"}
                </button>
                <select
                    style={SELECT_STYLE}
                    value={shaderId}
                    onChange={(e) => setShaderId(e.target.value)}
                >
                    <option value="none">None (source)</option>
                    {SHADERS.map((s) => (
                        <option key={s.id} value={s.id}>
                            {s.name}
                        </option>
                    ))}
                </select>
                {shader && shader.Controls && (
                    <shader.Controls
                        params={shaderParams}
                        onChange={(next) =>
                            setParamsByShader((prev) => ({
                                ...prev,
                                [shader.id]: next,
                            }))
                        }
                    />
                )}
            </div>
        </div>
    );
}
