import { LiquidMetalV2 } from "../src/react/LiquidMetalV2";

export function App() {
    return (
        <div
            style={{
                position: "fixed",
                inset: 0,
                width: "100vw",
                height: "100vh",
            }}
        >
            <LiquidMetalV2 logShaderSource debug />
        </div>
    );
}
