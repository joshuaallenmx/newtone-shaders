import { useMemo } from "react";
import * as THREE from "three";
import { Pipeline } from "../core/pipeline";
import { createTimeSignal } from "../core/signals";
import type { PointerState, Signal } from "../core/signals";
import { makeEnvTexture } from "../core/textures";
import { createLiquidMetalV2 } from "../atoms/recipes/liquid-metal-v2";
import { ShaderCanvas } from "./ShaderCanvas";
import type { ShaderSetup, ShaderSetupResult } from "./useShaderPipeline";

export interface LiquidMetalV2Props {
    /**
     * Two-stop vertical gradient sampled by reflections as the environment.
     * Top stop = v=0 in the texture; bottom stop = v=1.
     * @default ["#ffffff", "#1a1a22"]
     */
    readonly envColors?: readonly [string, string];
    /**
     * Print the assembled fragment shader source for every pass to the
     * browser console on mount. @default false
     */
    readonly logShaderSource?: boolean;
    /**
     * Diagnostic mode — replace the production render chain with a
     * single atom that paints `state.xy` directly as RGB. Use to verify
     * the sim is producing symmetric velocity in all four directions.
     * @default false
     */
    readonly debug?: boolean;
}

const DEFAULT_ENV_COLORS: readonly [string, string] = ["#ffffff", "#1a1a22"];

/**
 * Atomized rebuild of `LiquidMetal`. The shader source is assembled at
 * write time from atoms in `src/atoms/`, then handed to the existing
 * `Pipeline` runtime unchanged.
 *
 * Inline pointer signaling (bypasses `core/signals/PointerSignal`) so
 * we can rule out the signal abstraction as the source of an
 * iMouse-stuck-at-(0,0) bug. A direct `pointermove` listener on
 * `window` writes positions into a Vector4 that's read each frame as
 * `iMouse`, and per-frame delta into a Vector2 read as `iPointerDelta`.
 */
export function LiquidMetalV2({
    envColors = DEFAULT_ENV_COLORS,
    logShaderSource = false,
    debug = false,
}: LiquidMetalV2Props) {
    const envTexture = useMemo(
        () => makeEnvTexture(envColors),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [envColors.join("|")],
    );

    const setup: ShaderSetup = ({ canvasHost, renderer }): ShaderSetupResult => {
        // Initialize the pointer state at the canvas center so the
        // debug marker has a sane starting position. Without this it
        // would sit at (0,0) — the bottom-left corner — until first
        // mouse move, which makes it ambiguous whether events are
        // landing or not.
        const initRect = canvasHost.getBoundingClientRect();
        const initDpr = renderer.getPixelRatio();
        const initX = (initRect.width * 0.5) * initDpr;
        const initY = (initRect.height * 0.5) * initDpr;

        const pointerState: PointerState = {
            position: new THREE.Vector4(initX, initY, initX, initY),
            delta: new THREE.Vector2(0, 0),
            velocity: new THREE.Vector2(0, 0),
            speed: 0,
            active: false,
        };
        const hasPointer = { current: false };

        const onMove = (e: PointerEvent) => {
            const rect = canvasHost.getBoundingClientRect();
            const dpr = renderer.getPixelRatio();
            const cssX = e.clientX - rect.left;
            const cssY = e.clientY - rect.top;
            const x = cssX * dpr;
            // GL Y is bottom-up; DOM Y is top-down.
            const y = (rect.height - cssY) * dpr;
            const prevX = pointerState.position.x;
            const prevY = pointerState.position.y;
            // Accumulate sub-frame deltas (don't overwrite). onFrame
            // zeros this each rAF tick, so multiple pointermoves
            // between frames sum up to the total frame motion.
            pointerState.delta.x += x - prevX;
            pointerState.delta.y += y - prevY;
            pointerState.position.set(x, y, prevX, prevY);
            pointerState.active =
                cssX >= 0 && cssY >= 0 && cssX <= rect.width && cssY <= rect.height;
            hasPointer.current = true;
        };
        window.addEventListener("pointermove", onMove);

        // Stub Signal interface so the recipe's typing is satisfied.
        // The recipe reads from this via the same signal-projector
        // mechanism as the original PointerSignal.
        const pointer: Signal<PointerState> = {
            id: "liquid-metal-v2-inline-pointer",
            get: () => pointerState,
            subscribe: () => () => {},
            start: () => {},
            stop: () => {},
        };

        // eslint-disable-next-line no-console
        console.log(
            "[LiquidMetalV2 v3 INLINE] setup ran. rect=",
            initRect,
            "init pos=(",
            initX,
            initY,
            ")",
        );
        // Heartbeat: log pointerState once per second so we can verify
        // it's being updated even if pointermove events fire silently.
        const heartbeat = window.setInterval(() => {
            // eslint-disable-next-line no-console
            console.log(
                `[heartbeat] pos=(${pointerState.position.x.toFixed(1)},${pointerState.position.y.toFixed(1)}) delta=(${pointerState.delta.x.toFixed(1)},${pointerState.delta.y.toFixed(1)}) hasPointer=${hasPointer.current}`,
            );
        }, 1000);
        let ptrLogCount = 0;
        const ptrLogListener = (e: PointerEvent) => {
            ptrLogCount++;
            if (ptrLogCount <= 3 || ptrLogCount % 120 === 0) {
                // eslint-disable-next-line no-console
                console.log(
                    `[ptr #${ptrLogCount}] client=(${e.clientX},${e.clientY}) → iMouse=(${pointerState.position.x.toFixed(1)},${pointerState.position.y.toFixed(1)}) delta=(${pointerState.delta.x.toFixed(1)},${pointerState.delta.y.toFixed(1)})`,
                );
            }
        };
        window.addEventListener("pointermove", ptrLogListener);

        const composed = createLiquidMetalV2({
            envTexture,
            pointer,
            hasPointer,
            debug,
        });
        if (logShaderSource) {
            for (const [passId, source] of Object.entries(composed.sources)) {
                // eslint-disable-next-line no-console
                console.log(`[LiquidMetalV2] pass "${passId}":\n${source}`);
            }
        }
        const pipeline = new Pipeline(renderer, composed.config);
        const time = createTimeSignal();
        return {
            pipeline,
            time,
            // No signals — we manage pointermove inline above.
            // After each frame, zero the accumulated pointer delta so
            // the next frame starts at "no motion" unless real events
            // land in between.
            onFrame: () => {
                pointerState.delta.set(0, 0);
            },
            dispose: () => {
                window.clearInterval(heartbeat);
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointermove", ptrLogListener);
                envTexture.dispose();
            },
        };
    };

    return <ShaderCanvas setup={setup} />;
}
