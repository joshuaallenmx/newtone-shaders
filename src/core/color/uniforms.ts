import * as THREE from "three";
import { cssColorToLinearRgb } from "./css";

/**
 * Set a `vec3` uniform from any CSS color string. The vec3 holds linear RGB,
 * matching the linear-space convention shaders here write to the framebuffer.
 */
export function setVec3FromColor(target: THREE.Vector3, color: string): void {
    const [r, g, b] = cssColorToLinearRgb(color);
    target.set(r, g, b);
}
