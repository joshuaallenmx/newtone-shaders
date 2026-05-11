import { describe, it, expect } from "vitest";
import { composeFragment, SHADERTOY_PRELUDE } from "./prelude";

describe("composeFragment", () => {
    it("includes the prelude verbatim", () => {
        const out = composeFragment(`void mainImage(out vec4 c, in vec2 p) { c = vec4(1.0); }`);
        expect(out).toContain(SHADERTOY_PRELUDE.trim());
    });
    it("appends a main() that calls mainImage", () => {
        const out = composeFragment(`void mainImage(out vec4 c, in vec2 p) { c = vec4(1.0); }`);
        expect(out).toContain("void main() {");
        expect(out).toContain("mainImage(outColor, gl_FragCoord.xy)");
    });
});
