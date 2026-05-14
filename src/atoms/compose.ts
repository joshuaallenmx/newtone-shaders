import type { PassConfig, PipelineConfig } from "../core/pipeline";
import { composeFragment } from "../core/pipeline";
import type { Atom, Recipe } from "./types";

/**
 * Assemble a recipe into a `PipelineConfig` consumable by the existing
 * `Pipeline` class. No new runtime — just text concatenation of atom GLSL
 * snippets into one `mainImage` per pass, then wrapped with the standard
 * Shadertoy-style prelude via `composeFragment`.
 *
 * The output fragment shader of each pass is exposed on the returned object
 * so callers can `console.log` it for verification.
 */
export function compose(recipe: Recipe): {
    config: PipelineConfig;
    sources: Readonly<Record<string, string>>;
} {
    const sources: Record<string, string> = {};
    const passes: PassConfig[] = recipe.passes.map((pass) => {
        const fragmentSource = assembleMainImage(
            pass.atoms,
            pass.target.kind === "screen",
        );
        const fragment = composeFragment(fragmentSource);
        sources[pass.id] = fragment;
        return {
            id: pass.id,
            fragment,
            target: pass.target,
            uniforms: pass.uniforms ?? {},
        };
    });
    return {
        config: {
            id: recipe.id,
            passes,
            outputs: recipe.outputs,
        },
        sources,
    };
}

/**
 * Build the `mainImage` body for one pass by concatenating atom contributions
 * in order. Uniforms and definitions are deduplicated by exact-text match —
 * two atoms can independently declare the same helper or uniform without
 * causing a GLSL redeclaration error.
 */
function assembleMainImage(
    atoms: readonly Atom[],
    isScreenTarget: boolean,
): string {
    const uniformLines = new Set<string>();
    const definitionBlocks = new Set<string>();
    const bodyParts: string[] = [];

    for (const atom of atoms) {
        if (atom.uniforms) {
            for (const line of atom.uniforms.split("\n")) {
                const trimmed = line.trim();
                if (trimmed.length > 0) uniformLines.add(trimmed);
            }
        }
        if (atom.definitions) {
            const trimmed = atom.definitions.trim();
            if (trimmed.length > 0) definitionBlocks.add(trimmed);
        }
        bodyParts.push(`    // atom: ${atom.id}\n    ${indent(atom.body, "    ")}`);
    }

    const uniforms = Array.from(uniformLines).join("\n");
    const definitions = Array.from(definitionBlocks).join("\n\n");
    const body = bodyParts.join("\n\n");
    // For screen targets the pass writes a final color sourced from `col`.
    // For pingpong/fixed targets it writes simulation state from `state`.
    // Atoms may overwrite fragColor directly; the auto-write here runs last
    // and uses whichever shared local matches the target kind.
    const finalWrite = isScreenTarget
        ? "fragColor = vec4(col, 1.0);"
        : "fragColor = state;";

    return `
${uniforms}

${definitions}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Shared render/sim locals. Atoms read and overwrite by name.
    vec2 pos = fragCoord;
    vec2 uv = fragCoord / iResolution.xy;
    vec4 state = vec4(0.0);
    vec3 col = vec3(1.0);
    vec3 n = vec3(0.0, 0.0, 1.0);

${body}

    ${finalWrite}
}
`;
}

function indent(source: string, prefix: string): string {
    return source
        .trim()
        .split("\n")
        .map((line, i) => (i === 0 ? line : `${prefix}${line}`))
        .join("\n");
}
