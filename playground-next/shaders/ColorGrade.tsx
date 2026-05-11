import type { CSSProperties } from "react";
import type {
    GpuPassSpec,
    ShaderControlsProps,
    ShaderEntry,
} from ".";

interface ColorGradeParams {
    // Tone
    /** Exposure adjustment in stops; result multiplied by 2^exposure. */
    readonly exposure: number;
    /** Contrast around 0.5: out = (c - 0.5) * contrast + 0.5. */
    readonly contrast: number;
    /** Gamma curve: out = pow(c, 1/gamma). */
    readonly gamma: number;
    /** Black point in 0..1; values below are clipped to 0 after remap. */
    readonly blackPoint: number;
    /** White point in 0..1; values above are clipped to 1 after remap. */
    readonly whitePoint: number;

    // Color
    /** Saturation: 0 = grayscale, 1 = unchanged, >1 = boost. */
    readonly saturation: number;
    /** Vibrance: -1..1, boost only the less-saturated pixels. */
    readonly vibrance: number;
    /** Hue rotation in degrees, applied in HSV. */
    readonly hueShift: number;

    // White balance
    /** Temperature -1..1: negative = cool (blue), positive = warm (orange). */
    readonly temperature: number;
    /** Tint -1..1: negative = green, positive = magenta. */
    readonly tint: number;

    // RGB channel gain
    readonly gainR: number;
    readonly gainG: number;
    readonly gainB: number;

    // Three-way grading (shadows / mids / highlights)
    /** Hex color centered on #808080 (neutral). */
    readonly shadowsColor: string;
    readonly shadowsAmount: number;
    readonly midsColor: string;
    readonly midsAmount: number;
    readonly highlightsColor: string;
    readonly highlightsAmount: number;

    // Vignette
    readonly vignetteAmount: number;
    /** 0 = sharp edge, 1 = soft falloff to corner. */
    readonly vignetteFalloff: number;
}

const NEUTRAL_GREY = "#808080";

const DEFAULT_PARAMS: ColorGradeParams = {
    exposure: 0,
    contrast: 1,
    gamma: 1,
    blackPoint: 0,
    whitePoint: 1,
    saturation: 1,
    vibrance: 0,
    hueShift: 0,
    temperature: 0,
    tint: 0,
    gainR: 1,
    gainG: 1,
    gainB: 1,
    shadowsColor: NEUTRAL_GREY,
    shadowsAmount: 0,
    midsColor: NEUTRAL_GREY,
    midsAmount: 0,
    highlightsColor: NEUTRAL_GREY,
    highlightsAmount: 0,
    vignetteAmount: 0,
    vignetteFalloff: 0.6,
};

// Order: exposure → levels → channel gain → white balance → gamma →
// three-way → contrast → hue → saturation → vibrance → vignette.
// Hue/sat/vibrance run after tone work so they operate on the perceptual
// final color rather than the linear input. Vignette is last so it
// dims the fully-graded pixel.
const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSource;

uniform float uExposure;
uniform float uContrast;
uniform float uGamma;
uniform float uBlackPoint;
uniform float uWhitePoint;

uniform float uSaturation;
uniform float uVibrance;
uniform float uHueShift;

uniform float uTemperature;
uniform float uTint;

uniform vec3 uGainRgb;

uniform vec3 uShadowsTint;
uniform float uShadowsAmount;
uniform vec3 uMidsTint;
uniform float uMidsAmount;
uniform vec3 uHighlightsTint;
uniform float uHighlightsAmount;

uniform float uVignetteAmount;
uniform float uVignetteFalloff;

const vec3 LUM_W = vec3(0.2126, 0.7152, 0.0722);

vec3 rgb2hsv(vec3 c) {
    float maxC = max(max(c.r, c.g), c.b);
    float minC = min(min(c.r, c.g), c.b);
    float d = maxC - minC;
    float h = 0.0;
    if (d > 1e-5) {
        if (maxC == c.r) h = mod((c.g - c.b) / d, 6.0);
        else if (maxC == c.g) h = (c.b - c.r) / d + 2.0;
        else h = (c.r - c.g) / d + 4.0;
        h *= 60.0;
        if (h < 0.0) h += 360.0;
    }
    float s = maxC > 1e-5 ? d / maxC : 0.0;
    return vec3(h, s, maxC);
}

vec3 hsv2rgb(vec3 hsv) {
    float h = hsv.x;
    float s = hsv.y;
    float v = hsv.z;
    float c1 = v * s;
    float x = c1 * (1.0 - abs(mod(h / 60.0, 2.0) - 1.0));
    float m = v - c1;
    vec3 rgb;
    if (h < 60.0) rgb = vec3(c1, x, 0.0);
    else if (h < 120.0) rgb = vec3(x, c1, 0.0);
    else if (h < 180.0) rgb = vec3(0.0, c1, x);
    else if (h < 240.0) rgb = vec3(0.0, x, c1);
    else if (h < 300.0) rgb = vec3(x, 0.0, c1);
    else rgb = vec3(c1, 0.0, x);
    return rgb + vec3(m);
}

void main() {
    vec4 src = texture(uSource, vUv);
    vec3 c = src.rgb;

    // Exposure (stops).
    c *= exp2(uExposure);

    // Black/white levels — guard the divisor.
    float range = max(uWhitePoint - uBlackPoint, 1e-3);
    c = (c - vec3(uBlackPoint)) / range;

    // Per-channel gain.
    c *= uGainRgb;

    // White balance — temperature is a coupled R↑/B↓ shift; tint is
    // a magenta(R+B) ↔ green(G) shift. Coefficients are gentle on
    // purpose; users wanting heavy shifts can layer with channel gain.
    c.r += uTemperature * 0.10;
    c.b -= uTemperature * 0.10;
    c.r += uTint * 0.05;
    c.g -= uTint * 0.10;
    c.b += uTint * 0.05;

    // Gamma.
    c = pow(max(c, vec3(0.0)), vec3(1.0 / max(uGamma, 1e-3)));

    // Three-way grading by luminance. Shadow/highlight masks are
    // smoothsteps that overlap in the middle; mid mask is the leftover.
    {
        float l = dot(clamp(c, 0.0, 1.0), LUM_W);
        float wShadow = 1.0 - smoothstep(0.0, 0.5, l);
        float wHighlight = smoothstep(0.5, 1.0, l);
        float wMid = max(0.0, 1.0 - wShadow - wHighlight);
        c += uShadowsTint * uShadowsAmount * wShadow
           + uMidsTint * uMidsAmount * wMid
           + uHighlightsTint * uHighlightsAmount * wHighlight;
    }

    // Contrast (around 0.5).
    c = (c - 0.5) * uContrast + 0.5;

    // Hue shift in HSV.
    if (abs(uHueShift) > 1e-3) {
        vec3 hsv = rgb2hsv(max(c, vec3(0.0)));
        hsv.x = mod(hsv.x + uHueShift, 360.0);
        if (hsv.x < 0.0) hsv.x += 360.0;
        c = hsv2rgb(hsv);
    }

    // Saturation.
    {
        float l = dot(c, LUM_W);
        c = mix(vec3(l), c, uSaturation);
    }

    // Vibrance — boost less-saturated pixels more.
    if (abs(uVibrance) > 1e-3) {
        float l = dot(c, LUM_W);
        float maxC = max(max(c.r, c.g), c.b);
        float sat = (maxC - l) / max(maxC, 1e-3);
        float boost = uVibrance * (1.0 - clamp(sat, 0.0, 1.0));
        c = mix(vec3(l), c, 1.0 + boost);
    }

    // Vignette — falloff scales the inner ring; amount sets max darkening.
    if (uVignetteAmount > 1e-3) {
        vec2 d = vUv - vec2(0.5);
        float dist = length(d) * 1.41421356;
        float v = smoothstep(uVignetteFalloff, 1.0, dist) * uVignetteAmount;
        c *= (1.0 - v);
    }

    outColor = vec4(c, src.a);
}
`;

// ─── Helpers ───────────────────────────────────────────────────────────

interface RGB {
    readonly r: number;
    readonly g: number;
    readonly b: number;
}

function parseHexColor(hex: string): RGB {
    const cleaned = hex.replace(/^#/, "");
    if (cleaned.length === 3) {
        return {
            r: parseInt(cleaned[0]! + cleaned[0]!, 16),
            g: parseInt(cleaned[1]! + cleaned[1]!, 16),
            b: parseInt(cleaned[2]! + cleaned[2]!, 16),
        };
    }
    if (cleaned.length === 6) {
        return {
            r: parseInt(cleaned.slice(0, 2), 16),
            g: parseInt(cleaned.slice(2, 4), 16),
            b: parseInt(cleaned.slice(4, 6), 16),
        };
    }
    return { r: 128, g: 128, b: 128 };
}

/** Convert a color picker hex into a centered tint vec3 in [-1, 1].
 *  Neutral grey (#808080) maps to (0,0,0) — no effect. */
function tintFromHex(hex: string): [number, number, number] {
    const c = parseHexColor(hex);
    return [(c.r - 128) / 127, (c.g - 128) / 127, (c.b - 128) / 127];
}

const colorGradeGpuSpec: GpuPassSpec = {
    fragSrc: FRAG_SRC,
    samplers: ["uSource"],
    uniforms: [
        "uExposure",
        "uContrast",
        "uGamma",
        "uBlackPoint",
        "uWhitePoint",
        "uSaturation",
        "uVibrance",
        "uHueShift",
        "uTemperature",
        "uTint",
        "uGainRgb",
        "uShadowsTint",
        "uShadowsAmount",
        "uMidsTint",
        "uMidsAmount",
        "uHighlightsTint",
        "uHighlightsAmount",
        "uVignetteAmount",
        "uVignetteFalloff",
    ],
    setUniforms: (gl, locs, params) => {
        const p: ColorGradeParams = {
            ...DEFAULT_PARAMS,
            ...(params as Partial<ColorGradeParams>),
        };
        gl.uniform1f(locs.get("uExposure")!, p.exposure);
        gl.uniform1f(locs.get("uContrast")!, p.contrast);
        gl.uniform1f(locs.get("uGamma")!, p.gamma);
        gl.uniform1f(locs.get("uBlackPoint")!, p.blackPoint);
        gl.uniform1f(locs.get("uWhitePoint")!, p.whitePoint);
        gl.uniform1f(locs.get("uSaturation")!, p.saturation);
        gl.uniform1f(locs.get("uVibrance")!, p.vibrance);
        gl.uniform1f(locs.get("uHueShift")!, p.hueShift);
        gl.uniform1f(locs.get("uTemperature")!, p.temperature);
        gl.uniform1f(locs.get("uTint")!, p.tint);
        gl.uniform3f(
            locs.get("uGainRgb")!,
            p.gainR,
            p.gainG,
            p.gainB,
        );
        const sh = tintFromHex(p.shadowsColor);
        const md = tintFromHex(p.midsColor);
        const hi = tintFromHex(p.highlightsColor);
        gl.uniform3f(locs.get("uShadowsTint")!, sh[0], sh[1], sh[2]);
        gl.uniform1f(locs.get("uShadowsAmount")!, p.shadowsAmount);
        gl.uniform3f(locs.get("uMidsTint")!, md[0], md[1], md[2]);
        gl.uniform1f(locs.get("uMidsAmount")!, p.midsAmount);
        gl.uniform3f(locs.get("uHighlightsTint")!, hi[0], hi[1], hi[2]);
        gl.uniform1f(locs.get("uHighlightsAmount")!, p.highlightsAmount);
        gl.uniform1f(locs.get("uVignetteAmount")!, p.vignetteAmount);
        gl.uniform1f(locs.get("uVignetteFalloff")!, p.vignetteFalloff);
    },
};

// ─── Controls ───────────────────────────────────────────────────────────

const SECTION_TITLE_STYLE: CSSProperties = {
    color: "#bdbdbd",
    fontSize: 11,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    marginTop: 12,
    marginBottom: 4,
    paddingTop: 8,
    borderTop: "1px solid #2a2a2a",
};

const FIRST_SECTION_TITLE_STYLE: CSSProperties = {
    ...SECTION_TITLE_STYLE,
    marginTop: 0,
    paddingTop: 0,
    borderTop: "none",
};

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 6,
    marginBottom: 2,
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 8,
};

const COLOR_INPUT_STYLE: CSSProperties = {
    width: 32,
    height: 22,
    padding: 0,
    border: "1px solid #333",
    borderRadius: 4,
    background: "transparent",
    cursor: "pointer",
    flexShrink: 0,
};

const BUTTON_STYLE: CSSProperties = {
    background: "#1a1a1a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "2px 8px",
    fontSize: 11,
    cursor: "pointer",
    fontFamily: "inherit",
};

interface SliderRowProps {
    readonly label: string;
    readonly value: number;
    readonly min: number;
    readonly max: number;
    readonly step: number;
    readonly defaultValue: number;
    readonly format?: (v: number) => string;
    readonly onChange: (v: number) => void;
}

function SliderRow({
    label,
    value,
    min,
    max,
    step,
    defaultValue,
    format,
    onChange,
}: SliderRowProps) {
    const display = format ? format(value) : value.toFixed(2);
    return (
        <>
            <div
                style={{
                    ...LABEL_STYLE,
                    display: "flex",
                    justifyContent: "space-between",
                }}
            >
                <span>{label}</span>
                <span style={{ color: "#bdbdbd" }}>{display}</span>
            </div>
            <div style={ROW_STYLE}>
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    style={{ flex: 1 }}
                />
                <button
                    type="button"
                    style={BUTTON_STYLE}
                    onClick={() => onChange(defaultValue)}
                    title="reset"
                >
                    ↺
                </button>
            </div>
        </>
    );
}

interface TintRowProps {
    readonly label: string;
    readonly color: string;
    readonly amount: number;
    readonly onColorChange: (hex: string) => void;
    readonly onAmountChange: (v: number) => void;
    readonly onReset: () => void;
}

function TintRow({
    label,
    color,
    amount,
    onColorChange,
    onAmountChange,
    onReset,
}: TintRowProps) {
    return (
        <>
            <div
                style={{
                    ...LABEL_STYLE,
                    display: "flex",
                    justifyContent: "space-between",
                }}
            >
                <span>{label}</span>
                <span style={{ color: "#bdbdbd" }}>{amount.toFixed(2)}</span>
            </div>
            <div style={ROW_STYLE}>
                <input
                    type="color"
                    style={COLOR_INPUT_STYLE}
                    value={color}
                    onChange={(e) => onColorChange(e.target.value)}
                    title={`${label} tint (neutral = #808080 = no effect)`}
                />
                <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={amount}
                    onChange={(e) => onAmountChange(parseFloat(e.target.value))}
                    style={{ flex: 1 }}
                />
                <button
                    type="button"
                    style={BUTTON_STYLE}
                    onClick={onReset}
                    title="reset"
                >
                    ↺
                </button>
            </div>
        </>
    );
}

function ColorGradeControls({ params, onChange }: ShaderControlsProps) {
    const cur: ColorGradeParams = {
        ...DEFAULT_PARAMS,
        ...(params as Partial<ColorGradeParams>),
    };
    const update = (patch: Partial<ColorGradeParams>) =>
        onChange({ ...cur, ...patch } satisfies ColorGradeParams);

    return (
        <div>
            <div style={ROW_STYLE}>
                <button
                    type="button"
                    style={{ ...BUTTON_STYLE, marginLeft: "auto" }}
                    onClick={() => onChange(DEFAULT_PARAMS)}
                    title="reset all parameters to defaults"
                >
                    reset all
                </button>
            </div>

            <div style={FIRST_SECTION_TITLE_STYLE}>tone</div>
            <SliderRow
                label="exposure (stops)"
                value={cur.exposure}
                min={-4}
                max={4}
                step={0.05}
                defaultValue={DEFAULT_PARAMS.exposure}
                format={(v) => v.toFixed(2)}
                onChange={(v) => update({ exposure: v })}
            />
            <SliderRow
                label="contrast"
                value={cur.contrast}
                min={0}
                max={2}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.contrast}
                onChange={(v) => update({ contrast: v })}
            />
            <SliderRow
                label="gamma"
                value={cur.gamma}
                min={0.2}
                max={3}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.gamma}
                onChange={(v) => update({ gamma: v })}
            />
            <SliderRow
                label="black point"
                value={cur.blackPoint}
                min={0}
                max={0.5}
                step={0.005}
                defaultValue={DEFAULT_PARAMS.blackPoint}
                onChange={(v) => update({ blackPoint: v })}
            />
            <SliderRow
                label="white point"
                value={cur.whitePoint}
                min={0.5}
                max={1.5}
                step={0.005}
                defaultValue={DEFAULT_PARAMS.whitePoint}
                onChange={(v) => update({ whitePoint: v })}
            />

            <div style={SECTION_TITLE_STYLE}>color</div>
            <SliderRow
                label="saturation"
                value={cur.saturation}
                min={0}
                max={2}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.saturation}
                onChange={(v) => update({ saturation: v })}
            />
            <SliderRow
                label="vibrance"
                value={cur.vibrance}
                min={-1}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.vibrance}
                onChange={(v) => update({ vibrance: v })}
            />
            <SliderRow
                label="hue shift (°)"
                value={cur.hueShift}
                min={-180}
                max={180}
                step={1}
                defaultValue={DEFAULT_PARAMS.hueShift}
                format={(v) => v.toFixed(0)}
                onChange={(v) => update({ hueShift: v })}
            />

            <div style={SECTION_TITLE_STYLE}>white balance</div>
            <SliderRow
                label="temperature (cool ↔ warm)"
                value={cur.temperature}
                min={-1}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.temperature}
                onChange={(v) => update({ temperature: v })}
            />
            <SliderRow
                label="tint (green ↔ magenta)"
                value={cur.tint}
                min={-1}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.tint}
                onChange={(v) => update({ tint: v })}
            />

            <div style={SECTION_TITLE_STYLE}>rgb gain</div>
            <SliderRow
                label="red"
                value={cur.gainR}
                min={0}
                max={2}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.gainR}
                onChange={(v) => update({ gainR: v })}
            />
            <SliderRow
                label="green"
                value={cur.gainG}
                min={0}
                max={2}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.gainG}
                onChange={(v) => update({ gainG: v })}
            />
            <SliderRow
                label="blue"
                value={cur.gainB}
                min={0}
                max={2}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.gainB}
                onChange={(v) => update({ gainB: v })}
            />

            <div style={SECTION_TITLE_STYLE}>shadows · mids · highlights</div>
            <TintRow
                label="shadows"
                color={cur.shadowsColor}
                amount={cur.shadowsAmount}
                onColorChange={(hex) => update({ shadowsColor: hex })}
                onAmountChange={(v) => update({ shadowsAmount: v })}
                onReset={() =>
                    update({
                        shadowsColor: DEFAULT_PARAMS.shadowsColor,
                        shadowsAmount: DEFAULT_PARAMS.shadowsAmount,
                    })
                }
            />
            <TintRow
                label="mids"
                color={cur.midsColor}
                amount={cur.midsAmount}
                onColorChange={(hex) => update({ midsColor: hex })}
                onAmountChange={(v) => update({ midsAmount: v })}
                onReset={() =>
                    update({
                        midsColor: DEFAULT_PARAMS.midsColor,
                        midsAmount: DEFAULT_PARAMS.midsAmount,
                    })
                }
            />
            <TintRow
                label="highlights"
                color={cur.highlightsColor}
                amount={cur.highlightsAmount}
                onColorChange={(hex) => update({ highlightsColor: hex })}
                onAmountChange={(v) => update({ highlightsAmount: v })}
                onReset={() =>
                    update({
                        highlightsColor: DEFAULT_PARAMS.highlightsColor,
                        highlightsAmount: DEFAULT_PARAMS.highlightsAmount,
                    })
                }
            />

            <div style={SECTION_TITLE_STYLE}>vignette</div>
            <SliderRow
                label="amount"
                value={cur.vignetteAmount}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.vignetteAmount}
                onChange={(v) => update({ vignetteAmount: v })}
            />
            <SliderRow
                label="falloff (inner radius)"
                value={cur.vignetteFalloff}
                min={0}
                max={1}
                step={0.01}
                defaultValue={DEFAULT_PARAMS.vignetteFalloff}
                onChange={(v) => update({ vignetteFalloff: v })}
            />
        </div>
    );
}

export const colorGradeEntry: ShaderEntry = {
    id: "colorGrade",
    name: "Color Grade",
    defaultParams: DEFAULT_PARAMS,
    Controls: ColorGradeControls,
    inputs: [{ id: "in", label: "image" }],
    gpu: colorGradeGpuSpec,
};
