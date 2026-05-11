import {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type CSSProperties,
    type SyntheticEvent,
} from "react";
import {
    EdgeDetect,
    SkinMask,
    PaletteMask,
    HairDetect,
    LowPoly,
    Silhouette,
    CircleDetect,
    Posterize,
    loadClassifier,
    loadDetector,
    loadSegmenter,
    loadHumanParser,
    classifySkin,
    parseToVisualizationCanvas,
    colorForParseClass,
    boxCenter,
    findChromaticFocus,
    findHairFocus,
    runPipeline,
    DEFAULT_PARSE_EXCLUSION,
    NUDENET_CLASSES,
    HUMAN_PARSE_CLASSES,
    createDefaultSkinMaskParams,
    createDefaultHairDetectParams,
    createDefaultLowPolyParams,
    createDefaultCircleDetectParams,
    type CircleDetectMode,
    type CircleDetectParams,
    type ClassificationProbabilities,
    type DetectedRegion,
    type HairDetectParams,
    type HumanParser,
    type ImageSegmenter,
    type LowPolyColorMode,
    type LowPolyMode,
    type LowPolyParams,
    type MediaKind,
    type NsfwClassifier,
    type NsfwDetector,
    type PaletteMaskHandle,
    type PaletteMaskMode,
    type ParseProgress,
    type ParseResult,
    type PipelineRegion,
    type RGB,
    type SegmentMask,
    type SegmentScale,
    type SegmenterProgress,
    type SilhouetteHandle,
    type SilhouetteMode,
    type SkinMaskParams,
} from "@newtonedev/shaders";

const PANEL_STYLE: CSSProperties = {
    position: "fixed",
    top: 12,
    left: 12,
    zIndex: 10,
    padding: "12px 14px",
    background: "rgba(15,15,15,0.85)",
    border: "1px solid #2a2a2a",
    borderRadius: 8,
    backdropFilter: "blur(6px)",
    fontSize: 12,
    lineHeight: 1.4,
    minWidth: 280,
    maxWidth: "min(360px, calc(100vw - 24px))",
    maxHeight: "calc(100vh - 24px)",
    overflowY: "auto",
};

const ROW_STYLE: CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: 8,
    margin: "6px 0",
};

const LABEL_STYLE: CSSProperties = {
    flex: "0 0 90px",
    color: "#bdbdbd",
};

const VALUE_STYLE: CSSProperties = {
    flex: "0 0 50px",
    textAlign: "right",
    color: "#f0f0f0",
    fontVariantNumeric: "tabular-nums",
};

const INPUT_STYLE: CSSProperties = {
    flex: 1,
    background: "#0a0a0a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 6px",
    fontSize: 12,
    minWidth: 0,
};

const BUTTON_STYLE: CSSProperties = {
    background: "#1a1a1a",
    color: "#f0f0f0",
    border: "1px solid #333",
    borderRadius: 4,
    padding: "4px 8px",
    fontSize: 12,
    cursor: "pointer",
};

const HINT_STYLE: CSSProperties = {
    color: "#888",
    fontSize: 11,
    marginTop: 8,
};

const STAGE_STYLE: CSSProperties = {
    width: "100vw",
    height: "100vh",
    display: "flex",
};

const SECTION_TITLE: CSSProperties = {
    fontWeight: 600,
    margin: "10px 0 4px",
    color: "#f0f0f0",
};

const DIVIDER: CSSProperties = {
    height: 1,
    background: "#1f1f1f",
    margin: "10px 0",
};

const SWATCH_GRID: CSSProperties = {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: 6,
    marginTop: 6,
};

const SWATCH_BASE: CSSProperties = {
    aspectRatio: "1 / 1",
    borderRadius: 6,
    cursor: "pointer",
    border: "2px solid transparent",
    boxSizing: "border-box",
    transition: "transform 100ms ease, border-color 100ms ease",
    position: "relative",
};

// Auto-list everything in playground/assets/. Vite re-evaluates on file
// add/remove, so dropping a new file appears in the dropdown without a
// page reload.
const ASSET_MODULES = import.meta.glob<string>(
    "./assets/*.{mp4,webm,mov,m4v,ogg,ogv,mkv,png,jpg,jpeg,gif,webp,avif,svg}",
    { query: "?url", import: "default", eager: true },
);
const ASSET_ENTRIES = Object.entries(ASSET_MODULES)
    .map(([key, url]) => ({ name: key.replace(/^.*\//, ""), url }))
    .sort((a, b) => a.name.localeCompare(b.name));

type ShaderKind =
    | "edge"
    | "skin"
    | "palette"
    | "hair"
    | "lowpoly"
    | "silhouette"
    | "circle"
    | "posterize"
    | "classify"
    | "detect"
    | "segment"
    | "skin-map"
    | "human-parse"
    | "focusPt"
    | "manual";

type FocusPtStatus =
    | "idle"
    | "loading-models"
    | "detecting"
    | "encoding"
    | "decoding"
    | "ready"
    | "error";

interface FocusPtHit {
    readonly box: readonly [number, number, number, number];
    readonly class: string;
    readonly score: number;
    /** SAM box-prompt mask for this region (always computed). */
    readonly boxMask: SegmentMask;
    /** Chromatic focus point in source pixels — only for target classes. */
    readonly point: { x: number; y: number } | null;
    /** SAM point-prompt small mask — only for target classes. */
    readonly pointMask: SegmentMask | null;
}

interface FocusClassConfig {
    /**
     * Which focus-search algorithm to use for this class. `"chromatic"` is
     * the default darkness × redness peak; `"hair"` swaps in
     * `hairScoreMap` for synthesized pubic-hair regions. @default "chromatic"
     */
    readonly searchMethod?: "chromatic" | "hair";
    /** Short anatomical label rendered on the canvas next to the crosshair. */
    readonly label: string;
    /**
     * Longer descriptive sub-label rendered under the short label — names
     * the *feature* the chromatic + line search is targeting (e.g. "areola
     * + nipple", "cleft / introitus"). Helps when one anatomical region
     * has multiple plausible foci.
     */
    readonly description?: string;
    /** @see ChromaticFocusOptions.centerBias — passed through. */
    readonly centerBias: number;
    /** Vertical Gaussian-center offset, normalized to box height. */
    readonly centerOffsetY?: number;
    /** Horizontal Gaussian-center offset, normalized to box width. */
    readonly centerOffsetX?: number;
    /**
     * Hair-texture suppression weight, 0..1. Per-class because pubic-hair
     * pollution is dominant for genitalia but rare for clean breast/anus
     * crops. @see ChromaticFocusOptions.hairPenalty
     */
    readonly hairPenalty?: number;
    /**
     * Vertical-line bonus weight, 0..1. Adds the (normalized) Sobel-X
     * magnitude into the chromatic score; useful for closed-vulva cleft
     * targeting where there's no spot peak. @see
     * ChromaticFocusOptions.verticalLineBonus
     */
    readonly verticalLineBonus?: number;
}

/**
 * Per-class config for the chromatic-focus + SAM-point-prompt pipeline.
 * The keys are NudeNet `*_EXPOSED` classes; absence from this map means
 * the class is *not* targeted for focus-point detection. Each entry tunes
 * the chromatic search for its anatomy: tight breast/anus boxes get a
 * neutral center bias, the looser `FEMALE_GENITALIA_EXPOSED` box gets a
 * weaker center prior with a downward offset (introitus is anatomically
 * posterior to the box center on most frontal poses).
 */
const FOCUS_CONFIG: Record<string, FocusClassConfig> = {
    FEMALE_BREAST_EXPOSED: {
        label: "nipple",
        description: "areola + nipple",
        centerBias: 0.4,
        hairPenalty: 0,
    },
    MALE_BREAST_EXPOSED: {
        label: "nipple",
        description: "areola + nipple",
        centerBias: 0.4,
        hairPenalty: 0.3,
    },
    ANUS_EXPOSED: {
        label: "anus",
        description: "anal opening",
        centerBias: 0.4,
        hairPenalty: 0.4,
    },
    FEMALE_GENITALIA_EXPOSED: {
        label: "vulva",
        description: "cleft / introitus",
        centerBias: 0.2,
        centerOffsetY: 0.15,
        hairPenalty: 0.7,
        verticalLineBonus: 0.5,
    },
    MALE_GENITALIA_EXPOSED: {
        label: "penis",
        description: "glans / shaft",
        centerBias: 0.3,
        hairPenalty: 0.6,
    },
    /**
     * Synthetic class — there's no NudeNet `PUBIC_HAIR` label. Regions
     * are derived geometrically from `*_GENITALIA_*` boxes (covered or
     * exposed) by extending upward, then segmented via the hair-density
     * focus search.
     */
    PUBIC_HAIR: {
        searchMethod: "hair",
        label: "pubic hair",
        description: "hair / mons",
        centerBias: 0.25,
    },
};

/**
 * NudeNet classes that trigger synthesis of a `PUBIC_HAIR` region. The
 * pubic mound sits above the genitalia regardless of whether the genitals
 * are covered or exposed, so we accept both states.
 */
const PUBIC_HAIR_TRIGGERS: ReadonlySet<string> = new Set([
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "MALE_GENITALIA_EXPOSED",
]);

interface SyntheticPubicRegion {
    readonly class: "PUBIC_HAIR";
    readonly box: readonly [number, number, number, number];
    readonly score: number;
}

type LandmarkType =
    | "female_nipple"
    | "male_nipple"
    | "female_navel"
    | "male_navel"
    | "female_anus"
    | "male_anus"
    | "clitoris"
    | "introitus"
    | "scrotum"
    | "glans"
    | "shaft"
    | "shaft_base"
    | "female_pubic_hair"
    | "male_pubic_hair"
    | "female_head_hair"
    | "male_head_hair"
    | "female_face"
    | "male_face"
    | "female_hand"
    | "male_hand"
    | "female_feet"
    | "male_feet"
    | "female_armpit"
    | "male_armpit"
    | "female_underwear"
    | "male_underwear";

type LandmarkGroup =
    | "face"
    | "breast"
    | "torso"
    | "limb"
    | "genital"
    | "hair"
    | "clothing";

interface CensorVisual {
    readonly shape: "ellipse";
    /** Fill color (CSS color string). */
    readonly color: string;
    /** 0..1, default 1 (opaque). */
    readonly opacity?: number;
    /**
     * Multiplier on the fitted radii. `1.0` = exact PCA fit (matches the
     * mask's second moment); `>1` pads outward, `<1` shrinks inward.
     * @default 1.0
     */
    readonly scale?: number;
}

interface LandmarkClassConfig {
    readonly label: string;
    readonly description?: string;
    readonly group: LandmarkGroup;
    readonly scale: SegmentScale;
    /**
     * Optional censorship visual rendered on the SVG layer above the SAM
     * mask. Absence = no censor for this landmark type (only the mask
     * shows during debugging).
     */
    readonly censor?: CensorVisual;
}

const RED_ELLIPSE: CensorVisual = { shape: "ellipse", color: "#dc2626" };
const BLACK_ELLIPSE: CensorVisual = { shape: "ellipse", color: "#000000" };
const DARK_ELLIPSE_70: CensorVisual = {
    shape: "ellipse",
    color: "#000000",
    opacity: 0.7,
};

const LANDMARK_CONFIG: Record<LandmarkType, LandmarkClassConfig> = {
    // face
    female_face: { label: "♀ face", group: "face", scale: "largest" },
    male_face: { label: "♂ face", group: "face", scale: "largest" },
    // breast — primary censorship target: red ellipse
    female_nipple: {
        label: "♀ nipple",
        group: "breast",
        scale: "smallest",
        censor: RED_ELLIPSE,
    },
    male_nipple: {
        label: "♂ nipple",
        group: "breast",
        scale: "smallest",
        censor: RED_ELLIPSE,
    },
    // torso
    female_navel: { label: "♀ navel", group: "torso", scale: "smallest" },
    male_navel: { label: "♂ navel", group: "torso", scale: "smallest" },
    // limb
    female_hand: { label: "♀ hand", group: "limb", scale: "largest" },
    male_hand: { label: "♂ hand", group: "limb", scale: "largest" },
    female_feet: { label: "♀ feet", group: "limb", scale: "largest" },
    male_feet: { label: "♂ feet", group: "limb", scale: "largest" },
    female_armpit: {
        label: "♀ armpit",
        group: "limb",
        scale: "smallest",
    },
    male_armpit: { label: "♂ armpit", group: "limb", scale: "smallest" },
    // genital — primary censorship targets
    female_anus: {
        label: "♀ anus",
        group: "genital",
        scale: "smallest",
        censor: RED_ELLIPSE,
    },
    male_anus: {
        label: "♂ anus",
        group: "genital",
        scale: "smallest",
        censor: RED_ELLIPSE,
    },
    clitoris: {
        label: "clitoris",
        group: "genital",
        scale: "smallest",
        censor: RED_ELLIPSE,
    },
    introitus: {
        label: "introitus",
        group: "genital",
        scale: "smallest",
        censor: RED_ELLIPSE,
    },
    scrotum: {
        label: "scrotum",
        group: "genital",
        scale: "medium",
        censor: RED_ELLIPSE,
    },
    glans: {
        label: "glans",
        group: "genital",
        scale: "smallest",
        censor: RED_ELLIPSE,
    },
    shaft: {
        label: "shaft",
        group: "genital",
        scale: "medium",
        censor: RED_ELLIPSE,
    },
    shaft_base: {
        label: "shaft base",
        group: "genital",
        scale: "smallest",
        censor: RED_ELLIPSE,
    },
    // hair — pubic gets covered, head doesn't (defaults; user can edit)
    female_pubic_hair: {
        label: "♀ pubic hair",
        group: "hair",
        scale: "medium",
        censor: BLACK_ELLIPSE,
    },
    male_pubic_hair: {
        label: "♂ pubic hair",
        group: "hair",
        scale: "medium",
        censor: BLACK_ELLIPSE,
    },
    female_head_hair: {
        label: "♀ head hair",
        group: "hair",
        scale: "largest",
    },
    male_head_hair: {
        label: "♂ head hair",
        group: "hair",
        scale: "largest",
    },
    // clothing
    female_underwear: {
        label: "♀ underwear",
        group: "clothing",
        scale: "largest",
    },
    male_underwear: {
        label: "♂ underwear",
        group: "clothing",
        scale: "largest",
    },
};

// (DARK_ELLIPSE_70 reserved for future per-type defaults — exported via
// the closure so future entries can opt in.)
void DARK_ELLIPSE_70;

const LANDMARK_GROUP_ORDER: ReadonlyArray<LandmarkGroup> = [
    "face",
    "breast",
    "torso",
    "limb",
    "genital",
    "hair",
    "clothing",
];

const LANDMARK_GROUP_TITLES: Record<LandmarkGroup, string> = {
    face: "face",
    breast: "breast",
    torso: "torso",
    limb: "limbs",
    genital: "genital",
    hair: "hair",
    clothing: "clothing",
};

const LANDMARK_TYPES_BY_GROUP: Record<
    LandmarkGroup,
    ReadonlyArray<LandmarkType>
> = (() => {
    const out: Record<LandmarkGroup, LandmarkType[]> = {
        face: [],
        breast: [],
        torso: [],
        limb: [],
        genital: [],
        hair: [],
        clothing: [],
    };
    for (const t of Object.keys(LANDMARK_CONFIG) as LandmarkType[]) {
        out[LANDMARK_CONFIG[t].group].push(t);
    }
    return out;
})();

const LANDMARK_DRAG_MIME = "application/landmark-type";

interface ManualLandmark {
    readonly id: string;
    readonly type: LandmarkType;
    readonly x: number;
    readonly y: number;
    readonly mask: SegmentMask | null;
}

type ManualStatus =
    | "idle"
    | "loading-models"
    | "encoding"
    | "decoding"
    | "ready"
    | "error";

function newLandmarkId(): string {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return crypto.randomUUID();
    }
    return `lm-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Derive a pubic-hair region from a genitalia detection: same X span,
 * extended upward by ~80% of the genitalia box height (capped at the
 * image top), with a small overlap into the genitalia top edge to catch
 * hair that bleeds into the upper portion of the labia / mons.
 */
function synthesizePubicHairBox(
    region: { readonly box: readonly [number, number, number, number] },
    imgWidth: number,
    imgHeight: number,
): readonly [number, number, number, number] {
    const [x, y, w, h] = region.box;
    const expandUp = Math.min(y, h * 0.8);
    const overlap = h * 0.2;
    const newY = Math.max(0, y - expandUp);
    const newH = Math.min(imgHeight - newY, expandUp + overlap);
    const newX = Math.max(0, x);
    const newW = Math.min(imgWidth - newX, w);
    return [newX, newY, newW, newH] as const;
}

type ParseStatus =
    | "idle"
    | "loading-model"
    | "parsing"
    | "ready"
    | "error";

type ClassifyStatus =
    | "idle"
    | "loading-model"
    | "classifying"
    | "ready"
    | "error";

type DetectStatus =
    | "idle"
    | "loading-model"
    | "detecting"
    | "ready"
    | "error";

type SegmentStatus =
    | "idle"
    | "loading-models"
    | "detecting"
    | "encoding"
    | "decoding"
    | "ready"
    | "error";

interface SegmentMaskState {
    readonly data: Uint8Array;
    readonly width: number;
    readonly height: number;
}

const DETECT_MODEL_URL = "/nudenet/320n.onnx";
const DETECT_DEFAULT_SCORE = 0.25;
const DETECT_DEFAULT_IOU = 0.45;

interface SliderRowProps {
    label: string;
    value: number;
    onChange: (v: number) => void;
    min: number;
    max: number;
    step: number;
}

function SliderRow({ label, value, onChange, min, max, step }: SliderRowProps) {
    return (
        <div style={ROW_STYLE}>
            <span style={LABEL_STYLE}>{label}</span>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => onChange(Number(e.target.value))}
                style={{ flex: 1 }}
            />
            <span style={VALUE_STYLE}>{value.toFixed(3)}</span>
        </div>
    );
}

function rgbCss([r, g, b]: RGB): string {
    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Convert a mouse / pointer event to source-pixel coordinates inside an
 * `<img>` rendered at any CSS size (handles `object-fit` scaling).
 */
function pixelCoordsFromEvent(
    e: { clientX: number; clientY: number },
    img: HTMLImageElement,
): { x: number; y: number } {
    const rect = img.getBoundingClientRect();
    return {
        x: ((e.clientX - rect.left) / rect.width) * img.naturalWidth,
        y: ((e.clientY - rect.top) / rect.height) * img.naturalHeight,
    };
}

interface FittedEllipse {
    readonly cx: number;
    readonly cy: number;
    readonly rx: number;
    readonly ry: number;
    /** Rotation in radians, counterclockwise from +x. */
    readonly theta: number;
}

/**
 * PCA fit of an ellipse to the "on" pixels of a binary mask. Returns the
 * ellipse with the same centroid and (uniform-density) second moment as
 * the mask. `null` when the mask is empty.
 *
 * The semi-axes use the `2·sqrt(λ)` rule, which is the exact relationship
 * between covariance and semi-axes for a uniform-density ellipse.
 */
function fitEllipseToMask(mask: {
    data: Uint8Array;
    width: number;
    height: number;
}): FittedEllipse | null {
    const { data, width, height } = mask;
    let count = 0;
    let sumX = 0;
    let sumY = 0;
    for (let y = 0; y < height; y++) {
        const row = y * width;
        for (let x = 0; x < width; x++) {
            if (data[row + x]) {
                sumX += x;
                sumY += y;
                count++;
            }
        }
    }
    if (count === 0) return null;
    const cx = sumX / count;
    const cy = sumY / count;

    let sxx = 0;
    let syy = 0;
    let sxy = 0;
    for (let y = 0; y < height; y++) {
        const row = y * width;
        const dy = y - cy;
        for (let x = 0; x < width; x++) {
            if (data[row + x]) {
                const dx = x - cx;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
        }
    }
    sxx /= count;
    syy /= count;
    sxy /= count;

    const trace = sxx + syy;
    const det = sxx * syy - sxy * sxy;
    const disc = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
    const lambda1 = trace / 2 + disc;
    const lambda2 = trace / 2 - disc;

    let theta: number;
    if (Math.abs(sxy) > 1e-10) {
        theta = Math.atan2(lambda1 - sxx, sxy);
    } else {
        theta = sxx >= syy ? 0 : Math.PI / 2;
    }

    return {
        cx,
        cy,
        rx: 2 * Math.sqrt(Math.max(0, lambda1)),
        ry: 2 * Math.sqrt(Math.max(0, lambda2)),
        theta,
    };
}

const PROB_BAR_TRACK: CSSProperties = {
    flex: 1,
    height: 8,
    background: "#1a1a1a",
    borderRadius: 4,
    overflow: "hidden",
    border: "1px solid #2a2a2a",
};

const PROB_LABEL: CSSProperties = {
    flex: "0 0 64px",
    color: "#bdbdbd",
    fontVariantNumeric: "tabular-nums",
};

const PROB_VALUE: CSSProperties = {
    flex: "0 0 56px",
    textAlign: "right",
    color: "#f0f0f0",
    fontVariantNumeric: "tabular-nums",
};

interface ClassifyPanelProps {
    readonly status: ClassifyStatus;
    readonly probs: ClassificationProbabilities | null;
    readonly error: string | null;
}

const PROB_ORDER: ReadonlyArray<{
    key: keyof ClassificationProbabilities;
    label: string;
    color: string;
}> = [
    { key: "porn", label: "porn", color: "#ff5577" },
    { key: "sexy", label: "sexy", color: "#ff9966" },
    { key: "hentai", label: "hentai", color: "#cc66ff" },
    { key: "drawing", label: "drawing", color: "#66aaff" },
    { key: "neutral", label: "neutral", color: "#66cc99" },
];

function ClassifyPanel({ status, probs, error }: ClassifyPanelProps) {
    return (
        <>
            <div style={SECTION_TITLE}>NSFWJS classification</div>
            <div style={{ ...HINT_STYLE, marginTop: 0 }}>
                {status === "loading-model"
                    ? "loading model…"
                    : status === "classifying"
                      ? "classifying…"
                      : status === "ready"
                        ? "ready"
                        : status === "error"
                          ? `error: ${error ?? "unknown"}`
                          : "load an image"}
            </div>
            {probs ? (
                <div style={{ marginTop: 8 }}>
                    {PROB_ORDER.map(({ key, label, color }) => {
                        const v = probs[key];
                        return (
                            <div key={key} style={ROW_STYLE}>
                                <span style={PROB_LABEL}>{label}</span>
                                <div style={PROB_BAR_TRACK}>
                                    <div
                                        style={{
                                            width: `${(v * 100).toFixed(2)}%`,
                                            height: "100%",
                                            background: color,
                                            transition:
                                                "width 120ms ease-out",
                                        }}
                                    />
                                </div>
                                <span style={PROB_VALUE}>
                                    {(v * 100).toFixed(1)}%
                                </span>
                            </div>
                        );
                    })}
                </div>
            ) : null}
            <div style={HINT_STYLE}>
                Static images only. Tier 1 of the detection pipeline — pick
                an asset above to classify it.
            </div>
        </>
    );
}

// Hash a class label to a stable hue so the same body-part class always
// renders in the same color across images and runs.
function hueForClass(name: string): number {
    let h = 0;
    for (let i = 0; i < name.length; i++) {
        h = (h * 31 + name.charCodeAt(i)) >>> 0;
    }
    return h % 360;
}

function colorForClass(name: string): string {
    return `hsl(${hueForClass(name)}, 95%, 60%)`;
}

// Same hash → RGB triple for per-pixel ImageData painting (where CSS
// strings aren't usable). Mirrors `colorForClass`'s 95%/60% saturation
// + lightness pair so canvas-stroke and canvas-fill colors agree.
function rgbForClass(name: string): [number, number, number] {
    const h = hueForClass(name) / 360;
    const s = 0.95;
    const l = 0.6;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
    const m = l - c / 2;
    let r = 0;
    let g = 0;
    let b = 0;
    if (h < 1 / 6) [r, g, b] = [c, x, 0];
    else if (h < 2 / 6) [r, g, b] = [x, c, 0];
    else if (h < 3 / 6) [r, g, b] = [0, c, x];
    else if (h < 4 / 6) [r, g, b] = [0, x, c];
    else if (h < 5 / 6) [r, g, b] = [x, 0, c];
    else [r, g, b] = [c, 0, x];
    return [
        Math.round((r + m) * 255),
        Math.round((g + m) * 255),
        Math.round((b + m) * 255),
    ];
}

interface DetectPanelProps {
    readonly status: DetectStatus;
    readonly regions: ReadonlyArray<DetectedRegion>;
    readonly error: string | null;
    readonly scoreThreshold: number;
    readonly onScoreThresholdChange: (v: number) => void;
}

function DetectPanel({
    status,
    regions,
    error,
    scoreThreshold,
    onScoreThresholdChange,
}: DetectPanelProps) {
    return (
        <>
            <div style={SECTION_TITLE}>NudeNet detection</div>
            <div style={{ ...HINT_STYLE, marginTop: 0 }}>
                {status === "loading-model"
                    ? "loading model…"
                    : status === "detecting"
                      ? "detecting…"
                      : status === "ready"
                        ? `${regions.length} region${regions.length === 1 ? "" : "s"}`
                        : status === "error"
                          ? `error: ${error ?? "unknown"}`
                          : "load an image"}
            </div>
            <SliderRow
                label="score ≥"
                value={scoreThreshold}
                onChange={onScoreThresholdChange}
                min={0.05}
                max={0.9}
                step={0.01}
            />
            {regions.length > 0 ? (
                <div style={{ marginTop: 8 }}>
                    {regions.map((r, i) => (
                        <div
                            key={`${i}-${r.class}`}
                            style={{
                                ...ROW_STYLE,
                                margin: "4px 0",
                                gap: 6,
                            }}
                        >
                            <span
                                style={{
                                    flex: "0 0 12px",
                                    height: 12,
                                    borderRadius: 3,
                                    background: colorForClass(r.class),
                                }}
                            />
                            <span
                                style={{
                                    flex: 1,
                                    color: "#bdbdbd",
                                    fontSize: 11,
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                    whiteSpace: "nowrap",
                                }}
                                title={r.class}
                            >
                                {r.class}
                            </span>
                            <span
                                style={{
                                    flex: "0 0 44px",
                                    textAlign: "right",
                                    color: "#f0f0f0",
                                    fontSize: 11,
                                    fontVariantNumeric: "tabular-nums",
                                }}
                            >
                                {(r.score * 100).toFixed(1)}%
                            </span>
                        </div>
                    ))}
                </div>
            ) : null}
            <div style={HINT_STYLE}>
                Tier 2 — bounding boxes from NudeNet 320n. Boxes feed SAM
                as prompts in tier 3. {NUDENET_CLASSES.length} classes total.
            </div>
        </>
    );
}

interface SegmentPanelProps {
    readonly status: SegmentStatus;
    readonly progress: SegmenterProgress | null;
    readonly stageInfo: string | null;
    readonly error: string | null;
    readonly regions: ReadonlyArray<PipelineRegion>;
    readonly hasMask: boolean;
    readonly onRerun: () => void;
    readonly skinFilter: boolean;
    readonly onSkinFilterChange: (v: boolean) => void;
    readonly skinThreshold: number;
    readonly onSkinThresholdChange: (v: number) => void;
    readonly parseFilter: boolean;
    readonly onParseFilterChange: (v: boolean) => void;
    readonly parserLoading: boolean;
}

function SegmentPanel({
    status,
    progress,
    stageInfo,
    error,
    regions,
    hasMask,
    onRerun,
    skinFilter,
    onSkinFilterChange,
    skinThreshold,
    onSkinThresholdChange,
    parseFilter,
    onParseFilterChange,
    parserLoading,
}: SegmentPanelProps) {
    const progressLine = progress
        ? progress.progress != null
            ? `${progress.status} ${progress.file ?? ""} ${(
                  progress.progress as number
              ).toFixed(0)}%`
            : `${progress.status} ${progress.file ?? ""}`
        : null;
    const summary =
        status === "loading-models"
            ? progressLine ?? "loading models… (~150MB first run)"
            : status === "detecting"
              ? "detecting regions…"
              : status === "encoding"
                ? "encoding image…"
                : status === "decoding"
                  ? `segmenting ${stageInfo ?? "regions"}…`
                  : status === "ready"
                    ? hasMask
                        ? `${regions.length} mask${regions.length === 1 ? "" : "s"} ready`
                        : "no NSFW regions found"
                    : status === "error"
                      ? `error: ${error ?? "unknown"}`
                      : "load an image";
    return (
        <>
            <div style={SECTION_TITLE}>SAM 2 segmentation</div>
            <div style={{ ...HINT_STYLE, marginTop: 0 }}>{summary}</div>
            <div style={{ ...ROW_STYLE, marginTop: 8 }}>
                <button
                    type="button"
                    onClick={onRerun}
                    disabled={
                        status === "loading-models" ||
                        status === "detecting" ||
                        status === "encoding" ||
                        status === "decoding"
                    }
                    style={{ ...BUTTON_STYLE, flex: 1 }}
                >
                    re-run pipeline
                </button>
            </div>
            <div style={ROW_STYLE}>
                <span style={LABEL_STYLE}>skin filter</span>
                <button
                    type="button"
                    onClick={() => onSkinFilterChange(!skinFilter)}
                    style={{
                        ...BUTTON_STYLE,
                        flex: 1,
                        background: skinFilter ? "#2a2a2a" : "#0a0a0a",
                        borderColor: skinFilter ? "#4a8" : "#333",
                        color: skinFilter ? "#9fd4b3" : "#888",
                    }}
                >
                    {skinFilter ? "on (SAM ∩ skin)" : "off (raw SAM)"}
                </button>
            </div>
            {skinFilter ? (
                <SliderRow
                    label="skin ≥"
                    value={skinThreshold}
                    onChange={onSkinThresholdChange}
                    min={0.05}
                    max={0.95}
                    step={0.01}
                />
            ) : null}
            <div style={ROW_STYLE}>
                <span style={LABEL_STYLE}>parse filter</span>
                <button
                    type="button"
                    onClick={() => onParseFilterChange(!parseFilter)}
                    style={{
                        ...BUTTON_STYLE,
                        flex: 1,
                        background: parseFilter ? "#2a2a2a" : "#0a0a0a",
                        borderColor: parseFilter ? "#4a8" : "#333",
                        color: parseFilter ? "#9fd4b3" : "#888",
                    }}
                >
                    {parseFilter
                        ? parserLoading
                            ? "loading parser…"
                            : "on (− accessories / face / hair)"
                        : "off"}
                </button>
            </div>
            {regions.length > 0 ? (
                <div style={{ marginTop: 8 }}>
                    {regions.map((r, i) => (
                        <div
                            key={`${i}-${r.class}`}
                            style={{
                                ...ROW_STYLE,
                                margin: "4px 0",
                                gap: 6,
                            }}
                        >
                            <span
                                style={{
                                    flex: "0 0 12px",
                                    height: 12,
                                    borderRadius: 3,
                                    background: colorForClass(r.class),
                                }}
                            />
                            <span
                                style={{
                                    flex: 1,
                                    color: "#bdbdbd",
                                    fontSize: 11,
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                    whiteSpace: "nowrap",
                                }}
                                title={r.class}
                            >
                                {r.class}
                            </span>
                            <span
                                style={{
                                    flex: "0 0 60px",
                                    textAlign: "right",
                                    color: "#f0f0f0",
                                    fontSize: 11,
                                    fontVariantNumeric: "tabular-nums",
                                }}
                                title="detect score / SAM IoU"
                            >
                                {(r.score * 100).toFixed(0)}/
                                {(r.maskScore * 100).toFixed(0)}%
                            </span>
                        </div>
                    ))}
                </div>
            ) : null}
            <div style={HINT_STYLE}>
                Pipeline: NudeNet boxes ➜ SAM 2 mask decoder ➜ unioned mask.
                Encoder runs once per image, decoder once per detected box.
            </div>
        </>
    );
}

const DEFAULT_SRC = ASSET_ENTRIES[0]?.url ?? "";
const DEFAULT_PALETTE_SIZE = 8;
const PALETTE_SAMPLE_SEED = 42;

export function App() {
    const [advancedMode, setAdvancedMode] = useState(false);
    const [shader, setShader] = useState<ShaderKind>("focusPt");
    const [src, setSrc] = useState(DEFAULT_SRC);
    const [label, setLabel] = useState(
        ASSET_ENTRIES[0]?.name ?? "(no asset selected)",
    );
    const [draftSrc, setDraftSrc] = useState(DEFAULT_SRC);
    const [forcedKind, setForcedKind] = useState<MediaKind | undefined>(
        undefined,
    );
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Edge knobs
    const [edgeStrength, setEdgeStrength] = useState(1);
    const [edgeThreshold, setEdgeThreshold] = useState(0);
    const [edgeKnee, setEdgeKnee] = useState(1);

    // Skin knobs
    const [skinParams, setSkinParams] = useState<SkinMaskParams>(() =>
        createDefaultSkinMaskParams(),
    );
    const setSkin = (key: keyof SkinMaskParams) => (v: number) =>
        setSkinParams((p) => ({ ...p, [key]: v }));

    // Hair knobs
    const [hairParams, setHairParams] = useState<HairDetectParams>(() =>
        createDefaultHairDetectParams(),
    );
    const setHair = (key: keyof HairDetectParams) => (v: number) =>
        setHairParams((p) => ({ ...p, [key]: v }));

    // Low-poly knobs
    const [lowPolyParams, setLowPolyParams] = useState<LowPolyParams>(() =>
        createDefaultLowPolyParams(),
    );
    const setLowPolyNum = (key: keyof LowPolyParams) => (v: number) =>
        setLowPolyParams((p) => ({ ...p, [key]: v }));

    // Circle detect knobs
    const [circleParams, setCircleParams] = useState<CircleDetectParams>(() =>
        createDefaultCircleDetectParams(),
    );
    const setCircleNum = (key: keyof CircleDetectParams) => (v: number) =>
        setCircleParams((p) => ({ ...p, [key]: v }));
    const [circleStroke, setCircleStroke] = useState<RGB>([255, 80, 120]);
    const [circleStrokeWidth, setCircleStrokeWidth] = useState(2);
    const [circleStrokeOpacity, setCircleStrokeOpacity] = useState(1);
    const [circleFill, setCircleFill] = useState<RGB | null>(null);
    const [circleFillOpacity, setCircleFillOpacity] = useState(0.15);
    const [circleMaxCircles, setCircleMaxCircles] = useState(32);
    const [circleReadbackEvery, setCircleReadbackEvery] = useState(1);

    // Posterize knobs (single control: levels per RGB channel).
    const [posterizeLevels, setPosterizeLevels] = useState(4);

    // Classify (tier 1) state. The model is loaded lazily on first entry to
    // classify mode and reused across image swaps.
    const classifierRef = useRef<NsfwClassifier | null>(null);
    const [classifyStatus, setClassifyStatus] = useState<ClassifyStatus>("idle");
    const [classifyProbs, setClassifyProbs] =
        useState<ClassificationProbabilities | null>(null);
    const [classifyError, setClassifyError] = useState<string | null>(null);

    // Detect (tier 2) state. NudeNet model + last-seen regions for the
    // currently loaded image. The score-threshold slider re-runs the full
    // detect pipeline (no postprocess-only path is exposed).
    const detectorRef = useRef<NsfwDetector | null>(null);
    const detectImgRef = useRef<HTMLImageElement>(null);
    const detectCanvasRef = useRef<HTMLCanvasElement>(null);
    const [detectStatus, setDetectStatus] = useState<DetectStatus>("idle");
    const [detectRegions, setDetectRegions] = useState<DetectedRegion[]>([]);
    const [detectError, setDetectError] = useState<string | null>(null);
    const [detectScore, setDetectScore] = useState(DETECT_DEFAULT_SCORE);

    // Segment (tier 3) state. SAM 2 model is heavy (~150MB on first load)
    // so progress is surfaced. The pipeline orchestration is detect ➜
    // setImage (encoder) ➜ segmentBox per detected region ➜ compose. The
    // detector is shared with detect mode and lazy-loaded on first need.
    const segmenterRef = useRef<ImageSegmenter | null>(null);
    const segmentImgRef = useRef<HTMLImageElement>(null);
    const segmentCanvasRef = useRef<HTMLCanvasElement>(null);
    const segmentBoxCanvasRef = useRef<HTMLCanvasElement>(null);
    const [segmentStatus, setSegmentStatus] = useState<SegmentStatus>("idle");
    const [segmentMask, setSegmentMask] = useState<SegmentMaskState | null>(
        null,
    );
    const [segmentRegions, setSegmentRegions] = useState<PipelineRegion[]>([]);
    const [segmentError, setSegmentError] = useState<string | null>(null);
    const [segmentProgress, setSegmentProgress] =
        useState<SegmenterProgress | null>(null);
    const [segmentStageInfo, setSegmentStageInfo] = useState<string | null>(
        null,
    );
    const [detectorReady, setDetectorReady] = useState(
        () => detectorRef.current !== null,
    );
    const [segmenterReady, setSegmenterReady] = useState(false);
    const [segmentSkinFilter, setSegmentSkinFilter] = useState(true);
    const [skinThreshold, setSkinThreshold] = useState(0.5);
    const [segmentParseFilter, setSegmentParseFilter] = useState(true);
    const segmentProcessedSrcRef = useRef<string | null>(null);

    // Skin-map (JS classifier) mode — standalone preview of what the segment
    // pipeline's skin filter sees, sharing `skinThreshold` with segment mode.
    const skinMapImgRef = useRef<HTMLImageElement>(null);
    const skinMapCanvasRef = useRef<HTMLCanvasElement>(null);
    const [skinMapCoverage, setSkinMapCoverage] = useState<number | null>(null);

    // FocusPt mode — for each FEMALE_BREAST_EXPOSED region, feed the box
    // center to SAM as a positive point prompt and pick the smallest of
    // SAM's three multi-mask outputs (areola/focusPt scale).
    const focusPtImgRef = useRef<HTMLImageElement>(null);
    const focusPtBoxMaskCanvasRef = useRef<HTMLCanvasElement>(null);
    const focusPtMaskCanvasRef = useRef<HTMLCanvasElement>(null);
    const focusPtPointCanvasRef = useRef<HTMLCanvasElement>(null);
    const [focusPtStatus, setFocusPtStatus] = useState<FocusPtStatus>("idle");
    const [focusPtHits, setFocusPtHits] = useState<FocusPtHit[]>([]);
    const [focusPtError, setFocusPtError] = useState<string | null>(null);
    const [focusPtStageInfo, setFocusPtStageInfo] = useState<string | null>(null);
    const [focusPtScale, setFocusPtScale] = useState<SegmentScale>("smallest");
    const [focusPtDraggingIdx, setFocusPtDraggingIdx] = useState<
        number | null
    >(null);
    const focusPtProcessedSrcRef = useRef<string | null>(null);

    // Manual landmark mode — user drops named anatomical landmarks onto the
    // image, then runs SAM 2 against each as a positive point prompt.
    // Shares `segmenterRef` with focusPt mode so the encoder cache crosses
    // mode switches.
    const manualImgRef = useRef<HTMLImageElement>(null);
    const manualMaskCanvasRef = useRef<HTMLCanvasElement>(null);
    const manualOverlayCanvasRef = useRef<HTMLCanvasElement>(null);
    const [manualLandmarks, setManualLandmarks] = useState<ManualLandmark[]>(
        [],
    );
    const [manualDraggingId, setManualDraggingId] = useState<string | null>(
        null,
    );
    const [manualStatus, setManualStatus] = useState<ManualStatus>("idle");
    const [manualError, setManualError] = useState<string | null>(null);
    const manualEncodedSrcRef = useRef<string | null>(null);

    // Human-parse mode (Phase A) — visualizes per-pixel class assignments
    // from `mattmdjaga/segformer_b2_clothes`. Standalone view, no pipeline
    // integration yet.
    const parserRef = useRef<HumanParser | null>(null);
    const parseImgRef = useRef<HTMLImageElement>(null);
    const parseCanvasRef = useRef<HTMLCanvasElement>(null);
    const [parseStatus, setParseStatus] = useState<ParseStatus>("idle");
    const [parseResult, setParseResult] = useState<ParseResult | null>(null);
    const [parseError, setParseError] = useState<string | null>(null);
    const [parseProgress, setParseProgress] = useState<ParseProgress | null>(
        null,
    );
    const [parserReady, setParserReady] = useState(false);

    // Silhouette state
    const silhouetteRef = useRef<SilhouetteHandle>(null);
    const [silReference, setSilReference] = useState<RGB>([255, 255, 255]);
    const [silSmooth, setSilSmooth] = useState(4);
    const [silThreshold, setSilThreshold] = useState(0.18);
    const [silFeather, setSilFeather] = useState(0.04);
    const [silSpread, setSilSpread] = useState(0.1);
    const [silOutline, setSilOutline] = useState(1.5);
    const [silMode, setSilMode] = useState<SilhouetteMode>("outline");
    const [silOffMix, setSilOffMix] = useState(0.15);

    const sampleBackground = () => {
        const result = silhouetteRef.current?.sampleBackground();
        if (result) setSilReference(result);
        return !!result;
    };

    // Palette state
    const paletteRef = useRef<PaletteMaskHandle>(null);
    const [paletteSize, setPaletteSize] = useState(DEFAULT_PALETTE_SIZE);
    const [palette, setPalette] = useState<RGB[]>([]);
    const [enabled, setEnabled] = useState<boolean[]>([]);
    const [paletteMode, setPaletteMode] =
        useState<PaletteMaskMode>("posterize");
    const [offMix, setOffMix] = useState(0.15);

    const samplePalette = (size = paletteSize) => {
        const result = paletteRef.current?.samplePalette({
            paletteSize: size,
            seed: PALETTE_SAMPLE_SEED,
        });
        if (!result) return false;
        setPalette(result);
        setEnabled(result.map(() => true));
        return true;
    };

    // Auto-sample after src/size changes once the texture is ready. Retry
    // briefly to wait out video metadata loading.
    useEffect(() => {
        if (shader !== "palette") return;
        let cancelled = false;
        const tryAt = (delay: number, retriesLeft: number) => {
            const t = window.setTimeout(() => {
                if (cancelled) return;
                if (!samplePalette() && retriesLeft > 0) {
                    tryAt(800, retriesLeft - 1);
                }
            }, delay);
            return () => window.clearTimeout(t);
        };
        const cleanup = tryAt(400, 3);
        return () => {
            cancelled = true;
            cleanup();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [src, paletteSize, shader]);

    // Auto-sample background when entering silhouette mode or src changes.
    useEffect(() => {
        if (shader !== "silhouette") return;
        let cancelled = false;
        const tryAt = (delay: number, retriesLeft: number) => {
            const t = window.setTimeout(() => {
                if (cancelled) return;
                if (!sampleBackground() && retriesLeft > 0) {
                    tryAt(800, retriesLeft - 1);
                }
            }, delay);
            return () => window.clearTimeout(t);
        };
        const cleanup = tryAt(400, 3);
        return () => {
            cancelled = true;
            cleanup();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [src, shader]);

    // Revoke any prior blob URL when src changes / on unmount.
    useEffect(() => {
        return () => {
            if (src.startsWith("blob:")) URL.revokeObjectURL(src);
        };
    }, [src]);

    // Lazy-load the NSFW model on first entry to classify mode. Cached for
    // the lifetime of the page.
    useEffect(() => {
        if (shader !== "classify") return;
        if (classifierRef.current) return;
        let cancelled = false;
        setClassifyStatus("loading-model");
        setClassifyError(null);
        loadClassifier()
            .then((c) => {
                if (cancelled) {
                    c.dispose();
                    return;
                }
                classifierRef.current = c;
                setClassifyStatus("idle");
            })
            .catch((e: unknown) => {
                if (cancelled) return;
                setClassifyError(e instanceof Error ? e.message : String(e));
                setClassifyStatus("error");
            });
        return () => {
            cancelled = true;
        };
    }, [shader]);

    // Drop stale probabilities when the source swaps.
    useEffect(() => {
        setClassifyProbs(null);
    }, [src]);

    const onClassifyImgLoad = (e: SyntheticEvent<HTMLImageElement>) => {
        const c = classifierRef.current;
        if (!c) return;
        const img = e.currentTarget;
        setClassifyStatus("classifying");
        setClassifyError(null);
        c.classify(img)
            .then((probs) => {
                setClassifyProbs(probs);
                setClassifyStatus("ready");
            })
            .catch((err: unknown) => {
                setClassifyError(
                    err instanceof Error ? err.message : String(err),
                );
                setClassifyStatus("error");
            });
    };

    const onClassifyImgError = () => {
        setClassifyError("image failed to load");
        setClassifyStatus("error");
    };

    // Lazy-load the NudeNet detector on first entry to detect, segment, or
    // focusPt mode (all three use NudeNet boxes as their starting point).
    useEffect(() => {
        if (
            shader !== "detect" &&
            shader !== "segment" &&
            shader !== "focusPt"
        )
            return;
        if (detectorRef.current) {
            setDetectorReady(true);
            return;
        }
        let cancelled = false;
        setDetectStatus("loading-model");
        setDetectError(null);
        loadDetector({ modelUrl: DETECT_MODEL_URL })
            .then((d) => {
                if (cancelled) {
                    d.dispose();
                    return;
                }
                detectorRef.current = d;
                setDetectorReady(true);
                setDetectStatus("idle");
            })
            .catch((e: unknown) => {
                if (cancelled) return;
                setDetectError(e instanceof Error ? e.message : String(e));
                setDetectStatus("error");
            });
        return () => {
            cancelled = true;
        };
    }, [shader]);

    // Drop stale regions when the source swaps.
    useEffect(() => {
        setDetectRegions([]);
    }, [src]);

    const runDetect = (img: HTMLImageElement) => {
        const d = detectorRef.current;
        if (!d) return;
        setDetectStatus("detecting");
        setDetectError(null);
        d.detect(img, { scoreThreshold: detectScore })
            .then((regions) => {
                setDetectRegions(regions);
                setDetectStatus("ready");
            })
            .catch((err: unknown) => {
                setDetectError(
                    err instanceof Error ? err.message : String(err),
                );
                setDetectStatus("error");
            });
    };

    const onDetectImgLoad = (e: SyntheticEvent<HTMLImageElement>) => {
        runDetect(e.currentTarget);
    };

    const onDetectImgError = () => {
        setDetectError("image failed to load");
        setDetectStatus("error");
    };

    // Re-run detection when the score threshold changes (debounced).
    useEffect(() => {
        if (shader !== "detect") return;
        const img = detectImgRef.current;
        if (!img || !img.complete || !img.naturalWidth) return;
        if (!detectorRef.current) return;
        const t = window.setTimeout(() => runDetect(img), 150);
        return () => window.clearTimeout(t);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [detectScore]);

    // Lazy-load SAM 2 on first entry to segment, focusPt, or manual mode
    // (all use the segmenter's encoder cache). ~150MB download on first
    // run, browser-cached after.
    useEffect(() => {
        if (
            shader !== "segment" &&
            shader !== "focusPt" &&
            shader !== "manual"
        )
            return;
        if (segmenterRef.current) {
            setSegmenterReady(true);
            return;
        }
        let cancelled = false;
        setSegmentStatus("loading-models");
        setSegmentError(null);
        setSegmentProgress(null);
        loadSegmenter({
            onProgress: (e) => {
                if (cancelled) return;
                setSegmentProgress(e);
            },
        })
            .then((s) => {
                if (cancelled) {
                    s.dispose();
                    return;
                }
                segmenterRef.current = s;
                setSegmenterReady(true);
                setSegmentProgress(null);
                setSegmentStatus("idle");
            })
            .catch((e: unknown) => {
                if (cancelled) return;
                setSegmentError(e instanceof Error ? e.message : String(e));
                setSegmentStatus("error");
            });
        return () => {
            cancelled = true;
        };
    }, [shader]);

    // Drop stale results when the source swaps.
    useEffect(() => {
        setSegmentMask(null);
        setSegmentRegions([]);
        setSegmentStageInfo(null);
        segmentProcessedSrcRef.current = null;
    }, [src]);

    const runSegmentPipeline = useCallback(
        (img: HTMLImageElement) => {
            const detector = detectorRef.current;
            const segmenter = segmenterRef.current;
            if (!detector || !segmenter) return;
            setSegmentError(null);
            setSegmentMask(null);
            setSegmentRegions([]);
            setSegmentStageInfo(null);
            setSegmentStatus("detecting");
            const parser =
                segmentParseFilter && parserRef.current
                    ? parserRef.current
                    : null;
            runPipeline(
                img,
                { detector, segmenter },
                {
                    detectScoreThreshold: detectScore,
                    skinMask: segmentSkinFilter,
                    skinThreshold: skinThreshold,
                    parseExclusion: parser
                        ? {
                              parser,
                              classes: DEFAULT_PARSE_EXCLUSION,
                          }
                        : undefined,
                    onStage: (stage) => {
                        switch (stage.kind) {
                            case "classifying":
                            case "detecting":
                                setSegmentStatus("detecting");
                                setSegmentStageInfo(null);
                                break;
                            case "parsing":
                                setSegmentStatus("encoding");
                                setSegmentStageInfo("running human parser");
                                break;
                            case "skin-classifying":
                                setSegmentStatus("encoding");
                                setSegmentStageInfo("computing skin map");
                                break;
                            case "encoding":
                                setSegmentStatus("encoding");
                                setSegmentStageInfo(null);
                                break;
                            case "segmenting":
                                setSegmentStatus("decoding");
                                setSegmentStageInfo(
                                    `${stage.region.class} (${stage.index + 1}/${stage.total})`,
                                );
                                break;
                            case "composing":
                                setSegmentStatus("decoding");
                                setSegmentStageInfo("composing combined mask");
                                break;
                            case "done":
                                setSegmentStageInfo(null);
                                break;
                        }
                    },
                },
            )
                .then((result) => {
                    setSegmentRegions([...result.regions]);
                    setSegmentMask({
                        data: result.combinedMask.data,
                        width: result.combinedMask.width,
                        height: result.combinedMask.height,
                    });
                    setSegmentStatus("ready");
                })
                .catch((err: unknown) => {
                    setSegmentError(
                        err instanceof Error ? err.message : String(err),
                    );
                    setSegmentStatus("error");
                });
        },
        [detectScore, segmentSkinFilter, skinThreshold, segmentParseFilter],
    );

    // Trigger the pipeline once the image is loaded AND every required
    // model is ready (parser only required when its filter is enabled).
    // Exactly one trigger per `src` — deduped via a ref.
    useEffect(() => {
        if (shader !== "segment") return;
        if (!detectorReady || !segmenterReady) return;
        if (segmentParseFilter && !parserReady) return;
        const img = segmentImgRef.current;
        if (!img || !img.complete || !img.naturalWidth) return;
        if (segmentProcessedSrcRef.current === src) return;
        segmentProcessedSrcRef.current = src;
        runSegmentPipeline(img);
    }, [
        shader,
        src,
        detectorReady,
        segmenterReady,
        parserReady,
        segmentParseFilter,
        runSegmentPipeline,
    ]);

    const onSegmentImgLoad = (e: SyntheticEvent<HTMLImageElement>) => {
        if (!detectorReady || !segmenterReady) return;
        if (segmentParseFilter && !parserReady) return;
        if (segmentProcessedSrcRef.current === src) return;
        segmentProcessedSrcRef.current = src;
        runSegmentPipeline(e.currentTarget);
    };

    const onSegmentImgError = () => {
        setSegmentError("image failed to load");
        setSegmentStatus("error");
    };

    const segmentRerun = () => {
        const img = segmentImgRef.current;
        if (!img || !img.complete) return;
        segmentProcessedSrcRef.current = src;
        runSegmentPipeline(img);
    };

    // Drop stale focusPt results when the source swaps.
    useEffect(() => {
        setFocusPtHits([]);
        setFocusPtStageInfo(null);
        focusPtProcessedSrcRef.current = null;
    }, [src]);

    const runFocusPtPipeline = useCallback(
        (img: HTMLImageElement) => {
            const detector = detectorRef.current;
            const segmenter = segmenterRef.current;
            if (!detector || !segmenter) return;
            setFocusPtError(null);
            setFocusPtHits([]);
            setFocusPtStageInfo(null);
            setFocusPtStatus("detecting");
            (async () => {
                try {
                    const regions = await detector.detect(img, {
                        scoreThreshold: detectScore,
                    });
                    // Synthesize pubic-hair regions from any genitalia
                    // detections (covered or exposed) before walking the
                    // unified pipeline. Pure JS — no extra detector pass.
                    const synthetic: SyntheticPubicRegion[] = [];
                    for (const r of regions) {
                        if (PUBIC_HAIR_TRIGGERS.has(r.class)) {
                            synthetic.push({
                                class: "PUBIC_HAIR",
                                box: synthesizePubicHairBox(
                                    r,
                                    img.naturalWidth,
                                    img.naturalHeight,
                                ),
                                score: r.score,
                            });
                        }
                    }
                    const allRegions: ReadonlyArray<{
                        readonly class: string;
                        readonly box: readonly [
                            number,
                            number,
                            number,
                            number,
                        ];
                        readonly score: number;
                    }> = [...regions, ...synthetic];

                    if (allRegions.length === 0) {
                        setFocusPtStatus("ready");
                        return;
                    }
                    setFocusPtStatus("encoding");
                    await segmenter.setImage(img);
                    setFocusPtStatus("decoding");
                    const hits: FocusPtHit[] = [];
                    for (let i = 0; i < allRegions.length; i++) {
                        const r = allRegions[i];
                        setFocusPtStageInfo(
                            `${r.class} (${i + 1}/${allRegions.length})`,
                        );
                        const boxMask = await segmenter.segmentBox(r.box);
                        const cfg = FOCUS_CONFIG[r.class];
                        let point: FocusPtHit["point"] = null;
                        let pointMask: FocusPtHit["pointMask"] = null;
                        if (cfg) {
                            const candidates =
                                cfg.searchMethod === "hair"
                                    ? findHairFocus(img, {
                                          box: r.box,
                                          centerBias: cfg.centerBias,
                                          centerOffsetX: cfg.centerOffsetX,
                                          centerOffsetY: cfg.centerOffsetY,
                                      })
                                    : findChromaticFocus(img, {
                                          box: r.box,
                                          centerBias: cfg.centerBias,
                                          centerOffsetX: cfg.centerOffsetX,
                                          centerOffsetY: cfg.centerOffsetY,
                                          hairPenalty: cfg.hairPenalty,
                                          verticalLineBonus:
                                              cfg.verticalLineBonus,
                                      });
                            const fp = candidates[0] ?? boxCenter(r.box);
                            pointMask = await segmenter.segmentPoint(
                                [fp.x, fp.y],
                                { scale: focusPtScale },
                            );
                            point = { x: fp.x, y: fp.y };
                        }
                        hits.push({
                            box: r.box,
                            class: r.class,
                            score: r.score,
                            boxMask,
                            point,
                            pointMask,
                        });
                    }
                    setFocusPtHits(hits);
                    setFocusPtStageInfo(null);
                    setFocusPtStatus("ready");
                } catch (err: unknown) {
                    setFocusPtError(
                        err instanceof Error ? err.message : String(err),
                    );
                    setFocusPtStatus("error");
                }
            })();
        },
        // Per-class config drives the chromatic search; the global
        // `focusPtCenterBias` slider only exists in advanced mode.
        [detectScore, focusPtScale],
    );

    // Trigger focusPt pipeline once the image + detector + segmenter are
    // ready. Deduped per `src`; re-runs on `focusPtScale` change.
    useEffect(() => {
        if (shader !== "focusPt") return;
        if (!detectorReady || !segmenterReady) return;
        const img = focusPtImgRef.current;
        if (!img || !img.complete || !img.naturalWidth) return;
        if (focusPtProcessedSrcRef.current === src) return;
        focusPtProcessedSrcRef.current = src;
        runFocusPtPipeline(img);
    }, [shader, src, detectorReady, segmenterReady, runFocusPtPipeline]);

    const onFocusPtImgLoad = (e: SyntheticEvent<HTMLImageElement>) => {
        if (!detectorReady || !segmenterReady) return;
        if (focusPtProcessedSrcRef.current === src) return;
        focusPtProcessedSrcRef.current = src;
        runFocusPtPipeline(e.currentTarget);
    };

    const onFocusPtImgError = () => {
        setFocusPtError("image failed to load");
        setFocusPtStatus("error");
    };

    const onFocusPtMouseDown = (
        e: SyntheticEvent<HTMLDivElement, MouseEvent>,
    ) => {
        const img = focusPtImgRef.current;
        if (!img || !img.naturalWidth) return;
        const p = pixelCoordsFromEvent(e.nativeEvent, img);
        // Pixel hit radius scales with image size so big images don't need
        // pixel-perfect aim. ~2% of width, with a minimum.
        const hitRadius = Math.max(20, img.naturalWidth * 0.02);
        let bestIdx = -1;
        let bestDist2 = hitRadius * hitRadius;
        for (let i = 0; i < focusPtHits.length; i++) {
            const h = focusPtHits[i];
            if (!h.point) continue;
            const dx = h.point.x - p.x;
            const dy = h.point.y - p.y;
            const d2 = dx * dx + dy * dy;
            if (d2 <= bestDist2) {
                bestDist2 = d2;
                bestIdx = i;
            }
        }
        if (bestIdx >= 0) {
            setFocusPtDraggingIdx(bestIdx);
            e.preventDefault();
        }
    };

    const onFocusPtMouseMove = (
        e: SyntheticEvent<HTMLDivElement, MouseEvent>,
    ) => {
        if (focusPtDraggingIdx === null) return;
        const img = focusPtImgRef.current;
        if (!img) return;
        const p = pixelCoordsFromEvent(e.nativeEvent, img);
        const idx = focusPtDraggingIdx;
        setFocusPtHits((hits) =>
            hits.map((h, i) =>
                i === idx ? { ...h, point: { x: p.x, y: p.y } } : h,
            ),
        );
    };

    const onFocusPtMouseUp = (
        e: SyntheticEvent<HTMLDivElement, MouseEvent>,
    ) => {
        if (focusPtDraggingIdx === null) return;
        const idx = focusPtDraggingIdx;
        setFocusPtDraggingIdx(null);
        const img = focusPtImgRef.current;
        const segmenter = segmenterRef.current;
        if (!img || !segmenter) return;
        const p = pixelCoordsFromEvent(e.nativeEvent, img);
        // Final position update + re-prompt SAM at the dropped location.
        setFocusPtHits((hits) =>
            hits.map((h, i) =>
                i === idx ? { ...h, point: { x: p.x, y: p.y } } : h,
            ),
        );
        segmenter
            .segmentPoint([p.x, p.y], { scale: focusPtScale })
            .then((mask) => {
                setFocusPtHits((hits) =>
                    hits.map((h, i) =>
                        i === idx
                            ? {
                                  ...h,
                                  point: { x: p.x, y: p.y },
                                  pointMask: mask,
                              }
                            : h,
                    ),
                );
            })
            .catch((err: unknown) => {
                setFocusPtError(
                    err instanceof Error ? err.message : String(err),
                );
            });
    };

    const onFocusPtMouseLeave = () => {
        // Treat leaving the canvas during a drag as a cancel — the
        // last-set point stays where the cursor exited, no SAM re-prompt.
        if (focusPtDraggingIdx !== null) setFocusPtDraggingIdx(null);
    };

    // ─── Manual landmark mode ────────────────────────────────────────────

    // Reset placed landmarks when the source swaps. Also invalidate the
    // per-src encoder cache so the next `tryEncodeManual` call actually
    // re-runs `setImage` against the new image's pixels.
    useEffect(() => {
        setManualLandmarks([]);
        manualEncodedSrcRef.current = null;
    }, [src]);

    // Encode the image for SAM 2 once per src. Called from both an effect
    // (covers "img already loaded when entering manual mode" and "segmenter
    // ready arrived after img load") and the img onLoad handler (covers
    // "src changed; new img element loads after the effect ran").
    const tryEncodeManual = useCallback(() => {
        if (shader !== "manual") return;
        const segmenter = segmenterRef.current;
        const img = manualImgRef.current;
        if (!segmenter || !img || !img.complete || !img.naturalWidth) return;
        if (manualEncodedSrcRef.current === src) {
            setManualStatus("ready");
            return;
        }
        const cacheKey = src;
        setManualStatus("encoding");
        setManualError(null);
        segmenter
            .setImage(img)
            .then(() => {
                // Commit only if this img is still the rendered one. When
                // src swaps, the `<img key={src}>` is unmounted and
                // `manualImgRef.current` points to the new element — the
                // old encode then can't claim the cache slot.
                if (manualImgRef.current !== img) return;
                manualEncodedSrcRef.current = cacheKey;
                setManualStatus("ready");
            })
            .catch((err: unknown) => {
                if (manualImgRef.current !== img) return;
                setManualError(
                    err instanceof Error ? err.message : String(err),
                );
                setManualStatus("error");
            });
    }, [shader, src]);

    useEffect(() => {
        if (shader !== "manual") return;
        if (!segmenterReady) {
            setManualStatus("loading-models");
            return;
        }
        tryEncodeManual();
    }, [shader, segmenterReady, tryEncodeManual]);

    const onManualImgLoad = () => {
        tryEncodeManual();
    };

    // Run SAM 2 for one landmark id and update its mask. Reused by
    // addLandmark, drag-release, and the "run all SAM" batch.
    const promptLandmark = useCallback((id: string) => {
        const segmenter = segmenterRef.current;
        if (!segmenter) return;
        setManualLandmarks((current) => {
            const lm = current.find((l) => l.id === id);
            if (!lm) return current;
            const cfg = LANDMARK_CONFIG[lm.type];
            segmenter
                .segmentPoint([lm.x, lm.y], { scale: cfg.scale })
                .then((mask) => {
                    setManualLandmarks((latest) =>
                        latest.map((l) =>
                            l.id === id ? { ...l, mask } : l,
                        ),
                    );
                })
                .catch((err: unknown) => {
                    setManualError(
                        err instanceof Error ? err.message : String(err),
                    );
                    setManualStatus("error");
                });
            return current;
        });
    }, []);

    const addLandmark = (
        type: LandmarkType,
        x: number,
        y: number,
    ): string => {
        const id = newLandmarkId();
        setManualLandmarks((current) => [
            ...current,
            { id, type, x, y, mask: null },
        ]);
        // No auto-SAM here — the user explicitly clicks "run SAM" once
        // they're done placing / repositioning landmarks.
        return id;
    };

    const onManualDragOver = (e: SyntheticEvent<HTMLDivElement, DragEvent>) => {
        if (e.nativeEvent.dataTransfer?.types.includes(LANDMARK_DRAG_MIME)) {
            e.preventDefault();
            e.nativeEvent.dataTransfer.dropEffect = "copy";
        }
    };

    const onManualDrop = (e: SyntheticEvent<HTMLDivElement, DragEvent>) => {
        const dt = e.nativeEvent.dataTransfer;
        if (!dt) return;
        const type = dt.getData(LANDMARK_DRAG_MIME) as LandmarkType;
        if (!type || !LANDMARK_CONFIG[type]) return;
        e.preventDefault();
        const img = manualImgRef.current;
        if (!img || !img.naturalWidth) return;
        const p = pixelCoordsFromEvent(e.nativeEvent, img);
        addLandmark(type, p.x, p.y);
    };

    const onManualMouseDown = (
        e: SyntheticEvent<HTMLDivElement, MouseEvent>,
    ) => {
        const img = manualImgRef.current;
        if (!img || !img.naturalWidth) return;
        const p = pixelCoordsFromEvent(e.nativeEvent, img);
        const hitRadius = Math.max(20, img.naturalWidth * 0.02);
        let bestId: string | null = null;
        let bestDist2 = hitRadius * hitRadius;
        for (const lm of manualLandmarks) {
            const dx = lm.x - p.x;
            const dy = lm.y - p.y;
            const d2 = dx * dx + dy * dy;
            if (d2 <= bestDist2) {
                bestDist2 = d2;
                bestId = lm.id;
            }
        }
        if (bestId !== null) {
            setManualDraggingId(bestId);
            e.preventDefault();
        }
    };

    const onManualMouseMove = (
        e: SyntheticEvent<HTMLDivElement, MouseEvent>,
    ) => {
        if (manualDraggingId === null) return;
        const img = manualImgRef.current;
        if (!img) return;
        const p = pixelCoordsFromEvent(e.nativeEvent, img);
        const id = manualDraggingId;
        setManualLandmarks((lms) =>
            lms.map((l) =>
                l.id === id ? { ...l, x: p.x, y: p.y } : l,
            ),
        );
    };

    const onManualMouseUp = (e: SyntheticEvent<HTMLDivElement, MouseEvent>) => {
        if (manualDraggingId === null) return;
        const id = manualDraggingId;
        setManualDraggingId(null);
        const img = manualImgRef.current;
        if (!img) return;
        const p = pixelCoordsFromEvent(e.nativeEvent, img);
        // Drop position; clear stale mask so the user knows the
        // position has moved and SAM needs to re-run.
        setManualLandmarks((lms) =>
            lms.map((l) =>
                l.id === id
                    ? { ...l, x: p.x, y: p.y, mask: null }
                    : l,
            ),
        );
    };

    const onManualMouseLeave = () => {
        if (manualDraggingId !== null) setManualDraggingId(null);
    };

    const removeLandmark = (id: string) => {
        setManualLandmarks((lms) => lms.filter((l) => l.id !== id));
    };

    // PCA-fit an ellipse per landmark mask, only when the landmark
    // collection changes. Recomputed on every mask refresh.
    const manualCensorShapes = useMemo(() => {
        return manualLandmarks
            .map((lm) => {
                if (!lm.mask) return null;
                const cfg = LANDMARK_CONFIG[lm.type];
                if (!cfg.censor) return null;
                const fit = fitEllipseToMask(lm.mask);
                if (!fit) return null;
                return { id: lm.id, fit, censor: cfg.censor };
            })
            .filter(
                (
                    v,
                ): v is {
                    id: string;
                    fit: FittedEllipse;
                    censor: CensorVisual;
                } => v !== null,
            );
    }, [manualLandmarks]);

    const clearAllLandmarks = () => setManualLandmarks([]);

    const runAllManualSam = () => {
        // Always re-prompt every landmark — keeps it simple and predictable
        // (reposition + run SAM always shows fresh masks).
        for (const lm of manualLandmarks) promptLandmark(lm.id);
    };

    // Paint manual landmarks: SAM masks per landmark on the mask canvas
    // (per-class color), crosshairs + two-tier labels on the overlay.
    useEffect(() => {
        const maskCanvas = manualMaskCanvasRef.current;
        const overlayCanvas = manualOverlayCanvasRef.current;
        const img = manualImgRef.current;
        if (!maskCanvas || !overlayCanvas || !img || !img.naturalWidth) return;
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        for (const c of [maskCanvas, overlayCanvas]) {
            c.width = w;
            c.height = h;
        }
        const mctx = maskCanvas.getContext("2d");
        const octx = overlayCanvas.getContext("2d");
        if (!mctx || !octx) return;
        mctx.clearRect(0, 0, w, h);
        octx.clearRect(0, 0, w, h);

        if (manualLandmarks.length === 0) return;

        // Compose per-landmark masks with per-class color tint.
        const compose = mctx.createImageData(w, h);
        const out = compose.data;
        for (const lm of manualLandmarks) {
            if (!lm.mask) continue;
            if (lm.mask.width !== w || lm.mask.height !== h) continue;
            const [r, g, b] = rgbForClass(lm.type);
            const src = lm.mask.data;
            for (let i = 0; i < src.length; i++) {
                if (src[i]) {
                    const o = i * 4;
                    out[o] = out[o] ? Math.round((out[o] + r) / 2) : r;
                    out[o + 1] = out[o + 1]
                        ? Math.round((out[o + 1] + g) / 2)
                        : g;
                    out[o + 2] = out[o + 2]
                        ? Math.round((out[o + 2] + b) / 2)
                        : b;
                    out[o + 3] = Math.max(out[o + 3], 220);
                }
            }
        }
        mctx.putImageData(compose, 0, 0);

        // Crosshairs + two-tier labels.
        const crossLen = Math.max(12, w * 0.012);
        const lw = Math.max(2, w * 0.003);
        const fontPx = Math.max(11, w * 0.013);
        octx.lineCap = "round";
        octx.textBaseline = "alphabetic";
        for (const lm of manualLandmarks) {
            const cfg = LANDMARK_CONFIG[lm.type];
            const stroke = colorForClass(lm.type);

            // White cross with dark ring for contrast.
            octx.lineWidth = lw;
            octx.strokeStyle = "#fff";
            octx.beginPath();
            octx.moveTo(lm.x - crossLen, lm.y);
            octx.lineTo(lm.x + crossLen, lm.y);
            octx.moveTo(lm.x, lm.y - crossLen);
            octx.lineTo(lm.x, lm.y + crossLen);
            octx.stroke();
            octx.beginPath();
            octx.arc(lm.x, lm.y, crossLen * 0.45, 0, Math.PI * 2);
            octx.strokeStyle = "rgba(0,0,0,0.6)";
            octx.stroke();

            // Primary chip.
            const padX = fontPx * 0.4;
            const padY = fontPx * 0.25;
            octx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
            const tw = octx.measureText(cfg.label).width;
            const th = fontPx + padY * 2;
            const labelX = lm.x - (tw + padX * 2) / 2;
            const labelY = lm.y + crossLen + 4;
            octx.fillStyle = stroke;
            octx.fillRect(labelX, labelY, tw + padX * 2, th);
            octx.fillStyle = "#0a0a0a";
            octx.fillText(
                cfg.label,
                labelX + padX,
                labelY + fontPx + padY * 0.1,
            );

            // Secondary chip with description, if any.
            if (cfg.description) {
                const subFontPx = Math.max(9, fontPx * 0.78);
                const subPadX = subFontPx * 0.4;
                const subPadY = subFontPx * 0.25;
                octx.font = `500 ${subFontPx}px ui-sans-serif, system-ui, sans-serif`;
                const dw = octx.measureText(cfg.description).width;
                const dh = subFontPx + subPadY * 2;
                const descX = lm.x - (dw + subPadX * 2) / 2;
                const descY = labelY + th + 2;
                octx.fillStyle = "rgba(10, 10, 10, 0.85)";
                octx.fillRect(descX, descY, dw + subPadX * 2, dh);
                octx.fillStyle = stroke;
                octx.fillText(
                    cfg.description,
                    descX + subPadX,
                    descY + subFontPx + subPadY * 0.1,
                );
            }
        }
    }, [manualLandmarks]);

    // Re-run focusPt pipeline (debounced) when scale or center bias changes.
    useEffect(() => {
        if (shader !== "focusPt") return;
        if (!detectorReady || !segmenterReady) return;
        const img = focusPtImgRef.current;
        if (!img || !img.complete || !img.naturalWidth) return;
        if (focusPtProcessedSrcRef.current !== src) return;
        const t = window.setTimeout(() => runFocusPtPipeline(img), 200);
        return () => window.clearTimeout(t);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [focusPtScale]);

    // Paint three stacked layers:
    //   1. SAM box-prompt masks per region (low-alpha class color) — bottom
    //   2. SAM point-prompt small masks per target region (bright magenta) — middle
    //   3. NudeNet bounding boxes + focus crosshairs + labels — top
    useEffect(() => {
        const boxMaskCanvas = focusPtBoxMaskCanvasRef.current;
        const maskCanvas = focusPtMaskCanvasRef.current;
        const pointCanvas = focusPtPointCanvasRef.current;
        const img = focusPtImgRef.current;
        if (
            !boxMaskCanvas ||
            !maskCanvas ||
            !pointCanvas ||
            !img ||
            !img.naturalWidth
        )
            return;
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        for (const c of [boxMaskCanvas, maskCanvas, pointCanvas]) {
            c.width = w;
            c.height = h;
        }
        const bctx = boxMaskCanvas.getContext("2d");
        const mctx = maskCanvas.getContext("2d");
        const pctx = pointCanvas.getContext("2d");
        if (!bctx || !mctx || !pctx) return;
        bctx.clearRect(0, 0, w, h);
        mctx.clearRect(0, 0, w, h);
        pctx.clearRect(0, 0, w, h);

        if (focusPtHits.length === 0) return;

        // Layer 1: per-class translucent SAM box-prompt masks.
        const boxImage = bctx.createImageData(w, h);
        const boxOut = boxImage.data;
        for (const hit of focusPtHits) {
            const m = hit.boxMask;
            if (m.width !== w || m.height !== h) continue;
            const [r, g, b] = rgbForClass(hit.class);
            const src = m.data;
            for (let i = 0; i < src.length; i++) {
                if (src[i]) {
                    const o = i * 4;
                    // Average-blend on overlap so regions don't clip each other.
                    boxOut[o] = boxOut[o] ? Math.round((boxOut[o] + r) / 2) : r;
                    boxOut[o + 1] = boxOut[o + 1]
                        ? Math.round((boxOut[o + 1] + g) / 2)
                        : g;
                    boxOut[o + 2] = boxOut[o + 2]
                        ? Math.round((boxOut[o + 2] + b) / 2)
                        : b;
                    boxOut[o + 3] = Math.max(boxOut[o + 3], 255);
                }
            }
        }
        bctx.putImageData(boxImage, 0, 0);

        // Layer 2: bright point-prompt masks (only for target classes).
        const pointImage = mctx.createImageData(w, h);
        const pointOut = pointImage.data;
        for (const hit of focusPtHits) {
            if (!hit.pointMask) continue;
            const m = hit.pointMask;
            if (m.width !== w || m.height !== h) continue;
            const src = m.data;
            for (let i = 0; i < src.length; i++) {
                if (src[i]) {
                    const o = i * 4;
                    pointOut[o] = 255;
                    pointOut[o + 1] = 80;
                    pointOut[o + 2] = 200;
                    pointOut[o + 3] = Math.max(pointOut[o + 3], 220);
                }
            }
        }
        mctx.putImageData(pointImage, 0, 0);

        // Layer 3: bounding boxes + crosshairs + labels.
        const lw = Math.max(1.5, w * 0.0022);
        const crossLen = Math.max(12, w * 0.012);
        const fontPx = Math.max(11, w * 0.013);
        pctx.lineWidth = lw;
        pctx.lineCap = "round";
        pctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
        pctx.textBaseline = "alphabetic";

        for (const hit of focusPtHits) {
            const stroke = colorForClass(hit.class);
            const [bx, by, bw, bh] = hit.box;
            // Bounding box border.
            pctx.strokeStyle = stroke;
            pctx.strokeRect(bx, by, bw, bh);
            // Bounding-box label (top-left, like detect mode).
            const boxLabel = `${hit.class} ${(hit.score * 100).toFixed(0)}%`;
            const padX = fontPx * 0.4;
            const padY = fontPx * 0.25;
            const blw = pctx.measureText(boxLabel).width;
            const blh = fontPx + padY * 2;
            const blY = by - blh < 0 ? by : by - blh;
            pctx.fillStyle = stroke;
            pctx.fillRect(bx, blY, blw + padX * 2, blh);
            pctx.fillStyle = "#0a0a0a";
            pctx.fillText(boxLabel, bx + padX, blY + fontPx + padY * 0.1);
        }

        // Crosshair + anatomy label at each focus point (only for target classes).
        for (const hit of focusPtHits) {
            if (!hit.point) continue;
            const { x, y } = hit.point;
            pctx.strokeStyle = "#fff";
            pctx.beginPath();
            pctx.moveTo(x - crossLen, y);
            pctx.lineTo(x + crossLen, y);
            pctx.moveTo(x, y - crossLen);
            pctx.lineTo(x, y + crossLen);
            pctx.stroke();
            pctx.beginPath();
            pctx.arc(x, y, crossLen * 0.45, 0, Math.PI * 2);
            pctx.strokeStyle = "rgba(0,0,0,0.6)";
            pctx.stroke();

            const cfg = FOCUS_CONFIG[hit.class];
            const label = cfg?.label ?? hit.class.toLowerCase();
            const description = cfg?.description ?? null;
            const padX = fontPx * 0.4;
            const padY = fontPx * 0.25;
            const subFontPx = Math.max(9, fontPx * 0.78);
            const subPadX = subFontPx * 0.4;
            const subPadY = subFontPx * 0.25;

            // Primary chip — anatomical short-form (e.g. "vulva").
            const tw = pctx.measureText(label).width;
            const th = fontPx + padY * 2;
            const labelX = x - (tw + padX * 2) / 2;
            const labelY = y + crossLen + 4;
            pctx.fillStyle = colorForClass(hit.class);
            pctx.fillRect(labelX, labelY, tw + padX * 2, th);
            pctx.fillStyle = "#0a0a0a";
            pctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
            pctx.fillText(label, labelX + padX, labelY + fontPx + padY * 0.1);

            // Secondary chip — descriptive feature being targeted (e.g.
            // "cleft / introitus"). Smaller, dark background so it reads
            // as supporting info rather than another claim.
            if (description) {
                pctx.font = `500 ${subFontPx}px ui-sans-serif, system-ui, sans-serif`;
                const dw = pctx.measureText(description).width;
                const dh = subFontPx + subPadY * 2;
                const descX = x - (dw + subPadX * 2) / 2;
                const descY = labelY + th + 2;
                pctx.fillStyle = "rgba(10, 10, 10, 0.85)";
                pctx.fillRect(descX, descY, dw + subPadX * 2, dh);
                pctx.fillStyle = colorForClass(hit.class);
                pctx.fillText(
                    description,
                    descX + subPadX,
                    descY + subFontPx + subPadY * 0.1,
                );
            }
        }
    }, [focusPtHits]);

    // Re-paint the skin-map preview canvas (debounced) when threshold
    // or src changes. classifySkin is pure JS and fast (a few ms on small
    // images, ~100ms on 4K) so a short debounce is enough.
    const recomputeSkinMap = useCallback(() => {
        const img = skinMapImgRef.current;
        if (!img || !img.complete || !img.naturalWidth) return;
        const mask = classifySkin(img, { threshold: skinThreshold });
        const canvas = skinMapCanvasRef.current;
        if (!canvas) return;
        canvas.width = mask.width;
        canvas.height = mask.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const image = ctx.createImageData(mask.width, mask.height);
        const out = image.data;
        let count = 0;
        for (let i = 0; i < mask.data.length; i++) {
            if (mask.data[i]) {
                count++;
                const o = i * 4;
                out[o] = 80;
                out[o + 1] = 255;
                out[o + 2] = 140;
                out[o + 3] = 220;
            }
        }
        ctx.putImageData(image, 0, 0);
        setSkinMapCoverage(count / mask.data.length);
    }, [skinThreshold]);

    const onSkinMapImgLoad = () => recomputeSkinMap();

    useEffect(() => {
        if (shader !== "skin-map") return;
        const t = window.setTimeout(recomputeSkinMap, 100);
        return () => window.clearTimeout(t);
    }, [shader, skinThreshold, recomputeSkinMap]);

    // Lazy-load the human-parser on first entry to human-parse mode OR to
    // segment mode when the parse filter is enabled. ~100MB SegFormer-B2
    // download on first run, browser-cached after; shared across modes.
    useEffect(() => {
        const wantParser =
            shader === "human-parse" ||
            (shader === "segment" && segmentParseFilter);
        if (!wantParser) return;
        if (parserRef.current) {
            setParserReady(true);
            return;
        }
        let cancelled = false;
        setParseStatus("loading-model");
        setParseError(null);
        setParseProgress(null);
        loadHumanParser({
            onProgress: (e) => {
                if (cancelled) return;
                setParseProgress(e);
            },
        })
            .then((p) => {
                if (cancelled) {
                    p.dispose();
                    return;
                }
                parserRef.current = p;
                setParserReady(true);
                setParseProgress(null);
                setParseStatus("idle");
            })
            .catch((e: unknown) => {
                if (cancelled) return;
                setParseError(e instanceof Error ? e.message : String(e));
                setParseStatus("error");
            });
        return () => {
            cancelled = true;
        };
    }, [shader, segmentParseFilter]);

    // Drop stale parse results when the source swaps.
    useEffect(() => {
        setParseResult(null);
    }, [src]);

    const onParseImgLoad = (e: SyntheticEvent<HTMLImageElement>) => {
        const p = parserRef.current;
        if (!p) return;
        const img = e.currentTarget;
        setParseStatus("parsing");
        setParseError(null);
        p.parse(img)
            .then((result) => {
                setParseResult(result);
                setParseStatus("ready");
            })
            .catch((err: unknown) => {
                setParseError(
                    err instanceof Error ? err.message : String(err),
                );
                setParseStatus("error");
            });
    };

    const onParseImgError = () => {
        setParseError("image failed to load");
        setParseStatus("error");
    };

    // Paint the parse class-map as a translucent color overlay.
    useEffect(() => {
        const canvas = parseCanvasRef.current;
        if (!canvas) return;
        if (!parseResult) {
            canvas.width = 0;
            canvas.height = 0;
            return;
        }
        parseToVisualizationCanvas(parseResult, { canvas, alpha: 0.65 });
    }, [parseResult]);

    // Per-class coverage breakdown for the side-panel legend.
    const parseCoverage = useMemo(() => {
        if (!parseResult) return null;
        const counts = new Uint32Array(HUMAN_PARSE_CLASSES.length);
        const data = parseResult.classMap;
        for (let i = 0; i < data.length; i++) {
            const c = data[i];
            if (c < counts.length) counts[c]++;
        }
        const total = data.length;
        return parseResult.presentClasses
            .map((id) => ({
                id,
                name: HUMAN_PARSE_CLASSES[id] ?? `class ${id}`,
                fraction: counts[id] / total,
            }))
            .sort((a, b) => b.fraction - a.fraction);
    }, [parseResult]);

    // Re-run segment pipeline (debounced) when filter knobs change.
    useEffect(() => {
        if (shader !== "segment") return;
        if (!detectorReady || !segmenterReady) return;
        if (segmentParseFilter && !parserReady) return;
        const img = segmentImgRef.current;
        if (!img || !img.complete || !img.naturalWidth) return;
        if (segmentProcessedSrcRef.current !== src) return;
        const t = window.setTimeout(() => runSegmentPipeline(img), 200);
        return () => window.clearTimeout(t);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [segmentSkinFilter, skinThreshold, segmentParseFilter]);

    // Paint the binary mask onto the overlay canvas in source-pixel space.
    // CSS scales it to match the rendered image; opacity is applied via
    // canvas style so the underlying photo shows through.
    useEffect(() => {
        const canvas = segmentCanvasRef.current;
        if (!canvas) return;
        if (!segmentMask) {
            canvas.width = 0;
            canvas.height = 0;
            return;
        }
        canvas.width = segmentMask.width;
        canvas.height = segmentMask.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const image = ctx.createImageData(
            segmentMask.width,
            segmentMask.height,
        );
        const out = image.data;
        const src = segmentMask.data;
        for (let i = 0; i < src.length; i++) {
            const o = i * 4;
            out[o] = 255;
            out[o + 1] = 80;
            out[o + 2] = 200;
            out[o + 3] = src[i];
        }
        ctx.putImageData(image, 0, 0);
    }, [segmentMask]);

    // Paint detector bounding boxes onto the segment-mode overlay so the
    // SAM prompts are visible alongside their masks.
    useEffect(() => {
        const canvas = segmentBoxCanvasRef.current;
        const img = segmentImgRef.current;
        if (!canvas || !img || !img.naturalWidth) return;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (segmentRegions.length === 0) return;
        const lw = Math.max(1.5, canvas.width * 0.0022);
        const fontPx = Math.max(11, canvas.width * 0.013);
        ctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
        ctx.textBaseline = "alphabetic";
        for (const r of segmentRegions) {
            const [x, y, w, h] = r.box;
            const stroke = colorForClass(r.class);
            ctx.lineWidth = lw;
            ctx.strokeStyle = stroke;
            ctx.strokeRect(x, y, w, h);
            const label = `${r.class} ${(r.maskScore * 100).toFixed(0)}%`;
            const padX = fontPx * 0.4;
            const padY = fontPx * 0.25;
            const tw = ctx.measureText(label).width;
            const th = fontPx + padY * 2;
            const labelY = y - th < 0 ? y : y - th;
            ctx.fillStyle = stroke;
            ctx.fillRect(x, labelY, tw + padX * 2, th);
            ctx.fillStyle = "#0a0a0a";
            ctx.fillText(label, x + padX, labelY + fontPx + padY * 0.1);
        }
    }, [segmentRegions]);

    // Paint detected boxes onto the overlay canvas in source-pixel space.
    // CSS scales the canvas to match the rendered image.
    useEffect(() => {
        const canvas = detectCanvasRef.current;
        const img = detectImgRef.current;
        if (!canvas || !img || !img.naturalWidth) return;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (detectRegions.length === 0) return;
        const lw = Math.max(2, canvas.width * 0.003);
        const fontPx = Math.max(11, canvas.width * 0.014);
        ctx.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
        ctx.textBaseline = "alphabetic";
        for (const r of detectRegions) {
            const [x, y, w, h] = r.box;
            const stroke = colorForClass(r.class);
            ctx.lineWidth = lw;
            ctx.strokeStyle = stroke;
            ctx.strokeRect(x, y, w, h);
            const label = `${r.class} ${(r.score * 100).toFixed(0)}%`;
            const padX = fontPx * 0.4;
            const padY = fontPx * 0.25;
            const tw = ctx.measureText(label).width;
            const th = fontPx + padY * 2;
            const labelY = y - th < 0 ? y : y - th;
            ctx.fillStyle = stroke;
            ctx.fillRect(x, labelY, tw + padX * 2, th);
            ctx.fillStyle = "#0a0a0a";
            ctx.fillText(label, x + padX, labelY + fontPx + padY * 0.1);
        }
    }, [detectRegions]);

    const loadFromUrl = (url: string, displayName?: string) => {
        setSrc(url);
        setDraftSrc(url);
        setLabel(displayName ?? url);
        setForcedKind(undefined);
    };

    const loadFromFile = (file: File) => {
        const url = URL.createObjectURL(file);
        setSrc(url);
        setDraftSrc(file.name);
        setLabel(file.name);
        setForcedKind(file.type.startsWith("video/") ? "video" : "image");
    };

    const presetEntries = useMemo(() => ASSET_ENTRIES, []);

    const stepAsset = useCallback(
        (delta: number) => {
            if (presetEntries.length === 0) return;
            const idx = presetEntries.findIndex((e) => e.url === src);
            const len = presetEntries.length;
            const next =
                idx === -1
                    ? delta > 0
                        ? 0
                        : len - 1
                    : ((idx + delta) % len + len) % len;
            const entry = presetEntries[next];
            loadFromUrl(entry.url, entry.name);
        },
        // loadFromUrl is a closure over setters that are stable; deps on
        // presetEntries (memoized) and src cover the meaningful changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [presetEntries, src],
    );

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.metaKey || e.ctrlKey || e.altKey) return;
            const tgt = e.target;
            if (
                tgt instanceof HTMLInputElement ||
                tgt instanceof HTMLTextAreaElement ||
                tgt instanceof HTMLSelectElement
            ) {
                return;
            }
            if (e.key === "ArrowLeft") {
                e.preventDefault();
                stepAsset(-1);
            } else if (e.key === "ArrowRight") {
                e.preventDefault();
                stepAsset(1);
            }
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [stepAsset]);

    return (
        <>
            <div style={STAGE_STYLE}>
                {shader === "classify" ? (
                    <img
                        key={src}
                        src={src}
                        alt="classify subject"
                        crossOrigin="anonymous"
                        onLoad={onClassifyImgLoad}
                        onError={onClassifyImgError}
                        style={{
                            width: "100%",
                            height: "100%",
                            objectFit: "contain",
                            background: "#0a0a0a",
                        }}
                    />
                ) : shader === "detect" ? (
                    <div
                        style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#0a0a0a",
                        }}
                    >
                        <div
                            style={{
                                position: "relative",
                                maxWidth: "100%",
                                maxHeight: "100%",
                                display: "inline-block",
                            }}
                        >
                            <img
                                key={src}
                                ref={detectImgRef}
                                src={src}
                                alt="detect subject"
                                crossOrigin="anonymous"
                                onLoad={onDetectImgLoad}
                                onError={onDetectImgError}
                                style={{
                                    display: "block",
                                    maxWidth: "100vw",
                                    maxHeight: "100vh",
                                }}
                            />
                            <canvas
                                ref={detectCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                }}
                            />
                        </div>
                    </div>
                ) : shader === "segment" ? (
                    <div
                        style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#0a0a0a",
                        }}
                    >
                        <div
                            style={{
                                position: "relative",
                                maxWidth: "100%",
                                maxHeight: "100%",
                                display: "inline-block",
                            }}
                        >
                            <img
                                key={src}
                                ref={segmentImgRef}
                                src={src}
                                alt="segment subject"
                                crossOrigin="anonymous"
                                draggable={false}
                                onLoad={onSegmentImgLoad}
                                onError={onSegmentImgError}
                                style={{
                                    display: "block",
                                    maxWidth: "100vw",
                                    maxHeight: "100vh",
                                }}
                            />
                            <canvas
                                ref={segmentCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                    opacity: 0.55,
                                    mixBlendMode: "screen",
                                }}
                            />
                            <canvas
                                ref={segmentBoxCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                }}
                            />
                        </div>
                    </div>
                ) : shader === "skin-map" ? (
                    <div
                        style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#0a0a0a",
                        }}
                    >
                        <div
                            style={{
                                position: "relative",
                                maxWidth: "100%",
                                maxHeight: "100%",
                                display: "inline-block",
                            }}
                        >
                            <img
                                key={src}
                                ref={skinMapImgRef}
                                src={src}
                                alt="skin-map subject"
                                crossOrigin="anonymous"
                                onLoad={onSkinMapImgLoad}
                                style={{
                                    display: "block",
                                    maxWidth: "100vw",
                                    maxHeight: "100vh",
                                }}
                            />
                            <canvas
                                ref={skinMapCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                    mixBlendMode: "screen",
                                }}
                            />
                        </div>
                    </div>
                ) : shader === "human-parse" ? (
                    <div
                        style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#0a0a0a",
                        }}
                    >
                        <div
                            style={{
                                position: "relative",
                                maxWidth: "100%",
                                maxHeight: "100%",
                                display: "inline-block",
                            }}
                        >
                            <img
                                key={src}
                                ref={parseImgRef}
                                src={src}
                                alt="human-parse subject"
                                crossOrigin="anonymous"
                                onLoad={onParseImgLoad}
                                onError={onParseImgError}
                                style={{
                                    display: "block",
                                    maxWidth: "100vw",
                                    maxHeight: "100vh",
                                }}
                            />
                            <canvas
                                ref={parseCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                    opacity: 0.85,
                                }}
                            />
                        </div>
                    </div>
                ) : shader === "focusPt" ? (
                    <div
                        style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#0a0a0a",
                        }}
                    >
                        <div
                            onMouseDown={onFocusPtMouseDown}
                            onMouseMove={onFocusPtMouseMove}
                            onMouseUp={onFocusPtMouseUp}
                            onMouseLeave={onFocusPtMouseLeave}
                            style={{
                                position: "relative",
                                maxWidth: "100%",
                                maxHeight: "100%",
                                display: "inline-block",
                                cursor:
                                    focusPtDraggingIdx !== null
                                        ? "grabbing"
                                        : "default",
                                userSelect: "none",
                            }}
                        >
                            <img
                                key={src}
                                ref={focusPtImgRef}
                                src={src}
                                alt="focus subject"
                                crossOrigin="anonymous"
                                draggable={false}
                                onLoad={onFocusPtImgLoad}
                                onError={onFocusPtImgError}
                                style={{
                                    display: "block",
                                    maxWidth: "100vw",
                                    maxHeight: "100vh",
                                    pointerEvents: "none",
                                }}
                            />
                            <canvas
                                ref={focusPtBoxMaskCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                    opacity: 0.32,
                                    mixBlendMode: "multiply",
                                }}
                            />
                            <canvas
                                ref={focusPtMaskCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                    opacity: 0.62,
                                    mixBlendMode: "screen",
                                }}
                            />
                            <canvas
                                ref={focusPtPointCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                }}
                            />
                        </div>
                    </div>
                ) : shader === "manual" ? (
                    <div
                        style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#0a0a0a",
                        }}
                    >
                        <div
                            onDragOver={onManualDragOver}
                            onDrop={onManualDrop}
                            onMouseDown={onManualMouseDown}
                            onMouseMove={onManualMouseMove}
                            onMouseUp={onManualMouseUp}
                            onMouseLeave={onManualMouseLeave}
                            style={{
                                position: "relative",
                                maxWidth: "100%",
                                maxHeight: "100%",
                                display: "inline-block",
                                cursor:
                                    manualDraggingId !== null
                                        ? "grabbing"
                                        : "default",
                                userSelect: "none",
                            }}
                        >
                            <img
                                key={src}
                                ref={manualImgRef}
                                src={src}
                                alt="manual subject"
                                crossOrigin="anonymous"
                                draggable={false}
                                onLoad={onManualImgLoad}
                                style={{
                                    display: "block",
                                    maxWidth: "100vw",
                                    maxHeight: "100vh",
                                    pointerEvents: "none",
                                }}
                            />
                            <canvas
                                ref={manualMaskCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                    opacity: 0.42,
                                    mixBlendMode: "screen",
                                }}
                            />
                            {manualCensorShapes.length > 0 &&
                            manualImgRef.current?.naturalWidth ? (
                                <svg
                                    viewBox={`0 0 ${manualImgRef.current.naturalWidth} ${manualImgRef.current.naturalHeight}`}
                                    preserveAspectRatio="none"
                                    style={{
                                        position: "absolute",
                                        inset: 0,
                                        width: "100%",
                                        height: "100%",
                                        pointerEvents: "none",
                                    }}
                                >
                                    {manualCensorShapes.map(
                                        ({ id, fit, censor }) => {
                                            const scale = censor.scale ?? 1;
                                            const thetaDeg =
                                                (fit.theta * 180) / Math.PI;
                                            return (
                                                <ellipse
                                                    key={id}
                                                    cx={fit.cx}
                                                    cy={fit.cy}
                                                    rx={fit.rx * scale}
                                                    ry={fit.ry * scale}
                                                    transform={`rotate(${thetaDeg} ${fit.cx} ${fit.cy})`}
                                                    fill={censor.color}
                                                    fillOpacity={
                                                        censor.opacity ?? 1
                                                    }
                                                />
                                            );
                                        },
                                    )}
                                </svg>
                            ) : null}
                            <canvas
                                ref={manualOverlayCanvasRef}
                                style={{
                                    position: "absolute",
                                    inset: 0,
                                    width: "100%",
                                    height: "100%",
                                    pointerEvents: "none",
                                }}
                            />
                        </div>
                    </div>
                ) : shader === "edge" ? (
                    <EdgeDetect
                        src={src}
                        kind={forcedKind}
                        edgeStrength={edgeStrength}
                        edgeThreshold={edgeThreshold}
                        edgeKnee={edgeKnee}
                    />
                ) : shader === "skin" ? (
                    <SkinMask
                        src={src}
                        kind={forcedKind}
                        params={skinParams}
                    />
                ) : shader === "hair" ? (
                    <HairDetect
                        src={src}
                        kind={forcedKind}
                        params={hairParams}
                    />
                ) : shader === "lowpoly" ? (
                    <LowPoly
                        src={src}
                        kind={forcedKind}
                        params={lowPolyParams}
                    />
                ) : shader === "circle" ? (
                    <CircleDetect
                        src={src}
                        kind={forcedKind}
                        params={circleParams}
                        strokeColor={circleStroke}
                        strokeWidth={circleStrokeWidth}
                        strokeOpacity={circleStrokeOpacity}
                        fillColor={circleFill}
                        fillOpacity={circleFillOpacity}
                        maxCircles={circleMaxCircles}
                        readbackEvery={circleReadbackEvery}
                    />
                ) : shader === "posterize" ? (
                    <Posterize
                        src={src}
                        kind={forcedKind}
                        levels={posterizeLevels}
                    />
                ) : shader === "silhouette" ? (
                    <Silhouette
                        ref={silhouetteRef}
                        src={src}
                        kind={forcedKind}
                        referenceColor={silReference}
                        smoothRadius={silSmooth}
                        threshold={silThreshold}
                        feather={silFeather}
                        thresholdSpread={silSpread}
                        outlineThickness={silOutline}
                        mode={silMode}
                        offMix={silOffMix}
                    />
                ) : (
                    <PaletteMask
                        ref={paletteRef}
                        src={src}
                        kind={forcedKind}
                        palette={palette}
                        enabled={enabled}
                        mode={paletteMode}
                        offMix={offMix}
                    />
                )}
            </div>

            <div style={PANEL_STYLE}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>
                    newtone-shaders playground
                </div>

                {advancedMode ? (
                    <div style={ROW_STYLE}>
                        <span style={LABEL_STYLE}>shader</span>
                        <select
                            value={shader}
                            onChange={(e) =>
                                setShader(e.target.value as ShaderKind)
                            }
                            style={INPUT_STYLE}
                        >
                            <option value="edge">edge-detect</option>
                            <option value="skin">skin-mask (YCbCr)</option>
                            <option value="palette">palette-mask</option>
                            <option value="hair">hair-detect</option>
                            <option value="lowpoly">low-poly</option>
                            <option value="silhouette">silhouette</option>
                            <option value="circle">circle-detect</option>
                            <option value="posterize">posterize</option>
                            <option value="classify">classify (NSFWJS)</option>
                            <option value="detect">detect (NudeNet)</option>
                            <option value="segment">segment (SAM 2)</option>
                            <option value="skin-map">
                                skin map (JS pipeline)
                            </option>
                            <option value="human-parse">
                                human parse (SegFormer clothes)
                            </option>
                            <option value="focusPt">
                                focus points (SAM point prompts)
                            </option>
                            <option value="manual">manual landmarks</option>
                        </select>
                    </div>
                ) : null}

                {!advancedMode ? (
                    <div style={ROW_STYLE}>
                        <span style={LABEL_STYLE}>mode</span>
                        <button
                            type="button"
                            onClick={() => setShader("focusPt")}
                            style={{
                                ...BUTTON_STYLE,
                                flex: 1,
                                background:
                                    shader === "focusPt"
                                        ? "#2a2a2a"
                                        : "#0a0a0a",
                                borderColor:
                                    shader === "focusPt" ? "#4a8" : "#333",
                                color:
                                    shader === "focusPt"
                                        ? "#9fd4b3"
                                        : "#888",
                            }}
                        >
                            auto
                        </button>
                        <button
                            type="button"
                            onClick={() => setShader("manual")}
                            style={{
                                ...BUTTON_STYLE,
                                flex: 1,
                                background:
                                    shader === "manual"
                                        ? "#2a2a2a"
                                        : "#0a0a0a",
                                borderColor:
                                    shader === "manual" ? "#4a8" : "#333",
                                color:
                                    shader === "manual"
                                        ? "#9fd4b3"
                                        : "#888",
                            }}
                        >
                            manual
                        </button>
                    </div>
                ) : null}

                <div style={ROW_STYLE}>
                    <span style={LABEL_STYLE}>asset</span>
                    <button
                        type="button"
                        onClick={() => stepAsset(-1)}
                        disabled={presetEntries.length === 0}
                        title="previous (←)"
                        style={{
                            ...BUTTON_STYLE,
                            flex: "0 0 28px",
                            padding: "4px 0",
                        }}
                    >
                        ‹
                    </button>
                    <select
                        value={
                            presetEntries.find((e) => e.url === src)?.url ?? ""
                        }
                        onChange={(e) => {
                            const entry = presetEntries.find(
                                (x) => x.url === e.target.value,
                            );
                            if (entry) loadFromUrl(entry.url, entry.name);
                        }}
                        style={INPUT_STYLE}
                        disabled={presetEntries.length === 0}
                    >
                        {presetEntries.length === 0 ? (
                            <option value="">(no assets in /assets)</option>
                        ) : (
                            <>
                                {!presetEntries.some((e) => e.url === src) && (
                                    <option value="">(custom)</option>
                                )}
                                {presetEntries.map((entry) => (
                                    <option key={entry.url} value={entry.url}>
                                        {entry.name}
                                    </option>
                                ))}
                            </>
                        )}
                    </select>
                    <button
                        type="button"
                        onClick={() => stepAsset(1)}
                        disabled={presetEntries.length === 0}
                        title="next (→)"
                        style={{
                            ...BUTTON_STYLE,
                            flex: "0 0 28px",
                            padding: "4px 0",
                        }}
                    >
                        ›
                    </button>
                </div>

                {advancedMode ? (
                    <>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>upload</span>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*,video/*"
                                onChange={(e) => {
                                    const f = e.target.files?.[0];
                                    if (f) loadFromFile(f);
                                    e.target.value = "";
                                }}
                                style={{ display: "none" }}
                            />
                            <button
                                type="button"
                                onClick={() =>
                                    fileInputRef.current?.click()
                                }
                                style={{ ...BUTTON_STYLE, flex: 1 }}
                            >
                                choose local file…
                            </button>
                        </div>

                        <form
                            onSubmit={(e) => {
                                e.preventDefault();
                                loadFromUrl(draftSrc, draftSrc);
                            }}
                            style={ROW_STYLE}
                        >
                            <span style={LABEL_STYLE}>url</span>
                            <input
                                type="text"
                                value={draftSrc}
                                onChange={(e) => setDraftSrc(e.target.value)}
                                style={INPUT_STYLE}
                                placeholder="paste a URL and press Enter"
                            />
                        </form>
                    </>
                ) : null}

                <div
                    style={{
                        fontSize: 11,
                        color: "#888",
                        marginTop: 4,
                    }}
                >
                    loaded: <span style={{ color: "#bdbdbd" }}>{label}</span>
                    {forcedKind ? (
                        <>
                            {" "}
                            <span style={{ color: "#666" }}>
                                ({forcedKind})
                            </span>
                        </>
                    ) : null}
                </div>

                <div style={DIVIDER} />

                {shader === "classify" ? (
                    <ClassifyPanel
                        status={classifyStatus}
                        probs={classifyProbs}
                        error={classifyError}
                    />
                ) : shader === "detect" ? (
                    <DetectPanel
                        status={detectStatus}
                        regions={detectRegions}
                        error={detectError}
                        scoreThreshold={detectScore}
                        onScoreThresholdChange={setDetectScore}
                    />
                ) : shader === "segment" ? (
                    <SegmentPanel
                        status={segmentStatus}
                        progress={segmentProgress}
                        stageInfo={segmentStageInfo}
                        error={segmentError}
                        regions={segmentRegions}
                        hasMask={!!segmentMask}
                        onRerun={segmentRerun}
                        skinFilter={segmentSkinFilter}
                        onSkinFilterChange={setSegmentSkinFilter}
                        skinThreshold={skinThreshold}
                        onSkinThresholdChange={setSkinThreshold}
                        parseFilter={segmentParseFilter}
                        onParseFilterChange={setSegmentParseFilter}
                        parserLoading={
                            segmentParseFilter && !parserReady &&
                            parseStatus === "loading-model"
                        }
                    />
                ) : shader === "skin-map" ? (
                    <>
                        <div style={SECTION_TITLE}>skin map (JS pipeline)</div>
                        <div style={{ ...HINT_STYLE, marginTop: 0 }}>
                            {skinMapCoverage != null
                                ? `${(skinMapCoverage * 100).toFixed(1)}% of pixels classified as skin`
                                : "load an image"}
                        </div>
                        <SliderRow
                            label="skin ≥"
                            value={skinThreshold}
                            onChange={setSkinThreshold}
                            min={0.05}
                            max={0.95}
                            step={0.01}
                        />
                        <div style={HINT_STYLE}>
                            Same <code>classifySkin</code> the segment pipeline
                            uses for its SAM ∩ skin filter. Threshold here is
                            shared with segment mode — tune it visually, then
                            switch to <em>segment</em> to apply.
                        </div>
                    </>
                ) : shader === "focusPt" ? (
                    <>
                        <div style={SECTION_TITLE}>
                            focus points (SAM point prompts)
                        </div>
                        <div style={{ ...HINT_STYLE, marginTop: 0 }}>
                            {focusPtStatus === "loading-models"
                                ? "loading models…"
                                : focusPtStatus === "detecting"
                                  ? "detecting target regions…"
                                  : focusPtStatus === "encoding"
                                    ? "encoding image…"
                                    : focusPtStatus === "decoding"
                                      ? `point-prompting SAM ${focusPtStageInfo ?? ""}`
                                      : focusPtStatus === "ready"
                                        ? `${focusPtHits.length} focus point${focusPtHits.length === 1 ? "" : "s"} located`
                                        : focusPtStatus === "error"
                                          ? `error: ${focusPtError ?? "unknown"}`
                                          : "load an image"}
                        </div>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>scale</span>
                            <select
                                value={focusPtScale}
                                onChange={(e) =>
                                    setFocusPtScale(
                                        e.target.value as SegmentScale,
                                    )
                                }
                                style={INPUT_STYLE}
                            >
                                <option value="smallest">
                                    smallest (areola / orifice)
                                </option>
                                <option value="medium">medium</option>
                                <option value="largest">
                                    largest (whole region)
                                </option>
                                <option value="best">best IoU</option>
                            </select>
                        </div>
                        {focusPtHits.length > 0 ? (
                            <div style={{ marginTop: 8 }}>
                                {focusPtHits.map((hit, i) => {
                                    const m = hit.pointMask ?? hit.boxMask;
                                    let area = 0;
                                    for (let p = 0; p < m.data.length; p++)
                                        if (m.data[p]) area++;
                                    const total = m.width * m.height;
                                    const stroke = colorForClass(hit.class);
                                    const fcfg = FOCUS_CONFIG[hit.class];
                                    const labelText = fcfg?.label
                                        ? `${fcfg.label} (${hit.class.toLowerCase()})`
                                        : hit.class.toLowerCase();
                                    return (
                                        <div
                                            key={i}
                                            style={{
                                                ...ROW_STYLE,
                                                margin: "4px 0",
                                                gap: 6,
                                            }}
                                        >
                                            <span
                                                style={{
                                                    flex: "0 0 12px",
                                                    height: 12,
                                                    borderRadius: 3,
                                                    background: stroke,
                                                }}
                                                title={hit.class}
                                            />
                                            <span
                                                style={{
                                                    flex: 1,
                                                    color: "#f0f0f0",
                                                    fontSize: 11,
                                                    overflow: "hidden",
                                                    textOverflow: "ellipsis",
                                                    whiteSpace: "nowrap",
                                                }}
                                                title={hit.class}
                                            >
                                                {labelText}
                                            </span>
                                            <span
                                                style={{
                                                    flex: "0 0 70px",
                                                    textAlign: "right",
                                                    color: "#f0f0f0",
                                                    fontSize: 11,
                                                    fontVariantNumeric:
                                                        "tabular-nums",
                                                }}
                                            >
                                                {((area / total) * 100).toFixed(2)}%
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        ) : null}
                        <div style={HINT_STYLE}>
                            Targets:{" "}
                            <code>
                                {Object.values(FOCUS_CONFIG)
                                    .map((c) => c.label)
                                    .filter(
                                        (v, i, a) => a.indexOf(v) === i,
                                    )
                                    .join(", ")}
                            </code>
                            . Per-class chromatic focus + SAM 2 point-prompt;
                            each class gets its own center-bias and offset.
                            Smallest of SAM's three multi-mask outputs.
                        </div>
                    </>
                ) : shader === "manual" ? (
                    <>
                        <div style={SECTION_TITLE}>manual landmarks</div>
                        <div style={{ ...HINT_STYLE, marginTop: 0 }}>
                            {manualStatus === "loading-models"
                                ? "loading SAM 2…"
                                : manualStatus === "encoding"
                                  ? "encoding image…"
                                  : manualStatus === "error"
                                    ? `error: ${manualError ?? "unknown"}`
                                    : (() => {
                                          const total =
                                              manualLandmarks.length;
                                          const masked =
                                              manualLandmarks.filter(
                                                  (l) => l.mask,
                                              ).length;
                                          if (total === 0)
                                              return "drag a landmark onto the image";
                                          return `${masked}/${total} landmark${total === 1 ? "" : "s"} masked`;
                                      })()}
                        </div>

                        <div style={SECTION_TITLE}>toolbar</div>
                        <div
                            style={{
                                ...HINT_STYLE,
                                marginTop: 0,
                                marginBottom: 6,
                            }}
                        >
                            drag a chip onto the image
                        </div>
                        {LANDMARK_GROUP_ORDER.map((group) => (
                            <div key={group} style={{ marginBottom: 8 }}>
                                <div
                                    style={{
                                        fontSize: 10,
                                        textTransform: "uppercase",
                                        letterSpacing: 0.5,
                                        color: "#666",
                                        marginBottom: 4,
                                    }}
                                >
                                    {LANDMARK_GROUP_TITLES[group]}
                                </div>
                                <div
                                    style={{
                                        display: "flex",
                                        flexWrap: "wrap",
                                        gap: 4,
                                    }}
                                >
                                    {LANDMARK_TYPES_BY_GROUP[group].map(
                                        (type) => {
                                            const cfg = LANDMARK_CONFIG[type];
                                            return (
                                                <div
                                                    key={type}
                                                    draggable={true}
                                                    onDragStart={(e) => {
                                                        e.dataTransfer.setData(
                                                            LANDMARK_DRAG_MIME,
                                                            type,
                                                        );
                                                        e.dataTransfer.effectAllowed =
                                                            "copy";
                                                    }}
                                                    title={`drag onto image to drop a ${cfg.label}`}
                                                    style={{
                                                        ...BUTTON_STYLE,
                                                        cursor: "grab",
                                                        background:
                                                            colorForClass(type),
                                                        color: "#0a0a0a",
                                                        borderColor:
                                                            "transparent",
                                                        fontSize: 11,
                                                        fontWeight: 600,
                                                        padding: "3px 8px",
                                                    }}
                                                >
                                                    {cfg.label}
                                                </div>
                                            );
                                        },
                                    )}
                                </div>
                            </div>
                        ))}

                        <div style={{ ...ROW_STYLE, marginTop: 8 }}>
                            <button
                                type="button"
                                onClick={runAllManualSam}
                                disabled={
                                    manualLandmarks.length === 0 ||
                                    manualStatus === "loading-models" ||
                                    manualStatus === "encoding"
                                }
                                style={{ ...BUTTON_STYLE, flex: 1 }}
                            >
                                run SAM ({
                                    manualLandmarks.filter((l) => !l.mask)
                                        .length
                                } unmasked)
                            </button>
                            <button
                                type="button"
                                onClick={clearAllLandmarks}
                                disabled={manualLandmarks.length === 0}
                                style={{
                                    ...BUTTON_STYLE,
                                    flex: "0 0 64px",
                                }}
                            >
                                clear
                            </button>
                        </div>

                        {manualLandmarks.length > 0 ? (
                            <div style={{ marginTop: 8 }}>
                                <div style={SECTION_TITLE}>placed</div>
                                {manualLandmarks.map((lm) => {
                                    const cfg = LANDMARK_CONFIG[lm.type];
                                    return (
                                        <div
                                            key={lm.id}
                                            style={{
                                                ...ROW_STYLE,
                                                margin: "4px 0",
                                                gap: 6,
                                            }}
                                        >
                                            <span
                                                style={{
                                                    flex: "0 0 12px",
                                                    height: 12,
                                                    borderRadius: 3,
                                                    background:
                                                        colorForClass(lm.type),
                                                }}
                                            />
                                            <span
                                                style={{
                                                    flex: 1,
                                                    color: "#f0f0f0",
                                                    fontSize: 11,
                                                    overflow: "hidden",
                                                    textOverflow: "ellipsis",
                                                    whiteSpace: "nowrap",
                                                }}
                                                title={lm.type}
                                            >
                                                {cfg.label}
                                            </span>
                                            <span
                                                style={{
                                                    flex: "0 0 80px",
                                                    color: "#888",
                                                    fontSize: 10,
                                                    fontVariantNumeric:
                                                        "tabular-nums",
                                                    textAlign: "right",
                                                }}
                                            >
                                                {lm.mask
                                                    ? "masked"
                                                    : "no mask"}
                                            </span>
                                            <button
                                                type="button"
                                                onClick={() =>
                                                    removeLandmark(lm.id)
                                                }
                                                title="remove"
                                                style={{
                                                    ...BUTTON_STYLE,
                                                    flex: "0 0 24px",
                                                    padding: "2px 0",
                                                    fontSize: 12,
                                                    lineHeight: 1,
                                                }}
                                            >
                                                ×
                                            </button>
                                        </div>
                                    );
                                })}
                            </div>
                        ) : null}
                        <div style={HINT_STYLE}>
                            Drag a chip onto the image to place a landmark.
                            Drag an existing crosshair to reposition (SAM
                            re-prompts on release). Click <code>×</code> to
                            remove. Encoder caches per image — placing many
                            landmarks reuses the same embedding.
                        </div>
                    </>
                ) : shader === "human-parse" ? (
                    <>
                        <div style={SECTION_TITLE}>
                            human parse (SegFormer-B2 clothes)
                        </div>
                        <div style={{ ...HINT_STYLE, marginTop: 0 }}>
                            {parseStatus === "loading-model"
                                ? parseProgress?.progress != null
                                    ? `${parseProgress.status} ${parseProgress.file ?? ""} ${(
                                          parseProgress.progress as number
                                      ).toFixed(0)}%`
                                    : `loading model… (~100MB first run)`
                                : parseStatus === "parsing"
                                  ? "parsing image…"
                                  : parseStatus === "ready" && parseCoverage
                                    ? `${parseCoverage.length} class${parseCoverage.length === 1 ? "" : "es"} present`
                                    : parseStatus === "error"
                                      ? `error: ${parseError ?? "unknown"}`
                                      : "load an image"}
                        </div>
                        {parseCoverage ? (
                            <div style={{ marginTop: 8 }}>
                                {parseCoverage.map((c) => {
                                    const [r, g, b] = colorForParseClass(c.id);
                                    return (
                                        <div
                                            key={c.id}
                                            style={{
                                                ...ROW_STYLE,
                                                margin: "4px 0",
                                                gap: 6,
                                            }}
                                        >
                                            <span
                                                style={{
                                                    flex: "0 0 12px",
                                                    height: 12,
                                                    borderRadius: 3,
                                                    background: `rgb(${r}, ${g}, ${b})`,
                                                }}
                                            />
                                            <span
                                                style={{
                                                    flex: 1,
                                                    color: "#bdbdbd",
                                                    fontSize: 11,
                                                    overflow: "hidden",
                                                    textOverflow: "ellipsis",
                                                    whiteSpace: "nowrap",
                                                }}
                                                title={c.name}
                                            >
                                                {c.name}
                                            </span>
                                            <span
                                                style={{
                                                    flex: "0 0 56px",
                                                    textAlign: "right",
                                                    color: "#f0f0f0",
                                                    fontSize: 11,
                                                    fontVariantNumeric:
                                                        "tabular-nums",
                                                }}
                                            >
                                                {(c.fraction * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        ) : null}
                        <div style={HINT_STYLE}>
                            Phase A — visualization only. Trained on clothed
                            humans (LIP); bare torsos likely classify as
                            <code> Upper-clothes</code>. Cycle assets to
                            empirically check before wiring as a pipeline
                            filter.
                        </div>
                    </>
                ) : shader === "edge" ? (
                    <>
                        <SliderRow
                            label="strength"
                            value={edgeStrength}
                            onChange={setEdgeStrength}
                            min={0}
                            max={5}
                            step={0.05}
                        />
                        <SliderRow
                            label="threshold"
                            value={edgeThreshold}
                            onChange={setEdgeThreshold}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <SliderRow
                            label="knee"
                            value={edgeKnee}
                            onChange={setEdgeKnee}
                            min={0.001}
                            max={2}
                            step={0.005}
                        />
                    </>
                ) : shader === "skin" ? (
                    <>
                        <div style={SECTION_TITLE}>luminance (Y)</div>
                        <SliderRow
                            label="y min"
                            value={skinParams.yMin}
                            onChange={setSkin("yMin")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <SliderRow
                            label="y max"
                            value={skinParams.yMax}
                            onChange={setSkin("yMax")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <div style={SECTION_TITLE}>blue chroma (Cb)</div>
                        <SliderRow
                            label="cb min"
                            value={skinParams.cbMin}
                            onChange={setSkin("cbMin")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <SliderRow
                            label="cb max"
                            value={skinParams.cbMax}
                            onChange={setSkin("cbMax")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <div style={SECTION_TITLE}>red chroma (Cr)</div>
                        <SliderRow
                            label="cr min"
                            value={skinParams.crMin}
                            onChange={setSkin("crMin")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <SliderRow
                            label="cr max"
                            value={skinParams.crMax}
                            onChange={setSkin("crMax")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <div style={SECTION_TITLE}>feather</div>
                        <SliderRow
                            label="feather"
                            value={skinParams.feather}
                            onChange={setSkin("feather")}
                            min={0}
                            max={0.1}
                            step={0.001}
                        />
                    </>
                ) : shader === "hair" ? (
                    <>
                        <div style={SECTION_TITLE}>texture</div>
                        <SliderRow
                            label="kernel"
                            value={hairParams.kernelRadius}
                            onChange={setHair("kernelRadius")}
                            min={1}
                            max={4}
                            step={0.1}
                        />
                        <SliderRow
                            label="gain"
                            value={hairParams.textureGain}
                            onChange={setHair("textureGain")}
                            min={0}
                            max={20}
                            step={0.1}
                        />
                        <SliderRow
                            label="floor"
                            value={hairParams.textureFloor}
                            onChange={setHair("textureFloor")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <SliderRow
                            label="ceil"
                            value={hairParams.textureCeil}
                            onChange={setHair("textureCeil")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <div style={SECTION_TITLE}>color filters</div>
                        <SliderRow
                            label="sat max"
                            value={hairParams.saturationMax}
                            onChange={setHair("saturationMax")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <SliderRow
                            label="luma min"
                            value={hairParams.lumaMin}
                            onChange={setHair("lumaMin")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <SliderRow
                            label="luma max"
                            value={hairParams.lumaMax}
                            onChange={setHair("lumaMax")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                    </>
                ) : shader === "lowpoly" ? (
                    <>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>mode</span>
                            <select
                                value={lowPolyParams.mode}
                                onChange={(e) =>
                                    setLowPolyParams((p) => ({
                                        ...p,
                                        mode: e.target.value as LowPolyMode,
                                    }))
                                }
                                style={INPUT_STYLE}
                            >
                                <option value="facets-edges">
                                    facets + edges
                                </option>
                                <option value="facets">facets only</option>
                                <option value="wireframe">
                                    wireframe (lines)
                                </option>
                            </select>
                        </div>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>color</span>
                            <select
                                value={lowPolyParams.colorMode}
                                onChange={(e) =>
                                    setLowPolyParams((p) => ({
                                        ...p,
                                        colorMode: e.target
                                            .value as LowPolyColorMode,
                                    }))
                                }
                                style={INPUT_STYLE}
                            >
                                <option value="grayscale">
                                    grayscale (Lambertian)
                                </option>
                                <option value="hsv">
                                    HSV (color by direction)
                                </option>
                            </select>
                        </div>
                        <div style={SECTION_TITLE}>geometry</div>
                        <SliderRow
                            label="facets"
                            value={lowPolyParams.facets}
                            onChange={(v) =>
                                setLowPolyNum("facets")(Math.round(v))
                            }
                            min={4}
                            max={32}
                            step={1}
                        />
                        <SliderRow
                            label="smoothing"
                            value={lowPolyParams.smoothRadius}
                            onChange={setLowPolyNum("smoothRadius")}
                            min={1}
                            max={10}
                            step={0.1}
                        />
                        <div style={SECTION_TITLE}>edges</div>
                        <SliderRow
                            label="threshold"
                            value={lowPolyParams.edgeThreshold}
                            onChange={setLowPolyNum("edgeThreshold")}
                            min={0}
                            max={0.3}
                            step={0.001}
                        />
                        <SliderRow
                            label="width"
                            value={lowPolyParams.edgeWidth}
                            onChange={setLowPolyNum("edgeWidth")}
                            min={0}
                            max={0.2}
                            step={0.001}
                        />
                        <div style={SECTION_TITLE}>fake light</div>
                        <SliderRow
                            label="light x"
                            value={lowPolyParams.lightX}
                            onChange={setLowPolyNum("lightX")}
                            min={-1}
                            max={1}
                            step={0.01}
                        />
                        <SliderRow
                            label="light y"
                            value={lowPolyParams.lightY}
                            onChange={setLowPolyNum("lightY")}
                            min={-1}
                            max={1}
                            step={0.01}
                        />
                    </>
                ) : shader === "circle" ? (
                    <>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>mode</span>
                            <select
                                value={circleParams.mode}
                                onChange={(e) =>
                                    setCircleParams((p) => ({
                                        ...p,
                                        mode: e.target
                                            .value as CircleDetectMode,
                                    }))
                                }
                                style={INPUT_STYLE}
                            >
                                <option value="source">
                                    source (image + 2D overlay)
                                </option>
                                <option value="accumulator">
                                    accumulator (Hough heatmap)
                                </option>
                                <option value="mask">
                                    mask (above threshold)
                                </option>
                                <option value="overlay">
                                    overlay (heatmap on dimmed source)
                                </option>
                                <option value="edges">
                                    edges (debug — pass 1 output)
                                </option>
                            </select>
                        </div>
                        <div style={SECTION_TITLE}>circle</div>
                        <SliderRow
                            label="radius"
                            value={circleParams.radius}
                            onChange={setCircleNum("radius")}
                            min={5}
                            max={400}
                            step={1}
                        />
                        <SliderRow
                            label="spread"
                            value={circleParams.radiusSpread}
                            onChange={setCircleNum("radiusSpread")}
                            min={0}
                            max={80}
                            step={1}
                        />
                        <SliderRow
                            label="samples"
                            value={circleParams.samples}
                            onChange={(v) =>
                                setCircleNum("samples")(Math.round(v))
                            }
                            min={8}
                            max={128}
                            step={1}
                        />
                        <div style={SECTION_TITLE}>output</div>
                        <SliderRow
                            label="threshold"
                            value={circleParams.minScore}
                            onChange={setCircleNum("minScore")}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        {circleParams.mode === "overlay" ? (
                            <SliderRow
                                label="bg dim"
                                value={circleParams.offMix}
                                onChange={setCircleNum("offMix")}
                                min={0}
                                max={1}
                                step={0.005}
                            />
                        ) : null}
                        <div style={SECTION_TITLE}>vector overlay</div>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>stroke</span>
                            <div
                                style={{
                                    flex: "0 0 28px",
                                    height: 28,
                                    background: `rgb(${circleStroke.join(",")})`,
                                    border: "1px solid #333",
                                    borderRadius: 4,
                                }}
                            />
                            <input
                                type="color"
                                value={`#${circleStroke
                                    .map((v) =>
                                        v.toString(16).padStart(2, "0"),
                                    )
                                    .join("")}`}
                                onChange={(e) => {
                                    const hex = e.target.value.replace("#", "");
                                    setCircleStroke([
                                        parseInt(hex.slice(0, 2), 16),
                                        parseInt(hex.slice(2, 4), 16),
                                        parseInt(hex.slice(4, 6), 16),
                                    ] as RGB);
                                }}
                                style={{
                                    flex: 1,
                                    height: 28,
                                    background: "transparent",
                                    border: "1px solid #333",
                                    borderRadius: 4,
                                    padding: 0,
                                }}
                            />
                        </div>
                        <SliderRow
                            label="stroke px"
                            value={circleStrokeWidth}
                            onChange={setCircleStrokeWidth}
                            min={0.5}
                            max={12}
                            step={0.1}
                        />
                        <SliderRow
                            label="stroke α"
                            value={circleStrokeOpacity}
                            onChange={setCircleStrokeOpacity}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>fill</span>
                            <button
                                type="button"
                                onClick={() =>
                                    setCircleFill(
                                        circleFill ? null : [255, 80, 120],
                                    )
                                }
                                style={{ ...BUTTON_STYLE, flex: "0 0 56px" }}
                            >
                                {circleFill ? "on" : "off"}
                            </button>
                            {circleFill ? (
                                <input
                                    type="color"
                                    value={`#${circleFill
                                        .map((v) =>
                                            v.toString(16).padStart(2, "0"),
                                        )
                                        .join("")}`}
                                    onChange={(e) => {
                                        const hex = e.target.value.replace(
                                            "#",
                                            "",
                                        );
                                        setCircleFill([
                                            parseInt(hex.slice(0, 2), 16),
                                            parseInt(hex.slice(2, 4), 16),
                                            parseInt(hex.slice(4, 6), 16),
                                        ] as RGB);
                                    }}
                                    style={{
                                        flex: 1,
                                        height: 28,
                                        background: "transparent",
                                        border: "1px solid #333",
                                        borderRadius: 4,
                                        padding: 0,
                                    }}
                                />
                            ) : null}
                        </div>
                        {circleFill ? (
                            <SliderRow
                                label="fill α"
                                value={circleFillOpacity}
                                onChange={setCircleFillOpacity}
                                min={0}
                                max={1}
                                step={0.005}
                            />
                        ) : null}
                        <SliderRow
                            label="max"
                            value={circleMaxCircles}
                            onChange={(v) =>
                                setCircleMaxCircles(Math.round(v))
                            }
                            min={1}
                            max={64}
                            step={1}
                        />
                        <SliderRow
                            label="readback /N"
                            value={circleReadbackEvery}
                            onChange={(v) =>
                                setCircleReadbackEvery(Math.max(1, Math.round(v)))
                            }
                            min={1}
                            max={6}
                            step={1}
                        />
                    </>
                ) : shader === "posterize" ? (
                    <>
                        <SliderRow
                            label="levels"
                            value={posterizeLevels}
                            onChange={(v) =>
                                setPosterizeLevels(Math.max(2, Math.round(v)))
                            }
                            min={2}
                            max={8}
                            step={1}
                        />
                        <div style={HINT_STYLE}>
                            ≈ {Math.pow(posterizeLevels, 3).toLocaleString()}{" "}
                            possible colors ({posterizeLevels}³)
                        </div>
                    </>
                ) : shader === "silhouette" ? (
                    <>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>mode</span>
                            <select
                                value={silMode}
                                onChange={(e) =>
                                    setSilMode(
                                        e.target.value as SilhouetteMode,
                                    )
                                }
                                style={INPUT_STYLE}
                            >
                                <option value="outline">
                                    outline (line on black)
                                </option>
                                <option value="mask">
                                    mask (white = subject)
                                </option>
                                <option value="key">
                                    key (subject on black)
                                </option>
                                <option value="overlay">
                                    overlay (line over dimmed bg)
                                </option>
                                <option value="stable">
                                    stable (threshold-invariant edges)
                                </option>
                            </select>
                        </div>

                        <div style={SECTION_TITLE}>background reference</div>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>color</span>
                            <div
                                style={{
                                    flex: "0 0 28px",
                                    height: 28,
                                    background: `rgb(${silReference.join(",")})`,
                                    border: "1px solid #333",
                                    borderRadius: 4,
                                }}
                            />
                            <input
                                type="color"
                                value={`#${silReference
                                    .map((v) =>
                                        v.toString(16).padStart(2, "0"),
                                    )
                                    .join("")}`}
                                onChange={(e) => {
                                    const hex = e.target.value.replace("#", "");
                                    setSilReference([
                                        parseInt(hex.slice(0, 2), 16),
                                        parseInt(hex.slice(2, 4), 16),
                                        parseInt(hex.slice(4, 6), 16),
                                    ] as RGB);
                                }}
                                style={{
                                    flex: "0 0 36px",
                                    height: 28,
                                    background: "transparent",
                                    border: "1px solid #333",
                                    borderRadius: 4,
                                    padding: 0,
                                }}
                            />
                            <button
                                type="button"
                                onClick={() => sampleBackground()}
                                style={{ ...BUTTON_STYLE, flex: 1 }}
                            >
                                sample from corners
                            </button>
                        </div>

                        <div style={SECTION_TITLE}>matching</div>
                        <SliderRow
                            label="smoothing"
                            value={silSmooth}
                            onChange={setSilSmooth}
                            min={0}
                            max={20}
                            step={0.1}
                        />
                        <SliderRow
                            label="threshold"
                            value={silThreshold}
                            onChange={setSilThreshold}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <SliderRow
                            label="feather"
                            value={silFeather}
                            onChange={setSilFeather}
                            min={0}
                            max={0.3}
                            step={0.001}
                        />
                        {silMode === "stable" ? (
                            <SliderRow
                                label="spread"
                                value={silSpread}
                                onChange={setSilSpread}
                                min={0}
                                max={0.5}
                                step={0.005}
                            />
                        ) : null}
                        <div style={SECTION_TITLE}>outline</div>
                        <SliderRow
                            label="thickness"
                            value={silOutline}
                            onChange={setSilOutline}
                            min={0.5}
                            max={6}
                            step={0.1}
                        />
                        {silMode === "overlay" ? (
                            <SliderRow
                                label="bg dim"
                                value={silOffMix}
                                onChange={setSilOffMix}
                                min={0}
                                max={1}
                                step={0.005}
                            />
                        ) : null}
                    </>
                ) : (
                    <>
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>mode</span>
                            <select
                                value={paletteMode}
                                onChange={(e) =>
                                    setPaletteMode(
                                        e.target.value as PaletteMaskMode,
                                    )
                                }
                                style={INPUT_STYLE}
                            >
                                <option value="posterize">
                                    posterize (palette colors)
                                </option>
                                <option value="mask">mask (white/black)</option>
                                <option value="overlay">
                                    overlay (source dimmed)
                                </option>
                            </select>
                        </div>
                        <SliderRow
                            label="size"
                            value={paletteSize}
                            onChange={(v) => setPaletteSize(Math.round(v))}
                            min={2}
                            max={16}
                            step={1}
                        />
                        <SliderRow
                            label="off mix"
                            value={offMix}
                            onChange={setOffMix}
                            min={0}
                            max={1}
                            step={0.005}
                        />
                        <div style={ROW_STYLE}>
                            <span style={LABEL_STYLE}>sample</span>
                            <button
                                type="button"
                                onClick={() => samplePalette()}
                                style={{ ...BUTTON_STYLE, flex: 1 }}
                            >
                                sample palette from current frame
                            </button>
                        </div>

                        <div style={SECTION_TITLE}>palette</div>
                        {palette.length === 0 ? (
                            <div style={{ color: "#666", fontSize: 11 }}>
                                no palette yet — load media and click sample.
                            </div>
                        ) : (
                            <>
                                <div style={SWATCH_GRID}>
                                    {palette.map((rgb, i) => {
                                        const isOn = enabled[i] ?? true;
                                        return (
                                            <button
                                                key={`${i}-${rgb.join(",")}`}
                                                onClick={() =>
                                                    setEnabled((e) =>
                                                        e.map((v, j) =>
                                                            j === i ? !v : v,
                                                        ),
                                                    )
                                                }
                                                title={`rgb(${rgb.join(", ")})${isOn ? "" : " — off"}`}
                                                style={{
                                                    ...SWATCH_BASE,
                                                    background: rgbCss(rgb),
                                                    borderColor: isOn
                                                        ? "#f0f0f0"
                                                        : "#222",
                                                    opacity: isOn ? 1 : 0.35,
                                                    transform: isOn
                                                        ? "scale(1)"
                                                        : "scale(0.92)",
                                                }}
                                                aria-label={`palette slot ${i + 1}`}
                                            />
                                        );
                                    })}
                                </div>
                                <div style={ROW_STYLE}>
                                    <button
                                        type="button"
                                        onClick={() =>
                                            setEnabled(palette.map(() => true))
                                        }
                                        style={BUTTON_STYLE}
                                    >
                                        all
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() =>
                                            setEnabled(palette.map(() => false))
                                        }
                                        style={BUTTON_STYLE}
                                    >
                                        none
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() =>
                                            setEnabled((e) => e.map((v) => !v))
                                        }
                                        style={BUTTON_STYLE}
                                    >
                                        invert
                                    </button>
                                </div>
                            </>
                        )}
                    </>
                )}

                <div style={DIVIDER} />
                <div style={ROW_STYLE}>
                    <button
                        type="button"
                        onClick={() => setAdvancedMode((v) => !v)}
                        style={{
                            ...BUTTON_STYLE,
                            flex: 1,
                            opacity: advancedMode ? 1 : 0.55,
                        }}
                    >
                        {advancedMode
                            ? "advanced ▾ (hide)"
                            : "advanced ▸ (other shaders / upload / url)"}
                    </button>
                </div>
            </div>
        </>
    );
}
