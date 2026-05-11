import {
    useEffect,
    useRef,
    useState,
    type CSSProperties,
} from "react";
import {
    loadClassifier,
    type ClassificationProbabilities,
    type NsfwClassifier,
} from "../src/classify";
import { usePipelineHandle } from "./PipelineHandleContext";
import type { ChainSpec } from "./shaders";

// Lazy-load the classifier once per page; it's a ~3 MB model and the
// nsfwjs API has no global cache, so we keep our own.
let classifierPromise: Promise<NsfwClassifier> | null = null;
function getClassifier(): Promise<NsfwClassifier> {
    if (!classifierPromise) {
        classifierPromise = loadClassifier();
    }
    return classifierPromise;
}

interface NsfwStatusOverlayProps {
    /** The terminal ChainSpec (its `entry.id` should be `"nsfwCompare"`).
     *  We read `chain.inputs[0].nodeId` and `chain.inputs[1].nodeId` to
     *  know which pipeline nodes to capture. */
    readonly chain: ChainSpec;
}

type Status =
    | { kind: "idle" }
    | { kind: "loading-model" }
    | { kind: "classifying" }
    | {
        kind: "ready";
        before: ClassificationProbabilities | null;
        after: ClassificationProbabilities | null;
    }
    | { kind: "error"; message: string };

const OVERLAY_STYLE: CSSProperties = {
    position: "absolute",
    top: 12,
    left: 12,
    right: 12,
    pointerEvents: "none",
    fontFamily:
        "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
    fontSize: 12,
    lineHeight: 1.5,
    color: "#f5f5f5",
    textShadow: "0 1px 2px rgba(0,0,0,0.85)",
};

const ROW_STYLE: CSSProperties = {
    background: "rgba(15, 15, 15, 0.78)",
    backdropFilter: "blur(8px)",
    WebkitBackdropFilter: "blur(8px)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
    borderRadius: 6,
    padding: "8px 10px",
    marginBottom: 6,
    display: "grid",
    gridTemplateColumns: "auto 1fr auto",
    columnGap: 12,
    alignItems: "baseline",
};

const LABEL_STYLE: CSSProperties = {
    color: "#888",
    textTransform: "uppercase",
    letterSpacing: 0.6,
    fontSize: 10,
};

const HEADLINE_STYLE: CSSProperties = {
    fontSize: 13,
    fontWeight: 500,
};

const PROBS_STYLE: CSSProperties = {
    color: "#bdbdbd",
    fontSize: 11,
    textAlign: "right",
    fontVariantNumeric: "tabular-nums",
};

const STATUS_STYLE: CSSProperties = {
    background: "rgba(15, 15, 15, 0.78)",
    backdropFilter: "blur(8px)",
    WebkitBackdropFilter: "blur(8px)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
    borderRadius: 6,
    padding: "8px 10px",
    color: "#bdbdbd",
};

function summarize(p: ClassificationProbabilities | null): {
    headline: string;
    detail: string;
} {
    if (!p) return { headline: "—", detail: "" };
    // Headline: highest single class.
    const entries: Array<[keyof ClassificationProbabilities, number]> = [
        ["porn", p.porn],
        ["sexy", p.sexy],
        ["hentai", p.hentai],
        ["neutral", p.neutral],
        ["drawing", p.drawing],
    ];
    entries.sort((a, b) => b[1] - a[1]);
    const top = entries[0]!;
    const headline = `${top[0]} ${(top[1] * 100).toFixed(1)}%`;
    const detail = entries
        .filter(([, v]) => v > 0.01)
        .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
        .join(" · ");
    return { headline, detail };
}

export function NsfwStatusOverlay({ chain }: NsfwStatusOverlayProps) {
    const { handle, version } = usePipelineHandle();
    const [status, setStatus] = useState<Status>({ kind: "idle" });

    // Stable nodeIds to depend on across renders — chain identity changes
    // on every paramsByNode update, which would re-fire classification
    // far more often than we need. Pin to the inputs' nodeIds.
    const beforeId = chain.inputs[0]?.nodeId ?? null;
    const afterId = chain.inputs[1]?.nodeId ?? null;

    // Bumped after each classification finishes; the next chain identity
    // change triggers a fresh run.
    const generationRef = useRef(0);

    useEffect(() => {
        let cancelled = false;
        generationRef.current += 1;
        const myGen = generationRef.current;

        const run = async () => {
            if (!beforeId || !afterId) {
                setStatus({ kind: "ready", before: null, after: null });
                return;
            }

            // Model load — only on first run, but display the state.
            if (!classifierPromise) {
                setStatus({ kind: "loading-model" });
            }
            let classifier: NsfwClassifier;
            try {
                classifier = await getClassifier();
            } catch (e) {
                if (cancelled || myGen !== generationRef.current) return;
                setStatus({
                    kind: "error",
                    message: `failed to load NSFW model: ${
                        e instanceof Error ? e.message : String(e)
                    }`,
                });
                return;
            }
            if (cancelled || myGen !== generationRef.current) return;

            setStatus({ kind: "classifying" });

            // Capture both nodes from the pipeline. If either FBO isn't
            // ready yet, wait a frame and retry up to ~1 second.
            const pipeline = handle.pipeline;
            if (!pipeline) return;
            let beforeImg: ImageData | null = null;
            let afterImg: ImageData | null = null;
            for (let attempt = 0; attempt < 30; attempt++) {
                beforeImg = pipeline.captureNodeImageData(beforeId);
                afterImg = pipeline.captureNodeImageData(afterId);
                if (beforeImg && afterImg) break;
                await new Promise((r) => requestAnimationFrame(() => r(null)));
                if (cancelled || myGen !== generationRef.current) return;
            }

            if (!beforeImg || !afterImg) {
                setStatus({
                    kind: "error",
                    message: "inputs never became ready",
                });
                return;
            }

            try {
                const beforeProbs = await classifier.classify(beforeImg);
                if (cancelled || myGen !== generationRef.current) return;
                const afterProbs = await classifier.classify(afterImg);
                if (cancelled || myGen !== generationRef.current) return;
                setStatus({
                    kind: "ready",
                    before: beforeProbs,
                    after: afterProbs,
                });
            } catch (e) {
                if (cancelled || myGen !== generationRef.current) return;
                setStatus({
                    kind: "error",
                    message: `classification failed: ${
                        e instanceof Error ? e.message : String(e)
                    }`,
                });
            }
        };
        run();
        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [beforeId, afterId, chain, version]);

    if (status.kind === "idle" || status.kind === "loading-model") {
        return (
            <div className="newtone-nsfw-status" style={OVERLAY_STYLE}>
                <div style={STATUS_STYLE}>
                    {status.kind === "loading-model"
                        ? "loading NSFW model…"
                        : "preparing…"}
                </div>
            </div>
        );
    }
    if (status.kind === "classifying") {
        return (
            <div className="newtone-nsfw-status" style={OVERLAY_STYLE}>
                <div style={STATUS_STYLE}>classifying…</div>
            </div>
        );
    }
    if (status.kind === "error") {
        return (
            <div className="newtone-nsfw-status" style={OVERLAY_STYLE}>
                <div style={{ ...STATUS_STYLE, color: "#ff8888" }}>
                    {status.message}
                </div>
            </div>
        );
    }

    const before = summarize(status.before);
    const after = summarize(status.after);

    return (
        <div className="newtone-nsfw-status" style={OVERLAY_STYLE}>
            <div className="newtone-nsfw-row" style={ROW_STYLE}>
                <div style={LABEL_STYLE}>before</div>
                <div style={HEADLINE_STYLE}>{before.headline}</div>
                <div style={PROBS_STYLE}>{before.detail}</div>
            </div>
            <div className="newtone-nsfw-row" style={ROW_STYLE}>
                <div style={LABEL_STYLE}>after</div>
                <div style={HEADLINE_STYLE}>{after.headline}</div>
                <div style={PROBS_STYLE}>{after.detail}</div>
            </div>
        </div>
    );
}
