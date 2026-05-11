import * as nsfwjs from "nsfwjs";
import type {
    ClassificationProbabilities,
    ClassifyInput,
    LoadClassifierOptions,
    NsfwClassifier,
} from "./types";

const CLASS_KEY: Record<string, keyof ClassificationProbabilities> = {
    Porn: "porn",
    Sexy: "sexy",
    Hentai: "hentai",
    Neutral: "neutral",
    Drawing: "drawing",
};

function emptyProbs(): ClassificationProbabilities {
    return { porn: 0, sexy: 0, hentai: 0, neutral: 0, drawing: 0 };
}

export async function loadClassifier(
    options: LoadClassifierOptions = {},
): Promise<NsfwClassifier> {
    const model = options.modelUrl
        ? await nsfwjs.load(options.modelUrl)
        : await nsfwjs.load();
    let disposed = false;

    return {
        async classify(input: ClassifyInput) {
            if (disposed) throw new Error("classifier disposed");
            const predictions = await model.classify(input);
            const out: Record<keyof ClassificationProbabilities, number> =
                emptyProbs();
            for (const p of predictions) {
                const key = CLASS_KEY[p.className];
                if (key) out[key] = p.probability;
            }
            return out;
        },
        dispose() {
            disposed = true;
        },
    };
}
