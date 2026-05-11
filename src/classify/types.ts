/**
 * Tier 1 of the detection pipeline: image-level NSFW classification.
 *
 * Output is a fixed-shape probability distribution across the five NSFWJS
 * classes. Probabilities sum to ~1; consumers typically threshold against
 * `porn + sexy + hentai` to decide whether to invoke later tiers.
 */
export interface ClassificationProbabilities {
    readonly porn: number;
    readonly sexy: number;
    readonly hentai: number;
    readonly neutral: number;
    readonly drawing: number;
}

export type ClassifyInput =
    | HTMLImageElement
    | HTMLCanvasElement
    | HTMLVideoElement
    | ImageData;

export interface NsfwClassifier {
    classify(input: ClassifyInput): Promise<ClassificationProbabilities>;
    dispose(): void;
}

export interface LoadClassifierOptions {
    /**
     * Optional model URL override. When omitted, NSFWJS loads its default
     * graph from the bundled CDN. Pass a self-hosted URL to avoid the
     * CDN dependency in production.
     */
    readonly modelUrl?: string;
}
