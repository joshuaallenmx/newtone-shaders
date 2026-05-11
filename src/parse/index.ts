export { loadHumanParser } from "./segformer";
export {
    parseToMask,
    parseToVisualizationCanvas,
    colorForParseClass,
    type ParseVisualizationOptions,
} from "./mask";
export {
    HUMAN_PARSE_CLASSES,
    type HumanParseClass,
    type HumanParser,
    type LoadHumanParserOptions,
    type ParseDevice,
    type ParseDtype,
    type ParseProgress,
    type ParseResult,
    type ParseSource,
} from "./types";

/**
 * Default class set for the segment pipeline's parse-as-negative-filter:
 * classes the LIP-trained SegFormer is highly confident on AND that we
 * never want in a censor mask. Skips the four limb classes plus
 * `Upper-clothes`/`Skirt`/`Pants`/`Dress` because those are where bare
 * skin tends to misclassify on out-of-distribution NSFW imagery.
 *
 * Indexed by class id from `HUMAN_PARSE_CLASSES`:
 *   0  Background    9  Left-shoe
 *   1  Hat          10  Right-shoe
 *   2  Hair         11  Face
 *   3  Sunglasses   16  Bag
 *   8  Belt         17  Scarf
 */
export const DEFAULT_PARSE_EXCLUSION: ReadonlyArray<number> = [
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17,
];
