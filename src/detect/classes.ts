/**
 * NudeNet v3 class labels in the order the 320n / 640m ONNX exports emit
 * them. The class index from `argmax` of the per-row score channels maps
 * directly into this list.
 *
 * Source: https://github.com/notAI-tech/NudeNet/blob/v3/nudenet/nudenet.py
 */
export const NUDENET_CLASSES = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
] as const;

export type NudeNetClass = (typeof NUDENET_CLASSES)[number];
