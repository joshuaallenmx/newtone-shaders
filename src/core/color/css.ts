// Hex parsing is inline; everything else delegates to the browser's color
// parser via a hidden probe element so we get correct handling of modern
// formats (oklch, color(), hsl, named colors) without shipping a parser.

let probe: HTMLDivElement | null = null;

export function hexToRgbBytes(hex: string): [number, number, number] {
    const m = hex.replace(/^#/, "").trim();
    const expanded =
        m.length === 3
            ? m
                  .split("")
                  .map((d) => d + d)
                  .join("")
            : m.length === 6
                ? m
                : "ffffff";
    const v = parseInt(expanded, 16);
    if (Number.isNaN(v)) return [255, 255, 255];
    return [(v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];
}

export function cssColorToRgbBytes(color: string): [number, number, number] {
    if (color.startsWith("#")) return hexToRgbBytes(color);
    if (typeof document === "undefined") return [255, 255, 255];
    if (!probe) {
        probe = document.createElement("div");
        probe.style.position = "absolute";
        probe.style.left = "-9999px";
        probe.style.visibility = "hidden";
    }
    probe.style.color = "";
    probe.style.color = color;
    document.body.appendChild(probe);
    const computed = getComputedStyle(probe).color;
    document.body.removeChild(probe);
    const m = computed.match(/rgba?\(([^)]+)\)/i);
    if (!m) return [255, 255, 255];
    const parts = m[1]
        .split(/[\s,/]+/)
        .filter(Boolean)
        .map((s) => parseFloat(s));
    return [
        Math.round(Math.max(0, Math.min(255, parts[0] || 0))),
        Math.round(Math.max(0, Math.min(255, parts[1] || 0))),
        Math.round(Math.max(0, Math.min(255, parts[2] || 0))),
    ];
}

const SRGB_BREAKPOINT = 0.04045;

function srgbToLinear(byte: number): number {
    const x = byte / 255;
    return x <= SRGB_BREAKPOINT ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

export function cssColorToLinearRgb(color: string): [number, number, number] {
    const [r, g, b] = cssColorToRgbBytes(color);
    return [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)];
}
