/**
 * Make an element fill its parent visually even when the parent has padding
 * and content-driven height.
 *
 * Sets the parent to `height: 100%` (so it stretches inside a flex column),
 * then bleeds the element past the parent's padding via negative margins +
 * `width: calc(100% + ...)`. The element ends up edge-to-edge of the parent's
 * outer box while the parent still applies its padding to its other children.
 */
export interface BleedFitOptions {
    /** Force `parent.height: 100%` so the parent stretches. @default true */
    readonly forceParentHeight?: boolean;
    /** Bleed past the parent's padding via negative margins. @default true */
    readonly bleedParentPadding?: boolean;
}

export interface BleedFitController {
    /** Re-measure padding and re-apply (call from a ResizeObserver). */
    readonly update: () => void;
    /** Restore the parent's and element's inline styles. */
    readonly dispose: () => void;
}

interface ParentSnapshot {
    height: string;
    flex: string;
    minHeight: string;
}

export function applyBleedFit(
    el: HTMLElement,
    opts: BleedFitOptions = {},
): BleedFitController {
    const forceParentHeight = opts.forceParentHeight ?? true;
    const bleedParentPadding = opts.bleedParentPadding ?? true;
    const parent = el.parentElement;
    if (!parent) {
        return { update: () => {}, dispose: () => {} };
    }
    const original: ParentSnapshot = {
        height: parent.style.height,
        flex: parent.style.flex,
        minHeight: parent.style.minHeight,
    };
    if (forceParentHeight) {
        parent.style.height = "100%";
        parent.style.flex = "1 1 100%";
        parent.style.minHeight = "0";
    }

    const update = () => {
        if (!bleedParentPadding) return;
        const cs = getComputedStyle(parent);
        const pl = parseFloat(cs.paddingLeft) || 0;
        const pr = parseFloat(cs.paddingRight) || 0;
        const pt = parseFloat(cs.paddingTop) || 0;
        const pb = parseFloat(cs.paddingBottom) || 0;
        el.style.marginLeft = pl ? `-${pl}px` : "0";
        el.style.marginRight = pr ? `-${pr}px` : "0";
        el.style.marginTop = pt ? `-${pt}px` : "0";
        el.style.marginBottom = pb ? `-${pb}px` : "0";
        el.style.width = `calc(100% + ${pl + pr}px)`;
        el.style.height = `calc(100% + ${pt + pb}px)`;
    };
    update();

    return {
        update,
        dispose: () => {
            parent.style.height = original.height;
            parent.style.flex = original.flex;
            parent.style.minHeight = original.minHeight;
            el.style.marginLeft = "";
            el.style.marginRight = "";
            el.style.marginTop = "";
            el.style.marginBottom = "";
            el.style.width = "";
            el.style.height = "";
        },
    };
}
