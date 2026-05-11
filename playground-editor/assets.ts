const ASSET_MODULES = import.meta.glob<string>(
    "../playground/assets/*.{png,PNG,jpg,JPG,jpeg,JPEG,gif,GIF,webp,WEBP,avif,AVIF,svg,SVG}",
    { query: "?url", import: "default", eager: true },
);

export interface Asset {
    readonly name: string;
    readonly url: string;
}

export const ASSETS: readonly Asset[] = Object.entries(ASSET_MODULES)
    .map(([key, url]) => ({ name: key.replace(/^.*\//, ""), url }))
    .sort((a, b) => a.name.localeCompare(b.name));

export function findAssetByName(name: string | undefined): Asset | undefined {
    if (!name) return undefined;
    return ASSETS.find((a) => a.name === name);
}
