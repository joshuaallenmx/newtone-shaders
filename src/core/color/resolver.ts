/**
 * Translates a token reference like "$text" or "$fill.deep" into a concrete
 * CSS color. The shaders package never imports a token system; the consuming
 * app injects a resolver via context (see `react/ColorResolverProvider`).
 */
export type ColorResolver = (input: string) => string;

/** Pass-through resolver used when no resolver is provided. */
export const identityResolver: ColorResolver = (s) => s;
