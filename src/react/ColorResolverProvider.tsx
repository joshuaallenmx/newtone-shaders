import { createContext, useContext, type ReactNode } from "react";
import { identityResolver, type ColorResolver } from "../core/color";

const ColorResolverContext = createContext<ColorResolver>(identityResolver);

export interface ColorResolverProviderProps {
    readonly resolver: ColorResolver;
    readonly children: ReactNode;
}

/**
 * Inject a `ColorResolver` for descendant shaders. The shaders package never
 * imports a token system; consumers wrap a token-aware resolver here.
 *
 * @example
 * <ColorResolverProvider resolver={(s) => resolveTokenRef(s, surfaceTokens)}>
 *   <LiquidMetal tint="$text" />
 * </ColorResolverProvider>
 */
export function ColorResolverProvider({
    resolver,
    children,
}: ColorResolverProviderProps) {
    return (
        <ColorResolverContext.Provider value={resolver}>
            {children}
        </ColorResolverContext.Provider>
    );
}

export function useColorResolver(): ColorResolver {
    return useContext(ColorResolverContext);
}
