import { useCallback, useEffect, useLayoutEffect, useState } from "react";

export type Theme = "light" | "dark";

export const THEME_STORAGE_KEY = "scenestudio.theme";

export function isTheme(value: unknown): value is Theme {
  return value === "light" || value === "dark";
}

export function resolveTheme(stored: string | null): Theme {
  return isTheme(stored) ? stored : "dark";
}

function readStoredTheme(): Theme | null {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isTheme(stored) ? stored : null;
  } catch {
    return null;
  }
}

function initialTheme(): Theme {
  const bootTheme = document.documentElement.dataset.theme;
  if (isTheme(bootTheme)) return bootTheme;
  return resolveTheme(readStoredTheme());
}

function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

/** Theme state with dark as the first-run default and explicit choices persisted. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  // Layout timing keeps React-driven changes and Three.js token reads in the same
  // paint. The inline bootstrap in index.html handles the earlier, pre-React paint.
  useLayoutEffect(() => applyTheme(theme), [theme]);

  useEffect(() => {
    const followOtherTabs = (event: StorageEvent) => {
      if (event.key !== THEME_STORAGE_KEY) return;
      setTheme(resolveTheme(event.newValue));
    };
    window.addEventListener("storage", followOtherTabs);
    return () => {
      window.removeEventListener("storage", followOtherTabs);
    };
  }, []);

  const chooseTheme = useCallback((next: Theme) => {
    applyTheme(next);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Storage can be disabled; the in-memory choice still works for this session.
    }
    setTheme(next);
  }, []);

  const toggleTheme = useCallback(
    () => chooseTheme(theme === "light" ? "dark" : "light"),
    [chooseTheme, theme],
  );

  return { theme, chooseTheme, toggleTheme };
}

/** Read a theme token for Three.js, whose materials cannot consume CSS directly. */
export function themeToken(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  return (
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() ||
    fallback
  );
}
