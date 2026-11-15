import { useEffect, useState } from 'react';

const PREFERENCES_KEY = 'llm-reading-layout-v1';
const DESKTOP_QUERY = '(min-width: 901px)';
export const MIN_SIDEBAR_WIDTH = 220;
export const MAX_SIDEBAR_WIDTH = 400;
export const DEFAULT_SIDEBAR_WIDTH = 280;
export const MIN_READING_ZOOM = 80;
export const MAX_READING_ZOOM = 130;

function readPreferences() {
  const defaults = { width: DEFAULT_SIDEBAR_WIDTH, zoom: 100 };
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(PREFERENCES_KEY) ?? 'null');
    if (!saved || typeof saved !== 'object') return defaults;
    const { width, zoom } = saved as Record<string, unknown>;
    return {
      width:
        typeof width === 'number' && Number.isFinite(width)
          ? Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, width))
          : defaults.width,
      zoom:
        typeof zoom === 'number' && Number.isFinite(zoom)
          ? Math.max(MIN_READING_ZOOM, Math.min(MAX_READING_ZOOM, Math.round(zoom / 10) * 10))
          : defaults.zoom,
    };
  } catch {
    return defaults;
  }
}

/** Reading preferences stay on this device and never enter experiment share links. */
export function useReadingLayout() {
  const [preferences, setPreferences] = useState(readPreferences);
  const [desktop, setDesktop] = useState(() => window.matchMedia(DESKTOP_QUERY).matches);
  const [directoryOpen, setDirectoryOpen] = useState(desktop);

  useEffect(() => {
    const media = window.matchMedia(DESKTOP_QUERY);
    const change = () => {
      setDesktop(media.matches);
      setDirectoryOpen(media.matches);
    };
    media.addEventListener('change', change);
    return () => media.removeEventListener('change', change);
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(PREFERENCES_KEY, JSON.stringify(preferences));
    } catch {
      // Storage is optional; all controls still work for the current page.
    }
  }, [preferences]);

  return {
    ...preferences,
    desktop,
    directoryOpen,
    setDirectoryOpen,
    setWidth: (width: number) =>
      setPreferences((value) => ({
        ...value,
        width: Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, Math.round(width))),
      })),
    setZoom: (zoom: number) =>
      setPreferences((value) => ({
        ...value,
        zoom: Math.max(MIN_READING_ZOOM, Math.min(MAX_READING_ZOOM, zoom)),
      })),
  };
}
