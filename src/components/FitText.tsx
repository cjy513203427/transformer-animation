import { useLayoutEffect, useRef, useState, type SVGProps } from 'react';

type Props = Omit<SVGProps<SVGTextElement>, 'children' | 'ref'> & {
  children: string;
  /** Room for the text, in SVG user units. */
  width: number;
  /** Lines the text may wrap onto before it shrinks; 1 only shrinks. */
  maxLines?: number;
  /** Where extra lines go: centred on y, below it, or above it. */
  grow?: 'center' | 'down' | 'up';
  /** Smallest font scale; below it the text is left as is rather than becoming unreadable. */
  minScale?: number;
};

let context: CanvasRenderingContext2D | null | undefined;
function measurer() {
  if (context === undefined)
    context =
      typeof document === 'undefined' || !document.fonts
        ? null
        : document.createElement('canvas').getContext('2d');
  return context;
}

/** Splits after spaces, hyphens, slashes and “·”, keeping each break character on the line it ends. */
const pieces = (text: string) => text.split(/(?<=[\s\-/·])/);

function wrap(text: string, room: number, measure: (s: string) => number) {
  const lines: string[] = [];
  let line = '';
  for (const piece of pieces(text)) {
    if (line && measure((line + piece).trimEnd()) > room) {
      lines.push(line.trimEnd());
      line = piece.trimStart();
    } else line += piece;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

/**
 * SVG text that stays inside `width`. Text that already fits renders exactly like a plain
 * <text> (so the Chinese layout is untouched); longer translations wrap at spaces or hyphens
 * onto up to `maxLines` lines and shrink only as far as needed.
 */
export function FitText({
  children,
  width,
  maxLines = 2,
  grow = 'center',
  minScale = 0.72,
  x,
  y,
  style,
  ...rest
}: Props) {
  const ref = useRef<SVGTextElement>(null);
  const [layout, setLayout] = useState<{ lines: string[]; size: number } | null>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    const ctx = measurer();
    if (!el || !ctx) return;
    const fit = () => {
      const previous = el.style.fontSize;
      el.style.fontSize = '';
      const css = getComputedStyle(el);
      const base = parseFloat(css.fontSize);
      el.style.fontSize = previous;
      ctx.font = `${css.fontStyle} ${css.fontWeight} ${base}px ${css.fontFamily}`;
      const measure = (s: string) => ctx.measureText(s).width;
      if (measure(children) <= width) return setLayout(null);
      for (let scale = 1; scale >= minScale - 1e-9; scale -= 0.04) {
        const lines = wrap(children, width / scale, measure);
        if (lines.length <= maxLines && lines.every((line) => measure(line) * scale <= width))
          return setLayout({ lines, size: base * scale });
      }
      setLayout({ lines: wrap(children, width / minScale, measure), size: base * minScale });
    };
    let alive = true;
    const refit = () => alive && fit();
    fit();
    // Web fonts and breakpoint font sizes can change the measurement after the first paint.
    void document.fonts.ready.then(refit);
    window.addEventListener('resize', refit);
    return () => {
      alive = false;
      window.removeEventListener('resize', refit);
    };
  }, [children, width, maxLines, minScale]);

  if (!layout)
    return (
      <text ref={ref} x={x} y={y} style={style} {...rest}>
        {children}
      </text>
    );
  const step = layout.size * 1.15;
  const extra = (layout.lines.length - 1) * step;
  const first = grow === 'center' ? -extra / 2 : grow === 'up' ? -extra : 0;
  return (
    <text ref={ref} x={x} y={y} style={{ ...style, fontSize: layout.size }} {...rest}>
      {layout.lines.map((line, i) => (
        <tspan key={i} x={x} dy={i === 0 ? first : step}>
          {line}
        </tspan>
      ))}
    </text>
  );
}
