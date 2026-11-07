import { Fragment, useMemo, type CSSProperties } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

function render(tex: string, displayMode: boolean) {
  return katex.renderToString(tex, { throwOnError: false, displayMode, trust: false });
}

function escapeHtml(text: string) {
  return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

/** Inline math inside a sentence, e.g. <Tex tex="W_Q" />. */
export function Tex({ tex }: { tex: string }) {
  const html = useMemo(() => render(tex, false), [tex]);
  return <span className="tex" dangerouslySetInnerHTML={{ __html: html }} />;
}

const MATH = /\$([^$]+)\$/g;

/** Prose with inline math: "每个头除以 $\sqrt{d_k}$" renders the part between dollar signs as KaTeX. */
export function Rich({ text }: { text: string }) {
  const parts = text.split(MATH);
  // split() with one capture group alternates: plain, math, plain, math, ...
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? <Tex key={i} tex={part} /> : <Fragment key={i}>{part}</Fragment>,
      )}
    </>
  );
}

/** The same text with the math markers removed, for aria-labels and titles. */
export function plain(text: string) {
  return text.replace(MATH, (_, tex: string) => tex);
}

type Anchor = 'start' | 'middle' | 'end';

interface SvgBase {
  x: number;
  /** Text baseline, like SVG <text>. */
  y: number;
  anchor?: Anchor;
  size?: number;
  width?: number;
  className?: string;
  style?: CSSProperties;
}

/**
 * Math inside an <svg>. Rendered through foreignObject so it scales with the viewBox like the
 * neighbouring <text>. It never takes pointer events, so a clickable node underneath stays clickable.
 */
function SvgHtml({
  x,
  y,
  html,
  anchor = 'middle',
  size = 13,
  width = 200,
  className,
  style,
}: SvgBase & { html: string }) {
  const height = Math.round(size * 2.2);
  const left = anchor === 'middle' ? x - width / 2 : anchor === 'end' ? x - width : x;
  return (
    <foreignObject
      x={left}
      y={y - size * 1.4}
      width={width}
      height={height}
      pointerEvents="none"
      aria-hidden="true"
    >
      <div
        className={`svg-tex ${className ?? ''}`}
        style={{
          textAlign: anchor === 'middle' ? 'center' : anchor === 'end' ? 'right' : 'left',
          fontSize: size,
          lineHeight: `${height}px`,
          ...style,
        }}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </foreignObject>
  );
}

/** One formula, optionally preceded by plain text such as a circled step number. */
export function SvgTex({ tex, prefix = '', ...rest }: SvgBase & { tex: string; prefix?: string }) {
  const html = useMemo(() => escapeHtml(prefix) + render(tex, false), [prefix, tex]);
  return <SvgHtml {...rest} html={html} />;
}

/** A label that mixes plain text and $math$, like <Rich> but inside an <svg>. */
export function SvgRich({ text, ...rest }: SvgBase & { text: string }) {
  const html = useMemo(
    () =>
      text
        .split(MATH)
        .map((part, i) => (i % 2 === 1 ? render(part, false) : escapeHtml(part)))
        .join(''),
    [text],
  );
  return <SvgHtml {...rest} html={html} />;
}
