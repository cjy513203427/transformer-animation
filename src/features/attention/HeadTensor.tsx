import { Cuboid } from '../../components/Oblique';
import { formatNumber as fmt, type Matrix } from '../../math/attention';
import { viewHeads } from '../../math/multihead';
import { HEAD_COLORS } from './MultiHeadDiagram';
import { T } from '../../i18n';

const CW = 34;
const CH = 22;
/** Same cell size as the flat diagram above, so the two figures share one scale. */
const MAIN_WIDTH = 480;
const CARD: [number, number] = [6, 3];
/** Advance of an 11px Consolas label: 0.55 em per Latin glyph, a full em for CJK. */
const labelWidth = (text: string) =>
  [...text].reduce((w, ch) => w + (/[⺀-￿]/.test(ch) ? 11 : 6.05), 0);

/**
 * A [T, d_model] activation viewed as [T, num_heads, d_key], drawn with three real axes: token
 * (down), d_key (right) and head (up-right). Cell (t, h, d) has one fixed place; switching between
 * “token first” and “head first” only changes which slices are outlined, exactly as a transpose of
 * the first two axes changes the index order and nothing else.
 */
export function HeadTensor({
  matrix,
  numHeads,
  by,
  query,
  name,
  tone,
}: {
  matrix: Matrix;
  numHeads: number;
  by: 'token' | 'head';
  query: number;
  name: string;
  tone: 'query' | 'output';
}) {
  const view = viewHeads(matrix, numHeads),
    tokens = matrix.length,
    dKey = matrix[0].length / numHeads,
    gapX = dKey * CW + 18,
    gapY = Math.round(gapX * 0.42),
    ox = 58,
    oy = 40 + (numHeads - 1) * gapY,
    sx = (h: number) => ox + h * gapX,
    sy = (h: number) => oy - h * gapY,
    last = numHeads - 1,
    right = sx(last) + dKey * CW + CARD[0],
    sliceLabel = T('每片 {#0}_h：[{#1}, {#2}]', name, tokens, dKey),
    tokensLabel = T('{#0} 个 token 都在', tokens),
    // Room for the two side labels (11px monospace, wider CJK glyphs); never less than before.
    width = right + Math.max(124, 16 + Math.max(labelWidth(sliceLabel), labelWidth(tokensLabel))),
    height = oy + tokens * CH + 44;
  const heads = Array.from({ length: numHeads }, (_, h) => h),
    rows = Array.from({ length: tokens }, (_, t) => t);
  const band = (t: number) =>
    [
      [sx(0), sy(0) + t * CH],
      [sx(last), sy(last) + t * CH],
      [sx(last) + dKey * CW, sy(last) + t * CH],
      [sx(last) + dKey * CW, sy(last) + (t + 1) * CH],
      [sx(0) + dKey * CW, sy(0) + (t + 1) * CH],
      [sx(0), sy(0) + (t + 1) * CH],
    ]
      .map((p) => p.join(','))
      .join(' ');
  const corners: [number, number][] = [
    [0, 0],
    [dKey * CW, 0],
    [dKey * CW, tokens * CH],
    [0, tokens * CH],
  ];
  const shape =
    by === 'token' ? `[${tokens}, ${numHeads}, ${dKey}]` : `[${numHeads}, ${tokens}, ${dKey}]`;
  return (
    <svg
      className="ht-figure"
      viewBox={`0 0 ${width} ${height}`}
      style={{ ['--ht-width' as string]: `${(width / MAIN_WIDTH) * 100}%` }}
      role="img"
      aria-label={
        by === 'token'
          ? T(
              '{#0} 形状 {#1}：先按 token 取，{#2}',
              name,
              shape,
              rows
                .map((t) => T('{#0}[t{#1}] 是 {#2} 个头各 {#3} 维', name, t, numHeads, dKey))
                .join(T('，')),
            )
          : T(
              '{#0} 形状 {#1}：先按 head 取，{#2}',
              name,
              shape,
              heads
                .map((h) => T('head {#0} 是 {#1} 个 token 各 {#2} 维', h + 1, tokens, dKey))
                .join(T('，')),
            )
      }
    >
      <title>
        {T(
          '三条轴：token 向下、d_key 向右、head 向右上；每个格子的位置只由它的 (token, head, d) 决定',
        )}
      </title>
      {numHeads > 1 &&
        corners.map(([cx, cy], i) => (
          <line
            key={i}
            x1={sx(0) + cx}
            y1={sy(0) + cy}
            x2={sx(last) + cx}
            y2={sy(last) + cy}
            className="ht-rail"
          />
        ))}
      <g className="ht-overlay" style={{ opacity: by === 'token' ? 1 : 0 }} aria-hidden="true">
        {rows.map((t) => (
          <polygon
            key={t}
            points={band(t)}
            className={t === query ? 'ht-band ht-band-on' : 'ht-band'}
          />
        ))}
      </g>
      {heads
        .slice()
        .reverse()
        .map((h) => (
          <g key={h}>
            <Cuboid
              x={sx(h)}
              y={sy(h)}
              cols={dKey}
              rows={tokens}
              depth={1}
              cell={[CW, CH]}
              step={CARD}
              fill={() => `ht-cell ht-${tone}`}
              text={(c, r) => fmt(view[r][h][c], 2)}
              textClass={`ht-text ht-text-${tone}`}
            />
            {numHeads > 1 && (
              <rect x={sx(h)} y={sy(h) - 6} width={dKey * CW} height="3.5" fill={HEAD_COLORS[h]} />
            )}
            <rect
              className="mh-query-row"
              x={sx(h) - 1}
              y={sy(h) + query * CH - 1}
              width={dKey * CW + 2}
              height={CH + 2}
              rx="3"
            />
            <text x={sx(h)} y={sy(h) - 11} className="ht-label" style={{ fill: HEAD_COLORS[h] }}>
              head {h + 1}
            </text>
          </g>
        ))}
      <g className="ht-overlay" style={{ opacity: by === 'head' ? 1 : 0 }} aria-hidden="true">
        {heads.map((h) => (
          <g key={h}>
            <rect
              x={sx(h) - 5}
              y={sy(h) - 23}
              width={dKey * CW + CARD[0] + 10}
              height={tokens * CH + 44}
              rx="5"
              className="ht-frame"
              stroke={HEAD_COLORS[h]}
            />
            <text
              x={sx(h) + (dKey * CW) / 2}
              y={sy(h) + tokens * CH + 15}
              textAnchor="middle"
              className="ht-label"
              style={{ fill: HEAD_COLORS[h] }}
            >
              {name}
              {h + 1}
            </text>
          </g>
        ))}
        <text x={right + 8} y={sy(last) + 15} className="ht-band-label ht-band-label-on">
          {sliceLabel}
        </text>
        <text x={right + 8} y={sy(last) + 33} className="ht-band-label">
          {tokensLabel}
        </text>
      </g>
      {rows.map((t) => (
        <text
          key={t}
          x={ox - 7}
          y={oy + t * CH + 15}
          textAnchor="end"
          className="ht-label ht-muted"
        >
          t{t}
        </text>
      ))}
      <text x={ox - 7} y={oy - 1} textAnchor="end" className="ht-axis">
        token ↓
      </text>
      <text
        x={ox + (dKey * CW) / 2}
        y={oy + tokens * CH + 36}
        textAnchor="middle"
        className="ht-axis"
      >
        d_key = {dKey} →
      </text>
      <text x={right + 8} y={sy(last) - 8} className="ht-axis">
        {numHeads > 1 ? T('head ↗（{#0} 个）', numHeads) : T('head 轴长度为 1')}
      </text>
      <g className="ht-overlay" style={{ opacity: by === 'token' ? 1 : 0 }} aria-hidden="true">
        {rows.map((t) => (
          <text
            key={t}
            x={right + 8}
            y={sy(last) + t * CH + 15}
            className={t === query ? 'ht-band-label ht-band-label-on' : 'ht-band-label'}
          >
            {name}[t{t}] · [{numHeads}, {dKey}]
          </text>
        ))}
      </g>
    </svg>
  );
}
