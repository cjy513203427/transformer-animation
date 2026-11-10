import { useId } from 'react';
import { formatNumber as fmt } from '../../math/attention';
import type { Chain, Plot, Ranking, Strip } from '../../math/p1/shared';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import { FitText } from '../../components/FitText';

const W = 430;
/** Rough width of an 11px label: CJK glyphs are square, Latin glyphs about 0.6 em. */
export function textWidth(text: string, size = 11) {
  let width = 0;
  for (const ch of text) width += /[⺀-￿]/.test(ch) ? size : size * 0.62;
  return width;
}
const clamp = (v: number, low: number, high: number) => Math.min(high, Math.max(low, v));

/** About `count` round tick values (1, 2 or 5 × 10ⁿ apart) inside [min, max]. */
export function niceTicks(min: number, max: number, count = 5) {
  const span = max - min;
  if (!(span > 0)) return [min];
  const raw = span / count,
    magnitude = 10 ** Math.floor(Math.log10(raw)),
    norm = raw / magnitude,
    step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * magnitude,
    first = Math.ceil(min / step - 1e-9);
  const ticks: number[] = [];
  for (let i = first; i * step <= max + step * 1e-9; i++)
    ticks.push(Number((i * step).toPrecision(12)));
  return ticks;
}

/** Push label positions apart to at least `gap`, keeping their order, within [low, high]. */
function spread(values: number[], gap: number, low: number, high: number) {
  const order = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v),
    placed = order.map((o) => o.v);
  for (let k = 1; k < placed.length; k++) placed[k] = Math.max(placed[k], placed[k - 1] + gap);
  for (let k = placed.length - 1; k >= 0; k--) {
    placed[k] = Math.min(placed[k], k === placed.length - 1 ? high : placed[k + 1] - gap);
    placed[k] = Math.max(placed[k], low);
  }
  const result = new Array<number>(values.length);
  order.forEach((o, k) => (result[o.i] = placed[k]));
  return result;
}

/** Guide labels: coincident guides share one label, overlapping labels move to another row. */
function guideRows(guides: { px: number; label: string }[], left: number, right: number) {
  const merged: { px: number; label: string }[] = [];
  for (const g of [...guides].sort((a, b) => a.px - b.px)) {
    const last = merged[merged.length - 1];
    if (last && Math.abs(last.px - g.px) < 3) last.label += ',' + g.label;
    else merged.push({ ...g });
  }
  const ends: number[] = [];
  return merged.map((m) => {
    const width = textWidth(m.label),
      center = clamp(m.px, left + width / 2, right - width / 2);
    let row = ends.findIndex((end) => center - width / 2 > end + 6);
    if (row === -1) {
      row = ends.length;
      ends.push(center + width / 2);
    } else ends[row] = center + width / 2;
    return { ...m, center, row };
  });
}

const DASH = ['', '7 4', '2 3'];
function Marker({
  series = 0,
  x,
  y,
  size = 5,
}: {
  series?: number;
  x: number;
  y: number;
  size?: number;
}) {
  const cls = `p1-point p1-point-${series % 3}`;
  if (series % 3 === 1)
    return <rect className={cls} x={x - size} y={y - size} width={size * 2} height={size * 2} />;
  if (series % 3 === 2)
    return (
      <path
        className={cls}
        d={`M${x} ${y - size - 1} L${x + size + 1} ${y} L${x} ${y + size + 1} L${x - size - 1} ${y}Z`}
      />
    );
  return <circle className={cls} cx={x} cy={y} r={size} />;
}

/** x–y chart of curves sampled by the step, its current points and labelled reference lines. */
export function PlotDiagram({ plot }: { plot: Plot }) {
  const multi = plot.curves.length > 1,
    endWidth = multi ? Math.max(...plot.curves.map((c) => textWidth(c.label))) + 22 : 0,
    left = 46,
    right = W - 14 - endWidth;
  const xGuides = (plot.guides ?? []).filter((g) => g.axis === 'x'),
    yGuides = (plot.guides ?? []).filter((g) => g.axis === 'y');
  const sx = (v: number) => left + ((v - plot.x.min) / (plot.x.max - plot.x.min)) * (right - left);
  const labels = guideRows(
      xGuides.map((g) => ({ px: sx(g.value), label: g.label })),
      left,
      W - 8,
    ),
    rows = labels.reduce((n, l) => Math.max(n, l.row + 1), 0),
    top = 30 + rows * 13,
    bottom = top + 200,
    height = bottom + 40;
  const ySpan = plot.y.max - plot.y.min || 1,
    sy = (v: number) => bottom - ((v - plot.y.min) / ySpan) * (bottom - top);
  const path = (points: [number, number][]) =>
    points.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(2)} ${sy(y).toFixed(2)}`).join(' ');
  const ends = multi
    ? spread(
        plot.curves.map((c) => sy(c.points[c.points.length - 1][1])),
        13,
        top + 4,
        bottom,
      )
    : [];
  const clip = useId().replace(/:/g, '');
  /**
   * Point labels go right of their point (left near the right edge), switch sides rather than
   * run across a vertical guide line, and move below their point if they would overlap an
   * earlier label.
   */
  const placed: { x: number; y: number; w: number }[] = [];
  const crossesGuide = (from: number, to: number) =>
    labels.some((g) => g.px > from - 2 && g.px < to + 2);
  const pointLabels = plot.points.map((p) => {
    if (!p.label) return null;
    const x = sx(p.x),
      y = sy(clamp(p.y, plot.y.min, plot.y.max)),
      w = textWidth(p.label, 12);
    let end = x > right - 60,
      lx = end ? x - 9 - w : x + 9;
    if (crossesGuide(lx, lx + w)) {
      const other = end ? x + 9 : x - 9 - w;
      if (other >= left + 2 && other + w <= W - 4 && !crossesGuide(other, other + w)) {
        end = !end;
        lx = other;
      }
    }
    let ly = y - 8;
    if (placed.some((b) => Math.abs(b.y - ly) < 13 && lx < b.x + b.w && lx + w > b.x)) ly = y + 20;
    placed.push({ x: lx, y: ly, w });
    return { x: end ? x - 9 : x + 9, y: ly, anchor: end ? ('end' as const) : ('start' as const) };
  });
  return (
    <svg
      className="p1-svg"
      viewBox={`0 0 ${W} ${height}`}
      role="img"
      aria-label={T(
        '{#0}随{#1}变化的图：{#2}。{#3}',
        plot.y.label,
        plot.x.label,
        plot.curves.map((c) => c.label).join(T('、')),
        plot.caption,
      )}
    >
      <title>{T('曲线由本步同一个公式逐点计算，标记点是当前输入的结果')}</title>
      <defs>
        <clipPath id={clip}>
          <rect x={left} y={top - 2} width={right - left} height={bottom - top + 4} />
        </clipPath>
      </defs>
      <text x={8} y={16} className="p1-svg-small">
        ↑ {plot.y.label}
      </text>
      {niceTicks(plot.y.min, plot.y.max, 4).map((t) => (
        <g key={`y${t}`}>
          <line x1={left} x2={right} y1={sy(t)} y2={sy(t)} className="p1-grid" />
          <text x={left - 6} y={sy(t) + 4} textAnchor="end" className="p1-svg-small">
            {fmt(t, 3)}
          </text>
        </g>
      ))}
      {niceTicks(plot.x.min, plot.x.max, 5).map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} x2={sx(t)} y1={bottom} y2={bottom + 4} className="p1-axis" />
          <text x={sx(t)} y={bottom + 16} textAnchor="middle" className="p1-svg-small">
            {fmt(t, 3)}
          </text>
        </g>
      ))}
      <line x1={left} x2={right} y1={bottom} y2={bottom} className="p1-axis" />
      <line x1={left} x2={left} y1={top} y2={bottom} className="p1-axis" />
      <text x={right} y={bottom + 33} textAnchor="end" className="p1-svg-small">
        {plot.x.label} →
      </text>
      {yGuides.map((g) => (
        <g key={`gy${g.label}`}>
          <line x1={left} x2={right} y1={sy(g.value)} y2={sy(g.value)} className="p1-marker" />
          <text x={right - 4} y={sy(g.value) - 5} textAnchor="end" className="p1-svg-small">
            {g.label}
          </text>
        </g>
      ))}
      {labels.map((g) => (
        <g key={`gx${g.label}${g.px}`}>
          <line x1={g.px} x2={g.px} y1={top - 2} y2={bottom} className="p1-marker" />
          <text
            x={g.center}
            y={top - 8 - (rows - 1 - g.row) * 13}
            textAnchor="middle"
            className="p1-svg-small"
          >
            {g.label}
          </text>
        </g>
      ))}
      <g clipPath={`url(#${clip})`}>
        {/* The first curve is the main one: drawn last, so it stays on top where curves meet. */}
        {plot.curves
          .map((c, i) => (
            <path
              key={c.label}
              d={path(c.points)}
              className={`p1-curve p1-curve-${i % 3}`}
              strokeDasharray={DASH[i % 3] || undefined}
            />
          ))
          .reverse()}
      </g>
      {plot.points.map((p, i) => {
        const label = pointLabels[i];
        return (
          <g key={i}>
            <Marker series={p.series} x={sx(p.x)} y={sy(clamp(p.y, plot.y.min, plot.y.max))} />
            {label && (
              <text x={label.x} y={label.y} textAnchor={label.anchor} className="p1-svg-number">
                {p.label}
              </text>
            )}
          </g>
        );
      })}
      {multi &&
        plot.curves.map((c, i) => (
          <g key={`end${c.label}`}>
            <line
              x1={right + 4}
              x2={right + 16}
              y1={ends[i]}
              y2={ends[i]}
              className={`p1-curve p1-curve-${i % 3}`}
              strokeDasharray={DASH[i % 3] || undefined}
            />
            <text x={right + 20} y={ends[i] + 4} className="p1-svg-number">
              {c.label}
            </text>
          </g>
        ))}
    </svg>
  );
}

/**
 * Legend entries from left to right. Each starts no earlier than its `x` (the Chinese layout); a
 * longer translation pushes the following entries along, and an entry that would leave the
 * drawing starts a new row at `left`.
 */
function legendLayout(entries: { x: number; width: number }[], left: number, right: number) {
  let row = 0,
    end = -Infinity;
  return entries.map(({ x, width }) => {
    let at = Math.max(row ? left : x, end + 14);
    if (at + width > right && end > -Infinity) {
      row++;
      at = left;
    }
    end = at + width;
    return { x: at, y: row * 16, row };
  });
}

/** Sorted candidates: probability before (hollow) and after (filled) a renormalizing filter. */
export function RankingDiagram({ ranking }: { ranking: Ranking }) {
  const threshold = ranking.threshold,
    labels = [T('筛选前'), T('重新归一化后'), T('累计概率（从高到低）')],
    legend = legendLayout(
      [
        { x: 12, width: 16 + textWidth(labels[0]) },
        { x: 82, width: 16 + textWidth(labels[1]) },
        ...(threshold !== undefined ? [{ x: 196, width: 28 + textWidth(labels[2]) }] : []),
      ],
      12,
      W - 4,
    ),
    shift = Math.max(...legend.map((l) => l.row)) * 16,
    left = 44,
    right = W - 16,
    top = 40 + shift,
    bottom = 222 + shift,
    n = ranking.items.length,
    column = (right - left) / n,
    sy = (v: number) => bottom - v * (bottom - top),
    center = (i: number) => left + column * (i + 0.5),
    visited = ranking.items.filter((item) => item.before > 0),
    crossing =
      threshold !== undefined && threshold < 1
        ? ranking.items.findIndex((item) => item.cumulative >= threshold)
        : -1,
    status = ranking.items.map((item, i) =>
      item.kept ? (i === crossing ? T('保留（跨过阈值）') : T('保留')) : T('删去'),
    ),
    statusRoom = column - 8,
    // A status that wraps onto a second line needs one more line of room below the axis.
    height = 270 + shift + (status.some((s) => textWidth(s) > statusRoom) ? 13 : 0);
  return (
    <svg
      className="p1-svg"
      viewBox={`0 0 ${W} ${height}`}
      role="img"
      aria-label={T(
        '按概率排序：{#0}。{#1}',
        ranking.items
          .map(
            (item) =>
              `${item.name} ${fmt(item.before, 3)} → ${item.kept ? fmt(item.after, 3) : T('删去')}`,
          )
          .join(T('，')),
        ranking.caption,
      )}
    >
      <title>{T('柱高对应本步实际概率；累计折线由同一个筛选规则逐项累加')}</title>
      {['p1-bar-before', 'p1-bar-after'].map((swatch, i) => (
        <g key={swatch}>
          <rect x={legend[i].x} y={8 + legend[i].y} width={11} height={11} className={swatch} />
          <text x={legend[i].x + 16} y={18 + legend[i].y} className="p1-svg-small">
            {labels[i]}
          </text>
        </g>
      ))}
      {threshold !== undefined && (
        <>
          <line
            x1={legend[2].x}
            x2={legend[2].x + 22}
            y1={13 + legend[2].y}
            y2={13 + legend[2].y}
            className="p1-curve p1-curve-1"
            strokeDasharray="7 4"
          />
          <Marker series={1} x={legend[2].x + 11} y={13 + legend[2].y} size={3.5} />
          <FitText
            x={legend[2].x + 28}
            y={18 + legend[2].y}
            className="p1-svg-small"
            width={W - 4 - (legend[2].x + 28)}
            maxLines={1}
          >
            {labels[2]}
          </FitText>
        </>
      )}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={left} x2={right} y1={sy(t)} y2={sy(t)} className="p1-grid" />
          <text x={left - 6} y={sy(t) + 4} textAnchor="end" className="p1-svg-small">
            {t}
          </text>
        </g>
      ))}
      <line x1={left} x2={right} y1={bottom} y2={bottom} className="p1-axis" />
      {ranking.items.map((item, i) => {
        const c = center(i);
        return (
          <g key={item.name}>
            <rect
              x={c - 36}
              y={sy(item.before)}
              width={30}
              height={bottom - sy(item.before)}
              className={item.kept ? 'p1-bar-before' : 'p1-bar-dropped'}
            />
            <text x={c - 21} y={sy(item.before) - 5} textAnchor="middle" className="p1-svg-small">
              {fmt(item.before, 3)}
            </text>
            {item.kept ? (
              <>
                <rect
                  x={c + 6}
                  y={sy(item.after)}
                  width={30}
                  height={bottom - sy(item.after)}
                  className="p1-bar-after"
                />
                <text
                  x={c + 21}
                  y={sy(item.after) - 5}
                  textAnchor="middle"
                  className="p1-svg-number"
                >
                  {fmt(item.after, 3)}
                </text>
              </>
            ) : (
              <text x={c + 21} y={bottom - 6} textAnchor="middle" className="p1-svg-number">
                0
              </text>
            )}
            <text x={c} y={bottom + 18} textAnchor="middle" className="p1-svg-title">
              {Tx('第 {#0} 名 · {#1}', i + 1, item.name)}
            </text>
            <FitText
              x={c}
              y={bottom + 35}
              textAnchor="middle"
              className={item.kept ? 'p1-svg-small' : 'p1-svg-dropped'}
              width={statusRoom}
              grow="down"
            >
              {status[i]}
            </FitText>
          </g>
        );
      })}
      {threshold !== undefined && (
        <>
          <line x1={left} x2={right} y1={sy(threshold)} y2={sy(threshold)} className="p1-marker" />
          <text x={left + 4} y={sy(threshold) - 5} className="p1-svg-small">
            P = {fmt(threshold)}
          </text>
          <path
            d={visited
              .map((item, i) => `${i ? 'L' : 'M'}${center(i)} ${sy(item.cumulative)}`)
              .join(' ')}
            className="p1-curve p1-curve-1"
            strokeDasharray="7 4"
          />
          {visited.map((item, i) => (
            <g key={`c${item.name}`}>
              <Marker series={1} x={center(i)} y={sy(item.cumulative)} size={4} />
              {i === crossing && (
                <text x={center(i) + 8} y={sy(item.cumulative) + 14} className="p1-svg-number">
                  {Tx('累计 {#0}', fmt(item.cumulative, 3))}
                </text>
              )}
            </g>
          ))}
        </>
      )}
    </svg>
  );
}

/** [0, 1) laid out as consecutive segments; the uniform number u picks the segment it lands in. */
export function StripDiagram({ strip }: { strip: Strip }) {
  const left = 20,
    right = W - 20,
    y = 58,
    h = 36,
    sx = (v: number) => left + v * (right - left),
    u = sx(strip.u),
    empty = strip.segments.filter((s) => s.end - s.start <= 0),
    edges = [0, ...strip.segments.filter((s) => s.end > s.start).map((s) => s.end)];
  const shown: number[] = [];
  for (const e of edges) if (shown.every((v) => Math.abs(sx(v) - sx(e)) > 34)) shown.push(e);
  return (
    <svg
      className="p1-svg"
      viewBox={`0 0 ${W} 156`}
      role="img"
      aria-label={T(
        '区间 [0, 1) 按 {#0} 首尾相接；u = {#1}，选中 {#2}',
        strip.segments.map((s) => `${s.name} [${fmt(s.start, 3)}, ${fmt(s.end, 3)})`).join(T('、')),
        fmt(strip.u, 4),
        strip.segments[strip.chosen].name,
      )}
    >
      <title>{T('每段长度等于该候选的采样概率，顺序与采样代码逐项累加的顺序相同')}</title>
      {strip.segments.map((s, i) =>
        s.end > s.start ? (
          <g key={s.name}>
            <rect
              x={sx(s.start)}
              y={y}
              width={sx(s.end) - sx(s.start)}
              height={h}
              className={`p1-seg p1-seg-${i % 3}${i === strip.chosen ? ' p1-seg-chosen' : ''}`}
            />
            {sx(s.end) - sx(s.start) > 42 && (
              <text
                x={(sx(s.start) + sx(s.end)) / 2}
                y={y + 23}
                textAnchor="middle"
                className="p1-svg-number"
              >
                {s.name} · {fmt(s.end - s.start, 3)}
              </text>
            )}
          </g>
        ) : null,
      )}
      {shown.map((e) => (
        <g key={e}>
          <line x1={sx(e)} x2={sx(e)} y1={y + h} y2={y + h + 6} className="p1-axis" />
          <text
            x={sx(e)}
            y={y + h + 19}
            textAnchor={e === 0 ? 'start' : e >= 1 - 1e-9 ? 'end' : 'middle'}
            className="p1-svg-small"
          >
            {fmt(e, 3)}
          </text>
        </g>
      ))}
      <line x1={u} x2={u} y1={y - 12} y2={y + h + 4} className="p1-u-line" />
      <path d={`M${u - 6} ${y - 20} L${u + 6} ${y - 20} L${u} ${y - 11}Z`} className="p1-u-head" />
      <text
        x={clamp(u, left + 40, right - 40)}
        y={y - 26}
        textAnchor="middle"
        className="p1-svg-number"
      >
        u = {fmt(strip.u, 4)} → {strip.segments[strip.chosen].name}
      </text>
      <FitText x={left} y={y + h + 40} className="p1-svg-small" width={W - 2 * left} maxLines={1}>
        {empty.length
          ? T('{#0} 的区间长度为 0（已删去）', empty.map((s) => s.name).join(T('、')))
          : T('三段长度之和为 1；窄段同样可能被抽中')}
      </FitText>
    </svg>
  );
}

const SUB = '₀₁₂₃₄₅₆₇₈₉';
/** “+0.39” or “−0.16”: the write is added, so its sign is shown once. */
const signed = (v: number) => (v < 0 ? '−' + fmt(-v, 3) : '+' + fmt(v, 3));
/** A recurrence unrolled left to right: ×keep along the state line, +write from each input. */
export function ChainDiagram({ chain }: { chain: Chain }) {
  const arrow = useId().replace(/:/g, ''),
    column = 84,
    box = 58,
    inputBox = 74,
    rowHeight = 134,
    steps = Math.max(...chain.rows.map((r) => r.steps.length)),
    width = chain.collide ? W : Math.max(W, 20 + column * steps + box),
    height = chain.rows.length * rowHeight + 8,
    cx = (col: number) => 12 + col * column + box / 2,
    ready = (t: number) => chain.cursor === undefined || t <= chain.cursor;
  return (
    <svg
      className="p1-svg"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={chain.rows
        .map((row) =>
          T(
            '{#0}：初始 {#1}，{#2}',
            row.label,
            fmt(row.initial, 3),
            row.steps
              .map((s, t) =>
                ready(t)
                  ? T(
                      't{#0} 保留 ×{#1}、写入 {#2}，得 {#3}',
                      t,
                      fmt(s.keep, 3),
                      fmt(s.drive, 3),
                      fmt(s.state, 3),
                    )
                  : T('t{#0} 待计算', t),
              )
              .join(T('；')),
          ),
        )
        .join(T('。'))}
    >
      <title>{T('每个方框只存一个数：上一状态乘保留系数，再加这一步的写入量')}</title>
      <defs>
        <marker
          id={arrow}
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerUnits="userSpaceOnUse"
          markerWidth="9"
          markerHeight="9"
          orient="auto"
        >
          <path d="M0 0 L10 5 L0 10Z" className="p1-arrow-head" />
        </marker>
      </defs>
      {chain.rows.map((row, r) => {
        const y0 = r * rowHeight + 4,
          boxTop = y0 + 30,
          boxBottom = boxTop + 42,
          inputTop = boxBottom + 32;
        return (
          <g key={row.label}>
            <text x={12} y={y0 + 12} className="p1-svg-title">
              {row.label}
            </text>
            <rect
              x={cx(0) - box / 2}
              y={boxTop}
              width={box}
              height={42}
              className="p1-input-node"
            />
            <text x={cx(0)} y={boxTop + 16} textAnchor="middle" className="p1-svg-small">
              {T('初始 h')}
            </text>
            <text x={cx(0)} y={boxTop + 33} textAnchor="middle" className="p1-svg-number">
              {fmt(row.initial, 3)}
            </text>
            {row.steps.map((s, t) => {
              const c = cx(t + 1),
                on = ready(t),
                current = chain.cursor === t;
              return (
                <g key={t} opacity={on ? 1 : 0.45}>
                  <line
                    x1={cx(t) + box / 2}
                    x2={c - box / 2 - 1}
                    y1={boxTop + 21}
                    y2={boxTop + 21}
                    className={current ? 'p1-wire-on' : 'p1-wire'}
                    markerEnd={`url(#${arrow})`}
                  />
                  <text
                    x={(cx(t) + c) / 2}
                    y={boxTop - 6}
                    textAnchor="middle"
                    className="p1-svg-number"
                  >
                    {on ? `×${fmt(s.keep, 3)}` : '×?'}
                  </text>
                  <rect
                    x={c - box / 2}
                    y={boxTop}
                    width={box}
                    height={42}
                    className={
                      on
                        ? current
                          ? 'p1-output-node p1-node-current'
                          : 'p1-output-node'
                        : 'p1-future-node'
                    }
                  />
                  <text x={c} y={boxTop + 16} textAnchor="middle" className="p1-svg-small">
                    h{SUB[t]}
                  </text>
                  <text x={c} y={boxTop + 33} textAnchor="middle" className="p1-svg-number">
                    {on ? fmt(s.state, 3) : '?'}
                  </text>
                  <line
                    x1={c}
                    x2={c}
                    y1={inputTop}
                    y2={boxBottom + 2}
                    className={current ? 'p1-wire-on' : 'p1-wire'}
                    markerEnd={`url(#${arrow})`}
                  />
                  <text x={c + 5} y={boxBottom + 21} className="p1-svg-number">
                    {on ? signed(s.drive) : '+?'}
                  </text>
                  <rect
                    x={c - inputBox / 2}
                    y={inputTop}
                    width={inputBox}
                    height={24}
                    className="p1-input-node"
                  />
                  <text x={c} y={inputTop + 16} textAnchor="middle" className="p1-svg-number">
                    x{SUB[t]}={fmt(s.input, 3)}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
      {chain.collide && chain.rows.length === 2 && (
        <g>
          {[0, 1].map((r) => {
            const y = r * rowHeight + 4 + 30 + 21,
              x = cx(chain.rows[r].steps.length) + box / 2;
            return <line key={r} x1={x + 4} x2={x + 22} y1={y} y2={y} className="p1-marker" />;
          })}
          <line
            x1={cx(chain.rows[0].steps.length) + box / 2 + 22}
            x2={cx(chain.rows[0].steps.length) + box / 2 + 22}
            y1={4 + 30 + 21}
            y2={rowHeight + 4 + 30 + 21}
            className="p1-marker"
          />
          <text
            x={cx(chain.rows[0].steps.length) + box / 2 + 30}
            y={rowHeight / 2 + 4 + 30 + 17}
            className="p1-svg-title"
          >
            {T('同一个 h')}
          </text>
          <text
            x={cx(chain.rows[0].steps.length) + box / 2 + 30}
            y={rowHeight / 2 + 4 + 30 + 35}
            className="p1-svg-number"
          >
            {fmt(chain.rows[0].steps[chain.rows[0].steps.length - 1].state, 4)}
          </text>
        </g>
      )}
    </svg>
  );
}
