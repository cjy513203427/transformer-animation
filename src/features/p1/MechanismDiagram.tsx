import type { ReactNode } from 'react';
import { formatNumber as fmt } from '../../math/attention';
import type { Frame, NumberLine } from '../../math/p1/shared';
import { ChainDiagram, PlotDiagram, RankingDiagram, StripDiagram, textWidth } from './Charts';
const vec = (x: number[]) => '[' + x.map((v) => fmt(v)).join(', ') + ']';
const MAX_TICKS = 64;
/** True when the frame carries a dedicated figure instead of the generic bar view. */
export const hasMechanism = (f: Frame) =>
  !!(
    f.line ||
    f.plot ||
    f.ranking ||
    f.strip ||
    f.chain ||
    f.points ||
    f.connections ||
    f.branch ||
    f.tree
  );
function Captioned({ caption, children }: { caption: string; children: ReactNode }) {
  return (
    <>
      {children}
      {caption && <p className="p1-line-caption">{caption}</p>}
    </>
  );
}

function domain(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  let low = Math.min(0, ...finite),
    high = Math.max(0, ...finite);
  if (high - low < 1e-12) {
    low -= 1;
    high += 1;
  }
  const pad = (high - low) * 0.06;
  return [low - pad, high + pad];
}

/**
 * Points closer than a label's width share one label, so crowded values stay readable. Neighbouring
 * clusters alternate between two label rows, so half a label width (at least 14px) is enough.
 */
function labelClusters(positions: number[], names: string[]) {
  const gap = Math.max(14, (Math.max(...names.map((n) => textWidth(n))) + 4) / 2);
  const order = positions
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => Number.isFinite(p))
    .sort((a, b) => a.p - b.p);
  const clusters: { indices: number[]; x: number }[] = [];
  let last = -Infinity;
  for (const { p, i } of order) {
    const current = clusters[clusters.length - 1];
    if (current && p - last < gap) {
      current.indices.push(i);
      current.x = (current.x * (current.indices.length - 1) + p) / current.indices.length;
    } else clusters.push({ indices: [i], x: p });
    last = p;
  }
  for (const c of clusters) c.indices.sort((a, b) => a - b);
  return clusters;
}

/** Horizontal number lines: the same coordinate keeps its name and is linked across rows. */
function NumberLineDiagram({ line }: { line: NumberLine }) {
  const left = 26,
    right = 404,
    rowHeight = 104,
    top = 18,
    height = top + line.rows.length * rowHeight + 6;
  const extra = (row: number) => [
    ...(line.markers ?? []).filter((m) => m.row === row).map((m) => m.value),
    ...(line.ticks ?? []),
  ];
  const sharedDomain = domain([
    ...line.rows.flatMap((r) => r.values),
    ...(line.markers ?? []).map((m) => m.value),
    ...(line.ticks ?? []),
  ]);
  const domains = line.rows.map(
    (r, i) =>
      line.domains?.[i] ?? (line.shared ? sharedDomain : domain([...r.values, ...extra(i)])),
  );
  const x = (row: number, value: number) => {
    const [low, high] = domains[row];
    return left + ((value - low) / (high - low)) * (right - left);
  };
  const axisY = (row: number) => top + row * rowHeight + 52;
  const drawTicks = line.ticks && line.ticks.length <= MAX_TICKS;
  return (
    <svg
      className="p1-svg"
      viewBox={`0 0 430 ${height}`}
      role="img"
      aria-label={`数轴：${line.rows.map((r) => r.label).join('，')}。${line.caption}`}
    >
      <title>点的位置对应本步实际数值；同名的点是同一个坐标</title>
      {line.rows.slice(1).map((_, k) =>
        line.rows[k].values.map((v, i) => {
          const next = line.rows[k + 1].values[i];
          if (!Number.isFinite(v) || !Number.isFinite(next)) return null;
          return (
            <line
              key={`${k}-${i}`}
              x1={x(k, v)}
              y1={axisY(k) + 7}
              x2={x(k + 1, next)}
              y2={axisY(k + 1) - 7}
              className="p1-link"
            />
          );
        }),
      )}
      {line.rows.map((row, r) => {
        const y = axisY(r),
          [low, high] = domains[r];
        return (
          <g key={row.label}>
            <text x={left - 14} y={y - 34} className="p1-svg-title">
              {row.label}
            </text>
            {line.ticks && !drawTicks && r === 0 && (
              <text x={right} y={y - 34} textAnchor="end" className="p1-svg-small">
                格点 {line.ticks.length} 个，过密未画出
              </text>
            )}
            <line x1={left} x2={right} y1={y} y2={y} className="p1-wire" />
            {drawTicks &&
              line.ticks!.map((t) => (
                <line key={t} x1={x(r, t)} x2={x(r, t)} y1={y - 5} y2={y + 5} className="p1-tick" />
              ))}
            {(line.markers ?? [])
              .filter((m) => m.row === r)
              .map((m) => (
                <g key={m.label + m.value}>
                  <line
                    x1={x(r, m.value)}
                    x2={x(r, m.value)}
                    y1={y - 17}
                    y2={y + 17}
                    className="p1-marker"
                  />
                  <text x={x(r, m.value)} y={y - 21} textAnchor="middle" className="p1-svg-small">
                    {m.label}
                  </text>
                </g>
              ))}
            {row.values.map((v, i) =>
              Number.isFinite(v) ? (
                r === 0 ? (
                  <circle key={i} cx={x(r, v)} cy={y} r="5" className="p1-input-node" />
                ) : (
                  <rect
                    key={i}
                    x={x(r, v) - 5}
                    y={y - 5}
                    width="10"
                    height="10"
                    className="p1-output-node"
                  />
                )
              ) : null,
            )}
            {labelClusters(
              row.values.map((v) => x(r, v)),
              line.names,
            ).map((cluster, k) => (
              <text
                key={cluster.indices.join()}
                x={cluster.x}
                y={y + (k % 2 ? 31 : 19)}
                textAnchor="middle"
                className="p1-svg-small"
              >
                {cluster.indices.map((i) => line.names[i]).join(',')}
              </text>
            ))}
            <text x={left} y={y + 45} className="p1-svg-small">
              {fmt(low, 2)}
            </text>
            <text x={right} y={y + 45} textAnchor="end" className="p1-svg-small">
              {fmt(high, 2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function MechanismDiagram({ frame: f, fallback }: { frame: Frame; fallback: ReactNode }) {
  if (f.line)
    return (
      <Captioned caption={f.line.caption}>
        <NumberLineDiagram line={f.line} />
      </Captioned>
    );
  if (f.plot)
    return (
      <Captioned caption={f.plot.caption}>
        <PlotDiagram plot={f.plot} />
      </Captioned>
    );
  if (f.ranking)
    return (
      <Captioned caption={f.ranking.caption}>
        <RankingDiagram ranking={f.ranking} />
      </Captioned>
    );
  if (f.strip)
    return (
      <Captioned caption={f.strip.caption}>
        <StripDiagram strip={f.strip} />
      </Captioned>
    );
  if (f.chain)
    return (
      <Captioned caption={f.chain.caption}>
        <ChainDiagram chain={f.chain} />
      </Captioned>
    );
  if (f.points) {
    const { before, after } = f.points,
      extent = Math.max(1, ...before.flat().map(Math.abs), ...after.flat().map(Math.abs)),
      scale = 150 / extent,
      pos = ([x, y]: number[]) => [215 + x * scale, 215 - y * scale];
    return (
      <svg
        className="p1-svg"
        viewBox="0 0 430 430"
        role="img"
        aria-label="二维点集变换：圆形为原输入，方形为本步输出，相同行用线连接"
      >
        <title>真实二维坐标，不是高维降维</title>
        <text x="12" y="24" className="p1-svg-title">
          原输入（圆）→ 本步结果（方）
        </text>
        <line x1="35" x2="395" y1="215" y2="215" className="p1-wire" />
        <line x1="215" x2="215" y1="45" y2="385" className="p1-wire" />
        <text x="390" y="207" className="p1-svg-small">
          d0
        </text>
        <text x="222" y="48" className="p1-svg-small">
          d1
        </text>
        {[-extent, 0, extent].map((v) => (
          <g key={v}>
            <text x={215 + v * scale} y="235" textAnchor="middle" className="p1-svg-small">
              {fmt(v, 1)}
            </text>
            <text x="190" y={220 - v * scale} textAnchor="end" className="p1-svg-small">
              {fmt(v, 1)}
            </text>
          </g>
        ))}
        {before.map((v, i) => {
          const [x, y] = pos(v),
            [a, b] = pos(after[i]);
          return (
            <g key={i}>
              <line x1={x} y1={y} x2={a} y2={b} className="p1-wire" />
              <circle cx={x} cy={y} r="5" className="p1-input-node" />
              <rect x={a - 5} y={b - 5} width="10" height="10" className="p1-output-node" />
              <text x={a + 7} y={b - 7} className="p1-svg-small">
                t{i}
              </text>
            </g>
          );
        })}
        <text x="15" y="412" className="p1-svg-small">
          每条连线保留同一 token 身份；重合点的数值见表格
        </text>
      </svg>
    );
  }
  if (f.connections) {
    const weights = f.connections;
    return (
      <svg
        className="p1-svg"
        viewBox="0 0 430 370"
        role="img"
        aria-label="Query读取Key连接图，线宽对应当前注意力权重，零概率无连接"
      >
        <title>行 Query，列 Key；Mask 通过连接缺失表示</title>
        <text x="70" y="25" textAnchor="middle" className="p1-svg-title">
          Query
        </text>
        <text x="360" y="25" textAnchor="middle" className="p1-svg-title">
          Key
        </text>
        {weights.flatMap((row, i) =>
          row.map((p, j) =>
            p > 0 ? (
              <path
                key={`${i}-${j}`}
                d={`M102 ${75 + i * 70} C180 ${75 + i * 70} 250 ${75 + j * 70} 328 ${75 + j * 70}`}
                fill="none"
                stroke="#6c8c46"
                strokeWidth={1 + 5 * p}
                opacity={0.25 + 0.7 * p}
              />
            ) : null,
          ),
        )}
        {weights.map((_, i) => (
          <g key={i}>
            <circle cx="70" cy={75 + i * 70} r="25" className="p1-input-node" />
            <text x="70" y={79 + i * 70} textAnchor="middle" className="p1-svg-number">
              t{i}
            </text>
            <rect x="335" y={50 + i * 70} width="50" height="50" className="p1-key-node" rx="3" />
            <text x="360" y={79 + i * 70} textAnchor="middle" className="p1-svg-number">
              t{i}
            </text>
          </g>
        ))}
        <text x="15" y="350" className="p1-svg-small">
          线越粗权重越大；每个 Query 沿可见 Key 归一化
        </text>
      </svg>
    );
  }
  if (f.branch) {
    const { base, delta, total, baseLabel = '冻结底座 XW₀', totalLabel = '当前输出 Y' } = f.branch;
    return (
      <svg
        className="p1-svg"
        viewBox="0 0 430 360"
        role="img"
        aria-label="LoRA双支路：冻结底座输出与低秩更新输出相加"
      >
        <title>W₀ 保持固定，更新从独立支路进入</title>
        <text x="215" y="25" textAnchor="middle" className="p1-svg-title">
          同一份输入 X
        </text>
        <path d="M215 35 V50 H110 V75 M215 50 H320 V75" className="p1-wire" />
        {[
          { x: 15, label: baseLabel, values: base },
          { x: 230, label: '低秩更新 (α/r)XAB', values: delta },
        ].map((item) => (
          <g key={item.x}>
            <rect x={item.x} y="75" width="185" height="100" className="p1-input-node" />
            <text x={item.x + 92} y="98" textAnchor="middle" className="p1-svg-title">
              {item.label}
            </text>
            {item.values.map((row, i) => (
              <text
                x={item.x + 92}
                y={126 + i * 23}
                key={i}
                textAnchor="middle"
                className="p1-svg-number"
              >
                t{i} {vec(row)}
              </text>
            ))}
          </g>
        ))}
        <path d="M107 175 V207 H215 M322 175 V207 H215 V240" className="p1-wire" />
        <text x="215" y="222" textAnchor="middle" className="p1-svg-title">
          ＋
        </text>
        <rect x="95" y="240" width="240" height="100" className="p1-output-node" />
        <text x="215" y="265" textAnchor="middle" className="p1-svg-title">
          {totalLabel}
        </text>
        {total.map((row, i) => (
          <text key={i} x="215" y={292 + i * 22} textAnchor="middle" className="p1-svg-number">
            t{i} {vec(row)}
          </text>
        ))}
      </svg>
    );
  }
  if (f.tree) {
    const { root, conditional, joint, selected } = f.tree;
    return (
      <svg
        className="p1-svg"
        viewBox="0 0 430 385"
        role="img"
        aria-label={`两步概率树，当前Beam-2最优路径${selected}`}
      >
        <title>叶节点显示联合概率，内部节点显示第一步概率</title>
        <text x="215" y="25" textAnchor="middle" className="p1-svg-title">
          同一个 prompt
        </text>
        {root.map((p, i) => {
          const center = 75 + i * 140,
            on = selected[0] === 'abc'[i];
          return (
            <g key={i}>
              <line
                x1="215"
                y1="40"
                x2={center}
                y2="100"
                className={on ? 'p1-wire-on' : 'p1-wire'}
              />
              <rect
                x={center - 55}
                y="100"
                width="110"
                height="48"
                className={on ? 'p1-output-node' : 'p1-input-node'}
              />
              <text x={center} y="120" textAnchor="middle" className="p1-svg-title">
                {'abc'[i]}
              </text>
              <text x={center} y="138" textAnchor="middle" className="p1-svg-number">
                p={fmt(p, 3)}
              </text>
              {conditional[i].map((c, j) => {
                const x = center - 45 + j * 45,
                  path = 'abc'[i] + 'abc'[j],
                  active = path === selected;
                return (
                  <g key={j}>
                    <line
                      x1={center}
                      y1="148"
                      x2={x}
                      y2="245"
                      className={active ? 'p1-wire-on' : 'p1-wire'}
                    />
                    <text x={x} y="226" textAnchor="middle" className="p1-svg-small">
                      {fmt(c, 2)}
                    </text>
                    <rect
                      x={x - 20}
                      y="245"
                      width="40"
                      height="65"
                      className={active ? 'p1-output-node' : 'p1-input-node'}
                    />
                    <text x={x} y="268" textAnchor="middle" className="p1-svg-number">
                      {path}
                    </text>
                    <text x={x} y="293" textAnchor="middle" className="p1-svg-small">
                      {fmt(joint[i][j], 3)}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
        <text x="15" y="346" className="p1-svg-small">
          叶：联合概率；条件表见本页说明与数值
        </text>
        <text x="15" y="369" className="p1-svg-small">
          绿色：当前 Beam-2 找到的完整路径；不保证一般情形全局最优
        </text>
      </svg>
    );
  }
  return fallback;
}
