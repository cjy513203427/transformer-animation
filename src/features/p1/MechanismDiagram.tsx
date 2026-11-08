import type { ReactNode } from 'react';
import { formatNumber as fmt } from '../../math/attention';
import type { Frame } from '../../math/p1/shared';
const vec = (x: number[]) => '[' + x.map((v) => fmt(v)).join(', ') + ']';
export function MechanismDiagram({ frame: f, fallback }: { frame: Frame; fallback: ReactNode }) {
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
    const { base, delta, total } = f.branch;
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
          { x: 15, label: '冻结底座 XW₀', values: base },
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
          当前输出 Y
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
  if (f.scan) {
    const s = f.scan;
    return (
      <svg
        className="p1-svg"
        viewBox="0 0 430 510"
        role="img"
        aria-label={`状态扫描，当前计算到t${s.cursor}，未来状态尚未显示`}
      >
        <title>上一状态与当前输入共同决定下一状态</title>
        <text x="280" y="25" textAnchor="middle" className="p1-svg-title">
          初始状态 {fmt(s.initial)}
        </text>
        {s.input.map((v, i) => {
          const y = 70 + i * 105,
            ready = i <= s.cursor;
          return (
            <g key={i} opacity={ready ? 1 : 0.4}>
              <path d={`M280 ${i === 0 ? 35 : y - 49} V${y}`} className="p1-wire" />
              <rect x="15" y={y} width="125" height="57" className="p1-input-node" />
              <text x="77" y={y + 23} textAnchor="middle" className="p1-svg-small">
                输入 x{i}
              </text>
              <text x="77" y={y + 44} textAnchor="middle" className="p1-svg-number">
                {fmt(v)}
              </text>
              <path d={`M140 ${y + 28} H170`} className="p1-wire" />
              <rect x="170" y={y} width="220" height="57" className="p1-output-node" />
              <text x="280" y={y + 23} textAnchor="middle" className="p1-svg-title">
                t{i} · {ready ? 'h = ' + fmt(s.states[i], 4) : '待计算'}
              </text>
              <text x="280" y={y + 44} textAnchor="middle" className="p1-svg-small">
                {ready
                  ? `保留 ${fmt(s.retention[i], 3)} · 注入 ${fmt(s.drive[i], 3)}`
                  : '拖动时间步查看'}
              </text>
            </g>
          );
        })}
        <text x="15" y="500" className="p1-svg-small">
          每步：保留系数 × 上一状态 ＋ 输入注入
        </text>
      </svg>
    );
  }
  return fallback;
}
