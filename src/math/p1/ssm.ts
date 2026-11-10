import { attention, formatNumber as fmt } from '../attention';
import {
  check,
  frame,
  sample,
  softplus,
  tensor,
  type Chain,
  type Computation,
  type Parameters,
} from './shared';
import { T } from '../../i18n';
/** This page's selection rule: how Δ, B and C depend on the current input. A stays fixed. */
export function selectiveParams(value: number, delta: number, selective: boolean) {
  return {
    dt: delta * (selective ? softplus(0.5 * value) : 1),
    b: selective ? 1 + 0.25 * value : 1,
    c: selective ? 1 - 0.1 * value : 1,
  };
}
/** Zero-order hold: the old state keeps exp(ΔA); a unit of B writes (exp(ΔA) − 1) / A, or Δ at A = 0. */
export const retention = (dt: number, A: number) => Math.exp(dt * A);
export const writeScale = (dt: number, A: number) => (A === 0 ? dt : Math.expm1(dt * A) / A);
export function selectiveScan(
  x: number[],
  A: number,
  delta: number,
  selective: boolean,
  h0: number,
) {
  let h = h0;
  const keeps: number[] = [],
    drive: number[] = [],
    states: number[] = [],
    outputs: number[] = [],
    deltas: number[] = [],
    B: number[] = [],
    C: number[] = [];
  for (const value of x) {
    const { dt, b, c } = selectiveParams(value, delta, selective),
      keep = retention(dt, A),
      input = writeScale(dt, A) * b * value;
    h = keep * h + input;
    keeps.push(keep);
    drive.push(input);
    states.push(h);
    outputs.push(c * h);
    deltas.push(dt);
    B.push(b);
    C.push(c);
  }
  return { retention: keeps, drive, states, outputs, deltas, B, C };
}
type Scan = ReturnType<typeof selectiveScan>;
const chainRow = (label: string, input: number[], initial: number, s: Scan) => ({
  label,
  initial,
  steps: input.map((v, t) => ({
    input: v,
    keep: s.retention[t],
    drive: s.drive[t],
    state: s.states[t],
  })),
});
const SUB = '₀₁₂₃₄₅₆₇₈₉';
/** One guide per time step, named x₀, x₁…; the chart merges labels of guides that coincide. */
const tags = (prefix: string, values: number[]) =>
  values.map((value, t) => ({ value, label: prefix + SUB[t] }));
export function ssm(x: number[], p: Parameters): Computation {
  const selective = !!p.selective,
    s = selectiveScan(x, p.A, p.delta, selective, p.h0),
    fixed = selectiveScan(x, p.A, p.delta, false, p.h0),
    q = x.map((v) => [v]),
    attn = attention(q, q, q, true).output.flat(),
    lambda = retention(p.delta, p.A),
    a = selectiveScan([1, 0], p.A, p.delta, false, p.h0),
    b = selectiveScan([2, -lambda], p.A, p.delta, false, p.h0);
  const mode = selective ? T('输入相关') : T('固定参数'),
    xLow = Math.min(-3, Math.min(...x) - 1),
    xHigh = Math.max(3, Math.max(...x) + 1),
    params = (v: number) => selectiveParams(v, p.delta, selective),
    paramCurves = [
      { label: 'Δ', points: sample(xLow, xHigh, 81, (v) => params(v).dt) },
      { label: 'B', points: sample(xLow, xHigh, 81, (v) => params(v).b) },
      { label: 'C', points: sample(xLow, xHigh, 81, (v) => params(v).c) },
    ],
    paramValues = paramCurves.flatMap((c) => c.points.map(([, y]) => y)),
    dtHigh = Math.max(2.5, ...s.deltas.map((d) => d * 1.1)),
    keepCurve = sample(0, dtHigh, 101, (d) => retention(d, p.A)),
    writeCurve = sample(0, dtHigh, 101, (d) => writeScale(d, p.A));
  const scanChain = (cursor?: number): Chain => ({
    rows: [chainRow(mode, x, p.h0, s)],
    cursor,
    caption:
      cursor === undefined
        ? ''
        : T(
            '已算到 t{#0}：这一步只用上一步的状态 {#1} 和当前输入；后面的状态还没有算。',
            cursor,
            fmt(cursor === 0 ? p.h0 : s.states[cursor - 1]),
          ),
  });
  return {
    frames: [
      {
        ...frame(
          T('把过去折进一个状态，再读出输出'),
          [
            tensor(T('输入 x_t'), x),
            tensor(T('固定参数状态 h_t'), fixed.states),
            tensor(T('固定参数输出 y_t'), fixed.outputs),
          ],
          [
            [T('初始 h₀'), p.h0],
            [T('连续参数 A'), p.A],
          ],
          T('标量 SSM 教学示例；输入四个时间步，不代表真实模型的隐藏维度。'),
        ),
        chain: {
          rows: [chainRow(T('固定参数'), x, p.h0, fixed)],
          caption: T(
            '每一步：旧状态 × 保留系数 + 写入量 = 新状态。固定参数时每一步的保留系数都是 {#0}；箭头右边永远只有一个数，没有历史列表。',
            fmt(lambda, 4),
          ),
        },
      },
      {
        ...frame(
          T('选择性：Δ、B、C 由当前输入决定'),
          [tensor(T('Δ_t（始终为正）'), s.deltas), tensor('B_t', s.B), tensor('C_t', s.C)],
          [[T('模式'), mode]],
          T(
            '本例用 softplus(0.5x) 生成步长、线性投影生成 B/C。A 保持固定；省略 Mamba 的卷积、门控和扩展投影。',
          ),
        ),
        plot: {
          x: { label: T('输入 x'), min: xLow, max: xHigh },
          y: {
            label: T('参数值'),
            min: Math.min(0, ...paramValues),
            max: Math.max(...paramValues) * 1.05,
          },
          curves: paramCurves,
          points: x.flatMap((v) => {
            const r = params(v);
            return [
              { label: '', x: v, y: r.dt, series: 0 },
              { label: '', x: v, y: r.b, series: 1 },
              { label: '', x: v, y: r.c, series: 2 },
            ];
          }),
          guides: tags('x', x).map((g) => ({ axis: 'x' as const, ...g })),
          caption: selective
            ? T(
                '三条线是本页的选择规则：Δ = Δ·softplus(0.5x) 始终为正，B、C 随 x 线性变化。竖线是四个输入，交点就是表中的 Δ_t、B_t、C_t。',
              )
            : T('固定参数模式：三条线都是水平线，Δ、B、C 与输入无关，每一步都用同一组参数。'),
        },
      },
      {
        ...frame(
          T('零阶保持离散化：旧状态系数与新输入注入'),
          [tensor(T('保留系数 exp(ΔA)'), s.retention), tensor('B̄_t x_t', s.drive)],
          [],
          T('B̄ = expm1(ΔA)/A × B，A=0 时用极限 ΔB。不是把连续 B 直接拿来当离散 B̄。'),
        ),
        plot: {
          x: { label: T('步长 Δ'), min: 0, max: dtHigh },
          y: {
            label: T('系数'),
            min: 0,
            max: Math.max(1, ...writeCurve.map(([, y]) => y)) * 1.05,
          },
          curves: [
            { label: T('保留 Ā'), points: keepCurve },
            { label: T('写入 B̄/B'), points: writeCurve },
          ],
          points: s.deltas.flatMap((d) => [
            { label: '', x: d, y: retention(d, p.A), series: 0 },
            { label: '', x: d, y: writeScale(d, p.A), series: 1 },
          ]),
          guides: tags('Δ', s.deltas).map((g) => ({ axis: 'x' as const, ...g })),
          caption:
            p.A === 0
              ? T('A = 0：保留系数恒为 1，写入系数等于 Δ。')
              : T(
                  'A = {#0}：Δ 越大，旧状态的保留系数越小（实线），每单位 B 写入得越多（虚线）。竖线是四个时间步的 Δ_t。',
                  fmt(p.A),
                ),
        },
      },
      {
        ...frame(
          T('逐步扫描：h_t = Ā_t h_{t−1} + B̄_t x_t'),
          [
            tensor(T('已计算状态 h_t'), s.states.slice(0, p.cursor + 1)),
            tensor(T('已计算输出 C_t h_t'), s.outputs.slice(0, p.cursor + 1)),
          ],
          [
            [T('当前时间步（从0开始）'), p.cursor],
            [T('上一状态'), p.cursor === 0 ? p.h0 : s.states[p.cursor - 1]],
            [T('当前输入'), x[p.cursor]],
            [T('保留系数'), s.retention[p.cursor]],
            [T('输入注入'), s.drive[p.cursor]],
          ],
          T(
            '每一步读取前一步的状态；固定参数的训练可用等价并行计算，但本页演示的是可核对的串行递推。',
          ),
        ),
        chain: scanChain(p.cursor),
      },
      {
        ...frame(
          T('A < 0 时，较大 Δ 让旧状态保留系数更小'),
          [tensor(T('当前保留系数'), s.retention), tensor(T('固定参数保留系数'), fixed.retention)],
          [[T('基础步长 Δ'), p.delta]],
          T(
            '修正原书第32章 §32.3.4 的反向表述：小 Δ 更接近保留，大 Δ 旧状态衰减更快；这只描述旧状态项，新输入注入也同时变化。A=0 时保留系数恒为1。',
          ),
        ),
        plot: {
          x: { label: T('步长 Δ'), min: 0, max: dtHigh },
          y: { label: T('保留系数 e^(ΔA)'), min: 0, max: 1.05 },
          curves: [{ label: 'e^(ΔA)', points: keepCurve }],
          points: [
            ...s.deltas.map((d, t) => ({ label: '', x: d, y: s.retention[t], series: 0 })),
            { label: T('固定 Δ'), x: p.delta, y: lambda, series: 1 },
          ],
          guides: [{ axis: 'x', value: p.delta, label: T('基础 Δ = {#0}', fmt(p.delta)) }],
          caption:
            p.A === 0
              ? T('A = 0：曲线是 1 的水平线，Δ 不改变保留系数。')
              : T(
                  'A = {#0} < 0：曲线一路下降。小 Δ 的保留系数接近 1，旧状态留得多；大 Δ 衰减更快。圆点是{#1}的四个 Δ_t，方块是固定 Δ。',
                  fmt(p.A),
                  mode,
                ),
        },
      },
      {
        ...frame(
          T('固定状态有历史碰撞；Attention 直接读取历史'),
          [
            tensor(T('当前选择性 SSM 输出'), s.outputs),
            tensor(T('因果 Attention 输出（另一个算子）'), attn),
            tensor(T('碰撞历史 A / B'), [
              [1, 0],
              [2, -lambda],
            ]),
          ],
          [
            [T('历史 A 最终 h'), a.states[1]],
            [T('历史 B 最终 h'), b.states[1]],
          ],
          T(
            '碰撞示例使用固定参数，展示有限状态映射的边界；不能由此判定所有任务优劣。SSM 扫描与 Attention 历史访问是不同机制，本页不是完整 Mamba 模型。',
          ),
        ),
        chain: {
          rows: [
            chainRow(T('历史 A'), [1, 0], p.h0, a),
            chainRow(T('历史 B'), [2, -lambda], p.h0, b),
          ],
          collide: true,
          caption: T(
            '两段不同的输入 [1, 0] 和 [2, −{#0}]，固定参数扫两步后状态都是 {#1}：之后的计算再也分不清它们。',
            fmt(lambda, 4),
            fmt(a.states[1], 4),
          ),
        },
      },
    ],
    checks: [check(T('固定参数历史碰撞'), [a.states[1]], [b.states[1]])],
    signature: s.retention,
  };
}
