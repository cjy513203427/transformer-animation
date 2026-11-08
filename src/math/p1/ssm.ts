import { attention } from '../attention';
import { check, frame, softplus, tensor, type Computation, type Parameters } from './shared';
export function selectiveScan(
  x: number[],
  A: number,
  delta: number,
  selective: boolean,
  h0: number,
) {
  let h = h0;
  const retention: number[] = [],
    drive: number[] = [],
    states: number[] = [],
    outputs: number[] = [],
    deltas: number[] = [],
    B: number[] = [],
    C: number[] = [];
  for (const value of x) {
    const dt = delta * (selective ? softplus(0.5 * value) : 1),
      b = selective ? 1 + 0.25 * value : 1,
      c = selective ? 1 - 0.1 * value : 1,
      keep = Math.exp(dt * A),
      input = (A === 0 ? dt : Math.expm1(dt * A) / A) * b * value;
    h = keep * h + input;
    retention.push(keep);
    drive.push(input);
    states.push(h);
    outputs.push(c * h);
    deltas.push(dt);
    B.push(b);
    C.push(c);
  }
  return { retention, drive, states, outputs, deltas, B, C };
}
export function ssm(x: number[], p: Parameters): Computation {
  const s = selectiveScan(x, p.A, p.delta, !!p.selective, p.h0),
    fixed = selectiveScan(x, p.A, p.delta, false, p.h0),
    q = x.map((v) => [v]),
    attn = attention(q, q, q, true).output.flat(),
    lambda = Math.exp(p.delta * p.A),
    a = selectiveScan([1, 0], p.A, p.delta, false, p.h0),
    b = selectiveScan([2, -lambda], p.A, p.delta, false, p.h0);
  return {
    frames: [
      frame(
        '把过去折进一个状态，再读出输出',
        [
          tensor('输入 x_t', x),
          tensor('固定参数状态 h_t', fixed.states),
          tensor('固定参数输出 y_t', fixed.outputs),
        ],
        [
          ['初始 h₀', p.h0],
          ['连续参数 A', p.A],
        ],
        '标量 SSM 教学示例；输入四个时间步，不代表真实模型的隐藏维度。',
      ),
      frame(
        '选择性：Δ、B、C 由当前输入决定',
        [tensor('Δ_t（始终为正）', s.deltas), tensor('B_t', s.B), tensor('C_t', s.C)],
        [['模式', p.selective ? '输入相关' : '固定参数']],
        '本例用 softplus(0.5x) 生成步长、线性投影生成 B/C。A 保持固定；省略 Mamba 的卷积、门控和扩展投影。',
      ),
      frame(
        '零阶保持离散化：旧状态系数与新输入注入',
        [tensor('保留系数 exp(ΔA)', s.retention), tensor('B̄_t x_t', s.drive)],
        [],
        'B̄ = expm1(ΔA)/A × B，A=0 时用极限 ΔB。不是把连续 B 直接拿来当离散 B̄。',
      ),
      frame(
        '逐步扫描：h_t = Ā_t h_{t−1} + B̄_t x_t',
        [
          tensor('已计算状态 h_t', s.states.slice(0, p.cursor + 1)),
          tensor('已计算输出 C_t h_t', s.outputs.slice(0, p.cursor + 1)),
        ],
        [
          ['当前时间步（从0开始）', p.cursor],
          ['上一状态', p.cursor === 0 ? p.h0 : s.states[p.cursor - 1]],
          ['当前输入', x[p.cursor]],
          ['保留系数', s.retention[p.cursor]],
          ['输入注入', s.drive[p.cursor]],
        ],
        '每一步读取前一步的状态；固定参数的训练可用等价并行计算，但本页演示的是可核对的串行递推。',
      ),
      frame(
        'A < 0 时，较大 Δ 让旧状态保留系数更小',
        [tensor('当前保留系数', s.retention), tensor('固定参数保留系数', fixed.retention)],
        [['基础步长 Δ', p.delta]],
        '修正原书第32章 §32.3.4 的反向表述：小 Δ 更接近保留，大 Δ 旧状态衰减更快；这只描述旧状态项，新输入注入也同时变化。A=0 时保留系数恒为1。',
      ),
      frame(
        '固定状态有历史碰撞；Attention 直接读取历史',
        [
          tensor('当前选择性 SSM 输出', s.outputs),
          tensor('因果 Attention 输出（另一个算子）', attn),
          tensor('碰撞历史 A / B', [
            [1, 0],
            [2, -lambda],
          ]),
        ],
        [
          ['历史 A 最终 h', a.states[1]],
          ['历史 B 最终 h', b.states[1]],
        ],
        '碰撞示例使用固定参数，展示有限状态映射的边界；不能由此判定所有任务优劣。SSM 扫描与 Attention 历史访问是不同机制，本页不是完整 Mamba 模型。',
      ),
    ].map((item, index) =>
      index === 3
        ? {
            ...item,
            scan: {
              input: x,
              states: s.states,
              retention: s.retention,
              drive: s.drive,
              cursor: p.cursor,
              initial: p.h0,
            },
          }
        : item,
    ),
    checks: [check('固定参数历史碰撞', [a.states[1]], [b.states[1]])],
    signature: s.retention,
  };
}
