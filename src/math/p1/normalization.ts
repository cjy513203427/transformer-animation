import {
  check,
  frame,
  mean,
  softmax,
  sum,
  tensor,
  type Computation,
  type Parameters,
} from './shared';
export function normalize(x: number[], eps: number) {
  const mu = mean(x),
    centered = x.map((v) => v - mu),
    variance = mean(centered.map((v) => v * v)),
    squareMean = mean(x.map((v) => v * v));
  const ln = centered.map((v) => v / Math.sqrt(variance + eps)),
    rms = x.map((v) => v / Math.sqrt(squareMean + eps));
  return {
    mu,
    centered,
    variance,
    squareMean,
    ln,
    rms,
    lnDenominator: Math.sqrt(variance + eps),
    rmsDenominator: Math.sqrt(squareMean + eps),
  };
}
export function normalization(x: number[], p: Parameters): Computation {
  const n = normalize(x, p.eps),
    gamma = x.map((_, i) => (i === 0 ? p.gain : 1)),
    beta = x.map((_, i) => (i === 0 ? p.bias : 0));
  const ln = n.ln.map((v, i) => v * gamma[i] + beta[i]),
    rms = n.rms.map((v, i) => v * gamma[i]);
  const scores = x.map((v, i) =>
    p.mask === 2 || (p.mask === 1 && i === x.length - 1) ? -Infinity : v / p.temperature,
  );
  const m = Math.max(...scores),
    ex = scores.map((s) => (s === -Infinity ? 0 : Math.exp(s - m))),
    prob = softmax(scores);
  const changed = x.map((v) => v + p.shift),
    n2 = normalize(changed, p.eps),
    shiftedProb = softmax(scores.map((s) => (s === -Infinity ? s : s + p.shift / p.temperature)));
  return {
    frames: [
      frame(
        '沿特征轴统计，同一 token 的四个坐标',
        [tensor('x [1, 4]', x), tensor('x − μ', n.centered)],
        [
          ['μ', n.mu],
          ['方差（除以 D）', n.variance],
          ['均方', n.squareMean],
        ],
      ),
      frame(
        '中心化后，除以标准差与 ε 构成的分母',
        [tensor('x − μ', n.centered), tensor('LayerNorm 核心 z', n.ln)],
        [
          ['分母', n.lnDenominator],
          ['z 均值', mean(n.ln)],
          ['z 方差', mean(n.ln.map((v) => v * v))],
        ],
        'ε > 0 时方差通常略小于 1；常量向量的 z 为 0。',
      ),
      frame(
        'RMSNorm 不减均值，保留原向量的符号',
        [tensor('原 x', x), tensor('RMSNorm 核心 r', n.rms)],
        [
          ['RMS 分母', n.rmsDenominator],
          ['r 均值', mean(n.rms)],
          ['r 均方', mean(n.rms.map((v) => v * v))],
        ],
      ),
      frame(
        '逐坐标缩放；本例只编辑第 0 维参数',
        [
          tensor('γ', gamma),
          tensor('β（只用于 LN）', beta),
          tensor('LN 输出 y', ln),
          tensor('RMS 输出 y', rms),
        ],
        [
          ['LN 输出均值', mean(ln)],
          ['RMS 输出均值', mean(rms)],
        ],
        'γ、β 是可训练参数；这里仅做前向干预，没有优化器更新。',
      ),
      frame(
        '同一组数现在作为候选分数，沿候选轴 Softmax',
        [tensor('分数 / τ + Mask', scores), tensor('exp(s − m)', ex), tensor('概率 P', prob)],
        [
          ['最大值 m', m],
          ['指数和', sum(ex)],
          ['概率和', sum(prob)],
        ],
        p.mask === 2
          ? '全部屏蔽：数学上没有合法分布，本工具返回全零并明确标记。'
          : '减去最大值保证指数不溢出；不会改变概率。',
      ),
      frame(
        '整行平移后：LN 与 Softmax 保持，RMS 通常改变',
        [
          tensor('x′ = x + c', changed),
          tensor('LN(x′)', n2.ln),
          tensor('RMS(x′)', n2.rms),
          tensor('Softmax(x′)', shiftedProb),
        ],
        [['平移 c', p.shift]],
        '比较固定 ε、温度与 Mask，使用仿射之前的归一化结果。退化输入可能让 RMS 也恰好不变。',
      ),
    ],
    checks: [
      check('LN 平移不变', n.ln, n2.ln),
      check('Softmax 平移不变', prob, shiftedProb),
      check('合法概率和', [sum(prob)], [p.mask === 2 ? 0 : 1]),
    ],
    signature: n2.rms,
  };
}
