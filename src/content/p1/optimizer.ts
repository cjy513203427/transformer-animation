import { formatNumber as fmt } from '../../math/attention';
import {
  adamTrajectory,
  loss,
  lossGradient,
  sgdTrajectory,
  type OptimizerConfig,
} from '../../math/p1/optimizer';
import { frame, tensor, type Check, type Computation, type Parameters } from '../../math/p1/shared';
import { chapterUrl } from '../book';
import { choice, field, lesson, vec, type TopicDefinition } from './types';
import { T } from '../../i18n';

const CH17 = chapterUrl('第17章-学习率的理解-训练稳定的关键/');
const STEPS = 20;
const config = (p: Parameters): OptimizerConfig => ({
  eta: p.eta,
  beta1: p.beta1,
  beta2: p.beta2,
  lambda: p.lambda,
  curvature: p.curvature,
  mode: p.mode,
});
/** Error relative to the largest magnitude, so diverging runs compare fairly. */
function relative(label: string, a: number[], b: number[], tolerance = 1e-10): Check {
  const scale = Math.max(1, ...a.map(Math.abs), ...b.map(Math.abs));
  return { label, error: Math.max(0, ...a.map((x, i) => Math.abs(x - b[i]) / scale)), tolerance };
}
const logLoss = (w: number[], curvature: number) =>
  Math.min(12, Math.log10(Math.max(loss(w, curvature), 1e-12)));

function compute(input: number[], p: Parameters): Computation {
  const c = config(p),
    t = p.t;
  const adam = adamTrajectory(input, c, STEPS);
  const other = adamTrajectory(input, { ...c, mode: 1 - c.mode }, STEPS);
  const r = adam[t - 1],
    previous = adam[t - 2];
  const mPrev = previous ? previous.m : [0, 0],
    vPrev = previous ? previous.v : [0, 0];
  const sgdStep = lossGradient(r.w, c.curvature).map((g) => -c.eta * g);
  const ratio = (u: number[]) => (u[0] === 0 ? 0 : Math.abs(u[1] / u[0]));
  const metrics: [string, number][] = [
    ['L(w_{t-1})', loss(r.w, c.curvature)],
    ['L(w_t)', loss(r.next, c.curvature)],
  ];
  const mode = c.mode === 1 ? 'Adam + L2' : 'AdamW';
  const sgd = sgdTrajectory(input, c, STEPS, false);
  const curves = [
    { label: 'SGD', path: sgd },
    { label: mode, path: [input, ...adam.map((x) => x.next)] },
    { label: c.mode === 1 ? 'AdamW' : 'Adam + L2', path: [input, ...other.map((x) => x.next)] },
  ].map(({ label, path }) => ({
    label,
    points: path.map((w, i): [number, number] => [i, logLoss(w, c.curvature)]),
  }));
  const ys = curves.flatMap((curve) => curve.points.map(([, y]) => y));
  const frames = [
    frame(T('当前位置的梯度'), [
      tensor('w_{t-1}', r.w),
      tensor('∇L(w_{t-1})', r.lossGradient),
      tensor('g_t', r.g),
    ], metrics),
    frame(T('一阶矩：梯度的滑动平均'), [
      tensor('m_{t-1}', mPrev),
      tensor('g_t', r.g),
      tensor('m_t', r.m),
    ], metrics),
    frame(T('二阶矩：梯度平方的滑动平均'), [
      tensor('v_{t-1}', vPrev),
      tensor('g_t²', r.g.map((g) => g * g)),
      tensor('v_t', r.v),
    ], metrics),
    frame(T('偏差修正'), [
      tensor('[1−β₁ᵗ, 1−β₂ᵗ]', [1 - c.beta1 ** t, 1 - c.beta2 ** t]),
      tensor('m̂_t', r.mHat),
      tensor('v̂_t', r.vHat),
    ], metrics),
    frame(
      T('按坐标归一化后更新'),
      [
        tensor('−η m̂/(√v̂+ε)', r.adaptive),
        tensor('−ηλ w_{t-1}', r.decay),
        tensor('w_t', r.next),
        tensor('SGD: −η∇L(w_{t-1})', sgdStep),
      ],
      [...metrics, ['|Δw₁/Δw₀| Adam', ratio(r.adaptive)], ['|Δw₁/Δw₀| SGD', ratio(sgdStep)]],
    ),
    {
      ...frame(T('二十步轨迹对照'), [tensor('w_t', r.next)], [
        ...metrics,
        ['log₁₀ L₂₀ SGD', curves[0].points[STEPS][1]],
        ['log₁₀ L₂₀', curves[1].points[STEPS][1]],
      ]),
      plot: {
        x: { label: T('更新步数 t'), min: 0, max: STEPS },
        y: { label: 'log₁₀ L', min: Math.min(...ys), max: Math.max(...ys) + 0.5 },
        curves,
        points: [{ label: '', x: t, y: curves[1].points[t][1], series: 1 }],
        guides: [{ axis: 'x' as const, value: t, label: `t = ${t}` }],
        caption: T(
          '纵轴是损失的常用对数，上限截到 12。三条线从同一起点出发、使用同一个 η 与 λ；竖线上的标记是当前查看的第 t 步。',
        ),
      },
    },
  ];
  // Property checks with independent recomputations.
  const first = adam[0];
  const scaled = adamTrajectory(input, { ...c, lambda: 0, lossScale: 100 }, t)[t - 1];
  const unscaled = adamTrajectory(input, { ...c, lambda: 0 }, t)[t - 1];
  return {
    frames,
    signature: r.next,
    checks: [
      relative(T('第 1 步偏差修正后 m̂ = g、v̂ = g²'), [...first.mHat, ...first.vHat], [
        ...first.g,
        ...first.g.map((g) => g * g),
      ]),
      relative(
        T('SGD 中 L2 与解耦衰减逐步相同'),
        sgd.flat(),
        sgdTrajectory(input, c, STEPS, true).flat(),
      ),
      relative(
        T('损失整体放大 100 倍，Adam 的更新几乎不变（λ = 0）'),
        scaled.adaptive,
        unscaled.adaptive,
        1e-6,
      ),
    ],
  };
}

const symbolsW: [string, string] = ['w_{t-1}', T('第 t 步更新前的两个参数，行向量 [1, 2]')];
const topic: TopicDefinition = {
  id: 'optimizer',
  path: '/optimizer',
  title: T('优化器：Adam 怎样决定每一步走多远'),
  chapter: T('第 17 章'),
  intro: T(
    '同一个损失面、同一个学习率，SGD 与 Adam 走出完全不同的路线。逐步算出 Adam 的一阶矩、二阶矩、偏差修正和更新量，再看 AdamW 为什么要把权重衰减从梯度里拿出来。',
  ),
  question: T('这一页想弄明白：Adam 的一步更新由哪些量决定'),
  headline: T('梯度大小不同的两个方向，Adam 为什么走得差不多远？'),
  why: {
    question: T('学习率已经定了，为什么还需要 Adam？'),
    points: [
      T(
        'SGD 每一步走 $\\eta g$：梯度大的方向走得远，梯度小的方向几乎不动。两个方向曲率相差 10 倍时，同一个 η 要么让陡的方向来回振荡，要么让平的方向走得极慢。',
      ),
      T(
        'Adam 为每个参数记住梯度的滑动平均 $m$ 和平方的滑动平均 $v$，用 $m/\\sqrt{v}$ 决定方向和尺度，让各坐标的步长接近 η。原书第 17.5 节把 SGD、Adam、AdamW 并列；这一页在二维损失 $L=\\tfrac12(w_0^2+b\\,w_1^2)$ 上逐项算出来。',
      ),
    ],
    book: [T('17.5 · SGD、Adam 与 AdamW 的区别'), CH17],
  },
  input: [2, 2],
  inputNames: ['w₀', 'w₁'],
  fields: [
    field('eta', T('学习率 η'), 0.1, 0.01, 1, 0.01),
    choice('beta1', 'β₁', 0.9, [
      { value: 0, label: '0' },
      { value: 0.5, label: '0.5' },
      { value: 0.9, label: '0.9' },
    ]),
    choice('beta2', 'β₂', 0.999, [
      { value: 0.9, label: '0.9' },
      { value: 0.99, label: '0.99' },
      { value: 0.999, label: '0.999' },
    ]),
    field('lambda', T('权重衰减 λ'), 0.1, 0, 0.5, 0.05),
    choice('curvature', T('第二维曲率 b'), 10, [
      { value: 1, label: '1' },
      { value: 10, label: '10' },
      { value: 100, label: '100' },
    ]),
    choice('mode', T('权重衰减方式'), 0, [
      { value: 0, label: T('AdamW：解耦') },
      { value: 1, label: T('Adam + L2：加进梯度') },
    ]),
    field('t', T('查看第 t 步'), 3, 1, STEPS, 1, [0, 1, 2, 3, 4, 5]),
  ],
  steps: [
    lesson({
      name: T('梯度'),
      title: T('当前位置的梯度是多少？'),
      formula: String.raw`g_t=\nabla L(w_{t-1})=[\,w_0,\ b\,w_1\,]\quad\ (+\,\lambda w_{t-1},\ \mathrm{L2})`,
      symbols: [
        symbolsW,
        ['g_t', T('交给 Adam 的梯度；L2 方式会把 λw 加进来，AdamW 不会')],
        ['b', T('第二个坐标的曲率；第一个坐标的曲率固定为 1')],
      ],
      explanation: (c) =>
        T(
          '损失 L = ½(w₀² + b·w₁²) 在两个方向上的曲率是 1 和 b = {#0}，梯度的第二个分量要多乘一个 b。当前损失为 {#1}。“Adam + L2”方式下，λw 被加进梯度，之后会一起进入两个滑动平均。',
          fmt(c.config.curvature),
          fmt(c.metric('L(w_{t-1})'), 5),
        ),
      prompt: T('第二个坐标的梯度为什么要多乘 b？L2 方式和 AdamW 方式在这一步的 g_t 有什么不同？'),
      takeaway: T('梯度的两个分量尺度可以相差很多倍。'),
      book: [T('17.5.1 · SGD：公式最直观'), CH17],
      source: ['Adam · Algorithm 1', 'https://arxiv.org/abs/1412.6980'],
    }),
    lesson({
      name: T('一阶矩'),
      title: T('一阶矩怎样记住梯度的方向？'),
      formula: String.raw`m_t=\beta_1 m_{t-1}+(1-\beta_1)\,g_t`,
      symbols: [
        ['m_t', T('梯度的指数滑动平均，初值 m₀ = 0，形状与参数相同')],
        ['\\beta_1', T('旧平均保留的比例；β₁ = 0 时 m_t 就是当前梯度')],
      ],
      explanation: (c) =>
        T(
          '每一步保留 β₁ = {#0} 的旧平均，再混入 1 − β₁ 的新梯度。梯度方向稳定时 m 逐渐接近它；方向来回翻转时，正负相互抵消，m 变小。',
          fmt(c.config.beta1),
        ),
      prompt: T('β₁ = 0.9、从 m₀ = 0 开始，第 1 步的 m₁ 是 g₁ 的几倍？'),
      takeaway: T('一阶矩平滑方向，相当于带动量。'),
      book: [T('17.5.2 · Adam：先整理梯度的历史'), CH17],
      source: ['Adam · Algorithm 1', 'https://arxiv.org/abs/1412.6980'],
    }),
    lesson({
      name: T('二阶矩'),
      title: T('二阶矩怎样估计每个坐标的梯度尺度？'),
      formula: String.raw`v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^{2}`,
      symbols: [
        ['v_t', T('梯度逐元素平方的滑动平均，初值 v₀ = 0，始终非负')],
        ['\\beta_2', T('二阶矩的保留比例，通常比 β₁ 更接近 1')],
      ],
      explanation: (c) =>
        T(
          '平方逐元素计算，不是向量的模长平方。β₂ = {#0} 时，v 记住很长一段历史的梯度尺度。曲率大的坐标 v 大，后面会被除得更多。',
          fmt(c.config.beta2),
        ),
      prompt: T('g_t² 是逐元素平方还是向量的点积？v_t 能不能小于 0？'),
      takeaway: T('二阶矩按坐标记录梯度尺度。'),
      book: [T('17.5.2 · Adam：先整理梯度的历史'), CH17],
      source: ['Adam · Algorithm 1', 'https://arxiv.org/abs/1412.6980'],
    }),
    lesson({
      name: T('偏差修正'),
      title: T('前几步的滑动平均为什么偏小？'),
      formula: String.raw`\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{\,t}}`,
      symbols: [
        ['\\hat m_t,\\hat v_t', T('修正后的一阶、二阶矩估计')],
        ['1-\\beta^{\\,t}', T('从零初值开始累积的权重之和；t 增大时趋近 1')],
      ],
      explanation: (c) =>
        T(
          '两个平均都从 0 开始，前几步被拉向 0。除以已累积的权重 1 − βᵗ 把它们放大回来：第 {#0} 步时两个分母为 {#1}。第 1 步修正后正好 m̂ = g、v̂ = g²。',
          fmt(c.config.t),
          vec(c.values('[1−β₁ᵗ, 1−β₂ᵗ]')),
        ),
      prompt: T('t = 1 时修正后的 m̂、v̂ 等于什么？β₂ = 0.999 的修正为什么比 β₁ 慢得多才消失？'),
      takeaway: T('偏差修正抵消零初值，不改变长期行为。'),
      book: [T('17.6 · 为什么学习率还要变化？'), CH17],
      source: ['Adam · §3', 'https://arxiv.org/abs/1412.6980'],
    }),
    lesson({
      name: T('更新'),
      title: T('除以 √v̂ 之后，两个方向各走多远？'),
      formula: String.raw`w_t=(1-\eta\lambda)\,w_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}`,
      symbols: [
        ['\\eta', T('学习率；Adam 中每个坐标的步长大约不超过它')],
        ['\\lambda', T('权重衰减系数；AdamW 在归一化之外直接缩小参数')],
        ['\\epsilon', T('防止除零的小常数，本页取 1e−8')],
      ],
      explanation: (c) =>
        T(
          '除以 √v̂ 后，两个坐标的更新量之比为 {#0}；同一位置的 SGD 一步，比值是 {#1}。AdamW 把 −ηλw 单独加上，不经过归一化；“Adam + L2”方式下这一项为 0，λw 已经在梯度里被除过一次。',
          fmt(c.metric('|Δw₁/Δw₀| Adam'), 4),
          fmt(c.metric('|Δw₁/Δw₀| SGD'), 4),
        ),
      prompt: T('为什么 Adam 两个坐标的步长接近，SGD 的比值却随 b 变化？AdamW 的衰减项为什么不除以 √v̂？'),
      takeaway: T('Adam 按坐标归一化；AdamW 的衰减在归一化之外。'),
      book: [T('17.5.3 · AdamW：把 weight decay 分开'), CH17],
      source: ['Decoupled Weight Decay Regularization', 'https://arxiv.org/abs/1711.05101'],
    }),
    lesson({
      name: T('轨迹'),
      title: T('二十步之后，三种优化器分别到了哪里？'),
      formula: String.raw`L(w)=\tfrac12\,(w_0^2+b\,w_1^2)`,
      symbols: [
        ['L(w)', T('本页的二维二次损失；最小值在原点')],
        ['\\log_{10}L', T('纵轴使用的常用对数，便于同时显示收敛和发散')],
      ],
      explanation: (c) =>
        T(
          '二十步后 SGD 的 log₁₀ L 为 {#0}，当前方式为 {#1}。把 b 调到 100：SGD 用 η = 0.1 在陡的方向上每步大约乘以 1 − ηb = −9，开始发散；Adam 的步长由 η 控制，仍然稳定。',
          fmt(c.metric('log₁₀ L₂₀ SGD'), 3),
          fmt(c.metric('log₁₀ L₂₀'), 3),
        ),
      prompt: T('b = 100、η = 0.1 时 SGD 为什么发散？Adam 在同样设置下为什么不会？'),
      takeaway: T('同一个 η，在不同优化器里意味着不同的步长。'),
      book: [T('17.2 · 步子太小、太大与可用区间'), CH17],
      source: ['PyTorch AdamW', 'https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html'],
    }),
  ],
  compute,
  examples: [
    { name: T('两个方向曲率差 10 倍'), input: [2, 2] },
    { name: T('曲率相同'), input: [2, 2], parameters: { curvature: 1 } },
    { name: T('SGD 在 b = 100 发散'), input: [2, 2], parameters: { curvature: 100 } },
    { name: T('λ = 0：两种衰减相同'), input: [2, 2], parameters: { lambda: 0 } },
  ],
  prediction: {
    key: 'mode',
    target: 1,
    question: T('把解耦的权重衰减换成加进梯度的 L2，第 t 步后的参数会变吗？'),
    label: 'w_t',
  },
  sources: [
    { title: 'Adam: A Method for Stochastic Optimization', url: 'https://arxiv.org/abs/1412.6980' },
    { title: 'Decoupled Weight Decay Regularization (AdamW)', url: 'https://arxiv.org/abs/1711.05101' },
    {
      title: 'PyTorch AdamW',
      url: 'https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html',
    },
  ],
  limits: [
    T('二维二次损失、全批量确定性梯度，没有小批量噪声、梯度裁剪或学习率调度；结论用于理解更新规则，不代表真实网络的收敛速度。'),
    T('AdamW 按 PyTorch 的写法先乘 (1 − ηλ) 再做 Adam 更新；ε 固定为 1e−8。SGD 曲线使用与 L2 等价的衰减。'),
  ],
  verifiedDate: '2026-11-16',
};
export default topic;
