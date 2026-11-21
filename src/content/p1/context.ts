import { formatNumber as fmt } from '../../math/attention';
import { contextExtension, rotatedScore } from '../../math/p1/context';
import { check, frame, type Computation, type Parameters, type Tensor } from '../../math/p1/shared';
import { chapterUrl } from '../book';
import { choice, lesson, type TopicDefinition } from './types';
import { T } from '../../i18n';

const CH25 = chapterUrl('第25章-位置编码演进-Sinusoidal到RoPE到ALiBi/');
const table = (name: string, columns: string[], columnsValues: number[][]): Tensor => ({
  name,
  columns,
  rows: columnsValues[0].map((_, d) => `d=${d}`),
  values: columnsValues[0].map((_, d) => columnsValues.map((column) => column[d])),
});
const yes = (flags: boolean[]) => flags.map(Number);
const unseen = (flags: boolean[]) => flags.filter((f) => !f).length;

function compute(input: number[], p: Parameters): Computation {
  const r = contextExtension(p.dims, p.base, p.trained, p.scale);
  const L = p.trained,
    target = r.target;
  const q = input.slice(0, 2),
    k = input.slice(2, 4);
  const sizes: [string, number][] = [
    ['L', L],
    ["L'", target],
    ['s', p.scale],
  ];
  const extrapolated = r.seen(r.methods.extrapolate),
    pi = r.seen(r.methods.pi),
    ntk = r.seen(r.methods.ntk),
    yarn = r.seen(r.methods.yarn);
  const last = r.theta.length - 1;
  const methodNames = [T('直接外推'), 'PI', 'NTK-aware', 'YaRN'];
  const methodFrequencies = [r.methods.extrapolate, r.methods.pi, r.methods.ntk, r.methods.yarn];
  const methodSeen = [extrapolated, pi, ntk, yarn];
  const distance = target - 1;
  const lowest: Tensor = {
    name: T('最低频维度 d = {#0}，相对距离 Δ = L′ − 1', last),
    columns: [T('频率'), T('转角 Δ·f'), T('训练中见过'), 'q·R(Δf)k'],
    rows: methodNames,
    values: methodFrequencies.map((f, m) => [
      f[last],
      distance * f[last],
      Number(methodSeen[m][last]),
      rotatedScore(q, k, distance * f[last]) * (m === 3 ? r.temperatureFactor ** 2 : 1),
    ]),
  };
  const frames = [
    frame(
      T('每个维度对的频率与波长'),
      [
        table(T('频率、波长与训练中的转角'), ['θ_d', 'λ_d', 'L·θ_d', 'λ_d ≤ L'], [
          r.theta,
          r.wavelength,
          r.theta.map((t) => L * t),
          r.wavelength.map((l) => Number(l <= L)),
        ]),
      ],
      [...sizes, [T('没转满一圈的维度'), r.wavelength.filter((l) => l > L).length]],
    ),
    frame(
      T('直接外推到 L′'),
      [
        table(T('最远位置的转角'), ["(L'−1)·θ_d", T('训练中见过')], [
          r.theta.map((t) => distance * t),
          yes(extrapolated),
        ]),
      ],
      [...sizes, [T('没见过的维度'), unseen(extrapolated)]],
    ),
    frame(
      'Position Interpolation',
      [
        table(T('所有频率除以 s'), ['θ_d / s', "(L'−1)·θ_d / s", T('训练中见过')], [
          r.methods.pi,
          r.methods.pi.map((f) => distance * f),
          yes(pi),
        ]),
      ],
      [...sizes, [T('没见过的维度'), unseen(pi)], [T('最高频的相邻转角'), r.methods.pi[0]]],
    ),
    frame(
      'NTK-aware',
      [
        table(T('换底数后的频率'), ["θ'_d", "θ'_d / θ_d", T('训练中见过')], [
          r.methods.ntk,
          r.methods.ntk.map((f, d) => f / r.theta[d]),
          yes(ntk),
        ]),
      ],
      [...sizes, ["b'", r.ntkBase], [T('没见过的维度'), unseen(ntk)]],
    ),
    frame(
      T('YaRN：按维度混合，再调温度'),
      [
        table(T('按转数决定插值比例'), ['r(d)', 'γ(r)', 'h(θ_d)', T('训练中见过')], [
          r.ratio,
          r.gamma,
          r.methods.yarn,
          yes(yarn),
        ]),
      ],
      [...sizes, ['√(1/t)', r.temperatureFactor], [T('没见过的维度'), unseen(yarn)]],
    ),
    frame(T('同一个距离，四种方案的转角与分数'), [lowest], sizes),
  ];
  // Each step plots log₁₀ of the frequencies it compares; the guide marks λ = L (one full turn).
  const curve = (label: string, f: number[]) => ({
    label,
    points: f.map((v, d): [number, number] => [d, Math.log10(v)]),
  });
  const turn = Math.log10((2 * Math.PI) / L);
  const original = curve('θ', r.theta);
  const curvesByStep = [
    [original],
    [original],
    [original, curve('PI', r.methods.pi)],
    [original, curve('NTK-aware', r.methods.ntk)],
    [original, curve('PI', r.methods.pi), curve('YaRN', r.methods.yarn)],
    [curve('PI', r.methods.pi), curve('NTK-aware', r.methods.ntk), curve('YaRN', r.methods.yarn)],
  ];
  const ys = curvesByStep.flat().flatMap((c) => c.points.map(([, y]) => y));
  frames.forEach((f, step) => {
    const curves = curvesByStep[step];
    const shown = curves[curves.length - 1];
    f.plot = {
      x: { label: T('维度对 d'), min: 0, max: last },
      y: { label: 'log₁₀ θ', min: Math.min(turn, ...ys) - 0.3, max: Math.max(...ys) + 0.3 },
      curves,
      points: shown.points.map(([x, y]) => ({ label: '', x, y, series: curves.length - 1 })),
      guides: [{ axis: 'y', value: turn, label: 'λ = L' }],
      caption: T('纵轴是频率的常用对数；横线 λ = L 以下的维度在训练长度内没有转满一圈。'),
    };
  });
  return {
    frames,
    signature: [r.methods.yarn[0]],
    checks: [
      check(T('NTK-aware：最低频维度正好缩小 s 倍'), [r.methods.ntk[last] * p.scale], [r.theta[last]]),
      check(
        T('YaRN：γ = 1 的维度保持 θ，γ = 0 的维度等于 PI'),
        r.methods.yarn.filter((_, d) => r.gamma[d] === 1 || r.gamma[d] === 0),
        r.theta
          .map((t, d) => (r.gamma[d] === 1 ? t : t / p.scale))
          .filter((_, d) => r.gamma[d] === 1 || r.gamma[d] === 0),
      ),
      check(
        T('旋转不改变 k 的二维长度'),
        [Math.hypot(k[0], k[1])],
        [Math.hypot(rotatedScore([1, 0], k, 0.7), rotatedScore([0, 1], k, 0.7))],
      ),
    ],
  };
}

const topic: TopicDefinition = {
  id: 'context',
  path: '/context-extension',
  title: T('上下文扩展：PI、NTK-aware 与 YaRN'),
  chapter: T('第 25 章'),
  intro: T(
    '训练时只见过 L 个位置，怎样把 RoPE 用到 s·L？逐维算出频率与波长，比较直接外推、整体插值、换底数和按维度混合各让哪些维度转到训练中没见过的角度。',
  ),
  question: T('这一页想弄明白：把上下文拉长时，RoPE 的每个维度发生了什么'),
  headline: T('同样把窗口扩大 s 倍，四种做法改的是哪些频率？'),
  why: {
    question: T('RoPE 能算任意位置，为什么还要专门“扩展”上下文？'),
    points: [
      T(
        'RoPE 的公式对任何位置都能算出转角，但模型只在 0 到 $L-1$ 的位置上训练过。低频维度在训练长度内连一圈都没转满，位置超过 $L$ 后会转到从未见过的角度；原书 25.5.1 把这称为外推的困境。',
      ),
      T(
        '这一页把 $|D|$ 维拆成 $|D|/2$ 个维度对，逐对算出 $\\theta_d$、波长 $\\lambda_d$ 和最远位置的转角，再依次换成 PI、NTK-aware 与 YaRN，看每种方法让哪些维度保持原样、哪些被压缩。',
      ),
    ],
    book: [T('25.5 · YaRN：扩展 RoPE 的上下文长度'), CH25],
  },
  input: [1, 0, 1, 0],
  inputNames: ['q₀', 'q₁', 'k₀', 'k₁'],
  fields: [
    choice('dims', T('维度 |D|'), 8, [
      { value: 8, label: '8' },
      { value: 16, label: '16' },
    ]),
    choice('base', T('底数 b'), 10000, [
      { value: 10000, label: '10000' },
      { value: 500000, label: '500000' },
    ]),
    choice('trained', T('训练长度 L'), 4096, [
      { value: 2048, label: '2048' },
      { value: 4096, label: '4096' },
    ]),
    choice('scale', T('扩展倍数 s = L′/L'), 8, [
      { value: 2, label: '2' },
      { value: 4, label: '4' },
      { value: 8, label: '8' },
      { value: 16, label: '16' },
      { value: 32, label: '32' },
    ]),
  ],
  steps: [
    lesson({
      name: T('频率'),
      title: T('每个维度对转得多快？'),
      formula: String.raw`\theta_d=b^{-2d/|D|},\qquad \lambda_d=\frac{2\pi}{\theta_d}`,
      symbols: [
        ['\\theta_d', T('第 d 个维度对每前进一个位置转过的弧度，d = 0 … |D|/2 − 1')],
        ['\\lambda_d', T('波长：转满一圈 2π 需要的位置数')],
        ['b', T('RoPE 的底数，原始设定为 10000')],
      ],
      explanation: (c) =>
        T(
          'd = 0 的维度每个位置转 1 弧度，越往后转得越慢。训练长度 L = {#0} 时，有 {#1} 个维度对的波长超过 L：它们在训练中只见过一段角度，没有转满一圈。',
          fmt(c.config.trained),
          fmt(c.metric(T('没转满一圈的维度'))),
        ),
      prompt: T('哪些维度在训练长度内转满了好几圈？哪些连一圈都没转完？'),
      takeaway: T('高频维度转很多圈，低频维度可能一圈都没转满。'),
      book: [T('25.3.6 · 频率的选择'), CH25],
      source: ['YaRN · §2.1–2.2', 'https://arxiv.org/abs/2309.00071'],
    }),
    lesson({
      name: T('外推'),
      title: T('不做任何修改，位置 L′ − 1 会转到哪里？'),
      formula: String.raw`\phi_d=(L'-1)\,\theta_d\quad\text{vs}\quad L\,\theta_d`,
      symbols: [
        ["L'", T('扩展后的上下文长度 s·L')],
        ['\\phi_d', T('最远位置在第 d 个维度对上的转角')],
      ],
      explanation: (c) =>
        T(
          '转满过一圈的维度，任何角度在训练中都出现过；没转满的维度只见过 [0, L·θ_d)。直接外推时，有 {#0} 个维度对转到训练中没见过的角度。这是角度层面的判断，不是对模型质量的测量。',
          fmt(c.metric(T('没见过的维度'))),
        ),
      prompt: T('为什么高频维度外推没问题，低频维度却会遇到新角度？'),
      takeaway: T('外推的风险集中在低频维度。'),
      book: [T('25.5.1 · 问题：外推的困境'), CH25],
      source: ['YaRN · §3.2', 'https://arxiv.org/abs/2309.00071'],
    }),
    lesson({
      name: 'PI',
      title: T('把所有位置除以 s，代价是什么？'),
      formula: String.raw`m\mapsto \frac{m}{s}\quad\Longleftrightarrow\quad \theta_d\mapsto\frac{\theta_d}{s}`,
      symbols: [
        ['s', T('扩展倍数 L′/L')],
        ['m', T('token 的位置编号')],
      ],
      explanation: (c) =>
        T(
          'Position Interpolation 把位置 0 … L′ − 1 压回 0 … L 以内，所有维度都不再越界。代价是相邻 token 的转角也缩小 s 倍：最高频维度从 1 降到 {#0} 弧度，相近位置更难区分。',
          fmt(c.metric(T('最高频的相邻转角')), 4),
        ),
      prompt: T('PI 之后还有没见过的角度吗？相邻两个 token 的转角变成了多少？'),
      takeaway: T('PI 消除越界，但压缩了高频维度的分辨率。'),
      book: [T('25.5.2 · Position Interpolation (PI)'), CH25],
      source: ['Position Interpolation', 'https://arxiv.org/abs/2306.15595'],
    }),
    lesson({
      name: 'NTK-aware',
      title: T('只换底数，高频和低频各变多少？'),
      formula: String.raw`b'=b\cdot s^{|D|/(|D|-2)},\qquad \theta'_d=b'^{\,-2d/|D|}`,
      symbols: [
        ["b'", T('换算后的底数；指数让最后一个维度对正好缩小 s 倍')],
        ["\\theta'_d", T('用新底数算出的频率')],
      ],
      explanation: (c) =>
        T(
          '底数从 b 变为 {#0}。第 0 维频率不变，最后一维正好变为 θ/s，中间的维度逐渐过渡。高频保留分辨率，低频得到与 PI 相同的压缩；论文也指出，部分中间维度可能略微越界。',
          fmt(c.metric("b'"), 1),
        ),
      prompt: T('为什么指数取 |D|/(|D|−2)？第 0 维的频率变了吗？'),
      takeaway: T('NTK-aware 把压缩集中到低频维度。'),
      book: [T('25.5.3 · NTK-Aware Interpolation'), CH25],
      source: ['YaRN · §3.1, Appendix A.2', 'https://arxiv.org/abs/2309.00071'],
    }),
    lesson({
      name: 'YaRN',
      title: T('按每个维度转了几圈，决定插值多少'),
      formula: String.raw`r(d)=\frac{L}{\lambda_d},\quad h(\theta_d)=\bigl(1-\gamma(r)\bigr)\frac{\theta_d}{s}+\gamma(r)\,\theta_d`,
      symbols: [
        ['r(d)', T('训练长度内第 d 个维度对转过的圈数')],
        ['\\gamma(r)', T('斜坡：r < 1 时为 0（完全插值），r > 32 时为 1（不插值），中间线性过渡')],
        ['\\sqrt{1/t}', T('注意力温度：q 与 k 都乘以它，logits 因此乘以 1/t')],
      ],
      explanation: (c) =>
        T(
          '转了很多圈的维度保持原频率，不到一圈的维度按 PI 压缩，中间按圈数线性混合；α = 1、β = 32 是论文为 Llama 推荐的值。YaRN 再把 q、k 各乘 √(1/t) = 0.1·ln s + 1 = {#0}，用来抵消拉长后注意力分布变平。',
          fmt(c.metric('√(1/t)'), 4),
        ),
      prompt: T('γ = 1 和 γ = 0 的维度分别等于哪种方法？温度系数和 s 是什么关系？'),
      takeaway: T('YaRN 只插值低频，高频原样保留，再修正温度。'),
      book: [T('25.5.4 · YaRN 的完整方案'), CH25],
      source: ['YaRN · §3.2–3.3', 'https://arxiv.org/abs/2309.00071'],
    }),
    lesson({
      name: T('对照'),
      title: T('同一个距离，四种方案的分数一样吗？'),
      formula: String.raw`\mathrm{score}=q\cdot R\bigl(\Delta f\bigr)\,k,\qquad \Delta=L'-1`,
      symbols: [
        ['R(\\Delta f)', T('把 k 的二维对按转角 Δ·f 旋转的矩阵')],
        ['q,k', T('输入的两个二维内容向量；只看最低频这一个维度对')],
      ],
      explanation: T(
        '固定内容 q、k，只换频率规则，最低频维度上的转角与分数随之不同；YaRN 一行已乘上温度 1/t。修改 q、k 可以看到分数随内容变化，但“是否见过这个角度”只取决于频率规则。',
      ),
      prompt: T('改变 q、k 会改变“训练中见过”这一列吗？为什么？'),
      takeaway: T('扩展方法改变的是频率，内容向量不变。'),
      book: [T('25.5.5 · 温度调整'), CH25],
      source: ['YaRN · Eq. 14–15', 'https://arxiv.org/abs/2309.00071'],
    }),
  ],
  compute,
  examples: [
    { name: T('L = 4096，扩到 8 倍'), input: [1, 0, 1, 0] },
    { name: T('扩到 32 倍'), input: [1, 0, 1, 0], parameters: { scale: 32 } },
    { name: T('|D| = 16'), input: [1, 0, 1, 0], parameters: { dims: 16 } },
    { name: T('q 与 k 垂直'), input: [1, 0, 0, 1] },
  ],
  prediction: {
    key: 'scale',
    target: 16,
    question: T('扩展倍数从 8 改成 16，YaRN 中最高频那一维的频率会变吗？'),
    label: 'h(θ₀)',
  },
  sources: [
    { title: 'YaRN: Efficient Context Window Extension', url: 'https://arxiv.org/abs/2309.00071' },
    { title: 'Extending Context Window via Position Interpolation', url: 'https://arxiv.org/abs/2306.15595' },
    { title: 'RoFormer (RoPE)', url: 'https://arxiv.org/abs/2104.09864' },
  ],
  limits: [
    T('只比较频率与转角；“训练中见过”按角度范围判断，不是模型质量测量。真实扩展通常还需要少量继续训练，能算出远处的位置不等于模型能可靠使用整段上下文。'),
    T('PI 采用论文附录 A.1 的 m·L/L′ 形式；α = 1、β = 32 与温度公式是论文针对 Llama 系列给出的推荐值，其他模型可能不同。'),
  ],
  verifiedDate: '2026-11-16',
};
export default topic;
