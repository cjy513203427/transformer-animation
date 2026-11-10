import { formatNumber as fmt } from '../attention';
import {
  check,
  frame,
  sample,
  softmax,
  sum,
  tensor,
  uniforms,
  type Computation,
  type Parameters,
  type Ranking,
  type Strip,
} from './shared';
import { T } from '../../i18n';
const NAMES = 'abc';
const TEMPERATURE_RANGE: [number, number] = [0.25, 4];
export function filterTokens(prob: number[], k: number, threshold: number) {
  const ordered = prob
    .map((p, id) => ({ p, id }))
    .filter((item) => item.p > 0)
    .sort((a, b) => b.p - a.p || a.id - b.id)
    .slice(0, k);
  const mass = sum(ordered.map((v) => v.p));
  let total = 0;
  /** Cumulative probability after each ordered item, on the distribution normalized over them. */
  const running = ordered.map((item) => (total += item.p / mass));
  const cut = threshold < 1 ? running.findIndex((c) => c >= threshold) : -1,
    kept = ordered.slice(0, cut === -1 ? ordered.length : cut + 1).map((v) => v.id);
  const keptMass = sum(kept.map((id) => prob[id]));
  return {
    kept,
    probabilities: prob.map((v, id) => (kept.includes(id) ? v / keptMass : 0)),
    ordered,
    running,
    cumulative: cut === -1 ? total : running[cut],
  };
}
export function sampleToken(prob: number[], u: number) {
  let c = 0;
  for (let i = 0; i < prob.length; i++) {
    c += prob[i];
    if (u < c) return i;
  }
  for (let i = prob.length - 1; i >= 0; i--) if (prob[i] > 0) return i;
  return -1;
}
export function searchTree(root: number[]) {
  const conditional = [
      [0.4, 0.3, 0.3],
      [0.9, 0.05, 0.05],
      [0.5, 0.25, 0.25],
    ],
    joint = conditional.map((row, i) => row.map((c) => root[i] * c)),
    paths = root
      .flatMap((_, i) =>
        joint[i].map((value, j) => ({ path: `${NAMES[i]}${NAMES[j]}`, first: i, score: value })),
      )
      .sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
  const first = root.indexOf(Math.max(...root)),
    second = conditional[first].indexOf(Math.max(...conditional[first]));
  const roots = root
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p || a.i - b.i)
    .slice(0, 2)
    .map((v) => v.i);
  return {
    conditional,
    joint,
    paths,
    greedy: paths.find((v) => v.path === `${NAMES[first]}${NAMES[second]}`)!,
    beam: paths.find((v) => roots.includes(v.first))!,
    best: paths[0],
  };
}
/**
 * Candidates in the order the filter visits them: positive probabilities from high to low (ties by
 * id), then the ones an earlier filter already removed. `running` comes from the filter itself.
 */
function ranking(
  before: number[],
  result: ReturnType<typeof filterTokens>,
  caption: string,
  threshold?: number,
): Ranking {
  const visited = result.ordered.map((item, i) => ({ id: item.id, cumulative: result.running[i] })),
    rest = before
      .map((_, id) => id)
      .filter((id) => !visited.some((v) => v.id === id))
      .map((id) => ({ id, cumulative: visited.at(-1)?.cumulative ?? 0 }));
  return {
    items: [...visited, ...rest].map(({ id, cumulative }) => ({
      name: NAMES[id],
      before: before[id],
      after: result.probabilities[id],
      kept: result.kept.includes(id),
      cumulative,
    })),
    threshold,
    caption,
  };
}
/** Consecutive [start, end) segments in token order, the same order `sampleToken` walks. */
function strip(prob: number[], u: number, chosen: number): Strip {
  let start = 0;
  const segments = prob.map((p, id) => {
    const segment = { name: NAMES[id], start, end: start + p };
    start += p;
    return segment;
  });
  const hit = segments[chosen];
  return {
    segments,
    u,
    chosen,
    caption: T(
      'u = {#0} 落在 {#1} 的区间 [{#2}, {#3})，所以这次选 {#4}。被删去的候选区间长度为 0，不可能被选中。',
      fmt(u, 4),
      hit.name,
      fmt(hit.start, 3),
      fmt(hit.end, 3),
      hit.name,
    ),
  };
}
export function sampling(x: number[], p: Parameters, seed: number): Computation {
  const raw = softmax(x),
    prob = softmax(x.map((v) => v / p.temperature)),
    topK = filterTokens(prob, p.k, 1),
    filtered = filterTokens(topK.probabilities, x.length, p.topP),
    u = uniforms(seed, 1)[0],
    sample_ = sampleToken(filtered.probabilities, u),
    tree = searchTree(prob);
  const [low, high] = TEMPERATURE_RANGE,
    same = x.every((v) => Math.abs(v - x[0]) < 1e-12),
    keptMass = sum(topK.kept.map((id) => prob[id])),
    crossing = filtered.kept.at(-1)!;
  return {
    frames: [
      frame(
        T('候选分数先变成概率'),
        [tensor('logits', x), tensor('Softmax(logits)', raw)],
        [[T('概率和'), sum(raw)]],
        T('a、b、c 是玩具词表；没有调用语言模型。'),
      ),
      {
        ...frame(
          T('温度先除 logits，再归一化'),
          [
            tensor(
              'logits / τ',
              x.map((v) => v / p.temperature),
            ),
            tensor(T('温度分布'), prob),
          ],
          [[T('温度 τ'), p.temperature]],
          T('τ > 0；较小温度强化较高分数。全部同分时仍均匀。'),
        ),
        plot: {
          x: { label: T('温度 τ'), min: low, max: high },
          y: { label: T('概率'), min: 0, max: 1 },
          curves: [...NAMES].map((name, i) => ({
            label: name,
            points: sample(low, high, 76, (t) => softmax(x.map((v) => v / t))[i]),
          })),
          points: [...NAMES].map((name, i) => ({
            label: name,
            x: p.temperature,
            y: prob[i],
            series: i,
          })),
          guides: [
            { axis: 'x', value: p.temperature, label: T('当前 τ = {#0}', fmt(p.temperature)) },
            ...(p.temperature === 1 ? [] : [{ axis: 'x' as const, value: 1, label: 'τ = 1' }]),
          ],
          caption: same
            ? T('三个 logits 相同，三条线重合在 1/3：温度改变不了均匀分布。')
            : T(
                '每条线是一个候选的概率随 τ 的变化，竖线是当前 τ。往左（τ 小），最高分越来越占优；往右（τ 大），三条线互相靠近。',
              ),
        },
      },
      {
        ...frame(
          T('Top-K 按概率排序，恰好留下 K 个'),
          [tensor(T('Top-K 后的分布'), topK.probabilities)],
          [[T('保留 ID'), topK.kept.map((i) => NAMES[i]).join(T('、'))]],
          T('并列时按 token ID；这是恰好 K 个的教学规则，部分库按阈值保留并列项时可能超过 K。'),
        ),
        ranking: ranking(
          prob,
          topK,
          T(
            '从高到低排好，前 K = {#0} 个保留；空心柱是筛选前的概率，实心柱是除以保留项之和 {#1} 之后的概率。',
            fmt(p.k),
            fmt(keptMass, 4),
          ),
        ),
      },
      {
        ...frame(
          T('在 Top-K 分布上做 Top-P，保留跨过阈值的一项'),
          [
            tensor(T('筛选前'), topK.probabilities),
            tensor(T('筛选并重新归一化'), filtered.probabilities),
          ],
          [
            [T('阈值 P'), p.topP],
            [T('保留 ID'), filtered.kept.map((i) => NAMES[i]).join(T('、'))],
            [T('截止累计质量'), filtered.cumulative],
          ],
          T('本例明确先 K 后 P；P 的累计是在当时的归一化分布上计算。'),
        ),
        ranking: ranking(
          topK.probabilities,
          filtered,
          p.topP < 1
            ? T(
                '折线是从高到低的累计概率，虚线是 P = {#0}。累计第一次 ≥ P 的是 {#1}（累计 {#2}），它也保留，后面的删去。',
                fmt(p.topP),
                NAMES[crossing],
                fmt(filtered.cumulative, 4),
              )
            : T('P = 1：累计要到全部正概率项加完才达到 1，所以一项也不删。'),
          p.topP,
        ),
      },
      {
        ...frame(
          T('固定种子生成一次均匀数，落在哪段就选谁'),
          [tensor(T('采样分布'), filtered.probabilities)],
          [
            ['u ∈ [0,1)', u],
            [T('本次 token'), NAMES[sample_]],
          ],
          T('固定种子可复现一次抽样；一次结果不代表长期频率。'),
        ),
        strip: strip(filtered.probabilities, u, sample_),
      },
      {
        ...frame(
          T('两步树：局部最大不保证序列联合概率最大'),
          [
            {
              ...tensor(T('条件概率 P(y₂|y₁)'), tree.conditional),
              rows: ['a', 'b', 'c'],
              columns: ['a', 'b', 'c'],
            },
            {
              ...tensor(
                T('所有两步路径概率'),
                tree.paths.map((v) => [v.score]),
              ),
              rows: tree.paths.map((v) => v.path),
            },
          ],
          [
            ['Greedy', `${tree.greedy.path} · ${tree.greedy.score.toPrecision(4)}`],
            ['Beam-2', `${tree.beam.path} · ${tree.beam.score.toPrecision(4)}`],
            [T('穷举最优'), tree.best.path],
          ],
          T(
            '第一步用当前温度分布；条件表固定，不应用 Top-K/P。固定长度 2，无 EOS 与长度惩罚；Beam 在一般情形也可能剪掉全局最优。',
          ),
        ),
        tree: {
          root: prob,
          conditional: tree.conditional,
          joint: tree.joint,
          selected: tree.beam.path,
        },
      },
    ],
    checks: [
      check(T('筛选后概率和'), [sum(filtered.probabilities)], [1]),
      check(T('采样 token 位于保留集合'), [filtered.kept.includes(sample_) ? 1 : 0], [1]),
    ],
    signature: filtered.probabilities,
  };
}
