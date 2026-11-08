import {
  check,
  frame,
  softmax,
  sum,
  tensor,
  uniforms,
  type Computation,
  type Parameters,
} from './shared';
export function filterTokens(prob: number[], k: number, threshold: number) {
  const ordered = prob
      .map((p, id) => ({ p, id }))
      .filter((item) => item.p > 0)
      .sort((a, b) => b.p - a.p || a.id - b.id)
      .slice(0, k),
    kept: number[] = [];
  const mass = sum(ordered.map((v) => v.p));
  let cumulative = 0;
  for (const item of ordered) {
    kept.push(item.id);
    cumulative += item.p / mass;
    if (threshold < 1 && cumulative >= threshold) break;
  }
  const keptMass = sum(kept.map((id) => prob[id]));
  return {
    kept,
    probabilities: prob.map((v, id) => (kept.includes(id) ? v / keptMass : 0)),
    ordered,
    cumulative,
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
        joint[i].map((value, j) => ({ path: `${'abc'[i]}${'abc'[j]}`, first: i, score: value })),
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
    greedy: paths.find((v) => v.path === `${'abc'[first]}${'abc'[second]}`)!,
    beam: paths.find((v) => roots.includes(v.first))!,
    best: paths[0],
  };
}
export function sampling(x: number[], p: Parameters, seed: number): Computation {
  const raw = softmax(x),
    prob = softmax(x.map((v) => v / p.temperature)),
    topK = filterTokens(prob, p.k, 1),
    filtered = filterTokens(topK.probabilities, x.length, p.topP),
    u = uniforms(seed, 1)[0],
    sample = sampleToken(filtered.probabilities, u),
    tree = searchTree(prob);
  return {
    frames: [
      frame(
        '候选分数先变成概率',
        [tensor('logits', x), tensor('Softmax(logits)', raw)],
        [['概率和', sum(raw)]],
        'a、b、c 是玩具词表；没有调用语言模型。',
      ),
      frame(
        '温度先除 logits，再归一化',
        [
          tensor(
            'logits / τ',
            x.map((v) => v / p.temperature),
          ),
          tensor('温度分布', prob),
        ],
        [['温度 τ', p.temperature]],
        'τ > 0；较小温度强化较高分数。全部同分时仍均匀。',
      ),
      frame(
        'Top-K 按概率排序，恰好留下 K 个',
        [tensor('Top-K 后的分布', topK.probabilities)],
        [['保留 ID', topK.kept.map((i) => 'abc'[i]).join('、')]],
        '并列时按 token ID；这是恰好 K 个的教学规则，部分库按阈值保留并列项时可能超过 K。',
      ),
      frame(
        '在 Top-K 分布上做 Top-P，保留跨过阈值的一项',
        [tensor('筛选前', topK.probabilities), tensor('筛选并重新归一化', filtered.probabilities)],
        [
          ['阈值 P', p.topP],
          ['保留 ID', filtered.kept.map((i) => 'abc'[i]).join('、')],
          ['截止累计质量', filtered.cumulative],
        ],
        '本例明确先 K 后 P；P 的累计是在当时的归一化分布上计算。',
      ),
      frame(
        '固定种子生成一次均匀数，落在哪段就选谁',
        [tensor('采样分布', filtered.probabilities)],
        [
          ['u ∈ [0,1)', u],
          ['本次 token', 'abc'[sample]],
        ],
        '固定种子可复现一次抽样；一次结果不代表长期频率。',
      ),
      frame(
        '两步树：局部最大不保证序列联合概率最大',
        [
          {
            ...tensor('条件概率 P(y₂|y₁)', tree.conditional),
            rows: ['a', 'b', 'c'],
            columns: ['a', 'b', 'c'],
          },
          {
            ...tensor(
              '所有两步路径概率',
              tree.paths.map((v) => [v.score]),
            ),
            rows: tree.paths.map((v) => v.path),
          },
        ],
        [
          ['Greedy', `${tree.greedy.path} · ${tree.greedy.score.toPrecision(4)}`],
          ['Beam-2', `${tree.beam.path} · ${tree.beam.score.toPrecision(4)}`],
          ['穷举最优', tree.best.path],
        ],
        '第一步用当前温度分布；条件表固定，不应用 Top-K/P。固定长度 2，无 EOS 与长度惩罚；Beam 在一般情形也可能剪掉全局最优。',
      ),
    ].map((item, index) =>
      index === 5
        ? {
            ...item,
            tree: {
              root: prob,
              conditional: tree.conditional,
              joint: tree.joint,
              selected: tree.beam.path,
            },
          }
        : item,
    ),
    checks: [
      check('筛选后概率和', [sum(filtered.probabilities)], [1]),
      check('采样 token 位于保留集合', [filtered.kept.includes(sample) ? 1 : 0], [1]),
    ],
    signature: filtered.probabilities,
  };
}
