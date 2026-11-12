import { multiheadSymbols } from '../../content/notation/multihead';
import { chapterUrl } from '../../content/book';
import { useMemo, useState } from 'react';
import { Formula } from '../../components/Formula';
import { Matrix } from '../../components/Matrix';
import { plain, Rich, Tex } from '../../components/Tex';
import { formatNumber as fmt, matmul } from '../../math/attention';
import {
  attentionWeightCount,
  columnBlock,
  HEAD_OPTIONS,
  MH_WEIGHTS,
  MH_X,
  multiHeadAttention,
  scoreMultiplyAdds,
  type HeadCount,
} from '../../math/multihead';
import { HEAD_STEP_COUNT, type ExperimentState } from '../../state/experiment';
import { HEAD_COLORS, MultiHeadDiagram } from './MultiHeadDiagram';
import { HeadTensor } from './HeadTensor';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/multihead.css';

const words = [T('我'), T('喜欢'), T('学习')];
const TOLERANCE = 1e-12;
const KEY_TARGET = { token: 0, column: 3 };
const vector = (values: number[]) => `[${values.map((value) => fmt(value)).join(', ')}]`;

const stepNames = [
  T('投影'),
  T('切分多头'),
  T('矩阵变形'),
  T('每头 Attention'),
  'Concatenate',
  T('$W_O$ 融合'),
];
const titles = [
  T('先做一次大的投影，得到完整的 Q、K、V'),
  T('切分多头：把 $d_{\\mathrm{model}}$ 平均分给每个头'),
  T('矩阵变形：把“头”这一维换到前面'),
  T('每个头独立算一遍 Attention'),
  T('Concatenate：把各头的 A 并排拼回去'),
  T('乘 $W_O$：各头的信息第一次混合'),
];
const takeaways = [
  T('先一次投影，再切分；每个头的列都用到了 X 的全部 4 个输入维度。'),
  T('切分切的是特征维 $d_{\\mathrm{model}}$，不是把 token 切开，数值一个也没变。'),
  T('转置只交换“头”和“token”两个轴的顺序，第 t 行始终是同一个 token。'),
  T('每个头有自己的一张 P：同一套公式，读的是不同的列。'),
  T(
    '拼接把 [头, token, $d_{\\mathrm{key}}$] 摆回 [token, $d_{\\mathrm{model}}$]，此时各头还没有交流。',
  ),
  T(
    '$W_O$ 的每一列读取 A 的全部列：$\\text{Output}=\\sum_h A_h\\,W_O^{(h)}$，头的信息在这里融合。',
  ),
];
const prompts = [
  T('这一步的输入、输出形状各是什么？为什么说此时还没有“头”？'),
  T(
    '$d_{\\mathrm{key}}$ 怎样由 $d_{\\mathrm{model}}$ 和 $\\mathrm{num\\_heads}$ 算出？切分改变数值了吗？',
  ),
  T('转置前后，head 2 的第 3 行分别对应哪个 token 的哪些数？'),
  T(
    '每个头除以的是 $\\sqrt{d_{\\mathrm{model}}}$ 还是 $\\sqrt{d_{\\mathrm{key}}}$？两个头的 P 为什么不同？',
  ),
  T('拼接后的 A 是什么形状？这一步有没有让头之间交流？'),
  T('$W_O$ 为什么能让输出的每一列都受所有头影响？'),
];
const formulas = [
  'Q=XW_Q,\\quad K=XW_K,\\quad V=XW_V',
  'd_{\\mathrm{key}}=\\frac{d_{\\mathrm{model}}}{\\mathrm{num\\_heads}},\\qquad [T,\\,d_{\\mathrm{model}}]\\to[T,\\,\\mathrm{num\\_heads},\\,d_{\\mathrm{key}}]',
  '[T,\\,\\mathrm{num\\_heads},\\,d_{\\mathrm{key}}]\\xrightarrow{\\ \\text{transpose}\\ }[\\mathrm{num\\_heads},\\,T,\\,d_{\\mathrm{key}}]',
  'A_h=\\operatorname{softmax}\\!\\left(\\frac{Q_hK_h^\\top}{\\sqrt{d_{\\mathrm{key}}}}+M\\right)V_h',
  'A=\\operatorname{Concat}(A_1,\\dots,A_{\\mathrm{num\\_heads}})\\in\\mathbb{R}^{T\\times d_{\\mathrm{model}}}',
  '\\text{Output}=AW_O=\\sum_h A_h\\,W_O^{(h)}',
];
const chapterLinks: [string, string][] = [
  [T('第 11.2.4 节 · 两种切分方式'), '第11章-Multi-Head-Attention-多视角理解/'],
  [T('第 11.2.1 节 · 维度切分'), '第11章-Multi-Head-Attention-多视角理解/'],
  [T('第 11.2.2 节 · 为什么要转置'), '第11章-Multi-Head-Attention-多视角理解/'],
  [T('第 11.3 节 · 并行计算多个头'), '第11章-Multi-Head-Attention-多视角理解/'],
  [T('第 12.3 节 · Concatenate'), '第12章-QKV输出的本质/'],
  [T('第 12.4 节 · 线性变换 WO'), '第12章-QKV输出的本质/'],
];

interface Props {
  state: ExperimentState;
  onState: (state: ExperimentState) => void;
  onJump: (step: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  reducedMotion: boolean;
  onOpenSingleHead: () => void;
}

function differs(a: number, b: number) {
  return Math.abs(a - b) > TOLERANCE;
}

export function MultiHeadLesson({
  state,
  onState,
  onJump,
  playing,
  onTogglePlay,
  reducedMotion,
  onOpenSingleHead,
}: Props) {
  const { numHeads, query, step, keyDelta } = state.heads;
  const causal = state.config.causal;
  const dKey = 4 / numHeads;
  const hide = !state.view.explanations;
  const [draft, setDraft] = useState(keyDelta || -2);
  const [prediction, setPrediction] = useState<'all' | 'local' | 'mixed' | null>(null);
  const [verified, setVerified] = useState(false);
  const [recallOpen, setRecallOpen] = useState(false);

  const result = useMemo(
    () =>
      multiHeadAttention(MH_X, MH_WEIGHTS, numHeads, causal, { ...KEY_TARGET, delta: keyDelta }),
    [numHeads, causal, keyDelta],
  );
  const baseline = useMemo(
    () => multiHeadAttention(MH_X, MH_WEIGHTS, numHeads, causal),
    [numHeads, causal],
  );
  const changed = useMemo(
    () => ({
      key: result.k.map((row, i) => row.map((x, j) => differs(x, baseline.k[i][j]))),
      probabilities: result.heads.map((head, h) =>
        head.probabilities.map((row, i) =>
          row.map((p, j) => differs(p, baseline.heads[h].probabilities[i][j])),
        ),
      ),
      concat: result.concat.map((row, i) => row.map((x, j) => differs(x, baseline.concat[i][j]))),
      output: result.output.map((row, i) => row.map((x, j) => differs(x, baseline.output[i][j]))),
    }),
    [result, baseline],
  );
  // 方式一 check: each head's own Wq_h (a column block of Wq) reproduces the slice of Q.
  const methodsAgree = useMemo(
    () =>
      result.heads.every((head, h) =>
        matmul(MH_X, columnBlock(MH_WEIGHTS.wq, h * dKey, (h + 1) * dKey)).every((row, i) =>
          row.every((x, j) => !differs(x, head.q[i][j])),
        ),
      ),
    [result, dKey],
  );

  const heads = result.heads.map((_, h) => h);
  const owner = Math.floor(KEY_TARGET.column / dKey);
  const update = (patch: Partial<ExperimentState['heads']>) =>
    onState({ ...state, heads: { ...state.heads, ...patch } });
  const headTag = (h: number) => (
    <span className="mh-head-tag" style={{ borderColor: HEAD_COLORS[h], color: HEAD_COLORS[h] }}>
      h{h + 1}
    </span>
  );
  const changedHeads = heads.filter((h) =>
    changed.probabilities[h].some((row) => row.some(Boolean)),
  );
  const changedA = [0, 1, 2, 3].filter((j) => changed.concat.some((row) => row[j]));
  const changedOut = [0, 1, 2, 3].filter((j) => changed.output.some((row) => row[j]));

  const explanation = [
    T(
      'X 的形状是 [3, 4]：3 个 token，每个 4 维（$d_{\\mathrm{model}} = 4$）。和第 10 章一样，X 分别乘 $W_Q$、$W_K$、$W_V$，得到同形状的 Q、K、V。书中是 [4, 16, 512] @ [512, 512]；这里去掉 batch，缩小到可以手算。此时还没有“头”。',
    ),
    T(
      '把 Q 的最后一维 4 平均切成 {#0} 份，每份 $d_{\\mathrm{key}} = 4 \\div {#1} = {#2}$ 维，就像书中在向量上“砍 N 刀”。数值一个也没改，只是规定第几段属于第几个头。K、V 同样切分。',
      numHeads,
      numHeads,
      dKey,
    ),
    T(
      '转置把形状从 [3, {#0}, {#1}] 变成 [{#2}, 3, {#3}]。现在每个头拿到一张完整的 [3, {#4}] 小表：3 个 token 都在，每个 token 只保留属于这个头的 {#5} 维。',
      numHeads,
      dKey,
      numHeads,
      dKey,
      dKey,
      dKey,
    ),
    T(
      '每个头拿自己的 $Q_h$、$K_h$、$V_h$，独立走一遍第 10 章的流程：打分、除以 $\\sqrt{d_{\\mathrm{key}}}$（这里是 $\\sqrt{{#0}}$）、{#1}Softmax，再乘 $V_h$ 得到 $A_h$。这时头与头之间不交换任何数字。',
      dKey,
      causal ? T('Mask、') : '',
    ),
    T(
      '把 {#0} 个 $A_h$ 按头的顺序并排放回：先转置回 [3, {#1}, {#2}]，再 view 成 [3, 4]。得到的 A 与 X 同形状。拼接只是“摆在一起”，还没有融合。',
      numHeads,
      numHeads,
      dKey,
    ),
    T(
      '最后乘 [4, 4] 的 $W_O$。$W_O$ 的每一列都与 A 的整行做点积，所以 Output 的每个数都同时用到所有头的结果，这就是书中说的“融合不同头的信息”。之后结果进入残差相加。',
    ),
  ];
  const bookShapes: [string, string, string][] = [
    [T('投影'), '[3,4] @ [4,4] = [3,4]', '[4,16,512] @ [512,512] = [4,16,512]'],
    [T('切分多头'), `[3,4] → [3,${numHeads},${dKey}]`, '[4,16,512] → [4,16,4,128]'],
    [T('矩阵变形'), `→ [${numHeads},3,${dKey}]`, '→ [4,4,16,128]'],
    [
      T('每头 Attention'),
      T('P [{#0},3,3]，A [{#1},3,{#2}]', numHeads, numHeads, dKey),
      T('[4,4,16,16]，A [4,4,16,128]'),
    ],
    ['Concatenate', `→ [3,${numHeads},${dKey}] → [3,4]`, '→ [4,16,4,128] → [4,16,512]'],
    ['$W_O$', '[3,4] @ [4,4] = [3,4]', '[4,16,512] @ [512,512] = [4,16,512]'],
  ];

  function verify() {
    onState({ ...state, heads: { ...state.heads, keyDelta: draft, step: 5 } });
    setVerified(true);
  }

  return (
    <>
      <section className="lesson-panel mh-panel" aria-label={T('多头注意力：切分与合并')}>
        <div className="lesson-panel-heading">
          <div>
            <span className="lesson-eyebrow">{T('原书第 11–12 章 · Multi-Head Attention')}</span>
            <h2>{T('把一个 Attention 切成几个头，各算各的，再拼回去。')}</h2>
          </div>
          <span className="lesson-progress">
            {step + 1} / {HEAD_STEP_COUNT}
          </span>
        </div>
        <div className="mh-settings">
          <div className="mh-segment" role="group" aria-label={T('头数 num_heads')}>
            <span>{T('头数 num_heads')}</span>
            {HEAD_OPTIONS.map((option) => (
              <button
                key={option}
                aria-pressed={numHeads === option}
                onClick={() => {
                  setVerified(false);
                  update({ numHeads: option as HeadCount });
                }}
              >
                {option}
              </button>
            ))}
          </div>
          <span className="mh-dkey">
            <Tex tex={`d_{\\mathrm{key}} = 4 \\div ${numHeads} =`} /> <strong>{dKey}</strong>
          </span>
          <label className="lesson-switch mh-causal">
            <input
              type="checkbox"
              checked={causal}
              onChange={(e) => {
                setVerified(false);
                onState({ ...state, config: { ...state.config, causal: e.target.checked } });
              }}
            />
            {T('因果 Mask')}
          </label>
        </div>
        <div className="lesson-steps mh-steps" aria-label={T('多头步骤')}>
          {stepNames.map((name, i) => (
            <button
              key={name}
              aria-current={step === i ? 'step' : undefined}
              onClick={() => onJump(i)}
            >
              <span>{i + 1}</span>
              <Rich text={name} />
            </button>
          ))}
        </div>
        <div className="lesson-columns">
          <div className="diagram-panel mh-diagram-panel">
            <div className="diagram-caption">
              <strong>{T('同一份数据，换一种摆法')}</strong>
              <span>{reducedMotion ? T('已减少动态效果') : T('切换步骤时观察格子移动')}</span>
            </div>
            <MultiHeadDiagram
              result={result}
              step={step}
              numHeads={numHeads}
              query={query}
              changed={changed}
              showValues
            />
            <p className="mh-legend">
              {Tx(
                '{#0} 正在观察的 token 行{#1} 干预后变化的格子{#2} 被 Mask 的位置',
                <span className="mh-legend-row" />,
                <span className="mh-legend-changed" />,
                <span className="mh-legend-mask" />,
              )}
            </p>
            {(step === 1 || step === 2 || step === 4) && (
              <figure className="ht-panel">
                <figcaption className="ht-title">
                  {Tx('立体视图：{#0} 的三条轴（token、head、d_key）', step === 4 ? 'A' : 'Q')}
                </figcaption>
                <HeadTensor
                  matrix={step === 4 ? result.concat : result.q}
                  numHeads={numHeads}
                  by={step === 2 ? 'head' : 'token'}
                  query={query}
                  name={step === 4 ? 'A' : 'Q'}
                  tone={step === 4 ? 'output' : 'query'}
                />
                <p className="ht-caption">
                  {numHeads === 1
                    ? T(
                        '只有 1 个头：head 轴长度为 1，切分、转置和拼接都不改变任何排列。换成 2 个或 4 个头再看。',
                      )
                    : step === 1
                      ? T(
                          '立体地看 [3, {#0}, {#1}]：每个 token 的 4 个数按顺序切成 {#2} 段，沿 head 轴往后排开。虚线带是按第一条轴取出的一片 Q[t]，形状 [{#3}, {#4}]。数值一个也没变，K、V 同样处理。',
                          numHeads,
                          dKey,
                          numHeads,
                          numHeads,
                          dKey,
                        )
                      : step === 2
                        ? T(
                            '和上一步比，格子一个也没动：转置只把“先按 token 取”改成“先按 head 取”。彩色框是一片 Q_h，形状 [3, {#0}]：3 个 token 都在，第 t 行仍是同一个 token。',
                            dKey,
                          )
                        : T(
                            '反过来走一遍：各头的输出 A_h 沿 head 轴摆着，转置回“先按 token 取”（虚线带，每片 [{#0}, {#1}]），再把每片按顺序展平成 4 个数，就是拼接后的 A。',
                            numHeads,
                            dKey,
                          )}
                </p>
              </figure>
            )}
            <table className="mh-shapes" aria-label={T('本例与原书的形状对照')}>
              <thead>
                <tr>
                  <th>{T('步骤')}</th>
                  <th>{T('本例（无 batch）')}</th>
                  <th>{T('原书（batch = 4）')}</th>
                </tr>
              </thead>
              <tbody>
                {bookShapes.map(([name, mine, theirs], i) => (
                  <tr key={name} className={i === step ? 'current' : ''}>
                    <th scope="row">
                      <Rich text={name} />
                    </th>
                    <td>{mine}</td>
                    <td>{theirs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="lesson-reading">
            <div className="lesson-reading-top">
              <span>{T('跟着一个 token 看')}</span>
              <button
                onClick={() =>
                  onState({
                    ...state,
                    view: { ...state.view, explanations: !state.view.explanations },
                  })
                }
              >
                {hide ? T('显示说明') : T('隐藏说明')}
              </button>
            </div>
            <div className="lesson-tokens" role="group" aria-label={T('选择观察的 token')}>
              {words.map((word, i) => (
                <button key={word} aria-pressed={query === i} onClick={() => update({ query: i })}>
                  <small>t{i}</small>
                  <strong>{word}</strong>
                  {query === i && <span>{T('正在观察')}</span>}
                </button>
              ))}
            </div>
            <p className="lesson-token-note">
              {Tx(
                '固定教学数字，{#0}，不是训练好的模型参数。',
                <Tex tex="d_{\mathrm{model}} = 4" />,
              )}
            </p>
            <h3>
              <Rich text={titles[step]} />
            </h3>
            {hide ? (
              <div className="lesson-recall">
                <strong>{T('先自己说一遍')}</strong>
                <p>
                  <Rich text={prompts[step]} />
                </p>
              </div>
            ) : (
              <p className="lesson-explanation">
                <Rich text={explanation[step]} />
              </p>
            )}

            {step === 0 && (
              <div className="mh-detail">
                <div className="lesson-vector-label">
                  {Tx(
                    'X[{#0}]（{#1}）= {#2}',
                    query,
                    words[query],
                    <code>{vector(MH_X[query])}</code>,
                  )}
                </div>
                <table className="mh-table" aria-label={T('Q 的一行如何算出')}>
                  <thead>
                    <tr>
                      <th>{T('Q 的列')}</th>
                      <th>{Tx('X[{#0}] 逐项乘 {#1} 这一列，再相加', query, <Tex tex="W_Q" />)}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[0, 1, 2, 3].map((d) => (
                      <tr key={d}>
                        <th scope="row">
                          d{d} {numHeads > 1 && headTag(Math.floor(d / dKey))}
                        </th>
                        <td>
                          {MH_X[query]
                            .map((x, i) => `${fmt(x)}×${fmt(MH_WEIGHTS.wq[i][d])}`)
                            .join(' + ')}
                          <strong> = {fmt(result.q[query][d])}</strong>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mh-check" role="status">
                  {Tx(
                    '{#0} 书中“方式一”：每个头单独用 {#1} 的对应列（[4, {#2}]）计算 {#3}，与“方式二”切出的结果{#4}。',
                    methodsAgree ? '✓' : '✗',
                    <Tex tex="W_Q" />,
                    dKey,
                    <Tex tex="Q_h" />,
                    methodsAgree ? T('完全相同') : T('不一致'),
                  )}
                </p>
                <details className="mh-weights">
                  <summary>
                    {Tx(
                      '查看 {#0}、{#1}、{#2}（固定教学参数）',
                      <Tex tex="W_Q" />,
                      <Tex tex="W_K" />,
                      <Tex tex="W_V" />,
                    )}
                  </summary>
                  <div className="mh-weight-grid">
                    <Matrix name="W_Q" data={MH_WEIGHTS.wq} parameter tone="query" />
                    <Matrix name="W_K" data={MH_WEIGHTS.wk} parameter tone="key" />
                    <Matrix name="W_V" data={MH_WEIGHTS.wv} parameter tone="value" />
                  </div>
                </details>
              </div>
            )}

            {step === 1 && (
              <div className="mh-detail">
                <table className="mh-table mh-split" aria-label={T('一行 Q、K、V 的切分')}>
                  <thead>
                    <tr>
                      <th>{Tx('t{#0} 的行', query)}</th>
                      {heads.map((h) => (
                        <th key={h}>{headTag(h)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(['q', 'k', 'v'] as const).map((name) => (
                      <tr key={name}>
                        <th scope="row">{name.toUpperCase()}</th>
                        {heads.map((h) => (
                          <td key={h}>
                            {vector(result[name][query].slice(h * dKey, (h + 1) * dKey))}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mh-note">
                  {Tx(
                    '4 个数按顺序每 {#0} 个一段：第 1 段给 head 1{#1}。与 PyTorch 的{#2} 相同。',
                    dKey,
                    numHeads > 1 ? T('，第 {#0} 段给 head {#1}', numHeads, numHeads) : '',
                    <code> view(T, num_heads, d_key)</code>,
                  )}
                </p>
              </div>
            )}

            {step === 2 && (
              <div className="mh-detail mh-head-matrices">
                {heads.map((h) => (
                  <div key={h} className="mh-head-block" style={{ borderColor: HEAD_COLORS[h] }}>
                    <strong style={{ color: HEAD_COLORS[h] }}>
                      head {h + 1}
                      <small>
                        {Tx(
                          '取原来的 d{#0}{#1}',
                          h * dKey,
                          dKey > 1 ? `–d${(h + 1) * dKey - 1}` : '',
                        )}
                      </small>
                    </strong>
                    <div className="mh-mini-row">
                      <Matrix
                        name={`Q${h + 1}`}
                        data={result.heads[h].q}
                        tone="query"
                        row={query}
                      />
                      <Matrix name={`K${h + 1}`} data={result.heads[h].k} tone="key" row={query} />
                      <Matrix
                        name={`V${h + 1}`}
                        data={result.heads[h].v}
                        tone="value"
                        row={query}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {step === 3 && (
              <div className="mh-detail">
                {heads.map((h) => {
                  const head = result.heads[h];
                  return (
                    <div key={h} className="mh-head-block" style={{ borderColor: HEAD_COLORS[h] }}>
                      <strong style={{ color: HEAD_COLORS[h] }}>
                        head {h + 1}
                        <small>
                          Q{h + 1}[{query}] = {vector(head.q[query])}
                        </small>
                      </strong>
                      <table className="mh-table" aria-label={T('head {#0} 的打分与权重', h + 1)}>
                        <thead>
                          <tr>
                            <th>Key</th>
                            <th>
                              <Tex tex={`Q\\cdot K \\div \\sqrt{${dKey}}`} />
                            </th>
                            <th>{T('权重 P')}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {head.masked[query].map((score, j) => (
                            <tr key={j} className={score === -Infinity ? 'lesson-muted-row' : ''}>
                              <th scope="row">
                                t{j} {words[j]}
                              </th>
                              <td>{score === -Infinity ? T('−∞（未来）') : fmt(score)}</td>
                              <td>
                                <span className="mh-bar">
                                  <span
                                    style={{
                                      width: `${head.probabilities[query][j] * 100}%`,
                                      background: HEAD_COLORS[h],
                                    }}
                                  />
                                </span>
                                {(head.probabilities[query][j] * 100).toFixed(1)}%
                                {changed.probabilities[h][query][j] && (
                                  <em className="mh-changed">{T(' 已变')}</em>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <p className="mh-output-line">
                        A{h + 1}[{query}] = P · V{h + 1} = <code>{vector(head.output[query])}</code>
                      </p>
                    </div>
                  );
                })}
              </div>
            )}

            {step === 4 && (
              <div className="mh-detail">
                <div className="mh-concat" aria-label={T('A[{#0}] 由各头拼接', query)}>
                  {heads.map((h) => (
                    <span key={h} style={{ borderColor: HEAD_COLORS[h] }}>
                      <small style={{ color: HEAD_COLORS[h] }}>
                        A{h + 1}[{query}]
                      </small>
                      {vector(result.heads[h].output[query])}
                    </span>
                  ))}
                </div>
                <p className="mh-output-line">
                  {Tx(
                    'A[{#0}] = {#1}（{#2} × {#3} = 4 维）',
                    query,
                    <code>{vector(result.concat[query])}</code>,
                    numHeads,
                    dKey,
                  )}
                </p>
                <Matrix name="A" data={result.concat} tone="probability" row={query} />
              </div>
            )}

            {step === 5 && (
              <div className="mh-detail">
                <table className="mh-table mh-mix" aria-label={T('Output 每一列由各头贡献相加')}>
                  <thead>
                    <tr>
                      <th>{T('来源')}</th>
                      {[0, 1, 2, 3].map((j) => (
                        <th key={j}>Output d{j}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {heads.map((h) => (
                      <tr key={h}>
                        <th scope="row">
                          {Tx(
                            '{#0} · {#1} 行 {#2}{#3}',
                            numHeads > 1 ? headTag(h) : 'A',
                            <Tex tex="W_O" />,
                            h * dKey,
                            dKey > 1 ? `–${(h + 1) * dKey - 1}` : '',
                          )}
                        </th>
                        {[0, 1, 2, 3].map((j) => {
                          let part = 0;
                          for (let d = 0; d < dKey; d++)
                            part +=
                              result.heads[h].output[query][d] * MH_WEIGHTS.wo[h * dKey + d][j];
                          return <td key={j}>{fmt(part)}</td>;
                        })}
                      </tr>
                    ))}
                    <tr className="mh-total">
                      <th scope="row">{T('合计')}</th>
                      {result.output[query].map((value, j) => (
                        <td key={j}>
                          {fmt(value)}
                          {changed.output[query][j] && <em className="mh-changed">{T(' 已变')}</em>}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
                <p className="mh-note">
                  {Tx(
                    '参数量：{#0}、{#1}、{#2}、{#3} 各 [4, 4]，共 {#4} 个权重（忽略偏置）。按同一公式，书中 GPT-2 Small 的 {#5}，每层 {#6} 个。',
                    <Tex tex="W_Q" />,
                    <Tex tex="W_K" />,
                    <Tex tex="W_V" />,
                    <Tex tex="W_O" />,
                    <Tex tex={`4 \\times 4^2 = ${attentionWeightCount(4)}`} />,
                    <Tex tex="d_{\mathrm{model}} = 768" />,
                    <Tex
                      tex={`4 \\times 768^2 = ${attentionWeightCount(768).toLocaleString('en-US')}`}
                    />,
                  )}
                </p>
              </div>
            )}

            <div className="lesson-takeaway">
              <strong>{T('这一步，记住一句话')}</strong>
              <p>
                <Rich text={hide ? T('先自己复述，再点击“显示说明”核对。') : takeaways[step]} />
              </p>
            </div>
            <details className="lesson-formula">
              <summary>{T('公式与符号说明')}</summary>
              <Formula expression={formulas[step]} symbols={multiheadSymbols[step]} />
              {step === 3 && (
                <p>
                  {Tx(
                    '打分的乘加次数：{#0} 个头 × 3 × 3 × {#1} = {#2}，与头数无关，总是 {#3}（书中第 11.9 节）。',
                    numHeads,
                    dKey,
                    scoreMultiplyAdds(3, 4, numHeads),
                    <Tex tex="T^2 \times d_{\mathrm{model}}" />,
                  )}
                </p>
              )}
            </details>
            {step === 3 && (
              <p className="mh-caution">
                {T(
                  '书中第 11.8 节列举的“语法头”“指代头”等，是研究者在训练好的模型里观察到的模式，头的分工是学出来的，不是预先指定的。本例两个头的差异只来自不同的权重列。',
                )}
              </p>
            )}
            {step === 5 && (
              <p className="mh-caution">
                {Tx(
                  '与第 12 章对照：前向计算只是产生新的表示 Output，交给残差相加和后续层；嵌入表与 {#0}、{#1}、{#2}、{#3} 只在训练时由反向传播和优化器更新，推理时保持不变。',
                  <Tex tex="W_Q" />,
                  <Tex tex="W_K" />,
                  <Tex tex="W_V" />,
                  <Tex tex="W_O" />,
                )}
              </p>
            )}
            <a
              className="lesson-source"
              href={chapterUrl(chapterLinks[step][1])}
              target="_blank"
              rel="noreferrer"
            >
              {Tx('对应原书：{#0} ↗', chapterLinks[step][0])}
            </a>
          </div>
        </div>
        <div className="lesson-controls">
          <div>
            <button
              className="button secondary"
              disabled={step === 0}
              onClick={() => onJump(step - 1)}
            >
              {T('← 上一步')}
            </button>
            <button
              className="button primary"
              disabled={step === HEAD_STEP_COUNT - 1}
              onClick={() => onJump(step + 1)}
              aria-label={
                step === HEAD_STEP_COUNT - 1
                  ? T('已到最后一步')
                  : T('下一步：{#0} →', plain(stepNames[step + 1]).replace('W_O', 'Wo'))
              }
            >
              {step === HEAD_STEP_COUNT - 1 ? (
                T('已到最后一步')
              ) : (
                <Rich text={T('下一步：{#0} →', stepNames[step + 1])} />
              )}
            </button>
          </div>
          <div>
            <button className="lesson-text-button" onClick={onOpenSingleHead}>
              {T('回看单头')}
            </button>
            <button className="lesson-text-button" onClick={() => onJump(0)}>
              {T('从头看')}
            </button>
            <button
              className="lesson-text-button"
              onClick={() => {
                setVerified(false);
                setPrediction(null);
                setDraft(-2);
                onState({ ...state, heads: { step: 0, numHeads: 2, query: 2, keyDelta: 0 } });
              }}
            >
              {T('恢复默认示例')}
            </button>
            <button
              className="lesson-text-button"
              aria-label={playing ? T('暂停多头演示') : T('播放多头演示')}
              onClick={onTogglePlay}
            >
              {playing ? T('Ⅱ 暂停') : T('▶ 自动逐步')}
            </button>
            <label>
              {T('速度')}
              <select
                aria-label={T('多头演示速度')}
                value={state.view.speed}
                onChange={(e) =>
                  onState({ ...state, view: { ...state.view, speed: Number(e.target.value) } })
                }
              >
                <option value="0.5">0.5×</option>
                <option value="1">1×</option>
                <option value="2">2×</option>
              </select>
            </label>
          </div>
        </div>
        {reducedMotion && (
          <p className="lesson-motion">
            {T('已遵循减少动态效果偏好；格子直接切换到新位置，不做移动动画。')}
          </p>
        )}
      </section>

      <section className="experiment-card mh-experiment" aria-label={T('多头干预实验')}>
        <div className="card-heading">
          <span className="section-kicker">{T('动手验证')}</span>
          <span className="pill">{T('固定 Q、V 与权重')}</span>
        </div>
        <h2>{T('只改一个头的 K，哪些结果会变？')}</h2>
        {numHeads === 1 ? (
          <p>{T('只有一个头时，所有维度都属于同一个头。切换到 2 或 4 个头再做这个实验。')}</p>
        ) : (
          <>
            <p>
              {Tx(
                '在投影后给 K[t0, d3] 加上 ΔK。d3 属于 {#0}（第 {#1}–{#2} 维）。先预测，再执行。',
                <strong>head {owner + 1}</strong>,
                owner * dKey,
                (owner + 1) * dKey - 1,
              )}
            </p>
            <div className="range-heading">
              <label htmlFor="head-key-delta">{T('K[t0, d3] 的偏移 ΔK')}</label>
              <output htmlFor="head-key-delta">{fmt(draft)}</output>
            </div>
            <input
              id="head-key-delta"
              type="range"
              min="-4"
              max="4"
              step="0.5"
              value={draft}
              onChange={(e) => {
                setDraft(Number(e.target.value));
                setVerified(false);
              }}
            />
            <div className="mh-predictions" role="group" aria-label={T('预测哪些结果会变化')}>
              {(
                [
                  ['all', T('所有头的 P 都会变')],
                  ['local', T('只有 head {#0} 变，Output 也只变它那几列', owner + 1)],
                  ['mixed', T('只有 head {#0} 的 P 和 A 变，但 Output 每一列都可能变', owner + 1)],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  aria-pressed={prediction === id}
                  onClick={() => {
                    setPrediction(id);
                    setVerified(false);
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
            <button
              className="button primary"
              disabled={!prediction || draft === 0}
              onClick={verify}
            >
              {T('验证预测 →')}
            </button>
            <div className="prediction-feedback" role="status">
              {verified ? (
                <>
                  {Tx(
                    '{#0}P 发生变化的头：{#1}； A 变化的列：{#2}；Output 变化的列：{#3}。{#4}',
                    prediction === 'mixed' ? T('预测正确。') : T('再对照一次：'),
                    changedHeads.length
                      ? changedHeads.map((h) => `head ${h + 1}`).join(T('、'))
                      : T('无'),
                    changedA.length ? changedA.map((j) => `d${j}`).join(T('、')) : T('无'),
                    changedOut.length ? changedOut.map((j) => `d${j}`).join(T('、')) : T('无'),
                    changedHeads.length ? (
                      <>
                        {Tx(
                          ' 各头在 Concatenate 之前互不影响；{#0} 的每一列都读取 A 的全部列，所以变化扩散到 Output 的每一列。t0 只能看自己，它那一行不受影响。',
                          <Tex tex="W_O" />,
                        )}
                      </>
                    ) : (
                      T(' 当前设置下被改的分数没有影响任何可见权重。')
                    ),
                  )}
                </>
              ) : draft === 0 ? (
                T('ΔK 为 0 时没有可验证的变化。')
              ) : draft !== keyDelta ? (
                T('有待验证的改动；上方仍显示上一次执行的结果。')
              ) : (
                T('选择预测后点击验证。偏移限制在 [−4, 4]。')
              )}
            </div>
            {keyDelta !== 0 && (
              <button
                className="text-button"
                onClick={() => {
                  update({ keyDelta: 0 });
                  setVerified(false);
                }}
              >
                {T('撤销干预（恢复 ΔK = 0）')}
              </button>
            )}
          </>
        )}
      </section>

      <section className="recall-card">
        <div>
          <span className="section-kicker">{T('离开之前 · 试着复述')}</span>
          <h2>{T('不看图，讲清楚多头的五个动作。')}</h2>
          <p>
            {Tx(
              '切分切的是哪一维？为什么要转置？每个头除以多少？Concatenate 与 {#0} 各做什么？',
              <Tex tex="W_O" />,
            )}
          </p>
        </div>
        <button
          className="button secondary"
          aria-expanded={recallOpen}
          onClick={() => setRecallOpen((v) => !v)}
        >
          {recallOpen ? T('收起参考解释') : T('查看参考解释')}
        </button>
        {recallOpen && (
          <p className="recall-answer">
            {Tx(
              '先用 {#0}、{#1}、{#2} 一次投影得到 {#3} 的 Q、K、V；把最后一维切成 {#4} 份（{#5}），再转置成 {#6}，让每个头拿到全部 token 的一段特征。每个头独立做缩放点积 Attention（除以 {#7}）得到 {#8}；把各头的 {#9} 转置回去并拼接成 {#10}，最后乘 {#11}，让每个输出维度混合所有头的信息。',
              <Tex tex="W_Q" />,
              <Tex tex="W_K" />,
              <Tex tex="W_V" />,
              <Tex tex="[T,\,d_{\mathrm{model}}]" />,
              <Tex tex="\mathrm{num\_heads}" />,
              <Tex tex="d_{\mathrm{key}} = d_{\mathrm{model}} \div \mathrm{num\_heads}" />,
              <Tex tex="[\mathrm{num\_heads},\,T,\,d_{\mathrm{key}}]" />,
              <Tex tex="\sqrt{d_{\mathrm{key}}}" />,
              <Tex tex="A_h" />,
              <Tex tex="A_h" />,
              <Tex tex="[T,\,d_{\mathrm{model}}]" />,
              <Tex tex="W_O" />,
            )}
          </p>
        )}
      </section>
    </>
  );
}
