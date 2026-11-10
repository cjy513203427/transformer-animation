import { chapterUrl } from '../../content/book';
import { useMemo } from 'react';
import { Formula } from '../../components/Formula';
import { Rich, Tex } from '../../components/Tex';
import { steps } from '../../content/attention';
import { explainRow, formatNumber as fmt, type computeExperiment } from '../../math/attention';
import type { ExperimentState } from '../../state/experiment';
import { AttentionDiagram } from './AttentionDiagram';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/lesson.css';

const words = [T('我'), T('喜欢'), T('学习')];
const vector = (values: number[]) => `[${values.map((value) => fmt(value)).join(', ')}]`;
const titles = [
  T('同一个位置，分出三种用途'),
  T('用 Q 和每个 K 算一个匹配分数'),
  T('把不能看的位置挡住'),
  T('把分数换成“各取多少”'),
  T('按比例取出 V，再加起来'),
];
const chapterLinks = [
  [T('第 10 章 · QKV 的三个角色'), '第10章-QKV到底是什么-Attention的三个主角/'],
  [T('第 9 章 · 点积、缩放与注意力'), '第09章-Attention的几何逻辑-为什么是点积/'],
  [T('第 9 章 · 因果遮罩'), '第09章-Attention的几何逻辑-为什么是点积/'],
  [T('第 6 章 · Softmax'), '第06章-LayerNorm与Softmax-数字的缩放与概率化/'],
  [T('第 12 章 · QKV 输出的本质'), '第12章-QKV输出的本质/'],
];

interface Props {
  state: ExperimentState;
  result: ReturnType<typeof computeExperiment>;
  onState: (state: ExperimentState) => void;
  onJump: (step: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  onOpenNumbers: () => void;
  onOpenHeads: () => void;
  reducedMotion: boolean;
}

export function AttentionLesson({
  state,
  result,
  onState,
  onJump,
  playing,
  onTogglePlay,
  onOpenNumbers,
  onOpenHeads,
  reducedMotion,
}: Props) {
  const { query } = state.view;
  const trace = useMemo(() => explainRow(result, query), [result, query]);
  const visible = trace.keys.flatMap((key, j) => (key.maskedScore === -Infinity ? [] : [words[j]]));
  const masked = trace.keys.flatMap((key, j) => (key.maskedScore === -Infinity ? [words[j]] : []));
  const leading = trace.keys.reduce(
    (best, key, i) => (key.probability > trace.keys[best].probability ? i : best),
    0,
  );
  const updateQuery = (query: number) => onState({ ...state, view: { ...state.view, query } });
  const hide = !state.view.explanations;
  const explanations = [
    T(
      '先看“{#0}”这一行。它当前的表示是 X[{#1}] = {#2}。同一行分别乘三个参数矩阵，得到三条不同用途的向量；其他位置也做同样的事。',
      words[query],
      query,
      vector(state.input.x[query]),
    ),
    T(
      '现在把“{#0}”作为读取信息的一方。拿它的 Q，分别和“我”“喜欢”“学习”的 K 做点积。每次把对应维度相乘，再相加。{#1}',
      words[query],
      state.config.scaled
        ? T('最后除以 $\\sqrt{2}$，得到缩放后的分数 S。')
        : T('当前关闭缩放，直接用点积作分数；这是对照设置。'),
    ),
    state.config.causal
      ? T(
          '我们正在算 t{#0}“{#1}”。它可以看{#2}。{#3}',
          query,
          words[query],
          visible.map((w) => `“${w}”`).join(T('、')),
          masked.length
            ? T('{#0}在它后面，所以把分数改为 −∞。', masked.map((w) => `“${w}”`).join(T('、')))
            : T('它已经是最后一个位置，因此这一行没有未来 token 要屏蔽。'),
        )
      : T(
          '当前是双向读取对照。三个位置都可见，不屏蔽未来 token；这不符合生成式 Decoder 的因果约束。',
        ),
    T(
      'Softmax 把这一行可见的分数转成取用比例。对“{#0}”而言，{#1}的权重最高，为 {#2}%。权重之和为 100%，屏蔽位置为 0%。',
      words[query],
      words[leading],
      (trace.keys[leading].probability * 100).toFixed(1),
    ),
    T(
      '现在才用到 V。每个位置的 V 乘自己的权重，然后逐维相加。这给“{#0}”生成了一个含上下文信息的向量 O[{#1}] = {#2}。',
      words[query],
      query,
      vector(result.output[query]),
    ),
  ];
  return (
    <section className="lesson-panel" aria-label={T('跟图理解 Attention')}>
      <div className="lesson-panel-heading">
        <div>
          <span className="lesson-eyebrow">{T('Block → MHA → 一个头')}</span>
          <h2>{T('沿着两条支路，看懂信息怎样汇合。')}</h2>
        </div>
        <span className="lesson-progress">{state.step + 1} / 5</span>
      </div>
      <div className="lesson-steps" aria-label={T('导读步骤')}>
        {steps.map((step, i) => (
          <button
            key={step.title}
            aria-current={state.step === i ? 'step' : undefined}
            onClick={() => onJump(i)}
          >
            <span>{i + 1}</span>
            {[T('生成 QKV'), T('打分与缩放'), T('屏蔽未来'), T('分配比例'), T('汇总内容')][i]}
          </button>
        ))}
      </div>
      <div className="lesson-columns">
        <AttentionDiagram
          step={state.step}
          onStep={onJump}
          causal={state.config.causal}
          scaled={state.config.scaled}
          onOpenHeads={onOpenHeads}
        />
        <div className="lesson-reading">
          <div className="lesson-reading-top">
            <span>{T('点选一个词，切换观察位置')}</span>
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
          <div className="lesson-tokens" role="group" aria-label={T('选择观察位置')}>
            {words.map((word, i) => (
              <button key={word} aria-pressed={query === i} onClick={() => updateQuery(i)}>
                <small>t{i}</small>
                <strong>{word}</strong>
                {query === i && <span>{T('正在观察')}</span>}
              </button>
            ))}
          </div>
          <p className="lesson-token-note">
            {T('人为切分的词块，只用于对应位置；数值是二维教学例子。')}
          </p>
          <h3>{titles[state.step]}</h3>
          {hide ? (
            <div className="lesson-recall">
              <strong>{T('请指着图解释这一箭头')}</strong>
              <p>{T('哪条向量进来？做了什么？产生什么？V 有没有经过这一步？')}</p>
            </div>
          ) : (
            <p className="lesson-explanation">
              <Rich text={explanations[state.step]} />
            </p>
          )}

          {state.step === 0 && (
            <div className="lesson-roles">
              {(
                [
                  ['Q', T('要找什么'), T('用来发起匹配'), result.q[query], 'query'],
                  ['K', T('用什么来匹配'), T('供其他位置打分'), result.k[query], 'key'],
                  ['V', T('实际取出的内容'), T('最后参与加权汇总'), result.v[query], 'value'],
                ] as const
              ).map(([symbol, role, detail, values, tone]) => (
                <div className={`lesson-role tone-${tone}`} key={symbol}>
                  <strong>
                    {symbol}
                    <span>{role}</span>
                  </strong>
                  <code>{vector(values)}</code>
                  <small>{detail}</small>
                </div>
              ))}
              <p>
                {Tx(
                  '这里展示同一个 token 的三种表示；打分时，当前 Q 会去配{#0}的 K。',
                  <strong>{T('各个位置')}</strong>,
                )}
              </p>
              {(state.input.keyDelta !== 0 || state.input.valueDelta !== 0) && (
                <p className="lesson-intervention">
                  {Tx(
                    '当前还有投影后干预：K[1,0] += {#0}，V[1,0] += {#1}。',
                    fmt(state.input.keyDelta),
                    fmt(state.input.valueDelta),
                  )}
                </p>
              )}
            </div>
          )}

          {state.step === 1 && (
            <div className="lesson-score-table">
              <div className="lesson-vector-label">
                {Tx('当前 Q[{#0}] = {#1}', query, <code>{vector(result.q[query])}</code>)}
              </div>
              <table aria-label={T('逐位置点积')}>
                <thead>
                  <tr>
                    <th>{T('与谁匹配')}</th>
                    <th>{T('对应相乘再相加')}</th>
                    <th>
                      {state.config.scaled ? (
                        <>{Tx('÷ {#0} 后', <Tex tex="\sqrt{2}" />)}</>
                      ) : (
                        T('未缩放 S')
                      )}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {trace.keys.map((key, j) => (
                    <tr key={j}>
                      <th scope="row">
                        {words[j]}
                        <small>
                          K[{j}] {vector(result.k[j])}
                        </small>
                      </th>
                      <td>
                        {result.q[query]
                          .map((q, d) => `(${fmt(q)})×(${fmt(result.k[j][d])})`)
                          .join(' + ')}
                        <strong>= {fmt(key.rawScore)}</strong>
                      </td>
                      <td>{fmt(key.score)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>
                {T('这里先算所有配对；下一步再用 Mask 挡住未来位置。点积同时受方向和模长影响。')}
              </p>
            </div>
          )}

          {state.step === 2 && (
            <div className="lesson-mask">
              <div className="lesson-visibility">
                {trace.keys.map((key, j) => (
                  <div key={j} className={key.maskedScore === -Infinity ? 'blocked' : 'allowed'}>
                    <small>t{j}</small>
                    <strong>{words[j]}</strong>
                    <span>
                      {key.maskedScore === -Infinity ? T('× 未来，不能看') : T('✓ 当前可以看')}
                    </span>
                    <code>
                      {fmt(key.score)} → {fmt(key.maskedScore)}
                    </code>
                  </div>
                ))}
              </div>
              <label className="lesson-switch">
                <input
                  type="checkbox"
                  checked={state.config.causal}
                  onChange={(e) =>
                    onState({ ...state, config: { ...state.config, causal: e.target.checked } })
                  }
                />
                {T('开启因果 Mask')}
              </label>
              <p>
                {T(
                  '试试观察“喜欢”，再切换 Mask：未来的“学习”会被挡住。切到最后的“学习”，则没有未来位置。',
                )}
              </p>
            </div>
          )}

          {state.step === 3 && (
            <div className="lesson-proportions">
              {trace.keys.map((key, j) => (
                <div className="lesson-proportion" key={j}>
                  <span>{words[j]}</span>
                  <div className="lesson-bar-track">
                    <div style={{ width: `${key.probability * 100}%` }} />
                  </div>
                  <strong>{(key.probability * 100).toFixed(1)}%</strong>
                  {key.maskedScore === -Infinity && <small>{T('已屏蔽')}</small>}
                </div>
              ))}
              <div className="lesson-total">
                {Tx(
                  '合计 {#0}% {#1}',
                  (trace.probabilitySum * 100).toFixed(0),
                  <span>{T('这一行 P 的和为 1')}</span>,
                )}
              </div>
              <p>
                {T(
                  '这里回答“从哪个位置取多少信息”，不是在预测下一个词。模型最上方的词表 Softmax 是另一处计算。',
                )}
              </p>
              <details>
                <summary>{T('展开这行 Softmax 的数字')}</summary>
                <div className="lesson-softmax-numbers">
                  {trace.keys.map((key, j) => (
                    <p key={j}>
                      {Tx(
                        '{#0}：exp({#1} − {#2}) = {#3}；再除以 {#4} → {#5}',
                        words[j],
                        fmt(key.maskedScore),
                        fmt(trace.maximum),
                        fmt(key.exponential),
                        fmt(trace.denominator),
                        fmt(key.probability),
                      )}
                    </p>
                  ))}
                </div>
              </details>
            </div>
          )}

          {state.step === 4 && (
            <div className="lesson-merge">
              <table aria-label={T('Value 加权贡献')}>
                <thead>
                  <tr>
                    <th>{T('来源')}</th>
                    <th>{T('比例 × 内容 V')}</th>
                    <th>{T('贡献向量')}</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.keys.map((key, j) => (
                    <tr key={j} className={key.probability === 0 ? 'lesson-muted-row' : ''}>
                      <th scope="row">{words[j]}</th>
                      <td>
                        {fmt(key.probability)} × {vector(result.v[j])}
                      </td>
                      <td>{vector(key.contribution)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="lesson-result">
                <span>{T('逐维相加')}</span>
                <strong>
                  O[{query}] = {vector(result.output[query])}
                </strong>
                <small>{T('一个头的输出，不是词表概率')}</small>
              </div>
              <p>
                {Tx(
                  '完整 MHA 还要拼接各头输出，再乘 {#0}，才能回到总图中的残差加法。这里的 O 对应书中第 11–12 章 A 的一个头（一片 {#1}）。',
                  <Tex tex="W_O" />,
                  <Tex tex="[T,\,d_{\mathrm{key}}]" />,
                )}
              </p>
            </div>
          )}

          <div className="lesson-takeaway">
            <strong>{T('这一箭头，记住一句话')}</strong>
            <p>
              {hide
                ? T('先自己复述，再点击“显示说明”核对。')
                : [
                    T('同一份表示，生成用于提问、匹配和提供内容的 Q、K、V。'),
                    T('当前 Q 与每个 K 配对打分；V 此时还没有被使用。'),
                    T('先把未来分数设为 −∞，后面的 Softmax 才会给它 0 权重。'),
                    T('一行分数变成一组取用比例；P 是比例，不是内容。'),
                    T('用 P 决定取多少，用 V 提供取什么；合起来得到新向量。'),
                  ][state.step]}
            </p>
          </div>
          <details className="lesson-formula">
            <summary>{T('需要时再看公式与形状')}</summary>
            <Formula
              expression={
                state.step === 1 && !state.config.scaled ? 'S=QK^\\top' : steps[state.step].formula
              }
            />
            <p>
              <Rich text={steps[state.step].shape} />
            </p>
            <p>
              <Rich text={steps[state.step].caution} />
            </p>
          </details>
          <a
            className="lesson-source"
            href={chapterUrl(chapterLinks[state.step][1])}
            target="_blank"
            rel="noreferrer"
          >
            {Tx('对应原书：{#0} ↗', chapterLinks[state.step][0])}
          </a>
          <p className="lesson-precision">
            {T('数值显示时舍入至 3 位小数；计算保留精度，屏幕上的近似值不再用于后续运算。')}
          </p>
        </div>
      </div>
      <div className="lesson-controls">
        <div>
          <button
            className="button secondary"
            disabled={state.step === 0}
            onClick={() => onJump(state.step - 1)}
          >
            {T('← 上一步')}
          </button>
          <button
            className="button primary"
            onClick={state.step === 4 ? onOpenNumbers : () => onJump(state.step + 1)}
          >
            {state.step === 4
              ? T('改动 V，验证理解 →')
              : T(
                  '下一步：{#0} →',
                  ['', T('算匹配分数'), T('挡住未来'), T('分配比例'), T('汇总内容')][
                    state.step + 1
                  ],
                )}
          </button>
        </div>
        <div>
          <button className="lesson-text-button" onClick={() => onJump(0)}>
            {T('从头看')}
          </button>
          <button
            className="lesson-text-button"
            aria-label={playing ? T('暂停导读') : T('播放导读')}
            onClick={onTogglePlay}
          >
            {playing ? T('Ⅱ 暂停导读') : T('▶ 自动逐步')}
          </button>
          <label>
            {T('速度')}
            <select
              aria-label={T('导读速度')}
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
        <p className="lesson-motion">{T('已遵循减少动态效果偏好；仅切换离散计算步骤。')}</p>
      )}
    </section>
  );
}
