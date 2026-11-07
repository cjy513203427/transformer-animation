import { useMemo } from 'react';
import { Formula } from '../../components/Formula';
import { Rich, Tex } from '../../components/Tex';
import { steps } from '../../content/attention';
import { explainRow, formatNumber as fmt, type computeExperiment } from '../../math/attention';
import type { ExperimentState } from '../../state/experiment';
import { AttentionDiagram } from './AttentionDiagram';
import '../../styles/lesson.css';

const words = ['我', '喜欢', '学习'];
const vector = (values: number[]) => `[${values.map((value) => fmt(value)).join(', ')}]`;
const titles = [
  '同一个位置，分出三种用途',
  '用 Q 和每个 K 算一个匹配分数',
  '把不能看的位置挡住',
  '把分数换成“各取多少”',
  '按比例取出 V，再加起来',
];
const chapterLinks = [
  ['第 10 章 · QKV 的三个角色', '第10章-QKV到底是什么-Attention的三个主角/'],
  ['第 9 章 · 点积、缩放与注意力', '第09章-Attention的几何逻辑-为什么是点积/'],
  ['第 9 章 · 因果遮罩', '第09章-Attention的几何逻辑-为什么是点积/'],
  ['第 6 章 · Softmax', '第06章-LayerNorm与Softmax-数字的缩放与概率化/'],
  ['第 12 章 · QKV 输出的本质', '第12章-QKV输出的本质/'],
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
    `先看“${words[query]}”这一行。它当前的表示是 X[${query}] = ${vector(state.input.x[query])}。同一行分别乘三个参数矩阵，得到三条不同用途的向量；其他位置也做同样的事。`,
    `现在把“${words[query]}”作为读取信息的一方。拿它的 Q，分别和“我”“喜欢”“学习”的 K 做点积。每次把对应维度相乘，再相加。${state.config.scaled ? '最后除以 $\\sqrt{2}$，得到缩放后的分数 S。' : '当前关闭缩放，直接用点积作分数；这是对照设置。'}`,
    state.config.causal
      ? `我们正在算 t${query}“${words[query]}”。它可以看${visible.map((w) => `“${w}”`).join('、')}。${masked.length ? `${masked.map((w) => `“${w}”`).join('、')}在它后面，所以把分数改为 −∞。` : '它已经是最后一个位置，因此这一行没有未来 token 要屏蔽。'}`
      : '当前是双向读取对照。三个位置都可见，不屏蔽未来 token；这不符合生成式 Decoder 的因果约束。',
    `Softmax 把这一行可见的分数转成取用比例。对“${words[query]}”而言，${words[leading]}的权重最高，为 ${(trace.keys[leading].probability * 100).toFixed(1)}%。权重之和为 100%，屏蔽位置为 0%。`,
    `现在才用到 V。每个位置的 V 乘自己的权重，然后逐维相加。这给“${words[query]}”生成了一个含上下文信息的向量 O[${query}] = ${vector(result.output[query])}。`,
  ];
  return (
    <section className="lesson-panel" aria-label="跟图理解 Attention">
      <div className="lesson-panel-heading">
        <div>
          <span className="lesson-eyebrow">Block → MHA → 一个头</span>
          <h2>沿着两条支路，看懂信息怎样汇合。</h2>
        </div>
        <span className="lesson-progress">{state.step + 1} / 5</span>
      </div>
      <div className="lesson-steps" aria-label="导读步骤">
        {steps.map((step, i) => (
          <button
            key={step.title}
            aria-current={state.step === i ? 'step' : undefined}
            onClick={() => onJump(i)}
          >
            <span>{i + 1}</span>
            {['生成 QKV', '打分与缩放', '屏蔽未来', '分配比例', '汇总内容'][i]}
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
            <span>点选一个词，切换观察位置</span>
            <button
              onClick={() =>
                onState({
                  ...state,
                  view: { ...state.view, explanations: !state.view.explanations },
                })
              }
            >
              {hide ? '显示说明' : '隐藏说明'}
            </button>
          </div>
          <div className="lesson-tokens" role="group" aria-label="选择观察位置">
            {words.map((word, i) => (
              <button key={word} aria-pressed={query === i} onClick={() => updateQuery(i)}>
                <small>t{i}</small>
                <strong>{word}</strong>
                {query === i && <span>正在观察</span>}
              </button>
            ))}
          </div>
          <p className="lesson-token-note">人为切分的词块，只用于对应位置；数值是二维教学例子。</p>
          <h3>{titles[state.step]}</h3>
          {hide ? (
            <div className="lesson-recall">
              <strong>请指着图解释这一箭头</strong>
              <p>哪条向量进来？做了什么？产生什么？V 有没有经过这一步？</p>
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
                  ['Q', '要找什么', '用来发起匹配', result.q[query], 'query'],
                  ['K', '用什么来匹配', '供其他位置打分', result.k[query], 'key'],
                  ['V', '实际取出的内容', '最后参与加权汇总', result.v[query], 'value'],
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
                这里展示同一个 token 的三种表示；打分时，当前 Q 会去配<strong>各个位置</strong>的
                K。
              </p>
              {(state.input.keyDelta !== 0 || state.input.valueDelta !== 0) && (
                <p className="lesson-intervention">
                  当前还有投影后干预：K[1,0] += {fmt(state.input.keyDelta)}，V[1,0] +={' '}
                  {fmt(state.input.valueDelta)}。
                </p>
              )}
            </div>
          )}

          {state.step === 1 && (
            <div className="lesson-score-table">
              <div className="lesson-vector-label">
                当前 Q[{query}] = <code>{vector(result.q[query])}</code>
              </div>
              <table aria-label="逐位置点积">
                <thead>
                  <tr>
                    <th>与谁匹配</th>
                    <th>对应相乘再相加</th>
                    <th>
                      {state.config.scaled ? (
                        <>
                          ÷ <Tex tex="\sqrt{2}" /> 后
                        </>
                      ) : (
                        '未缩放 S'
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
              <p>这里先算所有配对；下一步再用 Mask 挡住未来位置。点积同时受方向和模长影响。</p>
            </div>
          )}

          {state.step === 2 && (
            <div className="lesson-mask">
              <div className="lesson-visibility">
                {trace.keys.map((key, j) => (
                  <div key={j} className={key.maskedScore === -Infinity ? 'blocked' : 'allowed'}>
                    <small>t{j}</small>
                    <strong>{words[j]}</strong>
                    <span>{key.maskedScore === -Infinity ? '× 未来，不能看' : '✓ 当前可以看'}</span>
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
                开启因果 Mask
              </label>
              <p>
                试试观察“喜欢”，再切换
                Mask：未来的“学习”会被挡住。切到最后的“学习”，则没有未来位置。
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
                  {key.maskedScore === -Infinity && <small>已屏蔽</small>}
                </div>
              ))}
              <div className="lesson-total">
                合计 {(trace.probabilitySum * 100).toFixed(0)}% <span>这一行 P 的和为 1</span>
              </div>
              <p>
                这里回答“从哪个位置取多少信息”，不是在预测下一个词。模型最上方的词表 Softmax
                是另一处计算。
              </p>
              <details>
                <summary>展开这行 Softmax 的数字</summary>
                <div className="lesson-softmax-numbers">
                  {trace.keys.map((key, j) => (
                    <p key={j}>
                      {words[j]}：exp({fmt(key.maskedScore)} − {fmt(trace.maximum)}) ={' '}
                      {fmt(key.exponential)}；再除以 {fmt(trace.denominator)} →{' '}
                      {fmt(key.probability)}
                    </p>
                  ))}
                </div>
              </details>
            </div>
          )}

          {state.step === 4 && (
            <div className="lesson-merge">
              <table aria-label="Value 加权贡献">
                <thead>
                  <tr>
                    <th>来源</th>
                    <th>比例 × 内容 V</th>
                    <th>贡献向量</th>
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
                <span>逐维相加</span>
                <strong>
                  O[{query}] = {vector(result.output[query])}
                </strong>
                <small>一个头的输出，不是词表概率</small>
              </div>
              <p>
                完整 MHA 还要拼接各头输出，再乘 <Tex tex="W_O" />
                ，才能回到总图中的残差加法。这里的 O 对应书中第 11–12 章 A 的一个头（一片{' '}
                <Tex tex="[T,\,d_{\mathrm{key}}]" />
                ）。
              </p>
            </div>
          )}

          <div className="lesson-takeaway">
            <strong>这一箭头，记住一句话</strong>
            <p>
              {hide
                ? '先自己复述，再点击“显示说明”核对。'
                : [
                    '同一份表示，生成用于提问、匹配和提供内容的 Q、K、V。',
                    '当前 Q 与每个 K 配对打分；V 此时还没有被使用。',
                    '先把未来分数设为 −∞，后面的 Softmax 才会给它 0 权重。',
                    '一行分数变成一组取用比例；P 是比例，不是内容。',
                    '用 P 决定取多少，用 V 提供取什么；合起来得到新向量。',
                  ][state.step]}
            </p>
          </div>
          <details className="lesson-formula">
            <summary>需要时再看公式与形状</summary>
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
            href={'https://waylandz.com/llm-transformer-book/' + chapterLinks[state.step][1]}
            target="_blank"
            rel="noreferrer"
          >
            对应原书：{chapterLinks[state.step][0]} ↗
          </a>
          <p className="lesson-precision">
            数值显示时舍入至 3 位小数；计算保留精度，屏幕上的近似值不再用于后续运算。
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
            ← 上一步
          </button>
          <button
            className="button primary"
            onClick={state.step === 4 ? onOpenNumbers : () => onJump(state.step + 1)}
          >
            {state.step === 4
              ? '改动 V，验证理解 →'
              : `下一步：${['', '算匹配分数', '挡住未来', '分配比例', '汇总内容'][state.step + 1]} →`}
          </button>
        </div>
        <div>
          <button className="lesson-text-button" onClick={() => onJump(0)}>
            从头看
          </button>
          <button
            className="lesson-text-button"
            aria-label={playing ? '暂停导读' : '播放导读'}
            onClick={onTogglePlay}
          >
            {playing ? 'Ⅱ 暂停导读' : '▶ 自动逐步'}
          </button>
          <label>
            速度
            <select
              aria-label="导读速度"
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
        <p className="lesson-motion">已遵循减少动态效果偏好；仅切换离散计算步骤。</p>
      )}
    </section>
  );
}
