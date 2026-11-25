import {
  projectionCellSymbols,
  scoreCellSymbols,
  outputCellSymbols,
} from '../../content/notation/attention';
import { useEffect, useMemo, useRef, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { RelatedNotes } from '../../components/RelatedNotes';
import { Formula } from '../../components/Formula';
import { Rich, Tex } from '../../components/Tex';
import { Matrix } from '../../components/Matrix';
import { sources, steps } from '../../content/attention';
import { TransformerOverview } from './TransformerOverview';
import { AttentionLesson } from './AttentionLesson';
import { MultiHeadLesson } from './MultiHeadLesson';
import { GuidedAttention } from './GuidedAttention';
import type { LessonMode } from '../../state/lesson';
import {
  computeExperiment,
  formatNumber as fmt,
  seededInput,
  W_K,
  W_Q,
  W_V,
} from '../../math/attention';
import {
  HEAD_STEP_COUNT,
  moveHeadStep,
  moveStep,
  presetState,
  restoreState,
  serializeState,
  STEP_COUNT,
  type ExperimentState,
  type Preset,
} from '../../state/experiment';
import { localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/editorial.css';
import '../../styles/attention-onboarding.css';

type LearningExperience = 'offer' | 'guided' | 'explore';
const GUIDE_SESSION_KEY = 'attention-guide-dismissed-v1';

function initialExperience(): LearningExperience {
  // A shared experiment always opens at its saved state, without a tour in front of it.
  if (window.location.hash) return 'explore';
  try {
    if (window.sessionStorage.getItem(GUIDE_SESSION_KEY) === '1') return 'explore';
  } catch {
    // The lab works normally when browser storage is unavailable.
  }
  return 'offer';
}

function rememberExploration() {
  try {
    window.sessionStorage.setItem(GUIDE_SESSION_KEY, '1');
  } catch {
    // Remembering the invitation is optional; no experiment data is stored here.
  }
}

function WeightFlow({ probabilities, query }: { probabilities: number[]; query: number }) {
  return (
    <svg
      className="weight-flow"
      viewBox="0 0 460 170"
      role="img"
      aria-label={T('Query t{#0} 对三个 Value 的取用比例', query)}
    >
      {probabilities.map((p, j) => (
        <g key={j}>
          <path
            d={`M110 85 C190 85 205 ${30 + j * 55} 290 ${30 + j * 55}`}
            fill="none"
            stroke={p ? '#9e3c28' : '#a89f92'}
            strokeWidth={p ? 1.5 + p * 9 : 1}
            strokeDasharray={p ? undefined : '4 4'}
            opacity={p ? 0.4 + p * 0.6 : 0.7}
          />
          <rect
            x="290"
            y={14 + j * 55}
            width="60"
            height="32"
            rx="4"
            fill="#ece4f1"
            stroke="#b49dc5"
          />
          <text x="320" y={35 + j * 55} textAnchor="middle" fill="#6b4b82">
            V t{j}
          </text>
          <text x="365" y={35 + j * 55} fill="#9e3c28">
            {(p * 100).toFixed(1)}%
          </text>
        </g>
      ))}
      <rect x="25" y="64" width="85" height="42" rx="4" fill="#e3ebf3" stroke="#9bb2cc" />
      <text x="67" y="90" textAnchor="middle" fill="#2e5785">
        Q t{query}
      </text>
    </svg>
  );
}

export default function AttentionLab() {
  const [restored] = useState(() => restoreState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [experience, setExperience] = useState(initialExperience);
  const guidedRef = useRef<HTMLDivElement>(null);
  const explorationRef = useRef<HTMLDivElement>(null);
  const inputsRef = useRef<HTMLDetailsElement>(null);
  const returnToExperiment = useRef(false);
  const [notice, setNotice] = useState(restored.warning ?? '');
  const [playing, setPlaying] = useState(false);
  const [draftValue, setDraftValue] = useState(state.input.valueDelta);
  const [prediction, setPrediction] = useState<'same' | 'change' | null>(null);
  const [verified, setVerified] = useState(false);
  const [manualLink, setManualLink] = useState('');
  const [recallOpen, setRecallOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const result = useMemo(
    () =>
      computeExperiment(
        state.input.x,
        state.input.keyDelta,
        state.input.valueDelta,
        state.config.causal,
        state.config.scaled,
      ),
    [state.input, state.config],
  );
  const baseline = useMemo(
    () =>
      computeExperiment(
        state.input.x,
        state.input.keyDelta,
        0,
        state.config.causal,
        state.config.scaled,
      ),
    [state.input.x, state.input.keyDelta, state.config],
  );
  const { query, key } = state.view;
  const step = steps[state.step];
  const selectCell = (query: number, key: number) => {
    setPlaying(false);
    setState((s) => ({ ...s, view: { ...s.view, query, key } }));
  };

  function replaceState(next: ExperimentState) {
    setPlaying(false);
    setState(next);
    setDraftValue(next.input.valueDelta);
    setPrediction(null);
    setVerified(false);
    setRecallOpen(false);
    setManualLink('');
  }

  function changeConfig(patch: Partial<ExperimentState['config']>, step: number) {
    setPlaying(false);
    setVerified(false);
    setState((s) => ({ ...s, config: { ...s.config, ...patch }, step }));
  }

  function jump(step: number) {
    setPlaying(false);
    setState((s) => ({ ...s, step }));
  }

  function jumpHead(step: number) {
    setPlaying(false);
    setState((s) => ({ ...s, heads: { ...s.heads, step } }));
  }

  function switchLesson(lesson: LessonMode) {
    setPlaying(false);
    if (experience === 'offer') {
      setExperience('explore');
      rememberExploration();
    }
    setState((s) => ({ ...s, view: { ...s.view, lesson } }));
  }

  function startGuide() {
    if (experience === 'guided') return;
    const example = presetState('basic');
    example.view.query = 0;
    replaceState(example);
    setNotice('');
    setExperience('guided');
  }

  function explore(completed = false) {
    setPlaying(false);
    returnToExperiment.current = experience === 'guided';
    setExperience('explore');
    rememberExploration();
    if (completed) {
      setNotice(T('已保留刚才的输入和计算结果。可以点选 P 的格子，或展开“输入与参数”继续修改 K。'));
    }
  }

  function openInputs() {
    const inputs = inputsRef.current;
    if (!inputs) return;
    inputs.open = true;
    inputs.querySelector<HTMLInputElement>('input')?.focus({ preventScroll: true });
    inputs.scrollIntoView({ block: 'start' });
  }

  useEffect(() => {
    const target =
      experience === 'guided'
        ? guidedRef.current
        : returnToExperiment.current
          ? explorationRef.current
          : null;
    if (!target) return;
    if (!target.contains(document.activeElement)) target.focus({ preventScroll: true });
    target.scrollIntoView({ block: 'start' });
    returnToExperiment.current = false;
  }, [experience]);

  function togglePlay() {
    if (state.view.lesson === 'heads') {
      if (!playing && state.heads.step === HEAD_STEP_COUNT - 1)
        setState((s) => ({ ...s, heads: { ...s.heads, step: 0 } }));
    } else if (state.step === STEP_COUNT - 1) setState((s) => ({ ...s, step: 0 }));
    setPlaying((p) => !p);
  }

  useEffect(() => {
    if (!playing) return;
    const heads = state.view.lesson === 'heads';
    const timer = window.setInterval(
      () => setState((s) => (heads ? moveHeadStep(s, 1) : moveStep(s, 1))),
      (state.view.lesson === 'numbers' ? 1600 : 7000) / state.view.speed,
    );
    return () => window.clearInterval(timer);
  }, [playing, state.view.speed, state.view.lesson]);

  useEffect(() => {
    if (state.step === STEP_COUNT - 1 && state.view.lesson !== 'heads') setPlaying(false);
  }, [state.step, state.view.lesson]);

  useEffect(() => {
    if (state.heads.step === HEAD_STEP_COUNT - 1 && state.view.lesson === 'heads')
      setPlaying(false);
  }, [state.heads.step, state.view.lesson]);

  useEffect(() => {
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    const restore = () => {
      const next = restoreState(window.location.hash);
      returnToExperiment.current = false;
      setExperience('explore');
      replaceState(next.state);
      setNotice(next.warning ?? T('已从链接恢复实验。'));
    };
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onMotion = () => {
      setReducedMotion(media.matches);
      setPlaying(false);
    };
    document.addEventListener('visibilitychange', pause);
    window.addEventListener('hashchange', restore);
    media.addEventListener('change', onMotion);
    return () => {
      document.removeEventListener('visibilitychange', pause);
      window.removeEventListener('hashchange', restore);
      media.removeEventListener('change', onMotion);
    };
  }, []);

  async function copyLink() {
    const url = new URL(localePath('/attention'), window.location.origin);
    url.hash = serializeState(state);
    try {
      await navigator.clipboard.writeText(url.href);
      setManualLink('');
      setNotice(T('已复制当前已计算状态的链接。'));
    } catch {
      setManualLink(url.href);
      setNotice(T('浏览器未允许剪贴板访问，请从下方选择并复制链接。'));
    }
  }

  const scoreExpression = `S_{${query},${key}}=\\frac{${result.q[query].map((q, d) => `(${fmt(q)})\\times(${fmt(result.k[key][d])})`).join('+')}}{${state.config.scaled ? '\\sqrt{2}' : '1'}}\\approx ${fmt(result.scores[query][key])}`;
  const outputExpression = `O_{${query},0}=${result.probabilities[query].map((p, j) => `${fmt(p)}\\times(${fmt(result.v[j][0])})`).join('+')}\\approx ${fmt(result.output[query][0])}`;

  return (
    <AppShell
      active="attention"
      crumb="Attention"
      title={T('Attention：信息怎样在词之间流动')}
      indicator={T('浏览器内计算')}
    >
      <section className="page-heading">
        <div>
          <div className="eyebrow">{Tx('模型原理 {#0} 第 3、9–12 章', <span>{T('／')}</span>)}</div>
          <h1>{Tx('{#0}：信息怎样在词之间流动', <span>Attention</span>)}</h1>
          <p>
            {T(
              '一个词怎样“读”到上下文里的其他词？先看 Attention 在整个模型里的位置，再把它拆成 Q、K、V，用三个词、每个词两个数，把整条计算亲手核对一遍。第一次来的话，点下面的“跟着做一次”。',
            )}
          </p>
        </div>
        <button className="button secondary share-button" onClick={() => void copyLink()}>
          {Tx('{#0} 复制实验链接', <span aria-hidden="true">↗</span>)}
        </button>
      </section>
      {notice && (
        <div className="notice" role="status">
          {notice}
          <button aria-label={T('关闭提示')} onClick={() => setNotice('')}>
            ×
          </button>
        </div>
      )}
      {manualLink && (
        <label className="manual-share">
          {T('手动复制链接')}
          <input readOnly value={manualLink} onFocus={(event) => event.target.select()} />
        </label>
      )}

      <section
        className={`attention-entry ${experience === 'offer' ? 'attention-entry-offer' : ''}`}
        aria-label={T('学习方式')}
      >
        {experience === 'offer' ? (
          <div className="attention-entry-intro">
            <strong>{T('一起试一次：只改变 V，注意力权重会变吗？')}</strong>
            <p>{T('选一个词、做出预测，再拖动数值验证。引导使用基础三词示例，随时可以退出。')}</p>
          </div>
        ) : (
          <span className="attention-entry-label">{T('学习方式')}</span>
        )}
        <div className="attention-entry-actions" role="group" aria-label={T('选择学习方式')}>
          <button
            className={`button ${experience !== 'explore' ? 'primary' : 'secondary'}`}
            aria-pressed={experience === 'guided'}
            onClick={startGuide}
          >
            {T('跟着做一次')}
          </button>
          <button
            className={`button ${experience === 'explore' ? 'primary' : 'secondary'}`}
            aria-pressed={experience === 'explore'}
            onClick={() => explore()}
          >
            {T('自由探索')}
          </button>
        </div>
        {experience === 'explore' && (
          <span className="attention-entry-hint">
            {T('随时可以重走引导；开始时会恢复基础示例。')}
          </span>
        )}
      </section>

      {experience === 'guided' ? (
        <div ref={guidedRef} className="attention-mode-content" tabIndex={-1}>
          <GuidedAttention
            state={state}
            onState={(next) => replaceState(next)}
            onFinish={() => explore(true)}
            onExit={() => explore()}
          />
        </div>
      ) : (
        <div ref={explorationRef} className="attention-mode-content" tabIndex={-1}>
          <nav className="lesson-tabs" aria-label={T('学习层次')}>
            <span className="lesson-tabs-label">{T('本篇内容')}</span>
            {(
              [
                ['overview', T('整体架构')],
                ['guide', T('拆开 Attention')],
                ['numbers', T('动手核对')],
                ['heads', T('多头注意力')],
              ] as const
            ).map(([mode, title]) => (
              <button
                className="lesson-tab"
                key={mode}
                aria-pressed={state.view.lesson === mode}
                onClick={() => switchLesson(mode)}
              >
                {title}
              </button>
            ))}
          </nav>

          {state.view.lesson === 'overview' && (
            <TransformerOverview
              selected={state.view.architectureNode}
              onSelect={(architectureNode) => {
                setPlaying(false);
                setState((s) => ({ ...s, view: { ...s.view, architectureNode } }));
              }}
              onExploreAttention={() => {
                switchLesson('guide');
                setState((s) => ({ ...s, step: 0 }));
              }}
            />
          )}
          {state.view.lesson === 'guide' && (
            <AttentionLesson
              state={state}
              result={result}
              onState={(next) => {
                setPlaying(false);
                setVerified(false);
                setState(next);
              }}
              onJump={jump}
              playing={playing}
              onTogglePlay={togglePlay}
              onOpenNumbers={() => switchLesson('numbers')}
              onOpenHeads={() => switchLesson('heads')}
              reducedMotion={reducedMotion}
            />
          )}
          {state.view.lesson === 'heads' && (
            <MultiHeadLesson
              state={state}
              onState={(next) => {
                setPlaying(false);
                setState(next);
              }}
              onJump={jumpHead}
              playing={playing}
              onTogglePlay={togglePlay}
              reducedMotion={reducedMotion}
              onOpenSingleHead={() => switchLesson('guide')}
            />
          )}

          {state.view.lesson === 'numbers' && (
            <>
              <div className="numbers-context">
                <strong>{T('现在核对图里的单头计算')}</strong>
                <span>
                  {Tx(
                    '当前输出 {#0} 是一个头的结果；切分、拼接与 {#1}在“多头注意力”中逐步演算。',
                    <Tex tex="O = PV" />,
                    <Tex tex="W_O" />,
                  )}
                </span>
                <button onClick={() => switchLesson('guide')}>{T('查看计算图 ↑')}</button>
                <button onClick={() => switchLesson('heads')}>{T('比较多头计算 →')}</button>
                <button onClick={openInputs}>{T('修改 X / K ↓')}</button>
              </div>

              <div className="scene-bar">
                <span className="section-kicker">{T('选择场景')}</span>
                <div className="scene-buttons">
                  {(
                    [
                      ['basic', T('基础计算')],
                      ['value', T('只改变 V')],
                      ['causal', T('因果 Mask')],
                      ['uniform', T('相同分数')],
                      ['bidirectional', T('双向读取对照')],
                    ] as [Preset, string][]
                  ).map(([id, title]) => (
                    <button
                      key={id}
                      onClick={() => {
                        replaceState(presetState(id));
                        setNotice(T('已恢复“{#0}”预设。', title));
                      }}
                    >
                      {title}
                      <span aria-hidden="true">↗</span>
                    </button>
                  ))}
                </div>
              </div>

              <section className="lab-panel" aria-label={T('Attention 逐步实验')}>
                <div className="lab-toolbar">
                  <div>
                    <span className="live-dot" />
                    <strong>{T('一条完整的计算链')}</strong>
                    <span className="subtle">{T('单头 · 3 个 token · 2 维')}</span>
                  </div>
                  <span className="example-tag">{T('教学示例')}</span>
                </div>
                <div className="pipeline" aria-label={T('计算步骤')}>
                  {steps.map((item, i) => (
                    <button
                      key={item.title}
                      aria-current={state.step === i ? 'step' : undefined}
                      className={`pipeline-step ${state.step === i ? 'current' : ''} ${state.step > i ? 'passed' : ''}`}
                      onClick={() => jump(i)}
                    >
                      <span className="step-top">
                        <span className="step-index">0{i + 1}</span>
                        {item.title}
                      </span>
                      <strong>{item.symbol}</strong>
                      {i < steps.length - 1 && (
                        <span className="step-arrow" aria-hidden="true">
                          →
                        </span>
                      )}
                    </button>
                  ))}
                </div>

                <div className="workspace">
                  <div className="calculation-pane">
                    <div className="section-heading">
                      <h2>
                        {state.step === 0
                          ? T('从隐藏状态到 Q、K、V')
                          : state.step === 1
                            ? T('每个 Query 与每个 Key 打分')
                            : state.step === 2
                              ? T('在归一化之前加入 Mask')
                              : state.step === 3
                                ? T('分数变为取用比例')
                                : T('按权重汇总 Value')}
                      </h2>
                      <span className="mono">{state.step + 1} / 5</span>
                    </div>
                    <div className="query-picker">
                      <span>{T('观察 Query')}</span>
                      {[0, 1, 2].map((i) => (
                        <button
                          key={i}
                          aria-pressed={query === i}
                          onClick={() => selectCell(i, key)}
                        >
                          t{i}
                          <span>{[T('我'), T('喜欢'), T('学习')][i]}</span>
                        </button>
                      ))}
                      <small>{T('词语仅作位置标签')}</small>
                    </div>
                    {state.step === 0 ? (
                      <>
                        <div className="matrix-row projections">
                          <Matrix name="Q" data={result.q} tone="query" row={query} />
                          <Matrix name="K" data={result.k} tone="key" row={query} />
                          <Matrix name="V" data={result.v} tone="value" row={query} />
                        </div>
                        <div className="operation-note">
                          {T(
                            '每一行始终属于同一个 token。三组投影参数分别展示在下方“输入与参数”中。',
                          )}
                        </div>
                        {(state.input.keyDelta !== 0 || state.input.valueDelta !== 0) && (
                          <div className="intervention-note">
                            {Tx(
                              '已在投影后干预激活：K[1,0] += {#0}；V[1,0] += {#1}。上方显示干预后的 K/V。',
                              fmt(state.input.keyDelta),
                              fmt(state.input.valueDelta),
                            )}
                          </div>
                        )}
                        <Formula
                          symbols={projectionCellSymbols}
                          expression={`Q_{${query},0}=(${fmt(state.input.x[query][0])})\\times1+(${fmt(state.input.x[query][1])})\\times0=${fmt(result.q[query][0])}`}
                        />
                      </>
                    ) : state.step < 3 ? (
                      <>
                        <div className="score-layout">
                          <Matrix
                            name={state.step === 1 ? 'S' : 'S + M'}
                            data={state.step === 1 ? result.scores : result.masked}
                            tone="key"
                            row={query}
                            column={key}
                            tokenColumns
                            onSelect={selectCell}
                          />
                          <div className="cell-inspector">
                            <span className="section-kicker">{T('选中格子')}</span>
                            <strong>
                              Query t{query} <span>→</span> Key t{key}
                            </strong>
                            <p>
                              {state.step === 2 && state.config.causal && key > query
                                ? T('这是未来位置，分数被设为 −∞。下一步权重将变为 0。')
                                : state.step === 2
                                  ? T('这个位置可见，保留原来的分数。')
                                  : T('按维度配对相乘，再相加。点击矩阵中任一格子查看计算。')}
                            </p>
                            <span className="large-value">
                              {fmt(
                                state.step === 1
                                  ? result.scores[query][key]
                                  : result.masked[query][key],
                              )}
                            </span>
                          </div>
                        </div>
                        <Formula expression={scoreExpression} symbols={scoreCellSymbols} />
                      </>
                    ) : (
                      <>
                        <div className="probability-layout">
                          <Matrix
                            name="P"
                            data={result.probabilities}
                            tone="probability"
                            row={query}
                            column={key}
                            tokenColumns
                            heatmap
                            onSelect={selectCell}
                          />
                          <div className="flow-block">
                            <span className="section-kicker">
                              {Tx('Query t{#0} 的取用比例', query)}
                            </span>
                            <WeightFlow probabilities={result.probabilities[query]} query={query} />
                            <div className="sum-check">
                              Σ P[{query}, j] ={' '}
                              {fmt(
                                result.probabilities[query].reduce((a, b) => a + b, 0),
                                6,
                              )}{' '}
                              <span>{T('已归一化')}</span>
                            </div>
                          </div>
                        </div>
                        {state.step === 4 ? (
                          <div className="output-block">
                            <div className="output-caption">
                              <span>{T('输出向量')}</span>
                              <strong className="mono">O[{query}]</strong>
                            </div>
                            <div className="output-values">
                              {result.output[query].map((value, i) => (
                                <div key={i}>
                                  <span>d{i}</span>
                                  <strong>{fmt(value)}</strong>
                                </div>
                              ))}
                            </div>
                            <span className="output-baseline">
                              {Tx(
                                '未干预 V 时{#0}{#1}',
                                <br />,
                                <span className="mono">
                                  [{baseline.output[query].map((x) => fmt(x)).join(', ')}]
                                </span>,
                              )}
                            </span>
                          </div>
                        ) : (
                          <div className="operation-note">
                            {T(
                              '横向读一行：当前 Query 向每个 Key 分配多少权重。线宽与权重成正比。',
                            )}
                          </div>
                        )}
                        {state.step === 4 && (
                          <Formula expression={outputExpression} symbols={outputCellSymbols} />
                        )}
                      </>
                    )}
                    <p className="precision-note">
                      {T('行向量约定 · 数值显示至小数点后 3 位，计算使用未舍入的双精度值。')}
                    </p>
                  </div>

                  <aside className="explanation-pane" aria-label={T('当前步骤说明')}>
                    <div className="explanation-heading">
                      <span className="section-kicker">{T('这一步发生了什么')}</span>
                      <button
                        onClick={() =>
                          setState((s) => ({
                            ...s,
                            view: { ...s.view, explanations: !s.view.explanations },
                          }))
                        }
                      >
                        {state.view.explanations ? T('隐藏说明') : T('显示说明')}
                      </button>
                    </div>
                    {state.view.explanations ? (
                      <>
                        <h3>{step.question}</h3>
                        <p>
                          <Rich
                            text={
                              state.step === 1 && !state.config.scaled
                                ? T(
                                    '当前为关闭缩放的对照：取 Q 的一行和 K 的一行，逐维相乘再相加，直接作为分数。标准 Attention 还会除以 $\\sqrt{d_k}$；重新开启缩放可比较同一输入的结果。',
                                  )
                                : step.explanation
                            }
                          />
                        </p>
                        <div className="step-formula">
                          <Formula
                            symbols={step.symbols.filter(
                              ([tex]) => state.step !== 1 || state.config.scaled || tex !== 'd_k',
                            )}
                            expression={
                              state.step === 1 && !state.config.scaled
                                ? T('S=QK^\\top\\quad\\text{（关闭缩放的对照）}')
                                : step.formula
                            }
                          />
                        </div>
                        <div className="shape-label">{T('输入 → 输出形状')}</div>
                        <div className="shape-value">
                          <Rich text={step.shape} />
                        </div>
                        <div className="remember">
                          <span>{T('留意')}</span>
                          <p>
                            <Rich text={step.caution} />
                          </p>
                        </div>
                      </>
                    ) : (
                      <div className="hidden-explanation">
                        <span>?</span>
                        <h3>{T('试着自己解释')}</h3>
                        <p>{step.question}</p>
                        <p>{T('指出输入、操作和输出，再说出改变哪个量会影响结果。')}</p>
                      </div>
                    )}
                  </aside>
                </div>

                <div className="transport">
                  <div className="transport-main">
                    <button
                      className="icon-button"
                      aria-label={T('重置步骤')}
                      onClick={() => jump(0)}
                    >
                      ↺
                    </button>
                    <button
                      className="icon-button"
                      aria-label={T('上一步')}
                      disabled={state.step === 0}
                      onClick={() => jump(state.step - 1)}
                    >
                      ←
                    </button>
                    <button
                      className="button primary play-button"
                      aria-label={playing ? T('暂停') : T('播放')}
                      onClick={() => {
                        if (state.step === STEP_COUNT - 1) setState((s) => ({ ...s, step: 0 }));
                        setPlaying((p) => !p);
                      }}
                    >
                      {playing ? T('Ⅱ 暂停') : T('▶ 播放')}
                    </button>
                    <button
                      className="icon-button"
                      aria-label={T('下一步')}
                      disabled={state.step === STEP_COUNT - 1}
                      onClick={() => jump(state.step + 1)}
                    >
                      →
                    </button>
                    <span className="transport-status" role="status">
                      {playing ? T('播放中') : T('已暂停')} · {state.step + 1}/5
                    </span>
                  </div>
                  <div className="transport-options">
                    <label>
                      {T('速度')}
                      <select
                        aria-label={T('播放速度')}
                        value={state.view.speed}
                        onChange={(e) => {
                          setPlaying(false);
                          setState((s) => ({
                            ...s,
                            view: { ...s.view, speed: Number(e.target.value) },
                          }));
                        }}
                      >
                        <option value="0.5">0.5×</option>
                        <option value="1">1×</option>
                        <option value="2">2×</option>
                      </select>
                    </label>
                    <button
                      className="text-button"
                      onClick={() => replaceState(presetState('basic'))}
                    >
                      {T('恢复基础示例')}
                    </button>
                  </div>
                </div>
                {reducedMotion && (
                  <p className="motion-note">
                    {T('已遵循减少动态效果偏好；播放只切换离散步骤，也可逐步查看。')}
                  </p>
                )}
              </section>

              <div className="below-grid">
                <section className="experiment-card">
                  <div className="card-heading">
                    <span className="section-kicker">{T('动手验证')}</span>
                    <span className="pill">{T('固定 Q / K')}</span>
                  </div>
                  <h2>{T('只改变 V，注意力权重会变吗？')}</h2>
                  <p>{T('在投影后的 V[1,0] 上加一个偏移。先预测，再执行。')}</p>
                  <div className="range-heading">
                    <label htmlFor="value-delta">{T('V[1,0] 的偏移 ΔV')}</label>
                    <output htmlFor="value-delta">{fmt(draftValue)}</output>
                  </div>
                  <input
                    id="value-delta"
                    type="range"
                    min="-4"
                    max="4"
                    step="0.5"
                    value={draftValue}
                    onChange={(e) => {
                      setPlaying(false);
                      setDraftValue(Number(e.target.value));
                      setVerified(false);
                    }}
                  />
                  <div className="range-scale">
                    <span>−4</span>
                    <span>0</span>
                    <span>+4</span>
                  </div>
                  <div className="prediction-controls">
                    <div
                      className="prediction-options"
                      role="group"
                      aria-label={T('预测注意力权重变化')}
                    >
                      <button
                        aria-pressed={prediction === 'same'}
                        onClick={() => {
                          setPrediction('same');
                          setVerified(false);
                        }}
                      >
                        {T('P 不会变')}
                      </button>
                      <button
                        aria-pressed={prediction === 'change'}
                        onClick={() => {
                          setPrediction('change');
                          setVerified(false);
                        }}
                      >
                        {T('P 会变')}
                      </button>
                    </div>
                    <button
                      className="button primary"
                      disabled={!prediction}
                      onClick={() => {
                        setPlaying(false);
                        setState((s) => ({
                          ...s,
                          input: { ...s.input, valueDelta: draftValue },
                          step: 4,
                        }));
                        setVerified(true);
                      }}
                    >
                      {T('验证预测 →')}
                    </button>
                  </div>
                  <div className="prediction-feedback" role="status">
                    {verified ? (
                      <>
                        {Tx(
                          '{#0} P 保持不变。当前 O[{#1},0] 相对未干预 V 时变化 {#2}。{#3}',
                          prediction === 'same' ? T('预测正确。') : T('再观察一次：'),
                          query,
                          fmt(result.output[query][0] - baseline.output[query][0]),
                          result.probabilities[query][1] === 0
                            ? T('t1 被屏蔽，因此对当前输出没有影响。')
                            : T('变化量等于 P[{#0},1] × ΔV。', query),
                        )}
                      </>
                    ) : draftValue !== state.input.valueDelta ? (
                      T('有待验证的改动；上方仍显示上一次计算结果。')
                    ) : (
                      T('选择预测后点击验证。偏移限制在 [−4, 4]，便于手算。')
                    )}
                  </div>
                </section>
                <section className="settings-card">
                  <div className="card-heading">
                    <span className="section-kicker">{T('计算条件')}</span>
                    <span className="pill muted">{T('即时重算')}</span>
                  </div>
                  <label className="toggle-row">
                    <span>
                      <strong>{T('因果 Mask')}</strong>
                      <small>{T('只允许访问自身及更早的 token')}</small>
                    </span>
                    <input
                      type="checkbox"
                      role="switch"
                      checked={state.config.causal}
                      onChange={(e) => changeConfig({ causal: e.target.checked }, 2)}
                    />
                    <span className="switch-track" />
                  </label>
                  <label className="toggle-row">
                    <span>
                      <strong>{Tx('缩放 {#0}', <Tex tex="1/\sqrt{d_k}" />)}</strong>
                      <small>{T('关闭后比较分数与概率的变化')}</small>
                    </span>
                    <input
                      type="checkbox"
                      role="switch"
                      aria-label={T('缩放 1/√d_k')}
                      checked={state.config.scaled}
                      onChange={(e) => changeConfig({ scaled: e.target.checked }, 1)}
                    />
                    <span className="switch-track" />
                  </label>
                  <div className="settings-footnote">
                    {Tx(
                      '调整计算条件会暂停播放，并跳到对应步骤。缩放的方差直觉假设 q/k 各分量独立、均值 0、方差 1，此时点积方差为 {#0}。',
                      <Tex tex="d_k" />,
                    )}
                  </div>
                </section>
              </div>

              <details className="details-panel" id="attention-inputs" ref={inputsRef}>
                <summary>
                  <span>{T('输入与参数')}</span>
                  <span className="summary-hint">{T('编辑 X · 查看固定 W · 对照修改 K')}</span>
                </summary>
                <div className="details-body">
                  <p>
                    {T(
                      'X 是人为设定的隐藏状态，不代表真实语义空间。各输入限制在 [−4, 4]。修改 X 会重新计算 Q/K/V，并返回第一步。',
                    )}
                  </p>
                  <div className="input-and-weights">
                    <div className="input-matrix">
                      <div className="matrix-heading">
                        <strong>X</strong>
                        <span>{T('3 × 2 · 可编辑')}</span>
                      </div>
                      <table aria-label={T('输入 X')}>
                        <thead>
                          <tr>
                            <th>token</th>
                            <th>d0</th>
                            <th>d1</th>
                          </tr>
                        </thead>
                        <tbody>
                          {state.input.x.map((row, i) => (
                            <tr key={i}>
                              <th>t{i}</th>
                              {row.map((value, j) => (
                                <td key={j}>
                                  <input
                                    type="number"
                                    aria-label={`X[${i},${j}]`}
                                    min="-4"
                                    max="4"
                                    step="0.5"
                                    value={value}
                                    onChange={(e) => {
                                      if (e.target.value === '') return;
                                      const next = Number(e.target.value);
                                      if (!Number.isFinite(next) || next < -4 || next > 4) return;
                                      setPlaying(false);
                                      setVerified(false);
                                      setState((s) => ({
                                        ...s,
                                        step: 0,
                                        input: {
                                          ...s.input,
                                          x: s.input.x.map((r, ri) =>
                                            r.map((x, ci) => (ri === i && ci === j ? next : x)),
                                          ),
                                        },
                                      }));
                                    }}
                                  />
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <Matrix name="W_Q" data={W_Q} parameter tone="query" />
                    <Matrix name="W_K" data={W_K} parameter tone="key" />
                    <Matrix name="W_V" data={W_V} parameter tone="value" />
                  </div>
                  <div className="advanced-controls">
                    <div>
                      <label htmlFor="key-delta">
                        {Tx('在投影后干预 K[1,0]：{#0}', fmt(state.input.keyDelta))}
                      </label>
                      <input
                        id="key-delta"
                        type="range"
                        min="-4"
                        max="4"
                        step="0.5"
                        value={state.input.keyDelta}
                        onChange={(e) => {
                          const keyDelta = Number(e.target.value);
                          setPlaying(false);
                          setVerified(false);
                          setState((s) => ({ ...s, input: { ...s.input, keyDelta }, step: 1 }));
                        }}
                      />
                      <small>
                        {Tx(
                          '可能改变 S 的 t1 列，以及各行归一化后的 P。参数 {#0} 保持不变。',
                          <Tex tex="W_K" />,
                        )}
                      </small>
                    </div>
                    <div>
                      <label htmlFor="seed">{T('可复现随机种子')}</label>
                      <div className="seed-control">
                        <input
                          id="seed"
                          type="number"
                          min="0"
                          max="999999"
                          step="1"
                          value={state.seed}
                          onChange={(e) => {
                            const seed = Number(e.target.value);
                            if (Number.isInteger(seed) && seed >= 0 && seed <= 999999) {
                              setPlaying(false);
                              setState((s) => ({ ...s, seed }));
                            }
                          }}
                        />
                        <button
                          className="button secondary"
                          onClick={() => {
                            const next = {
                              ...state,
                              step: 0,
                              input: { x: seededInput(state.seed), keyDelta: 0, valueDelta: 0 },
                            };
                            replaceState(next);
                          }}
                        >
                          {T('生成 X')}
                        </button>
                      </div>
                      <small>
                        {T('固定采样规则：每个元素取 −2 到 2 的半整数；种子与当前 X 一起分享。')}
                      </small>
                    </div>
                  </div>
                </div>
              </details>

              <section className="recall-card">
                <div>
                  <span className="section-kicker">{T('离开之前 · 试着复述')}</span>
                  <h2>{T('不用看公式，讲清楚 Q、K、V 各自做什么。')}</h2>
                  <p>{T('再解释：只改变 V 时为什么 P 不变？未来位置又是在哪一步被排除的？')}</p>
                </div>
                <button
                  className="button secondary"
                  aria-expanded={recallOpen}
                  onClick={() => {
                    setRecallOpen((v) => !v);
                    setState((s) => ({ ...s, view: { ...s.view, explanations: false } }));
                  }}
                >
                  {recallOpen ? T('收起参考解释') : T('查看参考解释')}
                </button>
                {recallOpen && (
                  <p className="recall-answer">
                    {T(
                      'Q 与各 K 点积产生分数；因果 Mask 在 Softmax 前排除未来位置。Softmax 沿 Key 轴把分数变成权重 P，最后 P 对 V 加权求和。固定 Q/K 和 Mask 时，改 V 不会改 P，但能按对应权重改变输出。',
                    )}
                  </p>
                )}
              </section>
            </>
          )}
        </div>
      )}

      <RelatedNotes path="/attention" />
      <details className="details-panel source-panel">
        <summary>
          <span>{T('适用条件与来源')}</span>
          <span className="summary-hint">{T('简化假设 · 常见误区 · 参考文献')}</span>
        </summary>
        <div className="details-body">
          <p>
            {Tx(
              '总图与模块顺序以原书第 3 章 Post-Norm 图为主，Attention 解释对应第 9–12 章。架构图是结构导航；单头实验忽略 batch、偏置、位置编码、Dropout 和残差，3 个 token、D = {#0}。多头实验对应第 11–12 章：{#1}，可选 1、2、4 个头，演算切分、转置、各头 Attention、Concatenate 与 {#2}；始终按 {#3} 缩放，与单头实验共用因果 Mask 开关。',
              <Tex tex="d_k = d_v = 2" />,
              <Tex tex="d_{\mathrm{model}} = 4" />,
              <Tex tex="W_O" />,
              <Tex tex="\sqrt{d_{\mathrm{key}}}" />,
            )}
          </p>
          <p>
            {T(
              '“只改变 V / K”是在投影之后人为干预一个激活，用于隔离因果关系；真实模型改变 X 时，Q、K、V 通常会同时变化。所有图形与说明为本项目编写。',
            )}
          </p>
          <ul>
            {sources.map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title} ↗
                </a>
              </li>
            ))}
          </ul>
          <p className="subtle">
            {T(
              '来源核对：单头 2026-11-05；多头 2026-11-06，对照原书 GitHub 源文件（commit 4a77788）。以原书组织图解和讲解，数值公式用原论文 §3.2 核对。第 3 章图的层堆叠统一标为 Block × N；与第 15 章 Pre-Norm 方案分别说明。',
            )}
          </p>
        </div>
      </details>
    </AppShell>
  );
}
