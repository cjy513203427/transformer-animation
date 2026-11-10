import { useEffect, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Formula } from '../../components/Formula';
import { Rich, Tex } from '../../components/Tex';
import { bookNotes, positionSources, positionSteps } from '../../content/position';
import { chapterUrl } from '../../content/book';
import {
  initialPositionState,
  MAX_POS,
  movePositionStep,
  POSITION_STEP_COUNT,
  restorePositionState,
  ROPE_MAX,
  serializePositionState,
  type PositionState,
} from '../../state/position';
import { PositionPipeline } from './PositionPipeline';
import {
  AdditionWidget,
  AlibiWidget,
  BlindnessWidget,
  RelativeWidget,
  RopeWidget,
  SinusoidalWidget,
  SummaryWidget,
} from './PositionWidgets';
import { localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/lesson.css';
import '../../styles/attention-diagram.css';
import '../../styles/position.css';
import '../../styles/editorial.css';

const explanations = [
  T(
    'Attention 打分只用到两个向量的点积。如果输入只有词嵌入，“我”放在第 0 位还是第 2 位都是同一个向量；把句子倒过来，每个词看到的仍是同一组向量，只是排列不同。结果是：模型分不清“我爱你”和“你爱我”。',
  ),
  T(
    '原始 Transformer 用 $\\sin$ 和 $\\cos$ 给每个位置造一个向量：第 $i$ 对维度按频率 $\\omega_i = 10000^{-2i/d_{\\mathrm{model}}}$ 转动。低维转得快，能区分相邻位置；高维转得慢，能区分相隔很远的位置。所有值都在 $[-1, 1]$ 之间，不会像整数编号那样越加越大。',
  ),
  T(
    '位置向量与词嵌入逐维相加（不是拼接），维度仍是 $d_{\\mathrm{model}}$。这一步发生在第一个 Block 之前：之后的 $W_Q$、$W_K$、$W_V$ 都作用在相加后的 X 上，所以 Q、K、V 都带位置。正弦表是固定的，嵌入表会在训练中更新。',
  ),
  T(
    'RoPE（LLaMA 等模型使用）不再把位置加到 X 上。投影得到 Q、K 之后，把它们每两维一组，按位置旋转 $m\\theta_i$。旋转只改变方向、不改变长度；V 不旋转。',
  ),
  T(
    '两个向量各自转了 $m\\theta_i$ 与 $n\\theta_i$，它们的点积只与夹角有关，而夹角的改变量只取决于 $(n-m)\\theta_i$。所以同时平移两个位置，分数不变：这就是“相对位置”。',
  ),
  T(
    'ALiBi（BLOOM、MPT 等使用）连 Q、K 都不动：算出 $QK^{\\top}/\\sqrt{d}$ 之后，每个 Key 按与 Query 的距离扣掉 $m\\times\\text{距离}$，再做 Softmax。不同的头用不同的斜率 $m$，有的头只看附近，有的头几乎不受距离影响。',
  ),
  T(
    '三种方法回答同一个问题“模型怎样知道顺序”，但加在计算的不同位置：加法在最底部改 X，RoPE 在中间改 Q、K，ALiBi 在 Softmax 前改分数。',
  ),
];

/** One tick of the in-step position animation; returns the same object at the boundary. */
function advancePosition(state: PositionState): PositionState {
  if (state.step === 1 || state.step === 2)
    return state.sin.pos < MAX_POS
      ? { ...state, sin: { ...state.sin, pos: state.sin.pos + 1 } }
      : state;
  if (state.step === 3)
    return state.rope.m < ROPE_MAX
      ? { ...state, rope: { ...state.rope, m: state.rope.m + 1 } }
      : state;
  if (state.step === 4)
    return Math.max(state.rope.m, state.rope.n) < ROPE_MAX
      ? { ...state, rope: { m: state.rope.m + 1, n: state.rope.n + 1 } }
      : state;
  return state;
}

function rewindPosition(state: PositionState): PositionState {
  if ((state.step === 1 || state.step === 2) && state.sin.pos >= MAX_POS)
    return { ...state, sin: { ...state.sin, pos: 0 } };
  if (state.step === 3 && state.rope.m >= ROPE_MAX)
    return { ...state, rope: { ...state.rope, m: 0 } };
  if (state.step === 4 && Math.max(state.rope.m, state.rope.n) >= ROPE_MAX) {
    const low = Math.min(state.rope.m, state.rope.n);
    return { ...state, rope: { m: state.rope.m - low, n: state.rope.n - low } };
  }
  return state;
}

export default function PositionLab() {
  const [restored] = useState(() => restorePositionState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.warning ?? '');
  const [playing, setPlaying] = useState(false);
  const [posPlaying, setPosPlaying] = useState(false);
  const [manualLink, setManualLink] = useState('');
  const [recallOpen, setRecallOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const step = positionSteps[state.step];
  const hide = !state.view.explanations;

  function stopAll() {
    setPlaying(false);
    setPosPlaying(false);
  }
  /** Any edit by the learner pauses both timelines, then rebuilds from the new inputs. */
  function edit(next: PositionState) {
    stopAll();
    setState(next);
  }
  function jump(target: number) {
    stopAll();
    setState((s) => ({ ...s, step: Math.max(0, Math.min(POSITION_STEP_COUNT - 1, target)) }));
  }
  function toggleStepPlay() {
    setPosPlaying(false);
    if (!playing && state.step === POSITION_STEP_COUNT - 1) setState((s) => ({ ...s, step: 0 }));
    setPlaying((p) => !p);
  }
  function togglePosPlay() {
    setPlaying(false);
    if (!posPlaying) setState((s) => rewindPosition(s));
    setPosPlaying((p) => !p);
  }

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setState((s) => movePositionStep(s, 1)),
      7000 / state.view.speed,
    );
    return () => window.clearInterval(timer);
  }, [playing, state.view.speed]);
  useEffect(() => {
    if (state.step === POSITION_STEP_COUNT - 1) setPlaying(false);
  }, [state.step]);

  useEffect(() => {
    if (!posPlaying) return;
    const timer = window.setInterval(
      () => setState((s) => advancePosition(s)),
      900 / state.view.speed,
    );
    return () => window.clearInterval(timer);
  }, [posPlaying, state.view.speed]);
  useEffect(() => {
    // Stop at the last position instead of looping, so the final frame stays readable.
    if (posPlaying && advancePosition(state) === state) setPosPlaying(false);
  }, [posPlaying, state]);

  useEffect(() => {
    const pause = () => {
      if (document.hidden) stopAll();
    };
    const restore = () => {
      const next = restorePositionState(window.location.hash);
      stopAll();
      setState(next.state);
      setRecallOpen(false);
      setManualLink('');
      setNotice(next.warning ?? T('已从链接恢复实验。'));
    };
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onMotion = () => {
      setReducedMotion(media.matches);
      stopAll();
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
    const url = new URL(localePath('/position'), window.location.origin);
    url.hash = serializePositionState(state);
    try {
      await navigator.clipboard.writeText(url.href);
      setManualLink('');
      setNotice(T('已复制当前实验状态的链接。'));
    } catch {
      setManualLink(url.href);
      setNotice(T('浏览器未允许剪贴板访问，请从下方选择并复制链接。'));
    }
  }

  const playProps = { posPlaying, onTogglePosPlay: togglePosPlay, reducedMotion };
  const widget = [
    <BlindnessWidget key="blind" state={state} onChange={edit} />,
    <SinusoidalWidget key="sin" state={state} onChange={edit} {...playProps} />,
    <AdditionWidget key="add" state={state} onChange={edit} />,
    <RopeWidget key="rope" state={state} onChange={edit} {...playProps} />,
    <RelativeWidget key="relative" state={state} onChange={edit} {...playProps} />,
    <AlibiWidget key="alibi" state={state} onChange={edit} />,
    <SummaryWidget key="summary" onJump={jump} />,
  ][state.step];

  return (
    <AppShell
      active="position"
      crumb={T('位置编码')}
      indicator={T('浏览器内计算')}
      title={T('位置编码')}
    >
      <section className="page-heading">
        <div>
          <div className="eyebrow">{Tx('原书导读 {#0} 第 5、14、25 章', <span>·</span>)}</div>
          <h1>{Tx('位置编码：顺序从{#0}进入计算', <span>{T('哪里')}</span>)}</h1>
          <p>
            {T(
              '在同一条 Attention 计算链上，对照加法位置编码、RoPE 与 ALiBi 各自加在哪一步、改了哪些数。',
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

      <section className="lesson-panel pos-panel" aria-label={T('位置编码逐步实验')}>
        <div className="lesson-panel-heading">
          <div>
            <span className="lesson-eyebrow">{T('同一个问题：模型怎样知道“谁在谁前面”')}</span>
            <h2>{T('一条计算链，三个可以加入位置的入口。')}</h2>
          </div>
          <span className="lesson-progress">
            {state.step + 1} / {POSITION_STEP_COUNT}
          </span>
        </div>
        <div className="lesson-steps pos-steps" aria-label={T('位置编码步骤')}>
          {positionSteps.map((item, i) => (
            <button
              key={item.name}
              aria-current={state.step === i ? 'step' : undefined}
              onClick={() => jump(i)}
            >
              <span>{i + 1}</span>
              {item.name}
            </button>
          ))}
        </div>
        <div className="lesson-columns pos-columns">
          <div className="diagram-panel">
            <PositionPipeline active={step.insertion} onSelect={jump} />
          </div>
          <div className="lesson-reading">
            <div className="lesson-reading-top">
              <span>{T('这一步')}</span>
              <button
                onClick={() =>
                  edit({
                    ...state,
                    view: { ...state.view, explanations: !state.view.explanations },
                  })
                }
              >
                {hide ? T('显示说明') : T('隐藏说明')}
              </button>
            </div>
            <h3>{step.title}</h3>
            {hide ? (
              <div className="lesson-recall">
                <strong>{T('先自己说一遍')}</strong>
                <p>
                  <Rich text={step.prompt} />
                </p>
              </div>
            ) : (
              <p className="lesson-explanation">
                <Rich text={explanations[state.step]} />
              </p>
            )}
            {widget}
            <div className="lesson-takeaway">
              <strong>{T('这一步，记住一句话')}</strong>
              <p>
                <Rich text={hide ? T('先自己复述，再点击“显示说明”核对。') : step.takeaway} />
              </p>
            </div>
            <details className="lesson-formula">
              <summary>{T('需要时再看公式')}</summary>
              <Formula expression={step.formula} />
            </details>
            <a
              className="lesson-source"
              href={chapterUrl(step.chapter[1])}
              target="_blank"
              rel="noreferrer"
            >
              {Tx('对应原书：{#0} ↗', step.chapter[0])}
            </a>
          </div>
        </div>
        <div className="lesson-controls">
          <div>
            <button
              className="button secondary"
              disabled={state.step === 0}
              onClick={() => jump(state.step - 1)}
            >
              {T('← 上一步')}
            </button>
            <button
              className="button primary"
              disabled={state.step === POSITION_STEP_COUNT - 1}
              onClick={() => jump(state.step + 1)}
            >
              {state.step === POSITION_STEP_COUNT - 1
                ? T('已到最后一步')
                : T('下一步：{#0} →', positionSteps[state.step + 1].name)}
            </button>
          </div>
          <div>
            <button className="lesson-text-button" onClick={() => jump(0)}>
              {T('从头看')}
            </button>
            <button
              className="lesson-text-button"
              onClick={() => {
                stopAll();
                setState({ ...initialPositionState(), view: state.view });
                setNotice(T('已恢复默认示例。'));
              }}
            >
              {T('恢复默认示例')}
            </button>
            <button
              className="lesson-text-button"
              aria-label={playing ? T('暂停自动逐步') : T('播放自动逐步')}
              onClick={toggleStepPlay}
            >
              {playing ? T('Ⅱ 暂停') : T('▶ 自动逐步')}
            </button>
            <label>
              {T('速度')}
              <select
                aria-label={T('播放速度')}
                value={state.view.speed}
                onChange={(e) =>
                  edit({ ...state, view: { ...state.view, speed: Number(e.target.value) } })
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
            {T('已遵循减少动态效果偏好；指针和向量直接跳到新角度，不做转动动画。')}
          </p>
        )}
      </section>

      <section className="recall-card">
        <div>
          <span className="section-kicker">{T('离开之前 · 试着复述')}</span>
          <h2>{T('不看图，说出三种位置编码分别加在哪一步。')}</h2>
          <p>
            {T(
              '再解释：没有位置编码时为什么分不清“我爱你”和“你爱我”？RoPE 的分数为什么只看相对距离？ALiBi 能直接算更长的序列，这又不保证什么？',
            )}
          </p>
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
              '没有位置信息时，Attention 只看到一组向量，交换顺序只会让输出跟着换行。加法位置编码在投影前把 PE 加到 X 上，所以 Q、K、V 都带位置；RoPE 在投影后按位置旋转 Q、K，内容固定时 (R_m q)·(R_n k) = q·(R_{n−m} k)，只取决于距离；ALiBi 在 Softmax 前给分数加上 −m × 距离，不改 Q、K、V。偏置公式对任何长度都能计算，但不保证模型能可靠利用超出训练长度的上下文。',
            )}
          </p>
        )}
      </section>

      <details className="details-panel source-panel">
        <summary>
          <span>{T('适用条件、与原书对照与来源')}</span>
          <span className="summary-hint">{T('简化假设 · 书中说法的修正 · 参考文献')}</span>
        </summary>
        <div className="details-body">
          <p>
            {Tx(
              '所有数值由浏览器实时计算，均为教学数字而不是训练好的模型参数；行向量约定，单头，忽略 batch。位置盲与四项分解令 {#0}（{#1}）。“L”的嵌入取书中第 5 章列出的前 4 维。RoPE 演示取 {#2}、{#3}的固定内容向量。ALiBi 演示取 6 个 token、8 个头的标准斜率 {#4}，并带因果 Mask；示例内容分数为人为设定。',
              <Tex tex="W_Q = W_K = W_V = I" />,
              <Tex tex="d_{\mathrm{model}} = 4" />,
              <Tex tex="d = 8" />,
              <Tex tex="\mathrm{base} = 10000" />,
              <Tex tex="2^{-h}" />,
            )}
          </p>
          <strong className="pos-notes-title">{T('与原书说法的对照')}</strong>
          <ul>
            {bookNotes.map((note) => (
              <li key={note}>
                <Rich text={note} />
              </li>
            ))}
          </ul>
          <ul>
            {positionSources.map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title} ↗
                </a>
              </li>
            ))}
          </ul>
          <p className="subtle">
            {T(
              '核对：2026-11-06。原书内容对照 GitHub 源文件（commit 4a77788）；公式按论文记法编写，本次环境无法访问 arXiv，未在线复核论文原文。',
            )}
          </p>
        </div>
      </details>
    </AppShell>
  );
}
