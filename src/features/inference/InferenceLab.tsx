import { useEffect, useMemo, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Formula } from '../../components/Formula';
import { Rich } from '../../components/Tex';
import { inferenceMyths, inferenceSources, inferenceSteps } from '../../content/inference';
import { BOOK_URL } from '../../content/topics';
import {
  generate,
  invalidatedByEdit,
  invalidatedByWindowSlide,
  MAX_DECODE_STEPS,
  TOY_MODEL,
  trainingWithEditedInput,
} from '../../math/inference';
import {
  INFERENCE_STEP_COUNT,
  initialInferenceState,
  moveInferenceStep,
  PROMPT,
  restoreInferenceState,
  serializeInferenceState,
  TRAIN_SEQUENCE,
  type InferenceState,
} from '../../state/inference';
import { CacheDiagram } from './CacheDiagram';
import {
  BaselineWidget,
  CompareWidget,
  DecodeWidget,
  InvalidationWidget,
  PrefillWidget,
  TrainingWidget,
} from './InferenceWidgets';
import '../../styles/lesson.css';
import '../../styles/attention-diagram.css';
import '../../styles/inference.css';
import '../../styles/instrument.css';

/** Scenes whose widget is driven by the number of Decode steps. */
const DECODE_SCENES = new Set([1, 3, 4]);

export default function InferenceLab() {
  const [restored] = useState(() => restoreInferenceState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.warning ?? '');
  const [playing, setPlaying] = useState(false);
  const [decodePlaying, setDecodePlaying] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [manualLink, setManualLink] = useState('');
  const [recallOpen, setRecallOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const step = inferenceSteps[state.step];
  const hide = !state.view.explanations;

  // Every number on the page is rebuilt from the state; nothing is carried over between edits.
  const generation = useMemo(
    () => generate(TOY_MODEL, PROMPT, state.decode.steps),
    [state.decode.steps],
  );
  // The cache after the shown Decode steps holds the prompt plus every token fed so far.
  const cachedIds = useMemo(
    () => generation.tokens.slice(0, PROMPT.length + state.decode.steps),
    [generation, state.decode.steps],
  );
  const { mode, pos, token } = state.cache;
  const flags = useMemo(
    () =>
      mode === 'edit'
        ? invalidatedByEdit(TOY_MODEL, cachedIds, pos, token)
        : invalidatedByWindowSlide(TOY_MODEL, cachedIds),
    [cachedIds, mode, pos, token],
  );
  const training = useMemo(() => {
    const { pos: editPos, token: editToken, applied } = state.train;
    return {
      inputs: TRAIN_SEQUENCE.slice(0, -1).map((id, i) =>
        applied && i === editPos ? editToken : id,
      ),
      targets: TRAIN_SEQUENCE.slice(1),
      changedRows: trainingWithEditedInput(TOY_MODEL, TRAIN_SEQUENCE, editPos, editToken)
        .changedRows,
      applied,
    };
  }, [state.train]);

  function stopAll() {
    setPlaying(false);
    setDecodePlaying(false);
  }
  /** Any edit by the learner pauses both timelines, then rebuilds from the new inputs. */
  function edit(next: InferenceState) {
    stopAll();
    if (
      next.cache.mode !== state.cache.mode ||
      next.cache.pos !== state.cache.pos ||
      next.cache.token !== state.cache.token ||
      next.decode.steps !== state.decode.steps
    )
      setRevealed(false);
    setState(next);
  }
  function jump(target: number) {
    stopAll();
    setState((s) => ({ ...s, step: Math.max(0, Math.min(INFERENCE_STEP_COUNT - 1, target)) }));
  }
  function toggleStepPlay() {
    setDecodePlaying(false);
    if (!playing && state.step === INFERENCE_STEP_COUNT - 1) setState((s) => ({ ...s, step: 0 }));
    setPlaying((p) => !p);
  }
  function toggleDecodePlay() {
    setPlaying(false);
    if (!decodePlaying && state.decode.steps >= MAX_DECODE_STEPS) {
      setRevealed(false);
      setState((s) => ({ ...s, decode: { steps: 1 } }));
    }
    setDecodePlaying((p) => !p);
  }

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setState((s) => moveInferenceStep(s, 1)),
      8000 / state.view.speed,
    );
    return () => window.clearInterval(timer);
  }, [playing, state.view.speed]);
  useEffect(() => {
    if (state.step === INFERENCE_STEP_COUNT - 1) setPlaying(false);
  }, [state.step]);

  useEffect(() => {
    if (!decodePlaying) return;
    const timer = window.setInterval(() => {
      setRevealed(false);
      setState((s) =>
        s.decode.steps < MAX_DECODE_STEPS ? { ...s, decode: { steps: s.decode.steps + 1 } } : s,
      );
    }, 1400 / state.view.speed);
    return () => window.clearInterval(timer);
  }, [decodePlaying, state.view.speed]);
  useEffect(() => {
    // Stop on the last step instead of looping, so the final frame stays readable.
    if (decodePlaying && state.decode.steps >= MAX_DECODE_STEPS) setDecodePlaying(false);
  }, [decodePlaying, state.decode.steps]);

  useEffect(() => {
    const pause = () => {
      if (document.hidden) stopAll();
    };
    const restore = () => {
      const next = restoreInferenceState(window.location.hash);
      stopAll();
      setState(next.state);
      setRevealed(false);
      setRecallOpen(false);
      setManualLink('');
      setNotice(next.warning ?? '已从链接恢复实验。');
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
    const url = new URL('/inference-cache', window.location.origin);
    url.hash = serializeInferenceState(state);
    try {
      await navigator.clipboard.writeText(url.href);
      setManualLink('');
      setNotice('已复制当前实验状态的链接。');
    } catch {
      setManualLink(url.href);
      setNotice('浏览器未允许剪贴板访问，请从下方选择并复制链接。');
    }
  }

  const widgetProps = { state, onChange: edit, generation };
  const widget = [
    <TrainingWidget key="train" state={state} onChange={edit} />,
    <BaselineWidget key="baseline" {...widgetProps} />,
    <PrefillWidget key="prefill" generation={generation} />,
    <DecodeWidget key="decode" {...widgetProps} />,
    <CompareWidget key="compare" {...widgetProps} />,
    <InvalidationWidget
      key="invalid"
      state={state}
      onChange={edit}
      ids={cachedIds}
      flags={flags}
      revealed={revealed}
      onReveal={() => {
        stopAll();
        setRevealed(true);
      }}
    />,
  ][state.step];

  return (
    <AppShell
      active="inference"
      crumb="训练、生成与缓存"
      indicator="浏览器内计算"
      title="训练、生成与缓存"
    >
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            原书导读 <span>·</span> 第 2、16、22 章
          </div>
          <h1>
            训练、生成与缓存：哪些结果<span>可以复用</span>
          </h1>
          <p>
            同一个两层小模型：训练时一次算出所有位置；生成时先 Prefill，再一步一个 Decode，并用 KV
            缓存跳过重复计算。
          </p>
        </div>
        <button className="button secondary share-button" onClick={() => void copyLink()}>
          <span aria-hidden="true">↗</span> 复制实验链接
        </button>
      </section>
      {notice && (
        <div className="notice" role="status">
          {notice}
          <button aria-label="关闭提示" onClick={() => setNotice('')}>
            ×
          </button>
        </div>
      )}
      {manualLink && (
        <label className="manual-share">
          手动复制链接
          <input readOnly value={manualLink} onFocus={(event) => event.target.select()} />
        </label>
      )}

      <section className="lesson-panel inf-panel" aria-label="训练、生成与缓存逐步实验">
        <div className="lesson-panel-heading">
          <div>
            <span className="lesson-eyebrow">同一个问题：哪些计算必须做，哪些可以复用</span>
            <h2>从训练到生成，一个两层小模型的完整账本。</h2>
          </div>
          <span className="lesson-progress">
            {state.step + 1} / {INFERENCE_STEP_COUNT}
          </span>
        </div>
        <div className="lesson-steps inf-steps" aria-label="训练与缓存步骤">
          {inferenceSteps.map((item, i) => (
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
        <div className="lesson-columns inf-columns">
          <div className="diagram-panel">
            <CacheDiagram
              step={state.step}
              generation={generation}
              decodeSteps={state.decode.steps}
              training={training}
              invalidation={{ mode, ids: cachedIds, flags, revealed }}
            />
            {DECODE_SCENES.has(state.step) && (
              <button
                className="button secondary inf-play"
                aria-label={decodePlaying ? '暂停逐步 Decode' : '播放逐步 Decode'}
                onClick={toggleDecodePlay}
              >
                {decodePlaying ? 'Ⅱ 暂停' : '▶ 逐步 Decode'}
              </button>
            )}
          </div>
          <div className="lesson-reading">
            <div className="lesson-reading-top">
              <span>这一步</span>
              <button
                onClick={() =>
                  edit({
                    ...state,
                    view: { ...state.view, explanations: !state.view.explanations },
                  })
                }
              >
                {hide ? '显示说明' : '隐藏说明'}
              </button>
            </div>
            <h3>{step.title}</h3>
            {hide ? (
              <div className="lesson-recall">
                <strong>先自己说一遍</strong>
                <p>
                  <Rich text={step.prompt} />
                </p>
              </div>
            ) : (
              <p className="lesson-explanation">
                <Rich text={step.explanation} />
              </p>
            )}
            {widget}
            <div className="lesson-takeaway">
              <strong>这一步，记住一句话</strong>
              <p>
                <Rich text={hide ? '先自己复述，再点击“显示说明”核对。' : step.takeaway} />
              </p>
            </div>
            <details className="lesson-formula">
              <summary>需要时再看公式</summary>
              <Formula expression={step.formula} />
            </details>
            <a
              className="lesson-source"
              href={BOOK_URL + step.chapter[1]}
              target="_blank"
              rel="noreferrer"
            >
              对应原书：{step.chapter[0]} ↗
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
              ← 上一步
            </button>
            <button
              className="button primary"
              disabled={state.step === INFERENCE_STEP_COUNT - 1}
              onClick={() => jump(state.step + 1)}
            >
              {state.step === INFERENCE_STEP_COUNT - 1
                ? '已到最后一步'
                : `下一步：${inferenceSteps[state.step + 1].name} →`}
            </button>
          </div>
          <div>
            <button className="lesson-text-button" onClick={() => jump(0)}>
              从头看
            </button>
            <button
              className="lesson-text-button"
              onClick={() => {
                stopAll();
                setRevealed(false);
                setState({ ...initialInferenceState(), view: state.view });
                setNotice('已恢复默认示例。');
              }}
            >
              恢复默认示例
            </button>
            <button
              className="lesson-text-button"
              aria-label={playing ? '暂停自动逐步' : '播放自动逐步'}
              onClick={toggleStepPlay}
            >
              {playing ? 'Ⅱ 暂停' : '▶ 自动逐步'}
            </button>
            <label>
              速度
              <select
                aria-label="播放速度"
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
            已遵循减少动态效果偏好；表格与缓存格子直接切换，不做过渡动画。
          </p>
        )}
      </section>

      <section className="recall-card">
        <div>
          <span className="section-kicker">离开之前 · 试着复述</span>
          <h2>不看图，讲清楚训练、Prefill、Decode 各算什么。</h2>
          <p>
            再解释：训练为什么能一次算所有位置，又不会偷看答案？缓存里存什么，为什么不存
            Q？改动一个提示 token 之后，哪些缓存要重算？
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
          {recallOpen ? '收起参考解释' : '查看参考解释'}
        </button>
        {recallOpen && (
          <p className="recall-answer">
            训练时整条序列已知，把它错开一位当作目标；因果 Mask 保证第 t 行只读
            x0…xt，所以一次前向就得到所有位置的预测和
            loss。推理时下一个输入要等这一步的输出，只能一步一个。Prefill
            并行处理提示，把每层每个位置的 K、V 写进缓存；之后每次 Decode 只为新 token 算
            Q、K、V，把 K、V 追加进缓存，用新 Q 读全部历史。旧 Q 的输出早已算完，所以不存
            Q。缓存与重算的结果相同，省掉的是重复的投影与打分。改动第 j 个 token，第 1 层只有位置 j
            失效，更高层从 j 起都要重算；位置编号整体平移时，旧条目都不能直接复用。
          </p>
        )}
      </section>

      <details className="details-panel source-panel">
        <summary>
          <span>适用条件与来源</span>
          <span className="summary-hint">简化假设 · 常见误区 · 参考文献</span>
        </summary>
        <div className="details-body">
          <p>
            所有数值由浏览器实时计算。玩具模型：词表 a–f 六个符号（没有语义），2 层单头 Attention，
            d = 2，加法正弦位置编码，每层只有 Attention 与残差（省略 W_O、FFN、LayerNorm、偏置与
            Dropout），输出头复用嵌入表，贪心解码。权重是固定教学数字，不是训练好的模型；训练一步只算前向和
            loss，不做反向传播与参数更新。缓存与重算的比较使用同一提示、权重、位置编号、Mask
            与运行模式，容差 1e-12。
          </p>
          <p>
            工作量只统计 Attention 的 K/V 投影与 Q·K 打分，用来说明哪些计算被跳过，不代表真实 GPU
            上的耗时。 缓存本身占用显存：第 22 章用 Llama 2 7B 手算了每个 token
            的缓存字节数；字节公式与 MHA、GQA、MLA 的对照会在“KV 架构对照”实验中展开（尚未实现）。
          </p>
          <strong className="inf-notes-title">常见误区</strong>
          <ul>
            {inferenceMyths.map((myth) => (
              <li key={myth}>
                <Rich text={myth} />
              </li>
            ))}
          </ul>
          <ul>
            {inferenceSources.map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title} ↗
                </a>
              </li>
            ))}
          </ul>
          <p className="subtle">来源核对日期：2026-11-07。图形与说明为本项目编写。</p>
        </div>
      </details>
    </AppShell>
  );
}
