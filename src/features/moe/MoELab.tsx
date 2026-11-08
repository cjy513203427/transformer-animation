import { useEffect, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Formula } from '../../components/Formula';
import { Rich } from '../../components/Tex';
import { StepSources, WhyIntro } from '../../components/WhyIntro';
import { computeMoE } from '../../math/moe';
import {
  moeBookNotes,
  moeMyths,
  moeSources,
  moeSteps,
  moeVerifiedDate,
  moeWhy,
} from '../../content/moe';
import {
  initialMoEState,
  MOE_STEP_COUNT,
  moveMoEStep,
  restoreMoEState,
  serializeMoEState,
  type MoEState,
} from '../../state/moe';
import { MoEDiagram } from './MoEDiagram';
import { MoEInputs, MoEWidget } from './MoEWidgets';
import '../../styles/lesson.css';
import '../../styles/attention-diagram.css';
import '../../styles/experiment.css';
import '../../styles/moe.css';
import '../../styles/editorial.css';

export default function MoELab() {
  const [restored] = useState(() => restoreMoEState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.warning ?? '');
  const [playing, setPlaying] = useState(false);
  const [revision, setRevision] = useState(0);
  const [manualLink, setManualLink] = useState('');
  const [recallOpen, setRecallOpen] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const step = moeSteps[state.step];
  const hide = !state.view.explanations;
  const result = computeMoE(state);

  /** Any edit by the learner pauses playback; every number is recomputed from the new state. */
  function edit(next: MoEState, preservePrediction = false) {
    setPlaying(false);
    if (!preservePrediction) setRevision((value) => value + 1);
    setState(next);
  }
  function jump(target: number) {
    setPlaying(false);
    setState((s) => ({ ...s, step: Math.max(0, Math.min(MOE_STEP_COUNT - 1, target)) }));
  }
  function togglePlay() {
    if (!playing && state.step === MOE_STEP_COUNT - 1) setState((s) => ({ ...s, step: 0 }));
    setPlaying((p) => !p);
  }

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setState((s) => moveMoEStep(s, 1)),
      8000 / state.view.speed,
    );
    return () => window.clearInterval(timer);
  }, [playing, state.view.speed]);
  useEffect(() => {
    if (state.step === MOE_STEP_COUNT - 1) setPlaying(false);
  }, [state.step]);

  useEffect(() => {
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    const restore = () => {
      const next = restoreMoEState(window.location.hash);
      setPlaying(false);
      setState(next.state);
      setRevision((value) => value + 1);
      setRecallOpen(false);
      setManualLink('');
      setNotice(next.warning ?? '已从链接恢复实验。');
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
    const url = new URL('/moe', window.location.origin);
    url.hash = serializeMoEState(state);
    try {
      await navigator.clipboard.writeText(url.href);
      setManualLink('');
      setNotice('已复制当前实验状态的链接。');
    } catch {
      setManualLink(url.href);
      setNotice('浏览器未允许剪贴板访问，请从下方选择并复制链接。');
    }
  }

  return (
    <AppShell active="moe" crumb="MoE 与专家并行" indicator="浏览器内计算" title="MoE 与专家并行">
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            原书导读 <span>·</span> 第 30 章 <span>·</span> 稀疏激活的计算链
          </div>
          <h1>
            MoE：一个 token <span>怎样经过多个专家</span>
          </h1>
          <p>
            4 个 token、4 个专家，从 Router 打分到加权合并逐步手算。 试着改变输入与
            Top-K，观察谁被选中、哪些参数参与计算，以及跨设备传了什么。
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

      <WhyIntro why={moeWhy} />
      <section className="lesson-panel moe-panel" aria-label="MoE 逐步实验">
        <div className="lesson-panel-heading">
          <div>
            <span className="lesson-eyebrow">动手验证：只改专家的输出，会改变路由吗？</span>
            <h2>打分 → 选专家 → 计算 → 按原 token 合并。</h2>
          </div>
          <span className="lesson-progress">
            {state.step + 1} / {MOE_STEP_COUNT}
          </span>
        </div>
        <div className="lesson-steps moe-steps" aria-label="MoE 步骤">
          {moeSteps.map((item, i) => (
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
        <MoEInputs state={state} onChange={edit} result={result} />
        <div className="lesson-columns moe-columns">
          <div className="diagram-panel">
            <MoEDiagram state={state} result={result} />
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
            <MoEWidget
              key={`${state.step}:${revision}`}
              state={state}
              result={result}
              onChange={edit}
            />
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
            <StepSources book={step.book} source={step.source} />
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
              disabled={state.step === MOE_STEP_COUNT - 1}
              onClick={() => jump(state.step + 1)}
            >
              {state.step === MOE_STEP_COUNT - 1
                ? '已到最后一步'
                : `下一步：${moeSteps[state.step + 1].name} →`}
            </button>
          </div>
          <div>
            <button className="lesson-text-button" onClick={() => jump(0)}>
              从头看
            </button>
            <button
              className="lesson-text-button"
              onClick={() => {
                setPlaying(false);
                setState({ ...initialMoEState(), view: state.view });
                setRevision((value) => value + 1);
                setNotice('已恢复默认示例。');
              }}
            >
              恢复默认示例
            </button>
            <button
              className="lesson-text-button"
              aria-label={playing ? '暂停自动逐步' : '播放自动逐步'}
              onClick={togglePlay}
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
          <p className="lesson-motion">已遵循减少动态效果偏好；图表直接切换，不做过渡动画。</p>
        )}
      </section>

      <section className="recall-card">
        <div>
          <span className="section-kicker">离开之前 · 试着复述</span>
          <h2>不看图，讲清一个 token 的 MoE 计算。</h2>
          <p>
            Router 和专家各自做什么？为什么 Top-2 要保留 token 身份？ 总参数、每 token
            激活参数、专家负载和通信量分别在数什么？
          </p>
        </div>
        <button
          className="button secondary"
          aria-expanded={recallOpen}
          onClick={() => {
            setPlaying(false);
            setRecallOpen((v) => !v);
            setState((s) => ({ ...s, view: { ...s.view, explanations: false } }));
          }}
        >
          {recallOpen ? '收起参考解释' : '查看参考解释'}
        </button>
        {recallOpen && (
          <p className="recall-answer">
            输入先乘 Router 权重，再沿专家轴做 Softmax。取分数最大的 K 个专家，
            本例在选中集合内重新归一化门控。每个选中专家分别做两层 ReLU FFN， 用门值加权后按 token
            身份求和。改变专家输出不改变这次前向的路由。 Top-K
            只减少参与计算的专家；全部权重仍需存储。多个 token 可能集中到同一专家，
            所以激活参数少也可能负载不均。专家并行把输入激活送往专家所在设备， 再把输出送回原 token
            所在设备；本例不传输权重。
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
            教学配置：4 个 token、D = 2、4 个专家，每个专家是中间维度 F = 3 的两层 ReLU FFN。
            行向量记法，忽略 batch 与偏置；只演示一个 MoE 子层，不含 Attention、残差、归一化与训练。
            种子固定时权重可复现，推理前向不更新权重，小向量不代表真实语义。 先沿全部专家做
            Softmax，再取 Top-K 并在选中集合内重新归一化；Top-1 门值为 1。
            此变体没有噪声、共享专家、容量限制、丢弃、重路由或负载均衡损失，不是 Switch 的完整复现。
          </p>
          <p>
            账本只统计本 MoE 子层。MAC 是一次乘加，未把 ReLU、Softmax、Top-K、门控合并计入矩阵乘加。
            通信以每元素 4 B 计数，仅包含跨设备的输入与输出激活；忽略元数据、协议、集体通信算法、
            带宽与延迟，不是 GPU 运行时间或显存实测。内部使用 float64，展示时才舍入，等价容差为
            1e-12。
          </p>
          <strong className="exp-notes-title">与原书说法的对照</strong>
          <ul>
            {moeBookNotes.map((note) => (
              <li key={note}>
                <Rich text={note} />
              </li>
            ))}
          </ul>
          <strong className="exp-notes-title">常见误区</strong>
          <ul>
            {moeMyths.map((myth) => (
              <li key={myth}>
                <Rich text={myth} />
              </li>
            ))}
          </ul>
          <ul>
            {moeSources.map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title} ↗
                </a>
              </li>
            ))}
          </ul>
          <p className="subtle">来源核对日期：{moeVerifiedDate}。图形与说明为本项目编写。</p>
        </div>
      </details>
    </AppShell>
  );
}
