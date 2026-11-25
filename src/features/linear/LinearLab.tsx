import { useEffect, useMemo, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Prediction, PredictionResult } from '../../components/Controls';
import { Formula } from '../../components/Formula';
import { RelatedNotes } from '../../components/RelatedNotes';
import { Rich } from '../../components/Tex';
import { StepSources, WhyIntro } from '../../components/WhyIntro';
import {
  linearLimits,
  linearSources,
  linearSteps,
  linearVerifiedDate,
  linearWhy,
} from '../../content/linear';
import { formatNumber as fmt } from '../../math/attention';
import {
  computeLinear,
  LINEAR_PRESETS,
  SECOND_MATRIX,
  type LinearPresetId,
  type Mat2,
  type Vec2,
} from '../../math/linear';
import {
  initialLinearState,
  LINEAR_STEP_COUNT,
  moveLinearStep,
  restoreLinearState,
  serializeLinearState,
  type LinearState,
} from '../../state/linear';
import { LinearPlane } from './LinearPlane';
import { localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/lesson.css';
import '../../styles/attention-diagram.css';
import '../../styles/experiment.css';
import '../../styles/editorial.css';
import '../../styles/linear.css';

const vec = (v: Vec2) => `[${fmt(v[0])}, ${fmt(v[1])}]`;

const PRESET_LIST: [LinearPresetId, string][] = [
  ['shear', T('水平剪切')],
  ['rotate90', T('旋转 90°')],
  ['stretch', T('各向缩放')],
  ['mirror', T('镜像翻转')],
  ['collapse', T('降维压扁 (det=0)')],
  ['identity', T('恒等矩阵 I')],
];

export default function LinearLab() {
  const [restored] = useState(() => restoreLinearState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.warning ?? '');
  const [playing, setPlaying] = useState(false);
  const [manualLink, setManualLink] = useState('');
  const [recallOpen, setRecallOpen] = useState(false);
  const [guess, setGuess] = useState<'both' | 'dot-only' | null>(null);
  const [checked, setChecked] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );

  const result = useMemo(() => computeLinear(state.w, state.x, state.k), [state.w, state.x, state.k]);
  const step = linearSteps[state.step];
  const hide = !state.view.explanations;

  function edit(next: LinearState) {
    setPlaying(false);
    setChecked(false);
    setNotice('');
    setManualLink('');
    setState(next);
  }
  function jump(target: number) {
    setPlaying(false);
    setState((s) => ({ ...s, step: Math.max(0, Math.min(LINEAR_STEP_COUNT - 1, target)) }));
  }
  function togglePlay() {
    if (!playing && state.step === LINEAR_STEP_COUNT - 1) setState((s) => ({ ...s, step: 0 }));
    setPlaying((p) => !p);
  }

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setState((s) => moveLinearStep(s, 1)),
      8000 / state.view.speed,
    );
    return () => window.clearInterval(timer);
  }, [playing, state.view.speed]);
  useEffect(() => {
    if (state.step === LINEAR_STEP_COUNT - 1) setPlaying(false);
  }, [state.step]);

  useEffect(() => {
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    const restore = () => {
      const next = restoreLinearState(window.location.hash);
      setPlaying(false);
      setState(next.state);
      setChecked(false);
      setRecallOpen(false);
      setManualLink('');
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
    const url = new URL(localePath('/linear-transform'), window.location.origin);
    url.hash = serializeLinearState(state);
    try {
      await navigator.clipboard.writeText(url.href);
      setManualLink('');
      setNotice(T('已复制当前实验状态的链接。'));
    } catch {
      setManualLink(url.href);
      setNotice(T('浏览器未允许剪贴板访问，请从下方选择并复制链接。'));
    }
  }

  function setMatrixCell(row: 0 | 1, col: 0 | 1, value: number) {
    const w: Mat2 = [
      [...state.w[0]],
      [...state.w[1]],
    ];
    w[row][col] = value;
    edit({ ...state, w });
  }

  function setVec(which: 'x' | 'k', index: 0 | 1, value: number) {
    const next: Vec2 = [...state[which]];
    next[index] = value;
    edit({ ...state, [which]: next });
  }

  function handleDrag(handle: 'e0' | 'e1' | 'x' | 'k', value: Vec2) {
    if (handle === 'e0') {
      edit({ ...state, w: [value, [...state.w[1]]] });
    } else if (handle === 'e1') {
      edit({ ...state, w: [[...state.w[0]], value] });
    } else {
      edit({ ...state, [handle]: value });
    }
  }

  const orientationText =
    result.orientation === 'collapsed'
      ? T('降维压扁（det = 0）')
      : result.orientation === 'flipped'
        ? T('镜像翻转（det < 0）')
        : T('保持定向（det > 0）');

  return (
    <AppShell
      active="linear"
      crumb={T('线性变换与几何')}
      indicator={T('浏览器内计算')}
      title={T('线性变换的几何意义')}
    >
      <section className="page-heading">
        <div>
          <div className="eyebrow">{Tx('原书导读 {#0} 第 8 章 {#1} 二维几何直觉', <span>·</span>, <span>·</span>)}</div>
          <h1>{Tx('线性变换：矩阵乘法{#0}', <span>{T('把空间变成了什么')}</span>)}</h1>
          <p>
            {T(
              '行向量约定下，y = xW 把矩阵的两行当作新基向量。拖动画布端点或编辑矩阵数值，实时看网格变形、行列式面积、点积投影以及两次变换的复合顺序。',
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
          <input readOnly value={manualLink} onFocus={(e) => e.target.select()} />
        </label>
      )}

      <WhyIntro why={linearWhy} />

      <section className="lesson-panel lin-panel" aria-label={T('二维线性变换逐步实验')}>
        <div className="lesson-panel-heading">
          <div>
            <span className="lesson-eyebrow">{T('动手验证：改矩阵的一行，整个网格怎样一起变')}</span>
            <h2>{T('先看基向量和网格，再看行列式、点积投影与连续变换。')}</h2>
          </div>
          <span className="lesson-progress">
            {state.step + 1} / {LINEAR_STEP_COUNT}
          </span>
        </div>
        <div className="lesson-steps lin-steps" aria-label={T('线性变换步骤')}>
          {linearSteps.map((item, i) => (
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

        <div className="lin-inputs">
          <div className="p1-presets" role="group" aria-label={T('变换预设')}>
            <span>{T('预设矩阵')}</span>
            {PRESET_LIST.map(([id, label]) => (
              <button
                key={id}
                onClick={() => {
                  const p = LINEAR_PRESETS[id];
                  edit({
                    ...state,
                    w: [
                      [...p.w[0]],
                      [...p.w[1]],
                    ],
                    x: [...p.x],
                    k: [...p.k],
                  });
                }}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="lin-sliders">
            <label>
              e₀&apos; · w₀₀
              <input
                type="range"
                aria-label="w00"
                min={-2}
                max={2}
                step={0.25}
                value={state.w[0][0]}
                onChange={(e) => setMatrixCell(0, 0, Number(e.target.value))}
              />
              <output>{fmt(state.w[0][0])}</output>
            </label>
            <label>
              e₀&apos; · w₀₁
              <input
                type="range"
                aria-label="w01"
                min={-2}
                max={2}
                step={0.25}
                value={state.w[0][1]}
                onChange={(e) => setMatrixCell(0, 1, Number(e.target.value))}
              />
              <output>{fmt(state.w[0][1])}</output>
            </label>
            <label>
              e₁&apos; · w₁₀
              <input
                type="range"
                aria-label="w10"
                min={-2}
                max={2}
                step={0.25}
                value={state.w[1][0]}
                onChange={(e) => setMatrixCell(1, 0, Number(e.target.value))}
              />
              <output>{fmt(state.w[1][0])}</output>
            </label>
            <label>
              e₁&apos; · w₁₁
              <input
                type="range"
                aria-label="w11"
                min={-2}
                max={2}
                step={0.25}
                value={state.w[1][1]}
                onChange={(e) => setMatrixCell(1, 1, Number(e.target.value))}
              />
              <output>{fmt(state.w[1][1])}</output>
            </label>
            <label>
              x₀
              <input
                type="range"
                aria-label="x0"
                min={-2}
                max={2}
                step={0.25}
                value={state.x[0]}
                onChange={(e) => setVec('x', 0, Number(e.target.value))}
              />
              <output>{fmt(state.x[0])}</output>
            </label>
            <label>
              x₁
              <input
                type="range"
                aria-label="x1"
                min={-2}
                max={2}
                step={0.25}
                value={state.x[1]}
                onChange={(e) => setVec('x', 1, Number(e.target.value))}
              />
              <output>{fmt(state.x[1])}</output>
            </label>
            {state.step === 2 && (
              <>
                <label>
                  k₀
                  <input
                    type="range"
                    aria-label="k0"
                    min={-2}
                    max={2}
                    step={0.25}
                    value={state.k[0]}
                    onChange={(e) => setVec('k', 0, Number(e.target.value))}
                  />
                  <output>{fmt(state.k[0])}</output>
                </label>
                <label>
                  k₁
                  <input
                    type="range"
                    aria-label="k1"
                    min={-2}
                    max={2}
                    step={0.25}
                    value={state.k[1]}
                    onChange={(e) => setVec('k', 1, Number(e.target.value))}
                  />
                  <output>{fmt(state.k[1])}</output>
                </label>
              </>
            )}
          </div>
        </div>

        <div className="lesson-columns lin-columns">
          <div className="diagram-panel">
            <LinearPlane step={state.step} result={result} onDrag={handleDrag} />
          </div>
          <div className="lesson-reading">
            <div className="lesson-reading-top">
              <span>{T('这一步')}</span>
              <button
                onClick={() =>
                  edit({ ...state, view: { ...state.view, explanations: !state.view.explanations } })
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
                <Rich text={step.explanation} />
              </p>
            )}

            <div className="exp-widget">
              {state.step === 0 && (
                <div className="exp-table-wrap">
                  <table className="exp-table" aria-label={T('基向量与输出核对')}>
                    <caption>{T('行向量乘法 y = x₀·e₀\' + x₁·e₁\'')}</caption>
                    <thead>
                      <tr>
                        <th scope="col">{T('量')}</th>
                        <th scope="col">d0</th>
                        <th scope="col">d1</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th scope="row">x</th>
                        <td>{fmt(state.x[0])}</td>
                        <td>{fmt(state.x[1])}</td>
                      </tr>
                      <tr>
                        <th scope="row">e₀&apos; = W[0,:]</th>
                        <td>{fmt(result.e0[0])}</td>
                        <td>{fmt(result.e0[1])}</td>
                      </tr>
                      <tr>
                        <th scope="row">e₁&apos; = W[1,:]</th>
                        <td>{fmt(result.e1[0])}</td>
                        <td>{fmt(result.e1[1])}</td>
                      </tr>
                      <tr className="exp-total">
                        <th scope="row">y = xW</th>
                        <td>{fmt(result.y[0])}</td>
                        <td>{fmt(result.y[1])}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}

              {state.step === 1 && (
                <>
                  <div className="lin-metrics">
                    <div>
                      <span>{T('行列式 det(W)')}</span>
                      <strong>{fmt(result.det, 4)}</strong>
                    </div>
                    <div>
                      <span>{T('面积倍率 |det(W)|')}</span>
                      <strong>{fmt(result.area, 4)}</strong>
                    </div>
                    <div>
                      <span>{T('定向状态')}</span>
                      <strong>{orientationText}</strong>
                    </div>
                  </div>
                  <p className="exp-verdict">
                    {Tx(
                      'w₀₀·w₁₁ − w₀₁·w₁₀ = {#0}×{#1} − {#2}×{#3} = {#4}。',
                      fmt(state.w[0][0]),
                      fmt(state.w[1][1]),
                      fmt(state.w[0][1]),
                      fmt(state.w[1][0]),
                      fmt(result.det, 4),
                    )}
                  </p>
                </>
              )}

              {state.step === 2 && (
                <>
                  <div className="lin-metrics">
                    <div>
                      <span>{T('代数点积 q·k')}</span>
                      <strong>{fmt(result.dotInfo.dot, 4)}</strong>
                    </div>
                    <div>
                      <span>{T('几何 ‖q‖‖k‖cosθ')}</span>
                      <strong>{fmt(result.dotInfo.geometricDot, 4)}</strong>
                    </div>
                    <div>
                      <span>{T('余弦相似度 cosθ')}</span>
                      <strong>
                        {result.dotInfo.cosine === null ? T('未定义') : fmt(result.dotInfo.cosine, 4)}
                      </strong>
                    </div>
                    <div>
                      <span>{T('夹角 θ')}</span>
                      <strong>
                        {result.dotInfo.angleDeg === null
                          ? T('未定义')
                          : `${fmt(result.dotInfo.angleDeg, 2)}°`}
                      </strong>
                    </div>
                  </div>
                  <div className="exp-experiment">
                    <strong>{T('先预测：只把 k 拉长到 2k，方向不变')}</strong>
                    <Prediction
                      question={T('点积 q·(2k) 与余弦相似度 cosθ 会怎样变化？')}
                      options={[
                        { value: 'dot-only', label: T('点积变 2 倍，余弦不变') },
                        { value: 'both', label: T('两者一起变 2 倍') },
                      ]}
                      choice={guess}
                      onChoose={(v) => {
                        setGuess(v);
                        setChecked(false);
                      }}
                    />
                    <button
                      className="button secondary"
                      disabled={!guess}
                      onClick={() => setChecked(true)}
                    >
                      {T('计算对照 2k →')}
                    </button>
                    {checked && (
                      <p className="exp-verdict ok" role="status">
                        {result.dotInfo.cosine === null
                          ? T('当前 q 或 k 为零向量，余弦未定义；请调出非零向量再核对。')
                          : Tx(
                              'q·k = {#0} → q·(2k) = {#1}；cosθ 保持 {#2}。',
                              fmt(result.dotInfo.dot, 4),
                              fmt(result.doubledDotInfo.dot, 4),
                              fmt(result.dotInfo.cosine, 4),
                            )}
                        {result.dotInfo.cosine !== null && (
                          <PredictionResult
                            label={
                              guess === 'dot-only' ? T('点积变 2 倍，余弦不变') : T('两者一起变 2 倍')
                            }
                            holds={guess === 'dot-only'}
                          />
                        )}
                      </p>
                    )}
                  </div>
                </>
              )}

              {state.step === 3 && (
                <>
                  <div className="exp-table-wrap">
                    <table className="exp-table" aria-label={T('两次变换与复合矩阵核对')}>
                      <caption>
                        {Tx(
                          '固定 W₂ = [[{#0}, {#1}], [{#2}, {#3}]]',
                          fmt(SECOND_MATRIX[0][0]),
                          fmt(SECOND_MATRIX[0][1]),
                          fmt(SECOND_MATRIX[1][0]),
                          fmt(SECOND_MATRIX[1][1]),
                        )}
                      </caption>
                      <thead>
                        <tr>
                          <th scope="col">{T('计算路径')}</th>
                          <th scope="col">{T('向量结果')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <th scope="row">xW₁</th>
                          <td>{vec(result.step1)}</td>
                        </tr>
                        <tr>
                          <th scope="row">(xW₁)W₂</th>
                          <td>{vec(result.step2)}</td>
                        </tr>
                        <tr className="exp-changed">
                          <th scope="row">x(W₁W₂)</th>
                          <td>{vec(result.composedY)}</td>
                        </tr>
                        <tr>
                          <th scope="row">{T('反序 x(W₂W₁)')}</th>
                          <td>{vec(result.reversedY)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <p className="exp-verdict ok">
                    {Tx(
                      '结合律核对 |(xW₁)W₂ − x(W₁W₂)| = {#0}（容差 1e-12）；交换顺序最大矩阵差 ‖W₁W₂ − W₂W₁‖ = {#1}。',
                      result.composeError === 0 ? '0' : result.composeError.toExponential(1),
                      fmt(result.commuteGap, 4),
                    )}
                  </p>
                </>
              )}

              {state.step === 4 && (
                <div className="lin-links">
                  <p className="exp-verdict">
                    {Tx(
                      '当前 x = {#0} 经 W 映到 y = {#1}（det = {#2}）。继续进入下游实验看线性变换在 Transformer 中的真实角色：',
                      vec(state.x),
                      vec(result.y),
                      fmt(result.det, 3),
                    )}
                  </p>
                  <p>
                    <a href={localePath('/attention')}>{T('前往 Attention：Q/K/V 投影与点积打分 →')}</a>
                  </p>
                  <p>
                    <a href={localePath('/ffn-residual')}>{T('前往 FFN 与残差：看非线性为何打破线性合并 →')}</a>
                  </p>
                </div>
              )}
            </div>

            <div className="lesson-takeaway">
              <strong>{T('这一步，记住一句话')}</strong>
              <p>
                <Rich text={hide ? T('先自己复述，再点击“显示说明”核对。') : step.takeaway} />
              </p>
            </div>
            <details className="lesson-formula">
              <summary>{T('公式与符号说明')}</summary>
              <Formula expression={step.formula} symbols={step.symbols} />
            </details>
            <StepSources book={step.book} />
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
              disabled={state.step === LINEAR_STEP_COUNT - 1}
              onClick={() => jump(state.step + 1)}
            >
              {state.step === LINEAR_STEP_COUNT - 1
                ? T('已到最后一步')
                : T('下一步：{#0} →', linearSteps[state.step + 1].name)}
            </button>
          </div>
          <div>
            <button className="lesson-text-button" onClick={() => jump(0)}>
              {T('从头看')}
            </button>
            <button
              className="lesson-text-button"
              onClick={() => {
                setPlaying(false);
                setChecked(false);
                setState({ ...initialLinearState(), view: state.view });
                setNotice(T('已恢复默认示例。'));
              }}
            >
              {T('恢复默认示例')}
            </button>
            <button
              className="lesson-text-button"
              aria-label={playing ? T('暂停自动逐步') : T('播放自动逐步')}
              onClick={togglePlay}
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
            {T('已遵循减少动态效果偏好；网格与向量直接更新，不做过渡动画。')}
          </p>
        )}
      </section>

      <section className="recall-card">
        <div>
          <span className="section-kicker">{T('离开之前 · 试着复述')}</span>
          <h2>{T('不看画布，说出矩阵乘法在几何上做了什么。')}</h2>
          <p>
            {T(
              '行向量约定下 W 的两行代表什么？det(W)=0 意味着什么？点积和余弦相似度差在哪里？为什么两次线性变换不能替代非线性？',
            )}
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
          {recallOpen ? T('收起参考解释') : T('查看参考解释')}
        </button>
        {recallOpen && (
          <p className="recall-answer">
            {T(
              '行向量约定 y = xW 中，W 的两行分别是标准基向量 e₀、e₁ 变换后的位置，任意输入 x 的输出就是这两行的加权和。行列式 det(W) 决定单位面积的缩放倍率与是否翻转定向，det = 0 表示空间被压扁降维。点积等于 ‖q‖‖k‖cosθ，同时依赖长度与方向，把向量拉长会改变点积但不改变余弦。连续两次线性变换满足 (xW₁)W₂ = x(W₁W₂)，合起来仍是单个矩阵，因此网络层间需要非线性激活。',
            )}
          </p>
        )}
      </section>

      <RelatedNotes path="/attention" />

      <details className="details-panel source-panel">
        <summary>
          <span>{T('适用条件与来源')}</span>
          <span className="summary-hint">{T('行向量约定 · 简化边界 · 参考文献')}</span>
        </summary>
        <div className="details-body">
          <ul>
            {linearLimits.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <ul>
            {linearSources.map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.title} ↗
                </a>
              </li>
            ))}
          </ul>
          <p className="subtle">
            {Tx('来源核对日期：{#0}。图形与数值实现为本项目编写。', linearVerifiedDate)}
          </p>
        </div>
      </details>
    </AppShell>
  );
}
