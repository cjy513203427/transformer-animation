import { useEffect, useMemo, useRef, useState } from 'react';
import { Rich, Tex } from '../../components/Tex';
import { formatNumber as fmt } from '../../math/attention';
import {
  addVectors,
  alibiAttention,
  alibiSlopes,
  applyRope,
  BOOK_L_EMBEDDING,
  crossTerms,
  identityAttention,
  norm,
  ROPE_K,
  ROPE_Q,
  ropeScore,
  ropeScoreParts,
  ropeTheta,
  sinusoidalFrequency,
  sinusoidalPE,
  TOY_EMBEDDINGS,
  truncate2,
} from '../../math/position';
import {
  ALIBI_HEADS,
  ALIBI_TOKENS,
  D_MODEL_OPTIONS,
  MAX_POS,
  ROPE_MAX,
  type DModel,
  type PositionState,
} from '../../state/position';
import { INSERTION_COLORS } from './PositionPipeline';

export interface WidgetProps {
  state: PositionState;
  onChange: (next: PositionState) => void;
}
interface PlayProps {
  posPlaying: boolean;
  onTogglePosPlay: () => void;
  reducedMotion: boolean;
}

const vector = (values: number[], digits = 3) =>
  `[${values.map((value) => fmt(value, digits)).join(', ')}]`;
const degrees = (radians: number) => (radians * 180) / Math.PI;

/** Animate rotations only for ±1 position steps; jumps snap so the hand never spins wildly. */
function useSmallStep(value: number) {
  const previous = useRef(value);
  const small = Math.abs(value - previous.current) === 1;
  useEffect(() => {
    previous.current = value;
  }, [value]);
  return small;
}

function PlayButton({
  playing,
  onToggle,
  label,
}: {
  playing: boolean;
  onToggle: () => void;
  label: string;
}) {
  return (
    <button
      className="pos-play"
      aria-label={playing ? `暂停${label}` : `播放${label}`}
      onClick={onToggle}
    >
      {playing ? 'Ⅱ 暂停' : `▶ ${label}`}
    </button>
  );
}

// ------------------------------------------------------------------ 1. 位置盲

export function BlindnessWidget({ state, onChange }: WidgetProps) {
  const { withPE, pair } = state.blind;
  const causal = pair === 'causal';
  const sentences: { words: ('我' | '爱' | '你')[]; focus: number }[] = causal
    ? [
        { words: ['我', '爱', '你'], focus: 2 },
        { words: ['爱', '我', '你'], focus: 2 },
      ]
    : [
        { words: ['我', '爱', '你'], focus: 0 },
        { words: ['你', '爱', '我'], focus: 2 },
      ];
  const runs = sentences.map(({ words, focus }) => {
    const x = words.map((word, pos) =>
      withPE ? addVectors(TOY_EMBEDDINGS[word], sinusoidalPE(pos, 4)) : [...TOY_EMBEDDINGS[word]],
    );
    const result = identityAttention(x, causal);
    return { words, focus, x, weights: result.probabilities[focus], output: result.output[focus] };
  });
  const difference = Math.max(
    ...runs[0].output.map((value, d) => Math.abs(value - runs[1].output[d])),
  );
  const same = difference < 1e-12;
  const focusWord = sentences[0].words[sentences[0].focus];
  const set = (patch: Partial<PositionState['blind']>) =>
    onChange({ ...state, blind: { ...state.blind, ...patch } });
  return (
    <div className="pos-widget">
      <div className="pos-controls">
        <div className="pos-segment" role="group" aria-label="选择对照的两句话">
          <button aria-pressed={!causal} onClick={() => set({ pair: 'book' })}>
            书中例子：我爱你 / 你爱我
          </button>
          <button aria-pressed={causal} onClick={() => set({ pair: 'causal' })}>
            加因果 Mask：我爱你 / 爱我你
          </button>
        </div>
        <label className="pos-switch">
          <input
            type="checkbox"
            checked={withPE}
            aria-label="加入正弦位置编码（d_model = 4）"
            onChange={(e) => set({ withPE: e.target.checked })}
          />
          加入正弦位置编码（
          <Tex tex="d_{\mathrm{model}} = 4" />）
        </label>
      </div>
      <div className="pos-sentences">
        {runs.map((run, s) => (
          <div className="pos-sentence" key={s}>
            <div className="pos-token-row" aria-label={`第 ${s + 1} 句：${run.words.join('')}`}>
              {run.words.map((word, pos) => (
                <span key={pos} className={pos === run.focus ? 'focus' : ''}>
                  <small>pos {pos}</small>
                  {word}
                </span>
              ))}
            </div>
            <dl>
              <div>
                <dt>“{focusWord}”的输入 x</dt>
                <dd>{vector(run.x[run.focus])}</dd>
              </div>
              <div>
                <dt>对各位置的权重</dt>
                <dd>
                  {run.weights
                    .map((p, j) => `${run.words[j]} ${(p * 100).toFixed(1)}%`)
                    .join(' · ')}
                </dd>
              </div>
              <div>
                <dt>输出 O</dt>
                <dd>
                  <strong>{vector(run.output)}</strong>
                </dd>
              </div>
            </dl>
          </div>
        ))}
      </div>
      <p className={`pos-verdict ${same ? 'same' : 'different'}`} role="status">
        {same
          ? `✓ 两句话里“${focusWord}”的输出完全相同（最大差 < 10⁻¹²）。`
          : `✗ 两句话里“${focusWord}”的输出不同（最大差 ${fmt(difference)}）。`}
      </p>
      <p className="pos-note">
        {withPE
          ? '加上位置向量后，同一个词在不同位置有不同的 x，打分和输出都随之改变。'
          : causal
            ? '因果 Mask 规定了每个位置能看见谁；但两句里的“你”都看见 {我、爱、你} 这同一组向量，先后顺序依然分不出来。'
            : '没有位置编码时，“我”在两句话里的向量一模一样，看到的也是同一组向量（只是排列不同），所以结果完全相同。'}
        为了只看位置的作用，这里令 <Tex tex="W_Q = W_K = W_V = I" />
        ，并且不加 Mask（除非选了因果对照）。
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ 2. 正弦位置表

/** Fastest pair, the book table's second pair, a moderate pair (ω = 0.1), a slow one and the slowest. */
export function clockPairs(dModel: number) {
  const pairs = dModel >= 16 ? [0, 1, dModel / 8, dModel / 4, dModel / 2 - 1] : [0, 1];
  return [...new Set(pairs)].filter((i) => Number.isInteger(i) && i >= 0 && i < dModel / 2);
}
/** One cycle over positions 0–63 reads like the book's wave figure; tiny d_model falls back to pair 1. */
export function defaultWavePair(dModel: number) {
  return dModel >= 16 ? dModel / 8 : 1;
}

function Clock({
  index,
  dModel,
  pos,
  selected,
  animate,
  onSelect,
}: {
  index: number;
  dModel: number;
  pos: number;
  selected: boolean;
  animate: boolean;
  onSelect: () => void;
}) {
  const omega = sinusoidalFrequency(index, dModel);
  const angle = degrees(pos * omega);
  const pe = sinusoidalPE(pos, dModel);
  return (
    <button
      className={`pos-clock ${selected ? 'selected' : ''}`}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <svg viewBox="-48 -48 96 96" aria-hidden="true">
        <circle r="38" className="pos-clock-face" />
        <line x1="0" y1="-38" x2="0" y2="-32" className="pos-clock-tick" />
        <g
          style={{ transform: `rotate(${angle}deg)`, transition: animate ? undefined : 'none' }}
          className="pos-clock-hand"
        >
          <line x1="0" y1="0" x2="0" y2="-34" />
          <circle cx="0" cy="-34" r="4" />
        </g>
        <circle r="3" className="pos-clock-center" />
      </svg>
      <strong>
        第 {index} 对 · d{2 * index}/d{2 * index + 1}
      </strong>
      <span>
        <Tex tex="\omega =" /> {omega < 0.001 ? omega.toExponential(2) : fmt(omega, 4)}，每步{' '}
        {fmt(degrees(omega), 2)}°
      </span>
      <code>
        ({fmt(pe[2 * index])}, {fmt(pe[2 * index + 1])})
      </code>
    </button>
  );
}

function Wave({ index, dModel, pos }: { index: number; dModel: number; pos: number }) {
  const width = 360;
  const height = 130;
  const x = (p: number) => 28 + (p / MAX_POS) * (width - 40);
  const y = (v: number) => 12 + ((1 - v) / 2) * (height - 34);
  const omega = sinusoidalFrequency(index, dModel);
  const curve = (fn: (a: number) => number) =>
    Array.from({ length: 253 }, (_, k) => {
      const p = (k / 252) * MAX_POS;
      return `${k ? 'L' : 'M'}${x(p).toFixed(1)} ${y(fn(p * omega)).toFixed(1)}`;
    }).join(' ');
  return (
    <svg
      className="pos-wave"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`第 ${index} 对维度随位置变化的 sin、cos 曲线，当前位置 ${pos}`}
    >
      {[-1, 0, 1].map((v) => (
        <g key={v}>
          <line x1={x(0)} x2={x(MAX_POS)} y1={y(v)} y2={y(v)} className="pos-grid" />
          <text x="4" y={y(v) + 4} className="pos-axis">
            {v}
          </text>
        </g>
      ))}
      {[0, 16, 32, 48, 63].map((p) => (
        <text key={p} x={x(p)} y={height - 4} textAnchor="middle" className="pos-axis">
          {p}
        </text>
      ))}
      <path d={curve(Math.sin)} className="pos-curve sin" />
      <path d={curve(Math.cos)} className="pos-curve cos" />
      {Array.from({ length: MAX_POS + 1 }, (_, p) => (
        <g key={p}>
          <circle
            cx={x(p)}
            cy={y(Math.sin(p * omega))}
            r={p === pos ? 4.5 : 1.6}
            className="pos-dot sin"
          />
          <circle
            cx={x(p)}
            cy={y(Math.cos(p * omega))}
            r={p === pos ? 4.5 : 1.6}
            className="pos-dot cos"
          />
        </g>
      ))}
      <line x1={x(pos)} x2={x(pos)} y1={y(1) - 4} y2={y(-1) + 4} className="pos-marker" />
    </svg>
  );
}

export function SinusoidalWidget({
  state,
  onChange,
  posPlaying,
  onTogglePosPlay,
  reducedMotion,
}: WidgetProps & PlayProps) {
  const { dModel, pos, pair } = state.sin;
  const animate = useSmallStep(pos) && !reducedMotion;
  const rows = [...new Set([0, 1, 2, 3, 4, 5, pos])].sort((a, b) => a - b);
  const pairs = clockPairs(dModel);
  const selectedPair = pair < dModel / 2 ? pair : 0;
  const set = (patch: Partial<PositionState['sin']>) =>
    onChange({ ...state, sin: { ...state.sin, ...patch } });
  return (
    <div className="pos-widget">
      <div className="pos-controls">
        <div className="pos-segment" role="group" aria-label="d_model">
          <span>
            <Tex tex="d_{\mathrm{model}}" />
          </span>
          {D_MODEL_OPTIONS.map((option) => (
            <button
              key={option}
              aria-pressed={dModel === option}
              onClick={() => set({ dModel: option as DModel, pair: defaultWavePair(option) })}
            >
              {option}
            </button>
          ))}
        </div>
        <label className="pos-range">
          位置 pos = <output>{pos}</output>
          <input
            type="range"
            min="0"
            max={MAX_POS}
            value={pos}
            aria-label="位置 pos"
            onChange={(e) => set({ pos: Number(e.target.value) })}
          />
        </label>
        <PlayButton playing={posPlaying} onToggle={onTogglePosPlay} label="位置逐个 +1" />
      </div>
      <div
        className="pos-clocks"
        role="group"
        aria-label="每一对维度像一根指针，点击选择要画波形的一对"
      >
        {pairs.map((i) => (
          <Clock
            key={i}
            index={i}
            dModel={dModel}
            pos={pos}
            selected={i === selectedPair}
            animate={animate}
            onSelect={() => set({ pair: i })}
          />
        ))}
      </div>
      <p className="pos-caption">
        每对 <Tex tex="(\sin,\ \cos)" /> 是单位圆上的一个点：pos 每加 1，第 <Tex tex="i" /> 对转{' '}
        <Tex tex="\omega_i" /> 弧度。指针转动只是为了看清方向；编码只在整数位置取值。
      </p>
      <Wave index={selectedPair} dModel={dModel} pos={pos} />
      <p className="pos-caption">
        第 {selectedPair} 对的 sin（实线）与
        cos（虚线）：曲线是函数本身，圆点才是各整数位置真正使用的数值。
      </p>
      <div className="pos-table-wrap">
        <table className="pos-table" aria-label="正弦位置编码表，前 4 维">
          <thead>
            <tr>
              <th>pos</th>
              <th>d0 sin</th>
              <th>d1 cos</th>
              <th>d2 sin</th>
              <th>d3 cos</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p} className={p === pos ? 'current' : ''}>
                <th scope="row">{p}</th>
                {sinusoidalPE(p, dModel)
                  .slice(0, 4)
                  .map((v, j) => (
                    <td key={j}>{fmt(v)}</td>
                  ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {dModel === 64 ? (
        <p className="pos-book">
          这正是书中第 5 章“LLM张老师”表格的设置：
          <Tex tex="d_{\mathrm{model}} = 64" />
          ，表中只列前 4 维。书中把数值截断到两位小数
          {pos <= 5 &&
            `，pos = ${pos} 那一行印作 ${sinusoidalPE(pos, 64)
              .slice(0, 4)
              .map((v) => truncate2(v).toFixed(2))
              .join('、')}`}
          。
        </p>
      ) : (
        <p className="pos-book">
          <Tex tex={`d_{\\mathrm{model}} = ${dModel}`} /> 时，第 1 对的频率是{' '}
          {fmt(sinusoidalFrequency(1, dModel), 4)}，所以 d2、d3 与书中表格（
          <Tex tex="d_{\mathrm{model}} = 64" />
          ）不同：频率由 <Tex tex="2i / d_{\mathrm{model}}" /> 决定。
        </p>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ 3. 相加

export function AdditionWidget({ state, onChange }: WidgetProps) {
  const { dModel, pos } = state.sin;
  const pe0 = sinusoidalPE(0, dModel).slice(0, 4);
  const pe = sinusoidalPE(pos, dModel).slice(0, 4);
  const x0 = addVectors(BOOK_L_EMBEDDING, pe0);
  const xp = addVectors(BOOK_L_EMBEDDING, pe);
  const same = x0.every((v, j) => Math.abs(v - xp[j]) < 1e-12);
  const terms = crossTerms(
    TOY_EMBEDDINGS.我,
    sinusoidalPE(0, 4),
    TOY_EMBEDDINGS.你,
    sinusoidalPE(2, 4),
  );
  // Plane of dimensions d0, d1 (pair 0 turns 1 radian per position for every d_model).
  const scale = 58;
  const px = (v: number) => 64 + v * scale;
  const py = (v: number) => 92 - v * scale;
  const [ex, ey] = BOOK_L_EMBEDDING;
  return (
    <div className="pos-widget">
      <div className="pos-controls">
        <label className="pos-range">
          第二个 L 的位置 pos = <output>{pos}</output>
          <input
            type="range"
            min="0"
            max={MAX_POS}
            value={pos}
            aria-label="第二个 L 的位置"
            onChange={(e) =>
              onChange({ ...state, sin: { ...state.sin, pos: Number(e.target.value) } })
            }
          />
        </label>
        <span className="pos-chip">
          <Tex tex={`d_{\\mathrm{model}} = ${dModel}`} />
          ，显示前 4 维
        </span>
      </div>
      <div className="pos-add-layout">
        <div className="pos-table-wrap">
          <table className="pos-table" aria-label="嵌入加位置编码">
            <thead>
              <tr>
                <th>行</th>
                <th>d0</th>
                <th>d1</th>
                <th>d2</th>
                <th>d3</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">E[L]（两个 L 相同）</th>
                {BOOK_L_EMBEDDING.map((v, j) => (
                  <td key={j}>{fmt(v)}</td>
                ))}
              </tr>
              <tr>
                <th scope="row">+ PE[0]</th>
                {pe0.map((v, j) => (
                  <td key={j}>{fmt(v)}</td>
                ))}
              </tr>
              <tr className="current">
                <th scope="row">= X（第一个 L）</th>
                {x0.map((v, j) => (
                  <td key={j}>{fmt(v)}</td>
                ))}
              </tr>
              <tr>
                <th scope="row">+ PE[{pos}]</th>
                {pe.map((v, j) => (
                  <td key={j}>{fmt(v)}</td>
                ))}
              </tr>
              <tr className="current">
                <th scope="row">= X（第二个 L）</th>
                {xp.map((v, j) => (
                  <td key={j}>{fmt(v)}</td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
        <svg
          className="pos-plane"
          viewBox="0 0 230 200"
          role="img"
          aria-label={`d0、d1 平面：E 加上位置 ${pos} 的向量`}
        >
          <line x1="0" x2="230" y1={py(0)} y2={py(0)} className="pos-grid" />
          <line y1="0" y2="200" x1={px(0)} x2={px(0)} className="pos-grid" />
          <circle cx={px(ex)} cy={py(ey)} r={scale} className="pos-orbit" />
          <line x1={px(0)} y1={py(0)} x2={px(ex)} y2={py(ey)} className="pos-arrow content" />
          <line
            x1={px(ex)}
            y1={py(ey)}
            x2={px(xp[0])}
            y2={py(xp[1])}
            className="pos-arrow position"
          />
          <line x1={px(0)} y1={py(0)} x2={px(xp[0])} y2={py(xp[1])} className="pos-arrow sum" />
          <circle cx={px(x0[0])} cy={py(x0[1])} r="4" className="pos-ghost" />
          <circle cx={px(xp[0])} cy={py(xp[1])} r="5" className="pos-tip" />
          <text x={px(ex) + 6} y={py(ey) + 14} className="pos-axis">
            E
          </text>
          <text x={px(xp[0]) + 7} y={py(xp[1]) - 6} className="pos-axis">
            X
          </text>
          <text x="226" y={py(0) - 4} textAnchor="end" className="pos-axis">
            d0
          </text>
          <text x={px(0) + 4} y="11" className="pos-axis">
            d1
          </text>
        </svg>
      </div>
      <p className={`pos-verdict ${same ? 'same' : 'different'}`} role="status">
        {same
          ? '两个 L 在同一位置：嵌入与位置都相同，X 也完全相同。'
          : `嵌入相同、位置不同 → 两个 L 的 X 不同。pos = 0、1 时就是书中“L(位置1)、L(位置2)”两行。`}
      </p>
      <p className="pos-note">
        右图只画 d0、d1 两维（第 0 对，每个位置转 1 弧度）：位置向量是一根长度为 1 的箭头，接在 E
        的末端绕虚线圆转动，所以 X 的长度和方向都会变。空心点是第一个 L。
        加法发生在投影之前，Q、K、V 三条支路都带上了位置，V 也不例外。
      </p>
      <details className="pos-details">
        <summary>展开：为什么书中说加法让内容和位置“混在一起”</summary>
        <p>
          取 <Tex tex="d_{\mathrm{model}} = 4" /> 的“我”（pos 0）与“你”（pos 2），令{' '}
          <Tex tex="W_Q = W_K = I" />
          ，它们的点积拆成四项（第 25.2.2 节）：
        </p>
        <table className="pos-table" aria-label="加法位置编码点积的四项">
          <tbody>
            <tr>
              <th scope="row">内容·内容 e·e</th>
              <td>{fmt(terms.contentContent)}</td>
            </tr>
            <tr>
              <th scope="row">内容·位置 e·p</th>
              <td>{fmt(terms.contentPosition)}</td>
            </tr>
            <tr>
              <th scope="row">位置·内容 p·e</th>
              <td>{fmt(terms.positionContent)}</td>
            </tr>
            <tr>
              <th scope="row">位置·位置 p·p</th>
              <td>{fmt(terms.positionPosition)}</td>
            </tr>
            <tr className="current">
              <th scope="row">合计 x_m·x_n</th>
              <td>{fmt(terms.total)}</td>
            </tr>
          </tbody>
        </table>
        <p>
          四项里有三项含位置；真实模型中 <Tex tex="W_Q" />、<Tex tex="W_K" />{' '}
          不是单位矩阵，各项还会被投影混合。
        </p>
      </details>
    </div>
  );
}

// ------------------------------------------------------------------ 4. RoPE 旋转

function RotationPlane({
  index,
  m,
  n,
  animateM,
  animateN,
}: {
  index: number;
  m: number;
  n: number;
  animateM: boolean;
  animateN: boolean;
}) {
  const theta = ropeTheta(index, 8);
  const scale = 26;
  const q: [number, number] = [ROPE_Q[2 * index], ROPE_Q[2 * index + 1]];
  const k: [number, number] = [ROPE_K[2 * index], ROPE_K[2 * index + 1]];
  const part = ropeScoreParts(ROPE_Q, ROPE_K, m, n)[index];
  // SVG y points down, so a counter-clockwise math rotation α is rotate(−α) here.
  const arrow = (v: [number, number], cls: string) => (
    <g className={cls}>
      <line x1="0" y1="0" x2={v[0] * scale} y2={-v[1] * scale} />
      <circle cx={v[0] * scale} cy={-v[1] * scale} r="3.5" />
    </g>
  );
  return (
    <figure className="pos-rot">
      <svg
        viewBox="-50 -50 100 100"
        role="img"
        aria-label={`第 ${index} 对：q 转 ${fmt(degrees(m * theta), 1)} 度，k 转 ${fmt(degrees(n * theta), 1)} 度`}
      >
        <circle r={scale * Math.SQRT2} className="pos-orbit" />
        <line x1="-48" x2="48" y1="0" y2="0" className="pos-grid" />
        <line y1="-48" y2="48" x1="0" x2="0" className="pos-grid" />
        {arrow(q, 'pos-vec query ghost')}
        {arrow(k, 'pos-vec key ghost')}
        <g
          style={{
            transform: `rotate(${-degrees(m * theta)}deg)`,
            transition: animateM ? undefined : 'none',
          }}
          className="pos-turn"
        >
          {arrow(q, 'pos-vec query')}
        </g>
        <g
          style={{
            transform: `rotate(${-degrees(n * theta)}deg)`,
            transition: animateN ? undefined : 'none',
          }}
          className="pos-turn"
        >
          {arrow(k, 'pos-vec key')}
        </g>
      </svg>
      <figcaption>
        <strong>
          第 {index} 对 <Tex tex={`\\theta_{${index}} =`} /> {fmt(theta, 4)}
        </strong>
        <span>
          q 转 {fmt(degrees(m * theta), 1)}°，k 转 {fmt(degrees(n * theta), 1)}°
        </span>
        <code>q′·k′ = {fmt(part)}</code>
      </figcaption>
    </figure>
  );
}

export function RopeWidget({
  state,
  onChange,
  posPlaying,
  onTogglePosPlay,
  reducedMotion,
}: WidgetProps & PlayProps) {
  const { m, n } = state.rope;
  const animateM = useSmallStep(m) && !reducedMotion;
  const animateN = useSmallStep(n) && !reducedMotion;
  const qr = applyRope(ROPE_Q, m);
  const kr = applyRope(ROPE_K, n);
  const score = ropeScore(ROPE_Q, ROPE_K, m, n);
  const set = (patch: Partial<PositionState['rope']>) =>
    onChange({ ...state, rope: { ...state.rope, ...patch } });
  return (
    <div className="pos-widget">
      <div className="pos-controls">
        <label className="pos-range">
          q 的位置 m = <output>{m}</output>
          <input
            type="range"
            min="0"
            max={ROPE_MAX}
            value={m}
            aria-label="q 的位置 m"
            onChange={(e) => set({ m: Number(e.target.value) })}
          />
        </label>
        <label className="pos-range">
          k 的位置 n = <output>{n}</output>
          <input
            type="range"
            min="0"
            max={ROPE_MAX}
            value={n}
            aria-label="k 的位置 n"
            onChange={(e) => set({ n: Number(e.target.value) })}
          />
        </label>
        <PlayButton playing={posPlaying} onToggle={onTogglePosPlay} label="m 逐个 +1" />
      </div>
      <div className="pos-rot-grid">
        {[0, 1, 2, 3].map((i) => (
          <RotationPlane key={i} index={i} m={m} n={n} animateM={animateM} animateN={animateN} />
        ))}
      </div>
      <p className="pos-caption">
        虚线是旋转前的内容向量（固定不变），实线是旋转后。
        <Tex tex="d = 8" /> 分成 4 对， <Tex tex="\theta" /> 依次为 1、0.1、0.01、0.001：第 0
        对每个位置转 57.3°，第 3 对只转 0.057°。
      </p>
      <div className="pos-table-wrap">
        <table className="pos-table" aria-label="RoPE 旋转前后的 q 与 k">
          <thead>
            <tr>
              <th>向量（d = 8）</th>
              <th>数值</th>
              <th>长度</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">q（内容，固定）</th>
              <td>{vector(ROPE_Q)}</td>
              <td>{fmt(norm(ROPE_Q))}</td>
            </tr>
            <tr className="current">
              <th scope="row">R_{m} q</th>
              <td>{vector(qr)}</td>
              <td>{fmt(norm(qr))}</td>
            </tr>
            <tr>
              <th scope="row">k（内容，固定）</th>
              <td>{vector(ROPE_K)}</td>
              <td>{fmt(norm(ROPE_K))}</td>
            </tr>
            <tr className="current">
              <th scope="row">R_{n} k</th>
              <td>{vector(kr)}</td>
              <td>{fmt(norm(kr))}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="pos-verdict same" role="status">
        <Tex tex="q'\cdot k'" /> = 四对之和 = <strong>{fmt(score)}</strong>，除以{' '}
        <Tex tex="\sqrt{8}" /> 后 = {fmt(score / Math.sqrt(8))} 进入 Softmax。旋转前后长度不变；V
        不旋转。
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ 5. 相对距离

function DistanceCurve({ distance }: { distance: number }) {
  const width = 380;
  const height = 150;
  const range = 20;
  const points = Array.from({ length: 2 * range + 1 }, (_, k) => k - range).map((d) => ({
    d,
    s: ropeScore(ROPE_Q, ROPE_K, 0, d),
  }));
  const max = Math.max(...points.map((p) => p.s));
  const min = Math.min(...points.map((p) => p.s));
  const x = (d: number) => 30 + ((d + range) / (2 * range)) * (width - 44);
  const y = (s: number) => 14 + ((max - s) / (max - min)) * (height - 40);
  const current = points.find((p) => p.d === distance);
  return (
    <svg
      className="pos-wave"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`RoPE 分数随相对距离 n − m 的变化，当前距离 ${distance}`}
    >
      <line
        x1={x(-range)}
        x2={x(range)}
        y1={y(points[range].s)}
        y2={y(points[range].s)}
        className="pos-grid"
      />
      {[-20, -10, 0, 10, 20].map((d) => (
        <text key={d} x={x(d)} y={height - 6} textAnchor="middle" className="pos-axis">
          {d}
        </text>
      ))}
      <text x="4" y={y(points[range].s) + 4} className="pos-axis">
        {fmt(points[range].s, 1)}
      </text>
      <polyline
        points={points.map((p) => `${x(p.d)},${y(p.s)}`).join(' ')}
        className="pos-curve rope"
      />
      {points.map((p) => (
        <circle
          key={p.d}
          cx={x(p.d)}
          cy={y(p.s)}
          r={p.d === distance ? 5 : 2.2}
          className={`pos-dot rope ${p.d === distance ? 'current' : ''}`}
        />
      ))}
      {current && (
        <text x={x(current.d)} y={y(current.s) - 9} textAnchor="middle" className="pos-axis strong">
          {fmt(current.s)}
        </text>
      )}
    </svg>
  );
}

export function RelativeWidget({
  state,
  onChange,
  posPlaying,
  onTogglePosPlay,
}: WidgetProps & PlayProps) {
  const { m, n } = state.rope;
  const [prediction, setPrediction] = useState<'same' | 'change' | null>(null);
  const [check, setCheck] = useState<{
    from: [number, number];
    to: [number, number];
    before: number;
    after: number;
  } | null>(null);
  const score = ropeScore(ROPE_Q, ROPE_K, m, n);
  const distance = n - m;
  const shiftLimit = ROPE_MAX - Math.max(m, n);
  const shift = shiftLimit >= 3 ? 3 : -Math.min(3, Math.min(m, n));
  const set = (nextM: number, nextN: number) =>
    onChange({ ...state, rope: { m: nextM, n: nextN } });
  return (
    <div className="pos-widget">
      <div className="pos-now">
        <span>
          m = <strong>{m}</strong>
        </span>
        <span>
          n = <strong>{n}</strong>
        </span>
        <span>
          距离 n − m = <strong>{distance}</strong>
        </span>
        <span>
          q′·k′ = <strong>{fmt(score)}</strong>
        </span>
      </div>
      <div className="pos-predict">
        <p>
          先预测：把 m 和 n <strong>同时{shift > 0 ? `加 ${shift}` : `减 ${-shift}`}</strong>
          ，分数会变吗？
        </p>
        <div className="pos-segment" role="group" aria-label="预测平移后的分数">
          <button
            aria-pressed={prediction === 'same'}
            onClick={() => {
              setPrediction('same');
              setCheck(null);
            }}
          >
            分数不变
          </button>
          <button
            aria-pressed={prediction === 'change'}
            onClick={() => {
              setPrediction('change');
              setCheck(null);
            }}
          >
            分数会变
          </button>
        </div>
        <button
          className="button primary"
          disabled={!prediction || shift === 0}
          onClick={() => {
            const after = ropeScore(ROPE_Q, ROPE_K, m + shift, n + shift);
            setCheck({ from: [m, n], to: [m + shift, n + shift], before: score, after });
            set(m + shift, n + shift);
          }}
        >
          执行平移并核对 →
        </button>
        {check && (
          <p className="pos-verdict same" role="status">
            {prediction === 'same' ? '预测正确。' : '再看一次：'}({check.from.join(', ')}) → (
            {check.to.join(', ')})，分数 {fmt(check.before, 6)} → {fmt(check.after, 6)}，差{' '}
            {Math.abs(check.after - check.before).toExponential(1)}
            （浮点误差）。两个向量转了同样的角度，夹角不变。
          </p>
        )}
      </div>
      <div className="pos-controls">
        <button
          className="pos-step"
          disabled={Math.max(m, n) >= ROPE_MAX}
          onClick={() => set(m + 1, n + 1)}
        >
          m、n 同时 +1
        </button>
        <button
          className="pos-step"
          disabled={Math.min(m, n) <= 0}
          onClick={() => set(m - 1, n - 1)}
        >
          m、n 同时 −1
        </button>
        <button className="pos-step" disabled={n >= ROPE_MAX} onClick={() => set(m, n + 1)}>
          只把 n +1（距离变）
        </button>
        <PlayButton playing={posPlaying} onToggle={onTogglePosPlay} label="同时平移播放" />
      </div>
      <DistanceCurve distance={distance} />
      <p className="pos-caption">
        横轴是距离 n − m，纵轴是固定内容下的 q′·k′。同一距离的所有 (m, n)
        落在同一个点上；但曲线上下起伏，距离 5 的分数就比距离 0 更高。
      </p>
      <p className="pos-note">
        成立条件：q、k 的内容固定，
        <Tex tex="m" />、<Tex tex="n" /> 用同一套 <Tex tex="\theta" /> 旋转。真实模型里改变 token
        也会改变内容向量，分数随之变化。
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ 6. ALiBi

const ALIBI_WORDS = ['L', 'L', 'M', '张', '老', '师'];
// Teaching content scores (already ÷√d); only the lower triangle is visible under the causal mask.
const EXAMPLE_SCORES = [
  [0, 0, 0, 0, 0, 0],
  [0.5, 0, 0, 0, 0, 0],
  [0.2, 1, 0, 0, 0, 0],
  [1.2, 0.1, 0.4, 0, 0, 0],
  [0.3, 0.9, 0.2, 0.6, 0, 0],
  [1, 0.2, 0.8, 0.1, 0.5, 0],
];

export function AlibiWidget({ state, onChange }: WidgetProps) {
  const { head, query, content } = state.alibi;
  const slopes = alibiSlopes(ALIBI_HEADS);
  const slope = slopes[head];
  const scores = useMemo(
    () => (content === 'same' ? EXAMPLE_SCORES.map((row) => row.map(() => 0)) : EXAMPLE_SCORES),
    [content],
  );
  const result = useMemo(() => alibiAttention(scores, slope), [scores, slope]);
  const plain = useMemo(() => alibiAttention(scores, 0), [scores]);
  const [prediction, setPrediction] = useState<'first' | 'last' | null>(null);
  const [compared, setCompared] = useState(false);
  const set = (patch: Partial<PositionState['alibi']>) =>
    onChange({ ...state, alibi: { ...state.alibi, ...patch } });
  const nearest = (h: number) =>
    alibiAttention(scores, slopes[h]).probabilities[ALIBI_TOKENS - 1][ALIBI_TOKENS - 1];
  return (
    <div className="pos-widget">
      <div className="pos-controls">
        <div className="pos-segment pos-heads" role="group" aria-label="选择头（不同斜率）">
          <span>头</span>
          {slopes.map((_, h) => (
            <button
              key={h}
              aria-pressed={head === h}
              aria-label={`head ${h + 1}，斜率 1/${2 ** (h + 1)}`}
              onClick={() => set({ head: h })}
            >
              h{h + 1}
            </button>
          ))}
        </div>
        <span className="pos-chip">
          斜率 m = 1/{2 ** (head + 1)} = {fmt(slope, 4)}
        </span>
        <div className="pos-segment" role="group" aria-label="内容分数">
          <button aria-pressed={content === 'same'} onClick={() => set({ content: 'same' })}>
            内容分数全相同
          </button>
          <button aria-pressed={content === 'example'} onClick={() => set({ content: 'example' })}>
            示例内容分数
          </button>
        </div>
      </div>
      <div className="pos-alibi-layout">
        <div className="pos-table-wrap">
          <table className="pos-table pos-bias" aria-label="ALiBi 偏置 m·B，行是 Query，列是 Key">
            <thead>
              <tr>
                <th>Q ↓ K →</th>
                {ALIBI_WORDS.map((w, j) => (
                  <th key={j}>
                    {w}
                    <small>{j}</small>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.bias.map((row, i) => (
                <tr key={i} className={i === query ? 'current' : ''}>
                  <th scope="row">
                    <button
                      className="pos-row-pick"
                      aria-pressed={i === query}
                      onClick={() => set({ query: i })}
                    >
                      {ALIBI_WORDS[i]} <small>{i}</small>
                    </button>
                  </th>
                  {row.map((b, j) => (
                    <td key={j} className={j > i ? 'masked' : ''}>
                      {j > i ? '−∞' : fmt(b)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="pos-caption">
            书中 25.4.2 的偏置矩阵 [0; −1 0; −2 −1 0; …] 乘以 m；右上方是因果 Mask。点行首选择
            Query。
          </p>
        </div>
        <div className="pos-row-detail">
          <strong>
            Query {query}（{ALIBI_WORDS[query]}）这一行
          </strong>
          <table className="pos-table" aria-label="内容分数加偏置后再 Softmax">
            <thead>
              <tr>
                <th>Key</th>
                <th>内容</th>
                <th>偏置</th>
                <th>权重 P</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: query + 1 }, (_, j) => (
                <tr key={j}>
                  <th scope="row">
                    {ALIBI_WORDS[j]} <small>距 {query - j}</small>
                  </th>
                  <td>{fmt(scores[query][j])}</td>
                  <td>{fmt(result.bias[query][j])}</td>
                  <td>
                    <span className="pos-bar" aria-hidden="true">
                      <span
                        className="plain"
                        style={{ width: `${plain.probabilities[query][j] * 100}%` }}
                      />
                      <span
                        style={{
                          width: `${result.probabilities[query][j] * 100}%`,
                          background: INSERTION_COLORS.alibi,
                        }}
                      />
                    </span>
                    {(result.probabilities[query][j] * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="pos-caption">彩色条是加偏置后的权重，灰色条是不加 ALiBi 的权重。</p>
        </div>
      </div>
      <div className="pos-predict">
        <p>
          先预测：在最后一个 token“师”这一行，<strong>head 1（m = 1/2）</strong>和{' '}
          <strong>head 8（m = 1/256）</strong>，哪个头给最近的自己分配更多权重？
        </p>
        <div className="pos-segment" role="group" aria-label="预测哪个头更关注附近">
          <button
            aria-pressed={prediction === 'first'}
            onClick={() => {
              setPrediction('first');
              setCompared(false);
            }}
          >
            head 1
          </button>
          <button
            aria-pressed={prediction === 'last'}
            onClick={() => {
              setPrediction('last');
              setCompared(false);
            }}
          >
            head 8
          </button>
        </div>
        <button className="button primary" disabled={!prediction} onClick={() => setCompared(true)}>
          对比两个头 →
        </button>
        {compared && (
          <p className="pos-verdict same" role="status">
            {prediction === 'first' ? '预测正确。' : '再看一次：'}head 1 给“师”自己{' '}
            {(nearest(0) * 100).toFixed(1)}%，head 8 只给{' '}
            {(nearest(ALIBI_HEADS - 1) * 100).toFixed(1)}
            %。斜率越大，远处被扣得越多，注意力越集中在附近；斜率小的头几乎只看内容分数。
          </p>
        )}
      </div>
      <p className="pos-note">
        Q、K、V 都没有改变，ALiBi 只在 Softmax
        之前给分数加偏置。偏置只取决于距离，任何长度都能直接算出来；但“能算更长的序列”不等于模型一定能可靠地利用更长的上下文。
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ 7. 对照总结

export function SummaryWidget({ onJump }: { onJump: (step: number) => void }) {
  const rows: [string, string, string, string, string, string, number | null][] = [
    [
      '正弦（第 5 章）',
      '投影前，加到 X',
      'X → Q、K、V 全部',
      '带位置',
      '绝对位置（固定表）',
      '无',
      1,
    ],
    [
      '可学习位置嵌入（GPT-2，第 14 章）',
      '投影前，加到 X',
      'X → Q、K、V 全部',
      '带位置',
      '绝对位置（训练得到）',
      '$\\text{max\\_len}\\times d_{\\mathrm{model}}$',
      null,
    ],
    ['RoPE（第 25 章）', '投影后，旋转 Q、K', '只有 Q、K', '不带位置', '分数只看相对距离', '无', 3],
    [
      'ALiBi（第 25 章）',
      'Softmax 前，加到分数',
      '只有分数 S',
      '不带位置',
      '按距离线性扣分',
      '无（每头一个固定斜率）',
      5,
    ],
  ];
  return (
    <div className="pos-widget">
      <div className="pos-table-wrap">
        <table className="pos-table pos-summary" aria-label="三种位置编码对照">
          <thead>
            <tr>
              <th>方法</th>
              <th>在哪里加入</th>
              <th>直接改变</th>
              <th>V</th>
              <th>位置形式</th>
              <th>额外参数</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, where, what, value, kind, params, step]) => (
              <tr key={name}>
                <th scope="row">
                  {step === null ? (
                    name
                  ) : (
                    <button className="pos-link" onClick={() => onJump(step)}>
                      {name} ↗
                    </button>
                  )}
                </th>
                <td>{where}</td>
                <td>{what}</td>
                <td>{value}</td>
                <td>{kind}</td>
                <td>
                  <Rich text={params} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="pos-myths">
        <li>
          位置编码不是把 1、2、3 直接加上去：书中 5.2 节的整数方案数值会越加越大，也没有可学的规律。
        </li>
        <li>RoPE 不加到 X 上，也不旋转 V；ALiBi 不改 Q、K、V，只改分数。</li>
        <li>
          RoPE 的“相对位置”成立需要固定内容、同一套 <Tex tex="\theta" />
          ；单个分数并不随距离单调下降。
        </li>
        <li>
          能计算更远的位置，不等于模型能可靠处理任意长的上下文（外推还取决于训练长度与方法）。
        </li>
      </ul>
    </div>
  );
}
