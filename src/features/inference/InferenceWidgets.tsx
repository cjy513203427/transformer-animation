import { useState } from 'react';
import {
  Prediction as SharedPrediction,
  PredictionResult as SharedResult,
  Segment,
} from '../../components/Controls';
import { Tex } from '../../components/Tex';
import { formatNumber as fmt } from '../../math/attention';
import {
  MAX_DECODE_STEPS,
  NUM_LAYERS,
  TOY_MODEL,
  trainingRows,
  trainingWithEditedInput,
  VOCAB,
  type Generation,
} from '../../math/inference';
import { PROMPT, TRAIN_SEQUENCE, type InferenceState } from '../../state/inference';

const name = (id: number) => VOCAB[id];
const vec = (values: number[]) => `[${values.map((v) => fmt(v)).join(', ')}]`;
const pct = (p: number) => `${(p * 100).toFixed(1)}%`;
const list = (indices: number[]) => (indices.length ? indices.join('、') : '无');
const diffText = (value: number) => (value === 0 ? '0' : value.toExponential(1));

interface WidgetProps {
  state: InferenceState;
  onChange: (next: InferenceState) => void;
}

function Tokens({
  ids,
  label,
  from = 0,
  highlight,
}: {
  ids: number[];
  label: string;
  from?: number;
  highlight?: number;
}) {
  return (
    <div className="exp-tokens" role="group" aria-label={label}>
      <span className="exp-tokens-label">{label}</span>
      {ids.map((id, i) => (
        <span key={i} className={`exp-chip ${highlight === from + i ? 'exp-chip-on' : ''}`}>
          <small>{from + i}</small>
          {name(id)}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- predictions

type Spread = 'only' | 'after' | 'all';
const SPREAD_LABEL: Record<Spread, string> = {
  only: '只有被改的这一个位置',
  after: '被改的位置及之后',
  all: '所有位置',
};
/** Does the predicted pattern match the computed changed/unchanged flags exactly? */
function predictionHolds(choice: Spread, changed: boolean[], pos: number): boolean {
  return changed.every(
    (flag, i) => flag === (choice === 'only' ? i === pos : choice === 'after' ? i >= pos : true),
  );
}

const SPREAD_OPTIONS = (['only', 'after', 'all'] as const).map((value) => ({
  value,
  label: SPREAD_LABEL[value],
}));

function Prediction({
  question,
  choice,
  onChoose,
}: {
  question: string;
  choice: Spread | null;
  onChoose: (choice: Spread) => void;
}) {
  return (
    <SharedPrediction
      question={question}
      options={SPREAD_OPTIONS}
      choice={choice}
      onChoose={onChoose}
    />
  );
}

function PredictionResult({ choice, holds }: { choice: Spread | null; holds: boolean }) {
  return <SharedResult label={choice && SPREAD_LABEL[choice]} holds={holds} />;
}

// ---------------------------------------------------------------- shared controls

function DecodeSlider({ state, onChange, label }: WidgetProps & { label: string }) {
  return (
    <label className="exp-range">
      <span>{label}</span>
      <input
        type="range"
        min={1}
        max={MAX_DECODE_STEPS}
        value={state.decode.steps}
        aria-label={label}
        onChange={(e) => onChange({ ...state, decode: { steps: Number(e.target.value) } })}
      />
      <output>{state.decode.steps}</output>
    </label>
  );
}

function GeneratedText({
  generation,
  decodeSteps,
}: {
  generation: Generation;
  decodeSteps: number;
}) {
  return (
    <div className="exp-stack">
      <Tokens ids={PROMPT} label="提示" />
      <Tokens
        ids={generation.tokens.slice(PROMPT.length, PROMPT.length + decodeSteps + 1)}
        label="生成"
        from={PROMPT.length}
      />
    </div>
  );
}

// ---------------------------------------------------------------- 1 训练并行

export function TrainingWidget({ state, onChange }: WidgetProps) {
  const [choice, setChoice] = useState<Spread | null>(null);
  const { pos, token, applied } = state.train;
  const base = trainingRows(TOY_MODEL, TRAIN_SEQUENCE);
  const edited = trainingWithEditedInput(TOY_MODEL, TRAIN_SEQUENCE, pos, token);
  const rows = applied ? edited.rows : base;
  const changed = applied ? edited.changedRows : base.map(() => false);
  const mean = rows.reduce((sum, row) => sum + row.loss, 0) / rows.length;
  const changedRows = edited.changedRows.flatMap((flag, i) => (flag ? [i] : []));
  const sameRows = edited.changedRows.flatMap((flag, i) => (flag ? [] : [i]));
  const noChange = TRAIN_SEQUENCE[pos] === token;
  const setTrain = (patch: Partial<InferenceState['train']>) => {
    setChoice(null);
    onChange({ ...state, train: { ...state.train, applied: false, ...patch } });
  };

  return (
    <div className="exp-widget">
      <div className="exp-stack">
        <Tokens ids={TRAIN_SEQUENCE.slice(0, -1)} label="输入 x" />
        <Tokens ids={TRAIN_SEQUENCE.slice(1)} label="目标 y" />
        <p className="exp-caption">
          同一条已知序列错开一位：第 t 个输入的目标，就是第 t + 1 个 token。
        </p>
      </div>
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label="每个位置的预测与 loss">
          <thead>
            <tr>
              <th>行 t</th>
              <th>输入</th>
              <th>能读到的输入</th>
              <th>目标</th>
              <th>p(目标)</th>
              <th>loss</th>
              {applied && <th>输出分布</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, t) => (
              <tr key={t} className={changed[t] ? 'exp-changed' : undefined}>
                <th scope="row">{t}</th>
                <td>{name(row.input)}</td>
                <td>{row.visible.map((i) => name(rows[i].input)).join(' ')}</td>
                <td>{name(row.target)}</td>
                <td>{pct(row.targetProb)}</td>
                <td>{fmt(row.loss)}</td>
                {applied && <td>{changed[t] ? '已变' : '逐位不变'}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-caption">
        平均 loss = {fmt(mean)}
        。权重是固定教学数字，没有训练过，概率大小本身没有意义；要看的是每一行用到了哪些输入。
      </p>

      <div className="exp-experiment">
        <strong>反事实：只改一个输入，目标不动</strong>
        <Segment
          label="改哪个输入"
          options={[0, 1, 2, 3]}
          value={pos}
          onSelect={(value) => setTrain({ pos: value })}
          render={(option) => `x${option}`}
        />
        <Segment
          label="换成"
          options={VOCAB.map((_, i) => i)}
          value={token}
          onSelect={(value) => setTrain({ token: value })}
          render={name}
        />
        <Prediction
          question={`把 x${pos} 换成 ${name(token)}，哪些行的输出分布会变？`}
          choice={choice}
          onChoose={setChoice}
        />
        <div className="exp-actions">
          <button
            className="button primary"
            disabled={applied}
            onClick={() => onChange({ ...state, train: { ...state.train, applied: true } })}
          >
            应用改动并核对 →
          </button>
          {applied && (
            <button className="button secondary" onClick={() => setTrain({})}>
              撤销改动
            </button>
          )}
        </div>
        {applied && (
          <p className="exp-verdict ok" role="status">
            {noChange ? (
              `x${pos} 本来就是 ${name(token)}，没有任何改动。`
            ) : (
              <>
                输出分布逐位不变的行：{list(sameRows)}；发生变化的行：{list(changedRows)}。
                <PredictionResult
                  choice={choice}
                  holds={choice ? predictionHolds(choice, edited.changedRows, pos) : false}
                />
              </>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- 2 无缓存基线

export function BaselineWidget({
  state,
  onChange,
  generation,
}: WidgetProps & { generation: Generation }) {
  const passes = [generation.prefill, ...generation.steps.map((s) => s.recompute)];
  const produced = [generation.first, ...generation.steps.map((s) => s.produced)];
  /** Largest difference between this pass and the previous one over the rows both computed. */
  const repeatDiff = (s: number) => {
    let max = 0;
    passes[s].layers.forEach((layer, l) =>
      passes[s - 1].layers[l].output.forEach((row, i) =>
        row.forEach((value, d) => {
          max = Math.max(max, Math.abs(value - layer.output[i][d]));
        }),
      ),
    );
    return max;
  };
  const worst = Math.max(...passes.slice(1).map((_, i) => repeatDiff(i + 1)));
  return (
    <div className="exp-widget">
      <DecodeSlider state={state} onChange={onChange} label="生成步数" />
      <GeneratedText generation={generation} decodeSteps={state.decode.steps} />
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label="不用缓存时每一步的工作">
          <thead>
            <tr>
              <th>步骤</th>
              <th>序列长度</th>
              <th>重算的行</th>
              <th>用到的行</th>
              <th>与上一步重复（最大差）</th>
              <th>选出</th>
            </tr>
          </thead>
          <tbody>
            {passes.map((pass, s) => {
              const n = pass.hidden.length;
              return (
                <tr key={s}>
                  <th scope="row">{s === 0 ? '第一次前向' : `第 ${s} 步`}</th>
                  <td>{n}</td>
                  <td>{n}</td>
                  <td>1（第 {n - 1} 行）</td>
                  <td>{s === 0 ? '—' : `${n - 1} 行（${diffText(repeatDiff(s))}）`}</td>
                  <td>{name(produced[s])}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="exp-verdict ok" role="status">
        每一步重算出的旧行，与上一步的结果相比，两层输出的最大差为 {diffText(worst)}
        。真正新的只有最后一行。
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- 3 Prefill

export function PrefillWidget({ generation }: { generation: Generation }) {
  const last = generation.prefill.probs[PROMPT.length - 1];
  return (
    <div className="exp-widget">
      <Tokens ids={PROMPT} label="提示" />
      <p className="exp-caption">
        3 个位置在一次因果前向里并行计算；每一层都把各位置的 K、V 写进缓存。
      </p>
      {generation.prefill.layers.map((layer, l) => (
        <div className="exp-table-wrap" key={l}>
          <table className="exp-table" aria-label={`第 ${l + 1} 层写入缓存的 K、V`}>
            <caption>
              第 {l + 1} 层写入缓存（这一层的输入是
              {l === 0 ? '词嵌入 + 位置编码' : '第 1 层的输出'}）
            </caption>
            <thead>
              <tr>
                <th>位置</th>
                <th>token</th>
                <th>K</th>
                <th>V</th>
              </tr>
            </thead>
            <tbody>
              {PROMPT.map((id, p) => (
                <tr key={p}>
                  <th scope="row">{p}</th>
                  <td>{name(id)}</td>
                  <td>{vec(layer.k[p])}</td>
                  <td>{vec(layer.v[p])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <div className="exp-bars" role="group" aria-label="最后一个提示位置的输出分布">
        <strong>只用最后一个位置（pos {PROMPT.length - 1}）的分布选第一个新 token</strong>
        {last.map((p, id) => (
          <div key={id} className={id === generation.first ? 'exp-bar exp-bar-on' : 'exp-bar'}>
            <span>{name(id)}</span>
            <i style={{ width: `${Math.max(1, p * 100)}%` }} />
            <em>{pct(p)}</em>
          </div>
        ))}
      </div>
      <p className="exp-verdict ok" role="status">
        贪心选概率最大的：第一个新 token 是 {name(generation.first)}
        。它还没有 K、V，要等下一次 Decode 把它喂进去才会算。
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- 4 Decode 一步

export function DecodeWidget({
  state,
  onChange,
  generation,
}: WidgetProps & { generation: Generation }) {
  const step = generation.steps[state.decode.steps - 1];
  const { decode } = step;
  return (
    <div className="exp-widget">
      <DecodeSlider state={state} onChange={onChange} label="看第几次 Decode" />
      <p className="exp-caption">
        这一步喂入 <strong>{name(step.input)}</strong>（位置 {decode.position}
        ）。缓存里已有 {decode.position} 个位置；每层只为新 token 算一组 <Tex tex="q,\ k,\ v" />。
      </p>
      {decode.layers.map((layer, l) => (
        <div className="exp-table-wrap" key={l}>
          <table className="exp-table" aria-label={`第 ${l + 1} 层：新 Query 读缓存`}>
            <caption>
              第 {l + 1} 层新算：
              <Tex tex="q" /> = {vec(layer.q)}，<Tex tex="k" /> = {vec(layer.k)}，<Tex tex="v" /> ={' '}
              {vec(layer.v)}
            </caption>
            <thead>
              <tr>
                <th>位置</th>
                <th>token</th>
                <th>K、V 的来源</th>
                <th>
                  <Tex tex="q\cdot k/\sqrt{2}" />
                </th>
                <th>权重</th>
              </tr>
            </thead>
            <tbody>
              {layer.weights.map((w, p) => (
                <tr key={p} className={p === decode.position ? 'exp-new-row' : undefined}>
                  <th scope="row">{p}</th>
                  <td>{name(generation.tokens[p])}</td>
                  <td>{p === decode.position ? '本步新算' : '读缓存'}</td>
                  <td>{fmt(layer.scores[p])}</td>
                  <td>{pct(w)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <p className="exp-verdict ok" role="status">
        本步每层：新算 1 组 Q/K/V，从缓存读 {decode.position} 组 K/V，打分 {decode.position + 1}{' '}
        次。输出分布选出 {name(step.produced)}（{pct(decode.probs[step.produced])}）。
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- 5 缓存对照重算

export function CompareWidget({
  state,
  onChange,
  generation,
}: WidgetProps & { generation: Generation }) {
  const prefill = generation.prefill.work;
  const totals = generation.steps.reduce(
    (sum, s) => ({
      recomputeKV: sum.recomputeKV + s.recompute.work.kvProjections,
      cacheKV: sum.cacheKV + s.decode.work.kvProjections,
      recomputeScores: sum.recomputeScores + s.recompute.work.scorePairs,
      cacheScores: sum.cacheScores + s.decode.work.scorePairs,
    }),
    { recomputeKV: 0, cacheKV: 0, recomputeScores: 0, cacheScores: 0 },
  );
  const maxDiff = Math.max(...generation.steps.map((s) => s.logitsDiff));
  const sameTokens = generation.steps.every((s) => {
    const row = s.recompute.probs[s.recompute.probs.length - 1];
    return row.indexOf(Math.max(...row)) === s.produced;
  });
  return (
    <div className="exp-widget">
      <DecodeSlider state={state} onChange={onChange} label="Decode 步数" />
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label="缓存与重算的工作量对照">
          <thead>
            <tr>
              <th rowSpan={2}>步骤</th>
              <th rowSpan={2}>长度</th>
              <th colSpan={2}>K/V 投影</th>
              <th colSpan={2}>打分次数</th>
              <th rowSpan={2}>logits 最大差</th>
            </tr>
            <tr>
              <th>重算</th>
              <th>缓存</th>
              <th>重算</th>
              <th>缓存</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Prefill（相同）</th>
              <td>{PROMPT.length}</td>
              <td>{prefill.kvProjections}</td>
              <td>{prefill.kvProjections}</td>
              <td>{prefill.scorePairs}</td>
              <td>{prefill.scorePairs}</td>
              <td>—</td>
            </tr>
            {generation.steps.map((s, i) => (
              <tr key={i}>
                <th scope="row">Decode {i + 1}</th>
                <td>{s.recompute.hidden.length}</td>
                <td>{s.recompute.work.kvProjections}</td>
                <td>{s.decode.work.kvProjections}</td>
                <td>{s.recompute.work.scorePairs}</td>
                <td>{s.decode.work.scorePairs}</td>
                <td>{diffText(s.logitsDiff)}</td>
              </tr>
            ))}
            <tr className="exp-total">
              <th scope="row">Decode 合计</th>
              <td>—</td>
              <td>{totals.recomputeKV}</td>
              <td>{totals.cacheKV}</td>
              <td>{totals.recomputeScores}</td>
              <td>{totals.cacheScores}</td>
              <td>{diffText(maxDiff)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="exp-verdict ok" role="status">
        {state.decode.steps} 步里两条路线
        {sameTokens ? '选出的 token 完全相同' : '选出的 token 不同'}，logits 最大差为{' '}
        {diffText(maxDiff)}（容差 1e-12）。
      </p>
      <p className="exp-caption">
        K/V 投影按“位置 × 层”计数。计数来自本页实现里实际执行的操作，并与公式 <Tex tex="L\cdot n" />
        、<Tex tex="L\cdot n(n+1)/2" /> 核对过（{NUM_LAYERS} 层、单头）。只统计 Attention 的 K/V
        投影与打分，不含 FFN、输出头和读写内存，也不是实测时间。
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- 6 缓存失效

export function InvalidationWidget({
  state,
  onChange,
  ids,
  flags,
  revealed,
  onReveal,
}: WidgetProps & {
  ids: number[];
  flags: (boolean | null)[][];
  revealed: boolean;
  onReveal: () => void;
}) {
  const [choice, setChoice] = useState<Spread | null>(null);
  const { mode, pos, token } = state.cache;
  const set = (patch: Partial<InferenceState['cache']>) => {
    setChoice(null);
    onChange({ ...state, cache: { ...state.cache, ...patch } });
  };
  const staleAt = (layer: number) => flags[layer].flatMap((flag, p) => (flag ? [p] : []));
  const noChange = mode === 'edit' && ids[pos] === token;
  // Compare the prediction with layer 2. After a slide the dropped column is not a cache entry.
  const upper = flags[NUM_LAYERS - 1];
  const holds = (option: Spread) =>
    mode === 'edit'
      ? predictionHolds(option, upper.map(Boolean), pos)
      : predictionHolds(option, upper.slice(1).map(Boolean), 0);

  return (
    <div className="exp-widget">
      <Segment
        label="改动方式"
        options={['edit', 'slide'] as const}
        value={mode}
        onSelect={(value) => set({ mode: value })}
        render={(option) => (option === 'edit' ? '改一个提示 token' : '滑动窗口')}
      />
      {mode === 'edit' ? (
        <>
          <Segment
            label="改哪个位置"
            options={[0, 1, 2]}
            value={pos}
            onSelect={(value) => set({ pos: value })}
            render={(option) => `pos ${option}`}
          />
          <Segment
            label="换成"
            options={VOCAB.map((_, i) => i)}
            value={token}
            onSelect={(value) => set({ token: value })}
            render={name}
          />
          <div className="exp-stack">
            <Tokens ids={ids} label="原序列" highlight={pos} />
            <Tokens
              ids={ids.map((id, i) => (i === pos ? token : id))}
              label="改动后"
              highlight={pos}
            />
          </div>
        </>
      ) : (
        <div className="exp-stack">
          <Tokens ids={ids} label="原序列" />
          <Tokens ids={ids.slice(1)} label="窗口右移" />
          <p className="exp-caption">丢掉最早的 token。后面的 token 不变，位置编号都减了 1。</p>
        </div>
      )}
      <Prediction question="第 2 层哪些位置的缓存会失效？" choice={choice} onChoose={setChoice} />
      <div className="exp-actions">
        <button className="button primary" disabled={revealed} onClick={onReveal}>
          重算并核对 →
        </button>
      </div>
      {revealed && (
        <p className="exp-verdict ok" role="status">
          {noChange ? (
            `pos ${pos} 本来就是 ${name(token)}，所有条目都可以复用。`
          ) : (
            <>
              {mode === 'slide' && '按原位置编号，'}第 1 层失效的位置：{list(staleAt(0))}；第 2
              层失效的位置：{list(staleAt(1))}。
              {mode === 'slide' && '位置编号整体变了，加法位置编码下旧条目都要重算。'}
              <PredictionResult choice={choice} holds={choice ? holds(choice) : false} />
            </>
          )}
        </p>
      )}
    </div>
  );
}
