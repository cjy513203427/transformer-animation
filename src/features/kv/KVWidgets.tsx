import { useState } from 'react';
import { Prediction, PredictionResult, Segment } from '../../components/Controls';
import { Tex } from '../../components/Tex';
import { formatNumber as fmt } from '../../math/attention';
import {
  coupledRopeRows,
  D_HEAD,
  D_LATENT,
  D_MODEL,
  D_ROPE,
  DEEPSEEK_V2,
  formatBytes,
  groupedAttention,
  groupOf,
  KV_OPTIONS,
  kvCacheBytes,
  mlaAbsorbed,
  mlaCacheBytes,
  mlaExpanded,
  MHA_WEIGHTS,
  MLA_WEIGHTS,
  projectionParams,
  Q_HEADS,
  repeatedKVReference,
  TOKENS,
  X,
  type KVHeads,
} from '../../math/kv';
import {
  BYTE_OPTIONS,
  CALC_PRESETS,
  COMPARE_TOKENS,
  kvHeadChoices,
  LIMITS,
  type CalcConfig,
  type KVState,
} from '../../state/kv';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';

interface WidgetProps {
  state: KVState;
  onChange: (next: KVState) => void;
}

const vec = (values: number[]) => `[${values.map((v) => fmt(v)).join(', ')}]`;
const pct = (p: number) => `${(p * 100).toFixed(1)}%`;
const diffText = (value: number) => (value === 0 ? '0' : value.toExponential(1));
const maxDiff = (a: number[][], b: number[][]) =>
  Math.max(...a.flatMap((row, i) => row.map((value, j) => Math.abs(value - b[i][j]))));
const kindOf = (g: number, h: number) => (g === h ? 'MHA' : g === 1 ? 'MQA' : `GQA-${g}`);
const thousands = (n: number) => n.toLocaleString('en-US');
const ROOT = Math.sqrt(D_HEAD + D_ROPE);
const headOptions = Array.from({ length: Q_HEADS }, (_, h) => h);
const tokenOptions = Array.from({ length: TOKENS }, (_, t) => t);

function NumberField({
  label,
  value,
  min,
  max,
  onCommit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onCommit: (value: number) => void;
}) {
  return (
    <label className="kv-field">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={1}
        value={value}
        aria-label={label}
        onChange={(e) => {
          const next = Number(e.target.value);
          if (Number.isInteger(next) && next >= min && next <= max) onCommit(next);
        }}
      />
    </label>
  );
}

// ---------------------------------------------------------------- 1 缓存账本

export function CalcWidget({ state, onChange }: WidgetProps) {
  const c = state.calc;
  const set = (patch: Partial<CalcConfig>) => {
    const next = { ...c, ...patch };
    // Keep the grouping valid: fall back to the largest KV head count that divides H_Q.
    if (next.qHeads % next.kvHeads)
      next.kvHeads = kvHeadChoices(next.qHeads).find((g) => g <= next.kvHeads) ?? 1;
    onChange({ ...state, calc: next });
  };
  const config = {
    batch: c.batch,
    layers: c.layers,
    tokens: c.tokens,
    kvHeads: c.kvHeads,
    dK: c.dHead,
    dV: c.dHead,
    bytesPerElement: c.bytes,
  };
  const bytes = kvCacheBytes(config);
  const mha = kvCacheBytes({ ...config, kvHeads: c.qHeads });
  const perToken = bytes / (c.tokens * c.batch);
  const preset = (Object.keys(CALC_PRESETS) as (keyof typeof CALC_PRESETS)[]).find((key) =>
    (Object.keys(c) as (keyof CalcConfig)[]).every(
      (field) => CALC_PRESETS[key].config[field] === c[field],
    ),
  );

  return (
    <div className="exp-widget">
      <div className="exp-segment" role="group" aria-label={T('预设')}>
        <span>{T('预设')}</span>
        {(Object.keys(CALC_PRESETS) as (keyof typeof CALC_PRESETS)[]).map((key) => (
          <button
            key={key}
            aria-pressed={preset === key}
            onClick={() => onChange({ ...state, calc: { ...CALC_PRESETS[key].config } })}
          >
            {CALC_PRESETS[key].label}
          </button>
        ))}
      </div>
      <div className="kv-fields">
        <NumberField
          label={T('批大小 B')}
          value={c.batch}
          min={1}
          max={LIMITS.batch}
          onCommit={(v) => set({ batch: v })}
        />
        <NumberField
          label={T('层数 L')}
          value={c.layers}
          min={1}
          max={LIMITS.layers}
          onCommit={(v) => set({ layers: v })}
        />
        <NumberField
          label={T('token 数 T')}
          value={c.tokens}
          min={1}
          max={LIMITS.tokens}
          onCommit={(v) => set({ tokens: v })}
        />
        <NumberField
          label={T('Query 头数')}
          value={c.qHeads}
          min={1}
          max={LIMITS.qHeads}
          onCommit={(v) => set({ qHeads: v })}
        />
        <NumberField
          label={T('每头维度 d')}
          value={c.dHead}
          min={1}
          max={LIMITS.dHead}
          onCommit={(v) => set({ dHead: v })}
        />
      </div>
      <Segment
        label={T('KV 头数')}
        options={kvHeadChoices(c.qHeads).slice(0, 8)}
        value={c.kvHeads}
        onSelect={(value) => set({ kvHeads: value })}
        render={(g) => T('{#0}（{#1}）', g, kindOf(g, c.qHeads))}
      />
      <Segment
        label={T('每个数的字节')}
        options={BYTE_OPTIONS}
        value={c.bytes}
        onSelect={(value) => set({ bytes: value })}
        render={(b) => (b === 4 ? T('4（FP32）') : b === 2 ? T('2（FP16/BF16）') : T('1（FP8）'))}
      />
      <div className="kv-result" role="status">
        <Tex
          tex={`${c.batch}\\times ${c.layers}\\times ${c.tokens}\\times ${c.kvHeads}\\times(${c.dHead}+${c.dHead})\\times ${c.bytes}=${thousands(bytes).replace(/,/g, '{,}')}\\ \\text{B}`}
        />
        <strong>= {formatBytes(bytes)}</strong>
        <span>
          {Tx(
            '每个 token {#0}；是同配置 MHA（{#1} 个 KV 头）的 {#2}。',
            formatBytes(perToken),
            c.qHeads,
            pct(bytes / mha),
          )}
        </span>
      </div>
      <p className="exp-caption">
        {Tx(
          '只统计 KV 缓存：不含模型权重、激活值和显存分配器的开销。{#0}；没有 batch 维的场景取 B = 1。',
          <Tex tex="d_k=d_v=d" />,
        )}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- 2 共享 KV 头

type ScoreGuess = 'less' | 'same' | 'more';
const SCORE_OPTIONS: { value: ScoreGuess; label: string }[] = [
  { value: 'less', label: T('变成原来的 1/4') },
  { value: 'same', label: T('不变') },
  { value: 'more', label: T('变多') },
];

export function GroupWidget({ state, onChange }: WidgetProps) {
  const [choice, setChoice] = useState<ScoreGuess | null>(null);
  const [revealed, setRevealed] = useState(false);
  const { kvHeads, head, query } = state.group;
  const set = (patch: Partial<KVState['group']>) =>
    onChange({ ...state, group: { ...state.group, ...patch } });
  const result = groupedAttention(X, MHA_WEIGHTS, kvHeads);
  const reference = repeatedKVReference(X, MHA_WEIGHTS, kvHeads);
  const diff = maxDiff(result.output, reference);
  const g = groupOf(head, Q_HEADS, kvHeads);
  const mates = headOptions.filter((h) => groupOf(h, Q_HEADS, kvHeads) === g);
  const all = KV_OPTIONS.map((option) => groupedAttention(X, MHA_WEIGHTS, option));
  const scoreAnswer: ScoreGuess =
    all[all.length - 1].scorePairs === all[0].scorePairs
      ? 'same'
      : all[all.length - 1].scorePairs < all[0].scorePairs
        ? 'less'
        : 'more';

  return (
    <div className="exp-widget">
      <Segment
        label={T('KV 头数 G')}
        options={KV_OPTIONS}
        value={kvHeads}
        onSelect={(value: KVHeads) => set({ kvHeads: value })}
        render={(option) => T('{#0}（{#1}）', option, kindOf(option, Q_HEADS))}
      />
      <Segment
        label={T('看哪个 Query 头')}
        options={headOptions}
        value={head}
        onSelect={(value) => set({ head: value })}
        render={(h) => `Q${h}`}
      />
      <Segment
        label={T('Query 位置 t')}
        options={tokenOptions}
        value={query}
        onSelect={(value) => set({ query: value })}
        render={(t) => `t${t}`}
      />
      <p className="exp-caption">
        {Tx(
          'Q{#0} 读第 {#1} 组 K/V；同组的 Query 头：{#2}。 它们查的是同一份 K，但各自的 Query 不同，权重也就不同。',
          head,
          g,
          mates.map((h) => `Q${h}`).join(T('、')),
        )}
      </p>
      <div className="exp-table-wrap">
        <table
          className="exp-table"
          aria-label={T('第 {#0} 组各 Query 头在位置 t{#1} 的权重', g, query)}
        >
          <thead>
            <tr>
              <th>{T('Key 位置')}</th>
              {mates.map((h) => (
                <th key={h} className={h === head ? 'kv-col-on' : undefined}>
                  {Tx('Q{#0} 的权重', h)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tokenOptions.map((j) => (
              <tr key={j}>
                <th scope="row">t{j}</th>
                {mates.map((h) => (
                  <td key={h} className={h === head ? 'kv-col-on' : undefined}>
                    {j > query ? T('屏蔽') : pct(result.probs[h][query][j])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('KV 头数改变了什么')}>
          <thead>
            <tr>
              <th>{T('设计')}</th>
              <th>{Tx('{#0} 参数', <Tex tex="W_Q" />)}</th>
              <th>{Tx('{#0} 参数', <Tex tex="W_K,W_V" />)}</th>
              <th>{Tx('{#0} 参数', <Tex tex="W_O" />)}</th>
              <th>{T('打分次数')}</th>
              <th>{T('缓存/token')}</th>
            </tr>
          </thead>
          <tbody>
            {all.map((r) => {
              const p = projectionParams({
                dModel: D_MODEL,
                qHeads: Q_HEADS,
                kvHeads: r.kvHeads,
                dHead: D_HEAD,
              });
              return (
                <tr key={r.kvHeads} className={r.kvHeads === kvHeads ? 'exp-new-row' : undefined}>
                  <th scope="row">{kindOf(r.kvHeads, Q_HEADS)}</th>
                  <td>{p.q}</td>
                  <td>{p.k + p.v}</td>
                  <td>{p.o}</td>
                  <td>{revealed ? r.scorePairs : '?'}</td>
                  <td>{r.cachePerToken}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Prediction
        question={T('KV 头从 4 减到 1，打分次数会怎样？')}
        options={SCORE_OPTIONS}
        choice={choice}
        onChoose={setChoice}
      />
      <div className="exp-actions">
        <button className="button primary" disabled={revealed} onClick={() => setRevealed(true)}>
          {T('核对打分次数 →')}
        </button>
      </div>
      {revealed && (
        <p className="exp-verdict ok" role="status">
          {Tx(
            '每个 Query 头仍要和每个可见 Key 打一次分：4 个头 × 10 个可见配对 = {#0} 次，与 KV 头数无关。',
            all[0].scorePairs,
          )}
          <PredictionResult
            label={choice && SCORE_OPTIONS.find((o) => o.value === choice)!.label}
            holds={choice === scoreAnswer}
          />
        </p>
      )}
      <p className="exp-verdict" role="status">
        {Tx(
          '按组查表的结果与“把每组 K/V 显式复制给每个 Query 头”的参考计算相比，输出最大差为 {#0}（容差 1e-12）。复制只用来核对正确性，真实实现不存这些副本。共享的 K/V 投影由组内各头取平均得到；论文在此之后还会继续训练，本页没有。',
          diffText(diff),
        )}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- 3 MLA 压缩

export function CompressWidget({ state, onChange }: WidgetProps) {
  const { head, query } = state.mla;
  const set = (patch: Partial<KVState['mla']>) =>
    onChange({ ...state, mla: { ...state.mla, ...patch } });
  const expanded = mlaExpanded(X, MLA_WEIGHTS);
  const { latent, ropeKey } = expanded.cache;
  return (
    <div className="exp-widget">
      <Segment
        label={T('看哪个位置 t')}
        options={tokenOptions}
        value={query}
        onSelect={(value) => set({ query: value })}
        render={(t) => `t${t}`}
      />
      <div className="kv-result" role="status">
        <span>
          {Tx(
            '缓存 {#0} = {#1}（{#2} 个数）',
            <Tex tex={`c_{${query}}=h_{${query}}W_{DKV}`} />,
            vec(latent[query]),
            D_LATENT,
          )}
        </span>
        <span>
          {Tx(
            '另存 {#0} = {#1}（{#2} 个数，下一步讲）',
            <Tex tex={`k^R_{${query}}`} />,
            vec(ropeKey[query]),
            D_ROPE,
          )}
        </span>
      </div>
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('由 c_{#0} 还原的各头 K、V', query)}>
          <caption>
            {Tx('用时由同一个 {#0} 还原各头的内容 K 和 V', <Tex tex={`c_{${query}}`} />)}
          </caption>
          <thead>
            <tr>
              <th>{T('头')}</th>
              <th>
                <Tex tex={`k^C=c\\,W_{UK}^{(i)}`} />
              </th>
              <th>
                <Tex tex={`v^C=c\\,W_{UV}^{(i)}`} />
              </th>
            </tr>
          </thead>
          <tbody>
            {headOptions.map((i) => (
              <tr key={i} className={i === head ? 'exp-new-row' : undefined}>
                <th scope="row">
                  <button
                    className="kv-row-pick"
                    aria-pressed={i === head}
                    onClick={() => set({ head: i })}
                  >
                    {Tx('头 {#0}', i)}
                  </button>
                </th>
                <td>{vec(expanded.kC[i][query])}</td>
                <td>{vec(expanded.vC[i][query])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-verdict ok" role="status">
        {Tx(
          '4 个头的 K 各不相同，但都由同一个 {#0} 维的 c 算出；MQA 里 4 个头读的是同一个 K。每个 token 每层：MHA 存 {#1} 个数，MLA 存 {#2} + {#3} = {#4} 个。',
          D_LATENT,
          Q_HEADS * 2 * D_HEAD,
          D_LATENT,
          D_ROPE,
          D_LATENT + D_ROPE,
        )}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- 4 解耦 RoPE

type ExactWhere = 'diagonal' | 'all' | 'none';
const WHERE_OPTIONS: { value: ExactWhere; label: string }[] = [
  { value: 'diagonal', label: T('只有 j = t') },
  { value: 'all', label: T('所有位置') },
  { value: 'none', label: T('没有位置') },
];

export function RopeWidget({ state, onChange }: WidgetProps) {
  const [choice, setChoice] = useState<ExactWhere | null>(null);
  const [revealed, setRevealed] = useState(false);
  const { head, query, coupled } = state.mla;
  const set = (patch: Partial<KVState['mla']>) => {
    setRevealed(false);
    onChange({ ...state, mla: { ...state.mla, ...patch } });
  };
  const expanded = mlaExpanded(X, MLA_WEIGHTS);
  const absorbed = mlaAbsorbed(X, MLA_WEIGHTS, expanded.cache);
  const rows = coupled
    ? coupledRopeRows(X, MLA_WEIGHTS, head, query)
    : tokenOptions.slice(0, query + 1).map((j) => {
        const exact = expanded.scores[head][query][j] * ROOT;
        const viaCache = absorbed.scores[head][query][j] * ROOT;
        return {
          key: j,
          distance: query - j,
          exact,
          absorbed: viaCache,
          error: Math.abs(exact - viaCache),
        };
      });
  // The prediction is about the coupled design, wherever the learner currently is.
  const coupledRows = coupledRopeRows(X, MLA_WEIGHTS, head, query);
  const exactAt = coupledRows.filter((row) => row.error < 1e-12).map((row) => row.key);
  const answer: ExactWhere =
    exactAt.length === coupledRows.length ? 'all' : exactAt.length === 0 ? 'none' : 'diagonal';

  return (
    <div className="exp-widget">
      <Segment
        label={T('RoPE 放在')}
        options={[true, false].map(String)}
        value={String(coupled)}
        onSelect={(value) => set({ coupled: value === 'true' })}
        render={(option) =>
          option === 'true' ? T('还原出的 K 上（不解耦）') : T('单独一支（解耦）')
        }
      />
      <Segment
        label={T('头')}
        options={headOptions}
        value={head}
        onSelect={(value) => set({ head: value })}
        render={(i) => T('头 {#0}', i)}
      />
      <Segment
        label={T('Query 位置 t')}
        options={tokenOptions}
        value={query}
        onSelect={(value) => set({ query: value })}
        render={(t) => `t${t}`}
      />
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('精确分数与吸收后分数的对照')}>
          <caption>
            {coupled
              ? T(
                  '精确值：RoPE 旋转后的 q 与还原并旋转后的 k 点积；吸收值：用同一个矩阵 W_UKᵀ 直接乘 c_j',
                )
              : T('精确值：先还原 K 再打分；吸收值：吸收后的 Query 乘 c_j，加上 q^R 与缓存的 k^R')}
          </caption>
          <thead>
            <tr>
              <th>{T('Key 位置 j')}</th>
              <th>{T('距离 t − j')}</th>
              <th>{T('精确 q·k')}</th>
              <th>{T('吸收后')}</th>
              <th>{T('误差')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              // In the coupled design the answer to the prediction stays hidden until checked.
              const hidden = coupled && !revealed;
              return (
                <tr
                  key={row.key}
                  className={!hidden && row.error > 1e-12 ? 'exp-changed' : undefined}
                >
                  <th scope="row">t{row.key}</th>
                  <td>{row.distance}</td>
                  <td>{fmt(row.exact)}</td>
                  <td>{hidden ? '?' : fmt(row.absorbed)}</td>
                  <td>{hidden ? '?' : row.error < 1e-12 ? '≤ 1e-12' : fmt(row.error)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Prediction
        question={T('不解耦时，同一个吸收矩阵在哪些位置还能算对？')}
        options={WHERE_OPTIONS}
        choice={choice}
        onChoose={setChoice}
      />
      <div className="exp-actions">
        <button className="button primary" disabled={revealed} onClick={() => setRevealed(true)}>
          {T('核对 →')}
        </button>
      </div>
      {revealed && (
        <p className="exp-verdict ok" role="status">
          {Tx(
            '不解耦时（头 {#0}，t = {#1}），误差为 0 的位置：{#2}。距离 t − j = 0 时两次旋转相互抵消；其他距离需要不同的矩阵。',
            head,
            query,
            exactAt.map((j) => `t${j}`).join(T('、')) || T('无'),
          )}
          <PredictionResult
            label={choice && WHERE_OPTIONS.find((o) => o.value === choice)!.label}
            holds={choice === answer}
          />
        </p>
      )}
      {!coupled && (
        <p className="exp-verdict" role="status">
          {Tx(
            '解耦后每个位置的误差都在 1e-12 以内：内容部分用吸收后的 Query 直接乘缓存的 {#0}，位置部分用缓存的 {#1}。代价是缓存每个 token 多存 {#2} 个数。',
            <Tex tex="c_j" />,
            <Tex tex="k^R_j" />,
            D_ROPE,
          )}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- 5 矩阵吸收

export function AbsorbWidget({ state, onChange }: WidgetProps) {
  const { head, query } = state.mla;
  const set = (patch: Partial<KVState['mla']>) =>
    onChange({ ...state, mla: { ...state.mla, ...patch } });
  const expanded = mlaExpanded(X, MLA_WEIGHTS);
  const absorbed = mlaAbsorbed(X, MLA_WEIGHTS, expanded.cache);
  const { latent, ropeKey } = expanded.cache;
  const outputDiff = maxDiff(absorbed.output, expanded.output);
  return (
    <div className="exp-widget">
      <Segment
        label={T('头')}
        options={headOptions}
        value={head}
        onSelect={(value) => set({ head: value })}
        render={(i) => T('头 {#0}', i)}
      />
      <Segment
        label={T('Query 位置 t')}
        options={tokenOptions}
        value={query}
        onSelect={(value) => set({ query: value })}
        render={(t) => `t${t}`}
      />
      <div className="kv-result" role="status">
        <span>
          {Tx(
            '{#0}： [{#1}, {#2}]，{#3}：[{#4}, {#5}]',
            <Tex tex={`\\tilde W_Q^{(${head})}=W_{QC}^{(${head})}{W_{UK}^{(${head})}}^{\\top}`} />,
            D_MODEL,
            D_LATENT,
            <Tex tex={`\\tilde W_O^{(${head})}=W_{UV}^{(${head})}W_O^{(${head})}`} />,
            D_LATENT,
            D_MODEL,
          )}
        </span>
        <span>
          {Tx(
            '{#0} = {#1}（{#2} 维，直接和缓存的 c 打分）',
            <Tex tex={`\\tilde q_{${query}}=h_{${query}}\\tilde W_Q^{(${head})}`} />,
            vec(absorbed.qLatent[head][query]),
            D_LATENT,
          )}
        </span>
      </div>
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('展开路径与吸收路径的打分对照')}>
          <thead>
            <tr>
              <th>{T('Key 位置')}</th>
              <th>{T('缓存 c_j')}</th>
              <th>{T('缓存 kᴿ_j')}</th>
              <th>{T('展开路径')}</th>
              <th>{T('吸收路径')}</th>
              <th>{T('权重')}</th>
            </tr>
          </thead>
          <tbody>
            {tokenOptions.slice(0, query + 1).map((j) => (
              <tr key={j}>
                <th scope="row">t{j}</th>
                <td>{vec(latent[j])}</td>
                <td>{vec(ropeKey[j])}</td>
                <td>{fmt(expanded.scores[head][query][j])}</td>
                <td>{fmt(absorbed.scores[head][query][j])}</td>
                <td>{pct(absorbed.probs[head][query][j])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-caption">
        {Tx(
          '打分已除以 {#0}。位置 t{#1} 的最终输出：展开 {#2}，吸收 {#3}。',
          <Tex tex="\sqrt{d+d_R}=2" />,
          query,
          vec(expanded.output[query]),
          vec(absorbed.output[query]),
        )}
      </p>
      <p className="exp-verdict ok" role="status">
        {Tx(
          '4 个位置、4 个头的输出，两条路径的最大差为 {#0}（容差 1e-12）。吸收路径没有还原任何 K、V，每个历史 token 只读缓存里的 {#1} 个数。',
          diffText(outputDiff),
          D_LATENT + D_ROPE,
        )}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- 6 对照

export function CompareWidget({ state, onChange }: WidgetProps) {
  const tokens = state.compare.tokens;
  const { layers, heads, dHead, latent, ropeDim } = DEEPSEEK_V2;
  const kv = (g: number) =>
    kvCacheBytes({
      batch: 1,
      layers,
      tokens,
      kvHeads: g,
      dK: dHead,
      dV: dHead,
      bytesPerElement: 2,
    });
  const mla = mlaCacheBytes({ batch: 1, layers, tokens, latent, ropeDim, bytesPerElement: 2 });
  const rows = [
    {
      name: T('MHA（{#0} 个 KV 头）', heads),
      perLayer: 2 * heads * dHead,
      bytes: kv(heads),
      cost: T('每个头各存 K、V，缓存最大。'),
    },
    {
      name: 'GQA-8',
      perLayer: 2 * 8 * dHead,
      bytes: kv(8),
      cost: T('组内共享 K、V；转换后需继续训练。'),
    },
    { name: 'MQA', perLayer: 2 * dHead, bytes: kv(1), cost: T('所有头共享一组 K、V。') },
    {
      name: T('MLA（DeepSeek-V2）'),
      perLayer: latent + ropeDim,
      bytes: mla,
      cost: T('每头仍有自己的 K、V；需要上投影或吸收后的更大 Query，RoPE 单独一支。'),
    },
  ];
  return (
    <div className="exp-widget">
      <Segment
        label={T('token 数')}
        options={COMPARE_TOKENS}
        value={tokens}
        onSelect={(value) => onChange({ ...state, compare: { tokens: value } })}
        render={(t) => `${t / 1024}K`}
      />
      <div className="exp-table-wrap">
        <table className="exp-table" aria-label={T('四种设计的 KV 缓存对照')}>
          <thead>
            <tr>
              <th>{T('设计')}</th>
              <th>{T('每 token 每层（个）')}</th>
              <th>{Tx('每 token（{#0} 层）', layers)}</th>
              <th>{thousands(tokens)} token</th>
              <th>{T('相对 MHA')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <th scope="row">{row.name}</th>
                <td>
                  {row.perLayer}
                  {row.name.startsWith('MLA') && T('（{#0} + {#1}）', latent, ropeDim)}
                </td>
                <td>{formatBytes(row.bytes / tokens)}</td>
                <td>{formatBytes(row.bytes)}</td>
                <td>{pct(row.bytes / rows[0].bytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="exp-verdict ok" role="status">
        {Tx(
          'MLA 每 token 每层 {#0} 个数 = {#1} 组 GQA 的量。只算 latent 会少算 {#2} 个，占 {#3}。',
          latent + ropeDim,
          (latent + ropeDim) / (2 * dHead),
          ropeDim,
          pct(ropeDim / (latent + ropeDim)),
        )}
      </p>
      <ul className="kv-costs">
        {rows.map((row) => (
          <li key={row.name}>{Tx('{#0}：{#1}', <strong>{row.name}</strong>, row.cost)}</li>
        ))}
      </ul>
      <p className="exp-caption">
        {Tx(
          'MHA、GQA-8、MQA 三行是按 DeepSeek-V2 的头数（{#0}）、头维（{#1}）和层数（{#2}）套公式得到的假想配置，只用来对照缓存；DeepSeek-V2 实际使用 MLA。BF16 每个数 2 字节，batch = 1。参数取自 DeepSeek-V2 论文 §3.1.2 与表 1。效果对比依赖具体训练与评测，本页不做排行。',
          heads,
          dHead,
          layers,
        )}
      </p>
    </div>
  );
}
