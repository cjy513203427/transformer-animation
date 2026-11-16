import { useEffect, useMemo, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Prediction, PredictionResult } from '../../components/Controls';
import { Formula } from '../../components/Formula';
import { RelatedNotes } from '../../components/RelatedNotes';
import { ShareState } from '../../components/ShareState';
import { Tex } from '../../components/Tex';
import { WhyIntro } from '../../components/WhyIntro';
import { chapterUrl } from '../../content/book';
import { budgetLedger, type BudgetConfig } from '../../math/budget';
import {
  BUDGET_LIMITS,
  BUDGET_PRESETS,
  budgetProblem,
  restoreBudgetState,
  serializeBudgetState,
  type BudgetIntegerKey,
  type BudgetPreset,
} from '../../state/budget';
import { T, localePath } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/editorial.css';
import '../../styles/chapters.css';
import '../../styles/experiment.css';
import '../../styles/training-tools.css';
import '../../styles/budget.css';

const VERIFIED = '2026-11-16';
const count = (n: number) => Math.round(n).toLocaleString('en-US');
const compact = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : count(n);
const flops = (n: number) => n.toExponential(3);
const share = (part: number, whole: number) => (whole ? ((100 * part) / whole).toFixed(1) : '0.0') + '%';

/** A number input that keeps what is typed; only valid values reach the configuration. */
function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    if (Number(text) !== value) setText(String(value));
    // Only an outside change (preset, link) should overwrite the typed text.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <label>
      {label}
      <input
        type="number"
        inputMode={step === 1 ? 'numeric' : 'decimal'}
        min={min}
        max={max}
        step={step}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          const v = e.target.valueAsNumber;
          if (Number.isFinite(v) && v >= min && v <= max && (step !== 1 || Number.isInteger(v)))
            onChange(v);
        }}
      />
    </label>
  );
}

function Choice<V extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: V;
  options: [V, string][];
  onChange: (value: V) => void;
}) {
  return (
    <label>
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value as V)}>
        {options.map(([v, text]) => (
          <option key={v} value={v}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function BudgetLab() {
  const [restored] = useState(() => restoreBudgetState(window.location.hash));
  const [state, setState] = useState(restored.state);
  /** What the inputs show; it becomes the configuration once it describes a real model. */
  const [draft, setDraft] = useState(restored.state.config);
  const [notice, setNotice] = useState(restored.invalid ? T('分享状态无效，已恢复默认示例。') : '');
  const [prediction, setPrediction] = useState<'less' | 'same' | null>(null);
  const [verdict, setVerdict] = useState<{ saved: number; change: number } | null>(null);
  const c = state.config;
  const r = useMemo(() => budgetLedger(c), [c]);
  const problem = budgetProblem(draft);
  const recall = state.view.recall;

  function configure(next: BudgetConfig) {
    setDraft(next);
    setNotice('');
    setVerdict(null);
    if (!budgetProblem(next)) setState((s) => ({ ...s, config: next }));
  }
  const set = <K extends keyof BudgetConfig>(key: K, value: BudgetConfig[K]) =>
    configure({ ...draft, [key]: value });
  useEffect(() => {
    const restore = () => {
      const next = restoreBudgetState(window.location.hash);
      setState(next.state);
      setDraft(next.state.config);
      setVerdict(null);
      setNotice(next.invalid ? T('分享状态无效，已恢复默认示例。') : '');
    };
    window.addEventListener('hashchange', restore);
    return () => window.removeEventListener('hashchange', restore);
  }, []);

  const presets: [BudgetPreset, string][] = [
    ['gpt2', 'GPT-2 Small'],
    ['book18', T('原书第 18 章模型')],
    ['llama7b', 'Llama 2 7B'],
    ['llama70b', T('Llama 2 70B（GQA）')],
  ];
  const integerFields: [BudgetIntegerKey, string][] = [
    ['vocab', T('词表大小 V')],
    ['dModel', T('隐藏宽度 d_model')],
    ['layers', T('层数 L')],
    ['heads', T('Query 头数 H')],
    ['kvHeads', T('KV 头数 H_kv')],
    ['dFF', T('FFN 中间宽度 d_ff')],
    ['positions', T('可学习位置表行数 P')],
    ['context', T('序列长度 s')],
  ];
  const ffnMatrices = c.ffn === 'swiglu' ? 3 : 2;
  const normTex = c.norm === 'layer' ? '2d' : 'd';
  const rows: { name: string; tex: string; value: number }[] = [
    { name: 'Token Embedding', tex: 'V\\cdot d', value: r.parts.embedding },
    { name: T('可学习位置表'), tex: 'P\\cdot d', value: r.parts.position },
    {
      name: T('Attention 投影 W_Q、W_K、W_V、W_O'),
      tex:
        'L\\,(2d^2+2d\\,H_{kv}d_h)' + (c.bias === 'all' ? '+L\\,(2d+2H_{kv}d_h)' : ''),
      value: r.parts.attention,
    },
    {
      name: 'FFN',
      tex:
        `L\\cdot ${ffnMatrices}\\,d\\,d_{ff}` +
        (c.bias === 'none' ? '' : c.ffn === 'swiglu' ? '+L\\,(2d_{ff}+d)' : '+L\\,(d_{ff}+d)'),
      value: r.parts.ffn,
    },
    { name: T('归一化 γ（β）'), tex: `(2L+1)\\cdot ${normTex}`, value: r.parts.norm },
    {
      name: c.tied ? T('输出投影（与 Embedding 共享）') : T('输出投影'),
      tex: c.tied ? '0' : 'V\\cdot d',
      value: r.parts.head,
    },
  ];
  const nonEmbeddingShare = share(r.nonEmbedding, r.total);
  const approximationGap = share(Math.abs(r.approximation - r.nonEmbedding), r.nonEmbedding);
  const f = r.flops;
  const flopRows: [string, number][] = [
    [T('各层线性投影 L·2sM'), f.linear],
    [T('各层配对项 L·4s²d'), f.pairs],
    [T('输出 logits 2sdV'), f.logits],
  ];

  function compareTying() {
    if (!prediction) return;
    const tied = budgetLedger({ ...c, tied: true }),
      untied = budgetLedger({ ...c, tied: false });
    setVerdict({
      saved: untied.total - tied.total,
      change: untied.flops.forward - tied.flops.forward,
    });
  }

  return (
    <AppShell active="budget" title={T('参数量与计算量')} crumb={T('计算器')} indicator={T('浏览器内计算')}>
      <article className="chapter-page budget-page">
        <header className="page-heading">
          <div>
            <div className="eyebrow">{T('第 7、11、15、18 章 · 附录 A · 计算器')}</div>
            <h1>{T('一个配置，能数出多少参数和计算量？')}</h1>
            <p>
              {T(
                '从词表、宽度、层数和头数出发，逐项数出权重矩阵的元素个数，再按矩阵乘法数出一次前向的 FLOPs。四组预设可与原书和公开配置核对；改一个量，先预测哪几项会变。',
              )}
            </p>
          </div>
          <ShareState getHash={() => serializeBudgetState(state)} />
        </header>
        {notice && (
          <p className="chapter-notice" role="status">
            {notice}
          </p>
        )}
        <WhyIntro
          why={{
            question: T('为什么要自己数参数？'),
            points: [
              T(
                '“7B”“124M”只是一个总数。原书在第 4、7、11、15、18 章和附录 A 各算过一部分：Embedding 是 $V\\times d$，每层 Attention 是 $4d^2$，FFN 是 $8d^2$。分开读容易记混，也看不出总数随哪个配置变化。',
              ),
              T(
                '本页把这些算式放进同一张表：参数决定要存多少权重，FLOPs 决定一次前向要做多少乘加。两者并不总是一起变化，下面的预测会碰到一个反例。',
              ),
            ],
            book: [T('附录 A.2 · 参数量估算'), chapterUrl('附录A-Scaling-Law与计算量估算/')],
          }}
        />

        <section id="config" className="chapter-section">
          <div className="eyebrow">1</div>
          <h2>{T('选择一个模型配置')}</h2>
          <div className="chapter-preset-buttons" role="group" aria-label={T('预设配置')}>
            {presets.map(([id, name]) => (
              <button
                key={id}
                aria-pressed={JSON.stringify(c) === JSON.stringify(BUDGET_PRESETS[id])}
                onClick={() => configure({ ...BUDGET_PRESETS[id] })}
              >
                {name}
              </button>
            ))}
          </div>
          <div className="budget-fields">
            {integerFields.map(([key, label]) => (
              <NumberField
                key={key}
                label={label}
                value={draft[key]}
                min={BUDGET_LIMITS[key][0]}
                max={BUDGET_LIMITS[key][1]}
                onChange={(v) => set(key, v)}
              />
            ))}
            <NumberField
              label={T('训练 token D（十亿）')}
              value={draft.tokensB}
              min={0.001}
              max={100000}
              step={0.001}
              onChange={(v) => set('tokensB', v)}
            />
            <Choice
              label={T('FFN 结构')}
              value={draft.ffn}
              options={[
                ['gelu', T('GELU · 2 个矩阵')],
                ['swiglu', T('SwiGLU · 3 个矩阵')],
              ]}
              onChange={(v) => set('ffn', v)}
            />
            <Choice
              label={T('归一化')}
              value={draft.norm}
              options={[
                ['layer', T('LayerNorm · γ 与 β')],
                ['rms', T('RMSNorm · 只有 γ')],
              ]}
              onChange={(v) => set('norm', v)}
            />
            <Choice
              label={T('偏置')}
              value={draft.bias}
              options={[
                ['none', T('无偏置')],
                ['ffn', T('仅 FFN')],
                ['all', T('Q/K/V/O 与 FFN')],
              ]}
              onChange={(v) => set('bias', v)}
            />
            <Choice
              label={T('输出投影')}
              value={draft.tied ? 'tied' : 'separate'}
              options={[
                ['tied', T('与 Embedding 共享')],
                ['separate', T('独立矩阵')],
              ]}
              onChange={(v) => set('tied', v === 'tied')}
            />
          </div>
          {problem && (
            <p className="chapter-notice budget-problem" role="alert">
              {Tx('{#0} 下方仍显示最近一次有效配置的结果。', problem)}
            </p>
          )}
          <p className="budget-derived">
            {Tx(
              '每头宽度 d_h = d_model / H = {#0}；W_K、W_V 的输出宽度 H_kv·d_h = {#1}。P = 0 表示正弦、RoPE 或 ALiBi，这些方案没有位置参数。',
              r.headDim,
              r.kvWidth,
            )}
          </p>
          {!recall && (
            <p className="budget-explain">
              {T(
                '预设取自公开配置文件，总数由本页公式算出，可与公开数字对照：GPT-2 Small 约 124.4M，Llama 2 7B 约 6.74B，70B 约 68.98B。原书第 18 章模型采用书中参数表的设定（V = 50,000、仅 FFN 有偏置）。',
              )}
            </p>
          )}
        </section>

        <section id="parameters" className="chapter-section">
          <div className="eyebrow">2</div>
          <h2>{T('参数从哪里来')}</h2>
          <div className="chapter-table-scroll">
            <table className="chapter-table budget-table">
              <caption>{T('逐项参数；占比条的长度等于该项占总数的比例')}</caption>
              <thead>
                <tr>
                  <th scope="col">{T('部分')}</th>
                  <th scope="col">{T('算式')}</th>
                  <th scope="col">{T('参数')}</th>
                  <th scope="col">{T('占比')}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.name}>
                    <th scope="row">{row.name}</th>
                    <td>
                      <Tex tex={row.tex} />
                    </td>
                    <td className="budget-number">{count(row.value)}</td>
                    <td>
                      <span className="budget-bar" aria-hidden="true">
                        <span style={{ width: share(row.value, r.total) }} />
                      </span>
                      {share(row.value, r.total)}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <th scope="row">{T('合计 N')}</th>
                  <td />
                  <td className="budget-number" data-testid="budget-total">
                    {count(r.total)}
                  </td>
                  <td>{compact(r.total)}</td>
                </tr>
              </tfoot>
            </table>
          </div>
          {!recall && (
            <p className="budget-explain">
              {Tx(
                'Embedding 和输出投影只和词表、宽度有关；每个 Block 的参数与 V 无关。Attention 的四个投影里，只有 W_K、W_V 随 KV 头数变窄。d_ff = 4d 的两矩阵 FFN 是 8d²，约为 Attention 的两倍。当前配置中，各层与归一化（非 Embedding 参数）占 {#0}。',
                nonEmbeddingShare,
              )}
            </p>
          )}
          <p data-testid="budget-approximation">
            {Tx(
              '常用近似 12·L·d² = {#0}，精确的非 Embedding 参数为 {#1}，相差 {#2}。近似假设 MHA、d_ff = 4d、两矩阵 FFN，并忽略偏置与归一化；SwiGLU、GQA 或其他 d_ff 都会让它偏离。',
              compact(r.approximation),
              compact(r.nonEmbedding),
              approximationGap,
            )}
          </p>
        </section>

        <section id="flops" className="chapter-section">
          <div className="eyebrow">3</div>
          <h2>{T('一次前向要算多少')}</h2>
          <Formula
            expression={String.raw`C_{\mathrm{fwd}}(s)=L\,(2s\,M+4s^2d)+2s\,dV`}
            symbols={[
              ['M', T('每层 Attention 与 FFN 的权重元素数，不含偏置和归一化')],
              ['s', T('一次前向处理的 token 数')],
              ['4s^2d', T('每层 QKᵀ 与 PV：H 个头各做 s²·d_h 次乘加，两步共 4s²d FLOPs')],
              ['2s\\,dV', T('输出 logits；训练时每个位置都要算')],
            ]}
          />
          <div className="chapter-table-scroll">
            <table className="chapter-table budget-table">
              <caption>{Tx('s = {#0} 时一次前向的 FLOPs（1 次乘加 = 2 FLOPs）', count(c.context))}</caption>
              <thead>
                <tr>
                  <th scope="col">{T('项')}</th>
                  <th scope="col">FLOPs</th>
                  <th scope="col">{T('占比')}</th>
                </tr>
              </thead>
              <tbody>
                {flopRows.map(([name, value]) => (
                  <tr key={name}>
                    <th scope="row">{name}</th>
                    <td className="budget-number">{flops(value)}</td>
                    <td>
                      <span className="budget-bar" aria-hidden="true">
                        <span style={{ width: share(value, f.forward) }} />
                      </span>
                      {share(value, f.forward)}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <th scope="row">{T('一次前向合计')}</th>
                  <td className="budget-number" data-testid="budget-forward">
                    {flops(f.forward)}
                  </td>
                  <td>{Tx('每 token {#0}', flops(f.forwardPerToken))}</td>
                </tr>
              </tfoot>
            </table>
          </div>
          {!recall && (
            <>
              <p className="budget-explain">
                {Tx(
                  '每层配对项在 s* = M / (2d) = {#0} 时追上线性项。当前 s = {#1}，配对项占前向的 {#2}。s 翻倍时线性项翻倍、配对项变为 4 倍；因果实现可以跳过约一半的配对。',
                  count(f.crossover),
                  count(c.context),
                  share(f.pairs, f.forward),
                )}
              </p>
              <p className="budget-explain">
                {Tx(
                  '生成时每个新 token 只算一行投影，但仍要读 s 个缓存的 K/V：约 {#0} FLOPs。这是一步 Decode，不等于整段前向除以 s。',
                  flops(f.decodeToken),
                )}
              </p>
              <p className="budget-explain">
                {Tx(
                  'D/N = {#0} 个 token/参数。Chinchilla 的拟合给出约 20 的粗略起点；它来自特定数据与模型族，不是必须遵守的比例，也没有计入部署后的推理成本。',
                  f.tokensPerParameter.toFixed(1),
                )}
              </p>
              <p className="budget-explain">
                {Tx(
                  '训练每 token 约为前向的 3 倍（反向约为前向的 2 倍）：{#0} FLOPs；D = {#1} 个 token 合计 {#2}。粗估 6ND = {#3}，与它相差 {#4}：6ND 计入了查表时不做乘法的 Embedding，又没有计入 s² 配对项。',
                  flops(f.trainingPerToken),
                  flops(c.tokensB * 1e9),
                  flops(f.training),
                  flops(f.sixND),
                  share(Math.abs(f.sixND - f.training), f.training),
                )}
              </p>
            </>
          )}
          <p>
            <a href={localePath('/kv-architectures')}>{T('KV 缓存字节另算：打开 KV 架构对照 →')}</a>
          </p>
        </section>

        <section id="tying" className="chapter-section">
          <div className="eyebrow">{T('对照')}</div>
          <h2>{T('权重共享改变了什么？')}</h2>
          <div className="note-tool">
            <Prediction
              question={T('把输出投影改为与 Embedding 共享，一次前向的 FLOPs 会减少吗？')}
              options={[
                { value: 'less', label: T('会减少') },
                { value: 'same', label: T('不变') },
              ]}
              choice={prediction}
              onChoose={(v) => {
                setPrediction(v);
                setVerdict(null);
              }}
            />
            <button disabled={!prediction} onClick={compareTying}>
              {T('计算两种设置')}
            </button>
            {verdict && (
              <p className="note-result" role="status" data-testid="budget-verdict">
                {Tx(
                  '共享后参数减少 V·d = {#0}；一次前向 FLOPs 变化 {#1}。输出层仍要用这张 [V, d] 矩阵为每个位置算 logits，共享只是少存一份权重。',
                  count(verdict.saved),
                  count(verdict.change),
                )}
                <PredictionResult
                  label={prediction === 'less' ? T('会减少') : T('不变')}
                  holds={(prediction === 'same') === (verdict.change === 0)}
                />
              </p>
            )}
          </div>
        </section>

        <section id="review" className="chapter-section">
          <div className="eyebrow">{T('复述')}</div>
          <h2>{T('合上说明，试着复述')}</h2>
          <button
            className="button secondary"
            onClick={() => setState((s) => ({ ...s, view: { recall: !s.view.recall } }))}
          >
            {recall ? T('显示说明') : T('隐藏说明')}
          </button>
          <ol>
            <li>{T('不看表，写出 GPT-2 Small 一层 Attention 和一层 FFN 的参数，并说明哪一项随词表变化。')}</li>
            <li>{T('把 KV 头从 64 减到 8，哪几项参数会变，哪几项不变？KV 缓存呢？')}</li>
            <li>{T('序列长度翻倍时，线性项和配对项分别变成几倍？')}</li>
            <li>{T('为什么权重共享省参数，却不省前向计算？')}</li>
          </ol>
          <details>
            <summary>{T('查看参考解释')}</summary>
            <p>
              {T(
                '每层 Attention 是 4d²（含偏置再加 4d），两矩阵 FFN 是 8d²（含偏置再加 5d），都与 V 无关；V 只出现在 Embedding 和输出投影里。',
              )}
            </p>
            <p>
              {T(
                '减少 KV 头只让 W_K、W_V 变窄：Attention 从 4d² 变成 2d² + 2d²·H_kv/H；W_Q、W_O、FFN 与 Embedding 不变。KV 缓存与 H_kv 成正比，模型总参数不会同比缩小。',
              )}
            </p>
            <p>
              {T(
                's 翻倍时线性项翻倍，配对项变为 4 倍。共享输出矩阵只少存一份 [V, d] 权重，logits 仍需要 2sdV FLOPs。',
              )}
            </p>
          </details>
        </section>

        <RelatedNotes path="/model-budget" />
        <details className="details-panel source-panel">
          <summary>
            <span>{T('适用条件与来源')}</span>
            <span className="summary-hint">{T('统计范围 · 估算口径 · 参考文献')}</span>
          </summary>
          <div className="details-body">
            <p>
              {T(
                '只统计稠密 Decoder-only 模型的权重与矩阵乘法：不含 MoE、MLA 与 Encoder–Decoder，也不含优化器状态、激活、KV 缓存和运行时间。FLOPs 不计归一化、Softmax、激活函数、偏置加法与查表；配对项按完整 s×s 计算。',
              )}
            </p>
            <p>
              {T(
                'Kaplan 等人表 1 的每 token 配对项 2·n_layer·n_ctx·d_attn，相当于本页的 4sd 按因果实现减半；他们的 N 不含 Embedding。“训练 ≈ 3 × 前向”是常用估算，实际取决于重计算与并行方式。这些数字不是硬件测量。',
              )}
            </p>
            <ul>
              <li>
                <a href={chapterUrl('附录A-Scaling-Law与计算量估算/')} target="_blank" rel="noreferrer">
                  {T('原书附录 A · Scaling Law 与计算量估算')}
                </a>
              </li>
              <li>
                <a href={chapterUrl('第15章-Transformer完整前向传播-从输入到输出/')} target="_blank" rel="noreferrer">
                  {T('原书第 15.7 节 · GPT-2 Small 参数量')}
                </a>
              </li>
              <li>
                <a href={chapterUrl('第18章-手写Model.py-模型定义/')} target="_blank" rel="noreferrer">
                  {T('原书第 18.11 节 · 绑定与不绑定的参数表')}
                </a>
              </li>
              <li>
                <a href="https://arxiv.org/abs/2001.08361" target="_blank" rel="noreferrer">
                  Scaling Laws for Neural Language Models · §2.1, Table 1
                </a>
              </li>
              <li>
                <a href="https://huggingface.co/openai-community/gpt2/blob/main/config.json" target="_blank" rel="noreferrer">
                  GPT-2 · config.json
                </a>
              </li>
              <li>
                <a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noreferrer">
                  Llama 2 · arXiv 2307.09288
                </a>
                {T('；层数、宽度与头数按 NousResearch 镜像的 config.json 核对（Meta 原仓库需要授权）。')}
              </li>
            </ul>
            <p className="subtle">{Tx('来源核对日期：{#0}。', VERIFIED)}</p>
          </div>
        </details>
      </article>
    </AppShell>
  );
}
