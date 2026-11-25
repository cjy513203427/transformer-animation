import { useEffect, useMemo, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Formula } from '../../components/Formula';
import { Matrix } from '../../components/Matrix';
import { Prediction } from '../../components/Controls';
import { ShareState } from '../../components/ShareState';
import {
  ARCHITECTURE_FAMILIES,
  ARCHITECTURE_SYMBOLS as symbols,
  ARCHITECTURE_VERIFIED_DATE,
  MAMBA_NODES,
} from '../../content/postTransformer';
import { TOPICS } from '../../content/topics';
import { chapterUrl } from '../../content/book';
import {
  architectureLedger,
  structuredRecurrence,
  type ArchitectureConfig,
  type Mixer,
} from '../../math/architecture';
import { formatNumber as fmt } from '../../math/attention';
import {
  initialArchitectureState,
  isArchitectureAnchor,
  restoreArchitectureState,
  serializeArchitectureState,
  type ArchitectureState,
} from '../../state/architecture';
import { localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/editorial.css';
import '../../styles/chapters.css';
import '../../styles/experiment.css';

const anchors = ['overview', 'mamba', 'ssd', 'hybrid', 'alternatives', 'review'];
const bytes = (value: number) => `${value.toLocaleString('en-US')} B`;

export default function PostTransformerChapter() {
  const [restored] = useState(() => restoreArchitectureState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.invalid ? T('分享状态无效，已恢复默认示例。') : '');
  const [prediction, setPrediction] = useState<'decrease' | 'same' | null>(null);
  const [comparison, setComparison] = useState<{
    before: ReturnType<typeof architectureLedger>;
    after: ReturnType<typeof architectureLedger>;
  } | null>(null);
  const ledger = useMemo(() => architectureLedger(state.config), [state.config]);
  const recurrence = useMemo(
    () => structuredRecurrence(state.input.signal, state.input.retention),
    [state.input],
  );
  const node = MAMBA_NODES[state.view.node];
  const edit = (next: ArchitectureState) => {
    setState(next);
    setComparison(null);
    setPrediction(null);
  };
  const configure = (config: ArchitectureConfig) => edit({ ...state, config });
  useEffect(() => {
    const visitAnchor = () => {
      if (!isArchitectureAnchor(window.location.hash)) return false;
      const target = document.getElementById(window.location.hash.slice(1));
      if (target instanceof HTMLDetailsElement) target.open = true;
      setState((s) =>
        s.view.explanations ? s : { ...s, view: { ...s.view, explanations: true } },
      );
      requestAnimationFrame(() => target?.scrollIntoView({ block: 'start' }));
      return true;
    };
    const restore = () => {
      if (visitAnchor()) return;
      const r = restoreArchitectureState(window.location.hash);
      edit(r.state);
      setNotice(r.invalid ? T('分享状态无效，已恢复默认示例。') : '');
    };
    visitAnchor();
    window.addEventListener('hashchange', restore);
    return () => window.removeEventListener('hashchange', restore);
  }, []);
  function replaceAttention() {
    if (!prediction) return;
    const index = state.config.mixers.indexOf('attention');
    if (index < 0) return;
    const mixers = state.config.mixers.map((m, i) => (i === index ? 'ssm' : m)) as Mixer[];
    const config = { ...state.config, mixers };
    setComparison({ before: ledger, after: architectureLedger(config) });
    setState({ ...state, config });
  }
  const sectionLabels = [
    T('架构全景'),
    T('完整 Mamba Block'),
    T('SSD 的联系'),
    T('混合架构账本'),
    T('其他架构路线'),
    T('复述与边界'),
  ];
  return (
    <AppShell
      active="hybrid"
      title={T('后 Transformer 架构')}
      crumb={T('后 Transformer 架构')}
      indicator={T('浏览器内计算')}
    >
      <article className={`chapter-page ${state.view.explanations ? '' : 'chapter-recall-mode'}`}>
        <header className="page-heading">
          <div>
            <div className="eyebrow">{T('架构全景 · 从单一机制走向混合模型')}</div>
            <h1>{T('历史怎样被读取、压缩和组合？')}</h1>
            <p>
              {T(
                '先把 Attention、状态空间、长卷积与混合架构放在同一张地图里，再拆开 Mamba Block，核对一次结构化递推，最后亲自配置混合层的历史存储。',
              )}
            </p>
          </div>
          <ShareState getHash={() => serializeArchitectureState(state)} />
        </header>
        {notice && (
          <p className="chapter-notice" role="status">
            {notice}
          </p>
        )}
        <nav className="chapter-section-nav" aria-label={T('本页小节')}>
          {anchors.map((id, i) => (
            <a key={id} href={`#${id}`}>
              {sectionLabels[i]}
            </a>
          ))}
        </nav>
        <section id="overview" className="chapter-section">
          <div className="eyebrow">01</div>
          <h2>{T('先区分两个设计问题')}</h2>
          <p>
            {T(
              '序列混合器决定当前位置怎样使用历史；FFN 或 MoE 决定每个位置经过怎样的特征加工。混合架构可以同时改变这两个维度。',
            )}
          </p>
          <div className="architecture-map">
            <div>
              <strong>{T('直接访问历史')}</strong>
              <p>Attention</p>
              <small>{T('Query 查询可见位置的 K/V')}</small>
              <a href={localePath('/attention')}>{T('复习 Attention →')}</a>
            </div>
            <div>
              <strong>{T('把历史压进状态')}</strong>
              <p>SSM · Mamba · RWKV · RetNet</p>
              <small>{T('携带可更新的聚合状态')}</small>
              <a href={localePath('/ssm-mamba')}>{T('手算状态递推 →')}</a>
            </div>
            <div>
              <strong>{T('卷积与门控')}</strong>
              <p>Hyena</p>
              <small>{T('沿序列应用长过滤器与数据相关门控')}</small>
              <a href="#alternatives">{T('查看机制定位 →')}</a>
            </div>
          </div>
          <div className="architecture-axis">
            <strong>{T('另一条设计轴：Dense FFN ↔ MoE')}</strong>
            <p>
              {T(
                '它可以与上面的序列混合器组合。专家稀疏激活不表示历史记忆变成稀疏，也不自动减少 Attention 的 KV 缓存。',
              )}
            </p>
            <a href={localePath('/moe')}>{T('复习 Router、专家与合并 →')}</a>
          </div>
          <p>
            {T(
              '这些分组用于理解操作，不是互斥的模型排行榜；SSD 等工作正是在研究它们之间的数学联系。',
            )}
          </p>
        </section>
        <section id="mamba" className="chapter-section">
          <div className="eyebrow">02</div>
          <h2>{T('SSM 只是 Mamba Block 中的一段')}</h2>
          <p>
            {T(
              '下面是带 Pre-Norm 残差包装的 Mamba-1 结构导读。选中一段看它的输入、操作和输出；状态递推的数值实验仍可单独打开。',
            )}
          </p>
          <div className="mamba-flow" role="group" aria-label={T('Mamba Block 数据流')}>
            <div className="mamba-main-flow">
              {MAMBA_NODES.slice(0, 4).map((n, i) => (
                <button
                  key={n.id}
                  aria-pressed={state.view.node === i}
                  onClick={() => setState((s) => ({ ...s, view: { ...s.view, node: i } }))}
                >
                  <span>{i + 1}</span>
                  {n.title}
                  <small>
                    {i === 0 ? '[u, z]' : i === 1 ? 'u → v' : i === 2 ? 'v → Δ, B, C' : 'v, h → y'}
                  </small>
                </button>
              ))}
            </div>
            <p className="mamba-gate-branch">
              {T('z 从输入投影分支直接到 SiLU 门控；主分支 y 从 SSM 到这里汇合。')}
            </p>
            <div className="mamba-output-flow">
              {MAMBA_NODES.slice(4).map((n, i) => (
                <button
                  key={n.id}
                  aria-pressed={state.view.node === i + 4}
                  onClick={() => setState((s) => ({ ...s, view: { ...s.view, node: i + 4 } }))}
                >
                  <span>{i + 5}</span>
                  {n.title}
                </button>
              ))}
            </div>
          </div>
          <div className="chapter-module-reading" aria-live="polite">
            <h3>{node.title}</h3>
            <p>{node.explanation}</p>
            <Formula expression={node.formula} symbols={node.symbols} />
          </div>
          <p className="chapter-notice">
            {T(
              '本图解释完整 Block 的连接方式。链接中的标量实验核对其中的选择性与离散化；短卷积、扩展通道和门控在这里作结构讲解。',
            )}
          </p>
          <a className="chapter-action-link" href={localePath('/ssm-mamba')}>
            {T('打开标量 SSM：改输入、改 Δ、逐步核对 →')}
          </a>
          <a
            className="chapter-source-inline"
            href="https://arxiv.org/html/2312.00752v2"
            target="_blank"
            rel="noreferrer"
          >
            Mamba · §3.2–3.4
          </a>
          <a
            className="chapter-source-inline"
            href="https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py"
            target="_blank"
            rel="noreferrer"
          >
            Mamba · LayerNorm / RMSNorm
          </a>
        </section>
        <section id="ssd" className="chapter-section">
          <div className="eyebrow">03</div>
          <h2>{T('同一段历史，可以递推，也可以写成结构化矩阵')}</h2>
          <p>
            {T(
              'SSD 将受约束的状态空间模型与结构化 Attention 联系起来。先验证它最小的标量切片：写入与读出系数为 1，初始状态为 0，保留比例 a 在本例各步相同。',
            )}
          </p>
          <div className="chapter-control-panel">
            <div className="chapter-inline-controls">
              {state.input.signal.map((x, t) => (
                <label key={t}>
                  {`x${t}`}
                  <input
                    type="number"
                    aria-label={T('结构化递推输入 x{#0}', t)}
                    min={-4}
                    max={4}
                    step={0.25}
                    value={x}
                    onChange={(e) => {
                      const x = Number(e.target.value);
                      if (Number.isFinite(x) && x >= -4 && x <= 4)
                        edit({
                          ...state,
                          input: {
                            ...state.input,
                            signal: state.input.signal.map((v, i) => (i === t ? x : v)),
                          },
                        });
                    }}
                  />
                </label>
              ))}
            </div>
            <label className="chapter-range">
              {Tx('保留比例 a = {#0}', fmt(state.input.retention, 2))}
              <input
                type="range"
                aria-label={T('结构化递推保留比例')}
                min={0}
                max={1}
                step={0.05}
                value={state.input.retention}
                onChange={(e) =>
                  edit({ ...state, input: { ...state.input, retention: Number(e.target.value) } })
                }
              />
            </label>
          </div>
          <Formula
            expression={String.raw`\begin{gathered}h_t=ah_{t-1}+x_t,\quad h_{-1}=0,\quad y=xL^\top\\L_{tj}=\begin{cases}a^{t-j}&j\le t\\0&j>t\end{cases}\end{gathered}`}
            symbols={[symbols.ssd, symbols.matrix]}
          />
          <div className="ssd-computation">
            <Matrix
              name="L"
              data={recurrence.matrix}
              tokenColumns
              axesLabel={T('输出 t ↓ 输入 j →')}
            />
            <div className="chapter-table-scroll">
              <table className="chapter-table" aria-label={T('递推与结构化矩阵核对')}>
                <thead>
                  <tr>
                    <th>{T('位置')}</th>
                    <th>{T('递推 h')}</th>
                    <th>{T('矩阵读出 y')}</th>
                  </tr>
                </thead>
                <tbody>
                  {recurrence.recurrent.map((value, t) => (
                    <tr key={t}>
                      <th>{`t${t}`}</th>
                      <td>{fmt(value, 6)}</td>
                      <td>{fmt(recurrence.expanded[t], 6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p role="status" data-testid="ssd-error">
                {Tx('最大绝对误差 {#0} · 容差 10⁻¹²', recurrence.error.toExponential(1))}
              </p>
            </div>
          </div>
          <p>
            {T(
              'a=0 只保留当前输入；a=1 累加全部输入。上三角为零，所以修改未来输入不改变之前的结果。真实 SSD 使用更一般的参数与分块算法；此 4×4 矩阵仅用于核对，不是高效 kernel。',
            )}
          </p>
          <p className="chapter-notice">
            {T(
              '这里没有 Softmax。这个等价核对不能推出“任意 SSM 等于标准 Softmax Attention”，也不证明两类模型在任务上表现相同。',
            )}
          </p>
          <a
            className="chapter-source-inline"
            href="https://arxiv.org/html/2405.21060v1#S2.SS4"
            target="_blank"
            rel="noreferrer"
          >
            Mamba-2 / SSD · §2.4, §5
          </a>
        </section>
        <section id="hybrid" className="chapter-section">
          <div className="eyebrow">04</div>
          <h2>{T('混合架构：每层留什么，实际算一遍')}</h2>
          <p>
            {T(
              '原始 Jamba 结合 Attention、Mamba 与部分层的 MoE，发布配置的 Attention:Mamba 比为 1:7。本账本只借用比例构造八层教学例子，可独立修改序列混合器与 FFN/MoE。',
            )}
          </p>
          <div className="chapter-preset-buttons">
            <button
              onClick={() =>
                configure({
                  ...state.config,
                  mixers: state.config.mixers.map(() => 'attention'),
                  moe: state.config.moe.map(() => false),
                })
              }
            >
              {T('全部 Attention')}
            </button>
            <button
              onClick={() =>
                configure({
                  ...state.config,
                  mixers: state.config.mixers.map(() => 'ssm'),
                  moe: state.config.moe.map(() => false),
                })
              }
            >
              {T('全部 SSM')}
            </button>
            <button onClick={() => edit(initialArchitectureState())}>
              {T('恢复 1:7 教学预设')}
            </button>
            <button onClick={() => configure({ ...state.config, tokens: 1 })}>
              {T('短序列对照：T=1')}
            </button>
          </div>
          <div className="hybrid-layer-grid" aria-label={T('八层混合架构配置')}>
            {state.config.mixers.map((m, i) => (
              <fieldset key={i}>
                <legend>{T('第 {#0} 层', i + 1)}</legend>
                <label>
                  {T('序列混合器')}
                  <select
                    aria-label={T('第 {#0} 层的序列混合器', i + 1)}
                    value={m}
                    onChange={(e) =>
                      configure({
                        ...state.config,
                        mixers: state.config.mixers.map((v, j) =>
                          j === i ? (e.target.value as Mixer) : v,
                        ),
                      })
                    }
                  >
                    <option value="attention">Attention</option>
                    <option value="ssm">SSM</option>
                  </select>
                </label>
                <label className="chapter-checkbox">
                  <input
                    type="checkbox"
                    aria-label={T('第 {#0} 层使用 MoE', i + 1)}
                    checked={state.config.moe[i]}
                    onChange={(e) =>
                      configure({
                        ...state.config,
                        moe: state.config.moe.map((v, j) => (j === i ? e.target.checked : v)),
                      })
                    }
                  />
                  {T('使用 MoE')}
                </label>
              </fieldset>
            ))}
          </div>
          <div className="chapter-control-panel chapter-inline-controls">
            <label>
              {T('已有 token 数 T')}
              <input
                type="number"
                min={1}
                max={131072}
                value={state.config.tokens}
                onChange={(e) => {
                  const tokens = Number(e.target.value);
                  if (Number.isInteger(tokens) && tokens >= 1 && tokens <= 131072)
                    configure({ ...state.config, tokens });
                }}
              />
            </label>
            <label>
              {T('批大小 B')}
              <input
                type="number"
                min={1}
                max={8}
                value={state.config.batch}
                onChange={(e) => {
                  const batch = Number(e.target.value);
                  if (Number.isInteger(batch) && batch >= 1 && batch <= 8)
                    configure({ ...state.config, batch });
                }}
              />
            </label>
            <label>
              {T('每元素字节 b')}
              <select
                value={state.config.bytes}
                onChange={(e) =>
                  configure({ ...state.config, bytes: Number(e.target.value) as 2 | 4 })
                }
              >
                <option value={2}>2</option>
                <option value={4}>4</option>
              </select>
            </label>
          </div>
          <div className="chapter-table-scroll">
            <table className="chapter-table" aria-label={T('混合架构历史存储账本')}>
              <thead>
                <tr>
                  <th>{T('历史存储项')}</th>
                  <th>{T('元素个数')}</th>
                  <th>{T('字节数')}</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th>{T('Attention 的 K/V')}</th>
                  <td>{ledger.kvElements}</td>
                  <td data-testid="hybrid-kv">{bytes(ledger.kvBytes)}</td>
                </tr>
                <tr>
                  <th>{T('SSM 状态')}</th>
                  <td>{ledger.ssmElements}</td>
                  <td>{bytes(ledger.ssmBytes)}</td>
                </tr>
                <tr>
                  <th>{T('短卷积历史')}</th>
                  <td>{ledger.convElements}</td>
                  <td>{bytes(ledger.convBytes)}</td>
                </tr>
                <tr>
                  <th>{T('上述历史合计')}</th>
                  <td>{ledger.kvElements + ledger.ssmElements + ledger.convElements}</td>
                  <td data-testid="hybrid-history">{bytes(ledger.historyBytes)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <Formula
            expression={String.raw`\mathrm{KV\ bytes}=B L_A T H_{KV}(d_k+d_v)b,\qquad \mathrm{SSM+conv\ bytes}=B L_S D_{inner}(N+k-1)b`}
            symbols={[symbols.ledger, symbols.ssmCache]}
          />
          <p>
            {T(
              '本例 D=4；Attention 有 2 个 Query 头、2 个 KV 头，d_k=d_v=2；SSM 扩展通道 D_inner=8、每通道状态 N=4、卷积宽度 k=4。按最少 k−1 个旧卷积输入计数，真实实现可能保留更多槽位。',
            )}
          </p>
          <div className="chapter-comparison">
            <strong>{T('再单独看 FFN / MoE 子层')}</strong>
            <dl className="chapter-metrics">
              <div>
                <dt>{T('全部 FFN/MoE 参数')}</dt>
                <dd data-testid="hybrid-all-parameters">{ledger.allFfnParameters}</dd>
              </div>
              <div>
                <dt>{T('每 token 激活的 FFN/MoE 参数')}</dt>
                <dd>{ledger.activeFfnParameters}</dd>
              </div>
            </dl>
            <p>
              {T(
                '每个 Dense FFN 的参数为 2DF=64；MoE 有 E=4 个专家，每 token 选 K=2 个，Router 为 DE=16 个参数。此处只数这些子层，不含序列混合器、Embedding、Norm 和偏置。切换 MoE 不改变上面的历史存储。',
              )}
            </p>
          </div>
          <Prediction
            question={T('将一个 Attention 层换成 SSM 层，KV 缓存会怎样？')}
            options={[
              { value: 'decrease' as const, label: T('减少') },
              { value: 'same' as const, label: T('不变') },
            ]}
            choice={prediction}
            onChoose={setPrediction}
          />
          <button
            className="button primary"
            disabled={!prediction || ledger.attentionLayers === 0}
            onClick={replaceAttention}
          >
            {T('替换一个 Attention 层并核对')}
          </button>
          {comparison && (
            <div className="chapter-comparison" role="status">
              <p>
                {prediction === 'decrease'
                  ? T('预测与计算一致。')
                  : T('KV 缓存实际减少了；请核对 Attention 层数。')}
              </p>
              <p>
                {Tx(
                  'KV：{#0} → {#1}；全部历史存储：{#2} → {#3}。',
                  bytes(comparison.before.kvBytes),
                  bytes(comparison.after.kvBytes),
                  bytes(comparison.before.historyBytes),
                  bytes(comparison.after.historyBytes),
                )}
              </p>
              <p>
                {comparison.after.historyBytes > comparison.before.historyBytes
                  ? T('短序列下，新增 SSM 与卷积状态可能比省下的 KV 更大；不要只看 KV 一项。')
                  : T('这里历史存储减少；这不是质量或运行速度的测量。')}
              </p>
            </div>
          )}
          <p className="chapter-notice">
            {T(
              '账本只计算声明配置下的历史状态和指定子层参数，不是显存实测。只要仍保留完整 Attention 层，全序列训练/Prefill 就仍有二次配对项；单步 Decode 的历史访问另算。',
            )}
          </p>
          <a
            className="chapter-source-inline"
            href="https://arxiv.org/html/2403.19887v2#S2"
            target="_blank"
            rel="noreferrer"
          >
            Jamba · §2–3
          </a>
        </section>
        <section id="alternatives" className="chapter-section">
          <div className="eyebrow">05</div>
          <h2>{T('其他路线：保留机制定位，也保留差异')}</h2>
          <p>
            {T(
              '展开任一项查看它怎样处理历史、怎样组织计算，以及当前讲解的边界。模型版本和论文设定已经标出，不使用一列“更强/更快”排名代替解释。',
            )}
          </p>
          <div className="architecture-families">
            {ARCHITECTURE_FAMILIES.map((f) => (
              <details key={f.id} id={`family-${f.id}`}>
                <summary>
                  <strong>{f.name}</strong>
                  <span>{f.headline}</span>
                </summary>
                <div>
                  <p>{f.mechanism}</p>
                  <p>{f.history}</p>
                  <p>{f.boundary}</p>
                  <a href={f.source} target="_blank" rel="noreferrer">
                    {T('原始论文 ↗')}
                  </a>
                  {f.topic && (
                    <a href={TOPICS.find((t) => t.id === f.topic)!.href}>{T('相关机制实验 →')}</a>
                  )}
                </div>
              </details>
            ))}
          </div>
        </section>
        <section id="review" className="chapter-section">
          <button
            className="button secondary"
            onClick={() =>
              setState((s) => ({ ...s, view: { ...s.view, explanations: !s.view.explanations } }))
            }
          >
            {state.view.explanations ? T('隐藏图与说明') : T('显示图与说明')}
          </button>
          <h2>{T('隐藏图后，试着解释三个问题')}</h2>
          <ol>
            <li>{T('选择性 SSM 以外，完整 Mamba Block 还有哪些操作？')}</li>
            <li>{T('混合层的 KV 缓存与 SSM 状态分别怎样随 token 数变化？')}</li>
            <li>{T('为什么 SSD 的等价形式不能直接推出所有模型效果相同？')}</li>
          </ol>
          <details>
            <summary>{T('显示参考解释')}</summary>
            <p>
              {T(
                'Mamba-1 还包含扩展投影、短因果卷积、激活、门控、输出投影及残差包装。混合模型中的 Attention 层仍追加 K/V，固定维度 SSM 的状态与短卷积历史不随序列逐项增长。SSD 的等价需要特定结构和相同参数；模型表达、训练结果与真实性能仍需单独验证。',
              )}
            </p>
          </details>
        </section>
        <details className="chapter-section">
          <summary>{T('采用的设定、尚未实现的部分与来源')}</summary>
          <p>
            {T(
              '本页包含架构导读、标量 SSD 等价核对和混合架构账本；完整 Mamba-1、Mamba-2、RWKV、RetNet、Hyena 的训练与模型级数值复现仍需到原论文和作者实现。本例不输出未经训练的模型成绩或 GPU 加速倍数。',
            )}
          </p>
          <p>
            {T(
              '原书中文版示意中的 Jamba 1:3 配比与原始发布论文配置不同；这里采用论文中的 1:7。教学账本采用小尺寸参数，不复刻真实模型的层序、参数量或显存。',
            )}
          </p>
          <p>{Tx('来源核对：{#0}。', ARCHITECTURE_VERIFIED_DATE)}</p>
          <a
            href={chapterUrl('第32章-后Transformer架构-Mamba与混合模型/')}
            target="_blank"
            rel="noreferrer"
          >
            {T('阅读原书第 32 章 ↗')}
          </a>
        </details>
      </article>
    </AppShell>
  );
}
