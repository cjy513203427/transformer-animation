import { useEffect, useMemo, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Prediction, PredictionResult } from '../../components/Controls';
import { Formula } from '../../components/Formula';
import { RelatedNotes } from '../../components/RelatedNotes';
import { ShareState } from '../../components/ShareState';
import { WhyIntro } from '../../components/WhyIntro';
import { chapterUrl } from '../../content/book';
import { encodeWord, learnBPE, utf8Bytes, type CorpusWord } from '../../math/bpe';
import { randomValues } from '../../math/p1/shared';
import {
  BPE_PRESETS,
  MAX_MERGES,
  restoreBPEState,
  serializeBPEState,
  validCount,
  validWord,
  type BPEState,
} from '../../state/bpe';
import { T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/editorial.css';
import '../../styles/chapters.css';
import '../../styles/experiment.css';
import '../../styles/training-tools.css';
import '../../styles/bpe.css';

const CH4 = chapterUrl('第04章-Tokenization-文字如何变成数字/');
const EMBEDDING_WIDTH = 4;
/** Toy embedding row for an ID: seeded teaching numbers, not trained weights. */
const embeddingRow = (id: number) => randomValues(1000 + id, EMBEDDING_WIDTH);

function Tokens({ tokens, highlight }: { tokens: string[]; highlight?: string }) {
  return (
    <span className="bpe-tokens">
      {tokens.map((t, i) => (
        <span key={i} className={t === highlight ? 'bpe-token bpe-new' : 'bpe-token'}>
          {t}
        </span>
      ))}
    </span>
  );
}

/** A corpus row keeps what is typed; only a valid, unused word reaches the corpus. */
function WordRow({
  entry,
  taken,
  onChange,
  onRemove,
}: {
  entry: CorpusWord;
  taken: string[];
  onChange: (entry: CorpusWord) => void;
  onRemove?: () => void;
}) {
  const [text, setText] = useState(entry.word);
  useEffect(() => setText(entry.word), [entry.word]);
  const problem =
    text !== entry.word && !validWord(text)
      ? T('只允许 1–12 个小写字母 a–z。')
      : text !== entry.word && taken.includes(text)
        ? T('这个词已经在语料里。')
        : '';
  return (
    <div className="bpe-word-row">
      <label>
        {T('单词')}
        <input
          value={text}
          maxLength={12}
          onChange={(e) => {
            const v = e.target.value;
            setText(v);
            if (validWord(v) && !taken.includes(v)) onChange({ ...entry, word: v });
          }}
        />
      </label>
      <label>
        {T('出现次数')}
        <input
          type="number"
          min={1}
          max={50}
          value={entry.count}
          onChange={(e) => {
            const v = e.target.valueAsNumber;
            if (validCount(v)) onChange({ ...entry, count: v });
          }}
        />
      </label>
      {onRemove && (
        <button aria-label={T('删除单词 {#0}', entry.word)} onClick={onRemove}>
          ×
        </button>
      )}
      {problem && (
        <p className="bpe-problem" role="alert">
          {problem}
        </p>
      )}
    </div>
  );
}

export default function BPELab() {
  const [restored] = useState(() => restoreBPEState(window.location.hash));
  const [state, setState] = useState(restored.state);
  const [notice, setNotice] = useState(restored.invalid ? T('分享状态无效，已恢复默认示例。') : '');
  const [playing, setPlaying] = useState(false);
  const [prediction, setPrediction] = useState<'one' | 'count' | null>(null);
  const [verdict, setVerdict] = useState<{ reduction: number; count: number } | null>(null);
  const [probeText, setProbeText] = useState(state.probe);
  const [bytesText, setBytesText] = useState('low 低');

  const learned = useMemo(() => learnBPE(state.corpus, MAX_MERGES), [state.corpus]);
  const last = learned.snapshots.length - 1;
  const k = Math.min(state.step, last);
  const snap = learned.snapshots[k];
  const next = learned.merges[k];
  const justMerged = k > 0 ? learned.merges[k - 1].token : undefined;
  const recall = state.view.recall;
  const encoded = encodeWord(state.probe, learned.merges, k);

  function edit(nextState: BPEState) {
    setState(nextState);
    setNotice('');
    setVerdict(null);
    setPrediction(null);
  }
  const go = (step: number) => edit({ ...state, step: Math.max(0, Math.min(last, step)) });
  const setCorpus = (corpus: CorpusWord[]) => {
    setPlaying(false);
    edit({ ...state, corpus, step: 0 });
  };

  useEffect(() => {
    if (!playing) return;
    if (k >= last) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => go(k + 1), 1400);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, k, last]);
  useEffect(() => {
    const pause = () => document.hidden && setPlaying(false);
    const restore = () => {
      const r = restoreBPEState(window.location.hash);
      setPlaying(false);
      setState(r.state);
      setProbeText(r.state.probe);
      setVerdict(null);
      setNotice(r.invalid ? T('分享状态无效，已恢复默认示例。') : '');
    };
    document.addEventListener('visibilitychange', pause);
    window.addEventListener('hashchange', restore);
    return () => {
      document.removeEventListener('visibilitychange', pause);
      window.removeEventListener('hashchange', restore);
    };
  }, []);

  function checkPrediction() {
    if (!prediction || !next) return;
    const reduction = snap.totalTokens - learned.snapshots[k + 1].totalTokens;
    setVerdict({ reduction, count: next.count });
    setState((s) => ({ ...s, step: k + 1 }));
  }

  const presets: [keyof typeof BPE_PRESETS, string][] = [
    ['sennrich', T('Sennrich 玩具词典')],
    ['shared', T('共同词根 play')],
    ['tie', T('平局：ab 与 ba')],
  ];

  return (
    <AppShell active="bpe" title={T('分词：BPE 合并与查表')} crumb={T('第 4 章实验')} indicator={T('浏览器内计算')}>
      <article className="chapter-page bpe-page">
        <header className="page-heading">
          <div>
            <div className="eyebrow">{T('第 4 章 · Tokenization')}</div>
            <h1>{T('一个新词，怎样被切成 token 并查到编号？')}</h1>
            <p>
              {T(
                '在一个小语料上从字符开始执行 BPE：数相邻对、合并最高频的一对、扩充词表。合并完成后，用学到的规则切分一个语料里没有的词，再查出它的 ID 和 Embedding 行。',
              )}
            </p>
          </div>
          <ShareState getHash={() => serializeBPEState({ ...state, step: k })} />
        </header>
        {notice && (
          <p className="chapter-notice" role="status">
            {notice}
          </p>
        )}
        <WhyIntro
          why={{
            question: T('文字怎样变成模型能查表的编号？'),
            points: [
              T(
                '模型只接受整数 ID。Tokenizer 先把文字切成 token，再查词表得到编号，Embedding 按编号取出一行向量。按字符切会让序列很长，按整词切又会遇到没见过的词；BPE 从字符开始，把经常相邻出现的片段逐步合并成新 token。',
              ),
              T(
                '这一页执行原始的字符级 BPE（Sennrich 等，2016）：每个词末尾加上词尾符号 </w>，统计相邻对，合并最高频的一对，再用学到的合并规则切分新词。GPT-2 和 tiktoken 用的是字节级版本，最后一节说明区别。',
              ),
            ],
            book: [T('4.2.3 · 方法二：字节级 BPE / tiktoken'), CH4],
          }}
        />

        <section id="corpus" className="chapter-section">
          <div className="eyebrow">1</div>
          <h2>{T('训练语料')}</h2>
          <div className="chapter-preset-buttons" role="group" aria-label={T('预设语料')}>
            {presets.map(([id, name]) => (
              <button key={id} onClick={() => setCorpus(BPE_PRESETS[id].map((w) => ({ ...w })))}>
                {name}
              </button>
            ))}
          </div>
          <div className="bpe-corpus">
            {state.corpus.map((entry, i) => (
              <WordRow
                key={i}
                entry={entry}
                taken={state.corpus.filter((_, j) => j !== i).map((e) => e.word)}
                onChange={(e) => setCorpus(state.corpus.map((old, j) => (j === i ? e : old)))}
                onRemove={
                  state.corpus.length > 1
                    ? () => setCorpus(state.corpus.filter((_, j) => j !== i))
                    : undefined
                }
              />
            ))}
          </div>
          {state.corpus.length < 6 && (
            <button
              className="button secondary"
              onClick={() => {
                const word = ['tree', 'trees', 'street', 'stream', 'dream', 'dreams'].find(
                  (w) => !state.corpus.some((e) => e.word === w),
                )!;
                setCorpus([...state.corpus, { word, count: 1 }]);
              }}
            >
              {T('添加单词')}
            </button>
          )}
          {!recall && (
            <p className="budget-explain">
              {T(
                '每个词先拆成字母，再在末尾加上 </w>，表示词到这里结束；这样 “est</w>” 只匹配词尾的 est。出现次数表示这个词在语料中重复了几次，统计相邻对时按次数加权。',
              )}
            </p>
          )}
        </section>

        <section id="merges" className="chapter-section">
          <div className="eyebrow">2</div>
          <h2>{T('逐次合并')}</h2>
          <div className="bpe-controls" role="group" aria-label={T('合并步骤控制')}>
            <button onClick={() => go(k - 1)} disabled={k === 0}>
              {T('← 上一步')}
            </button>
            <button onClick={() => setPlaying((p) => !p)} disabled={k >= last && !playing}>
              {playing ? T('暂停') : T('播放')}
            </button>
            <button onClick={() => go(k + 1)} disabled={k >= last}>
              {T('下一步 →')}
            </button>
            <button
              onClick={() => {
                setPlaying(false);
                go(0);
              }}
            >
              {T('重置')}
            </button>
            <label>
              {Tx('已合并 {#0} / {#1} 次', k, last)}
              <input
                type="range"
                aria-label={T('跳到第几次合并')}
                min={0}
                max={last}
                value={k}
                onChange={(e) => {
                  setPlaying(false);
                  go(Number(e.target.value));
                }}
              />
            </label>
          </div>
          <Formula
            expression={String.raw`\mathrm{count}(a,b)=\sum_{w} f(w)\cdot n_w(a,b)`}
            symbols={[
              ['f(w)', T('单词 w 在语料中出现的次数')],
              ['n_w(a,b)', T('当前切分下，a 紧接着 b 在 w 中出现的次数')],
            ]}
          />
          <div className="bpe-grid">
            <div className="chapter-table-scroll">
              <table className="chapter-table">
                <caption>{T('当前切分')}</caption>
                <thead>
                  <tr>
                    <th scope="col">{T('单词')}</th>
                    <th scope="col">{T('次数')}</th>
                    <th scope="col">token</th>
                    <th scope="col">{T('token 数')}</th>
                  </tr>
                </thead>
                <tbody>
                  {state.corpus.map((entry, i) => (
                    <tr key={entry.word}>
                      <th scope="row">{entry.word}</th>
                      <td>{entry.count}</td>
                      <td>
                        <Tokens tokens={snap.words[i]} highlight={justMerged} />
                      </td>
                      <td>{snap.words[i].length}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <th scope="row" colSpan={3}>
                      {T('语料 token 总数（按次数加权）')}
                    </th>
                    <td data-testid="bpe-total">{snap.totalTokens}</td>
                  </tr>
                </tfoot>
              </table>
            </div>
            <div className="chapter-table-scroll">
              <table className="chapter-table">
                <caption>{T('相邻对计数（前 8 名）')}</caption>
                <thead>
                  <tr>
                    <th scope="col">{T('相邻对')}</th>
                    <th scope="col">{T('加权次数')}</th>
                  </tr>
                </thead>
                <tbody>
                  {snap.pairs.slice(0, 8).map(({ pair, count }, i) => (
                    <tr key={pair.join(' ')} className={i === 0 ? 'bpe-best' : undefined}>
                      <th scope="row">
                        {pair[0]} + {pair[1]}
                        {i === 0 && <span className="bpe-mark">{T('下一次合并')}</span>}
                      </th>
                      <td>{count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p data-testid="bpe-next">
            {next
              ? Tx('下一次合并：{#0} + {#1} → {#2}，加权次数 {#3}。', next.pair[0], next.pair[1], next.token, next.count)
              : T('已经没有相邻对可以合并，或已达到本页的 12 次上限。')}
          </p>
          <div className="bpe-vocab" aria-label={T('词表')}>
            <strong>{Tx('词表 · {#0} 个 token', snap.vocab.length)}</strong>
            <ol start={0}>
              {snap.vocab.map((token, id) => (
                <li key={id} className={token === justMerged ? 'bpe-new' : undefined}>
                  <span>{id}</span> {token}
                </li>
              ))}
            </ol>
          </div>
          {!recall && (
            <p className="budget-explain">
              {T(
                '每次只合并一对：把所有词中这对相邻符号换成一个新 token，并把它追加到词表末尾，所以词表每次加 1。出现最多的一对先合并；次数相同时，本页取字典序最小的一对（原论文没有规定，只是为了结果可复现）。',
              )}
            </p>
          )}
          {next && (
            <div className="note-tool">
              <Prediction
                question={T('执行下一次合并后，语料的 token 总数会减少多少？')}
                options={[
                  { value: 'one', label: T('只减少 1') },
                  { value: 'count', label: T('减少这一对的加权次数') },
                ]}
                choice={prediction}
                onChoose={(v) => {
                  setPrediction(v);
                  setVerdict(null);
                }}
              />
              <button disabled={!prediction} onClick={checkPrediction}>
                {T('执行合并并核对')}
              </button>
            </div>
          )}
          {verdict && (
            <p className="note-result" role="status" data-testid="bpe-verdict">
              {Tx(
                '这一对的加权次数是 {#0}，token 总数实际减少 {#1}。每处相邻出现都被换成 1 个 token，所以总数按出现次数减少；同一字母连续出现（如 aaa）时，重叠的对只能从左到右合并一次，减少量会小于计数。',
                verdict.count,
                verdict.reduction,
              )}
              <PredictionResult
                label={prediction === 'one' ? T('只减少 1') : T('减少这一对的加权次数')}
                holds={(prediction === 'count') === (verdict.reduction === verdict.count)}
              />
            </p>
          )}
        </section>

        <section id="encode" className="chapter-section">
          <div className="eyebrow">3</div>
          <h2>{T('切分新词并查表')}</h2>
          <label className="bpe-probe">
            {T('新词（小写字母 a–z）')}
            <input
              value={probeText}
              maxLength={12}
              onChange={(e) => {
                const v = e.target.value;
                setProbeText(v);
                if (validWord(v)) edit({ ...state, probe: v });
              }}
            />
          </label>
          {probeText !== state.probe && (
            <p className="bpe-problem" role="alert">
              {T('只允许 1–12 个小写字母 a–z；下方仍显示上一个有效的词。')}
            </p>
          )}
          <p>
            {Tx('用前 {#0} 条合并规则切分 {#1}：', k, state.probe)}{' '}
            <span data-testid="bpe-encoded">
              <Tokens tokens={encoded} />
            </span>
          </p>
          <div className="chapter-table-scroll">
            <table className="chapter-table">
              <caption>{Tx('查表：ID 是词表中的行号，E 是 [{#0}, {#1}] 的 Embedding 矩阵', snap.vocab.length, EMBEDDING_WIDTH)}</caption>
              <thead>
                <tr>
                  <th scope="col">token</th>
                  <th scope="col">ID</th>
                  <th scope="col">{T('E[ID]（教学数值）')}</th>
                </tr>
              </thead>
              <tbody>
                {encoded.map((token, i) => {
                  const id = snap.vocab.indexOf(token);
                  return (
                    <tr key={i}>
                      <th scope="row">{token}</th>
                      <td>{id < 0 ? T('不在词表') : id}</td>
                      <td>{id < 0 ? '—' : '[' + embeddingRow(id).join(', ') + ']'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {!recall && (
            <p className="budget-explain">
              {T(
                '新词不需要在语料里出现过：按学到的顺序逐条套用合并规则即可。ID 只是行号，相邻编号不代表意思相近；E 的数值是按 ID 生成的教学数值，没有训练过。语料里从未出现的字母不在这个字符级词表里，只能标为“不在词表”。',
              )}
            </p>
          )}
        </section>

        <section id="bytes" className="chapter-section">
          <div className="eyebrow">4</div>
          <h2>{T('字节级 BPE 为什么没有“未知字符”')}</h2>
          <label className="bpe-probe">
            {T('任意文字（只在本机显示，不写入分享链接）')}
            <input value={bytesText} maxLength={24} onChange={(e) => setBytesText(e.target.value)} />
          </label>
          <div className="chapter-table-scroll">
            <table className="chapter-table">
              <caption>{T('每个字符的 UTF-8 字节（十六进制）')}</caption>
              <thead>
                <tr>
                  <th scope="col">{T('字符')}</th>
                  <th scope="col">{T('字节')}</th>
                  <th scope="col">{T('字节数')}</th>
                </tr>
              </thead>
              <tbody>
                {[...bytesText].map((ch, i) => {
                  const bytes = utf8Bytes(ch);
                  return (
                    <tr key={i}>
                      <th scope="row">{ch === ' ' ? '␠' : ch}</th>
                      <td>{bytes.map((b) => b.toString(16).toUpperCase().padStart(2, '0')).join(' ')}</td>
                      <td>{bytes.length}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {!recall && (
            <p className="budget-explain">
              {T(
                'GPT-2 与 tiktoken 先把文本编码成 UTF-8 字节，以 256 个字节值为起点再做合并，所以任何文本都能编码，不会出现未知字符。一个常用汉字是 3 个字节，合并前就是 3 个 token；合并后可能成为 1 个 token，也可能仍然跨几个 token，取决于词表。它们还会先做预切分，并且不使用 </w>。本页只显示字节，没有运行真实的 tokenizer。',
              )}
            </p>
          )}
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
            <li>{T('不看页面，对 low ×5、lower ×2 手算第一轮相邻对计数，说出第一次合并哪一对。')}</li>
            <li>{T('为什么每次合并后词表正好多 1 个 token？语料 token 总数减少多少？')}</li>
            <li>{T('编码新词时，为什么要按学到的顺序套用规则，而不是重新统计新词里的相邻对？')}</li>
            <li>{T('ID 为 7 和 8 的两个 token，意思一定相近吗？Embedding 怎样用到 ID？')}</li>
          </ol>
          <details>
            <summary>{T('查看参考解释')}</summary>
            <p>
              {T(
                'l+o 与 o+w 各出现 7 次，w+</w> 出现 5 次；平局时按字典序先合并 l+o。每次合并把一对换成一个新符号，词表加 1，语料 token 总数减少这一对（不重叠的）出现次数。',
              )}
            </p>
            <p>
              {T(
                '合并规则是在语料上学到的固定列表，编码时按顺序套用，同一个词在任何文本中都得到同样的切分。ID 只是行号，Embedding 用它取出 E 的一行；向量之间的关系来自训练，不来自编号的大小。',
              )}
            </p>
          </details>
        </section>

        <RelatedNotes path="/tokenization" />
        <details className="details-panel source-panel">
          <summary>
            <span>{T('适用条件与来源')}</span>
            <span className="summary-hint">{T('简化假设 · 与真实 Tokenizer 的差别 · 参考文献')}</span>
          </summary>
          <div className="details-body">
            <p>
              {T(
                '本页实现 Sennrich 等人的字符级 BPE：词尾符号 </w>、按出现次数加权的相邻对计数、从左到右不重叠地合并；平局按字典序处理，最多 12 次合并。真实 tokenizer 还包括字节级起点、预切分规则、特殊 token、正规化与模型专用的合并表，必须使用与模型匹配的版本。',
              )}
            </p>
            <ul>
              <li>
                <a href={CH4} target="_blank" rel="noreferrer">
                  {T('原书第 4 章 · Tokenization')}
                </a>
              </li>
              <li>
                <a href="https://arxiv.org/abs/1508.07909" target="_blank" rel="noreferrer">
                  Neural Machine Translation of Rare Words with Subword Units · Algorithm 1
                </a>
              </li>
              <li>
                <a href="https://github.com/openai/tiktoken" target="_blank" rel="noreferrer">
                  tiktoken
                </a>
              </li>
            </ul>
            <p className="subtle">{Tx('来源核对日期：{#0}。', '2026-11-16')}</p>
          </div>
        </details>
      </article>
    </AppShell>
  );
}
