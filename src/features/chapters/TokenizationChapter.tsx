import { useEffect, useMemo, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Formula } from '../../components/Formula';
import { Matrix } from '../../components/Matrix';
import { Prediction } from '../../components/Controls';
import { ShareState } from '../../components/ShareState';
import { chapterUrl } from '../../content/book';
import { tokenizationSteps } from '../../content/tokenization';
import { localePath, T } from '../../i18n';
import { CORPORA, MAX_MERGES, MAX_TEXT_LENGTH, tokenText, tokenize } from '../../math/tokenization';
import {
  initialTokenizationState,
  restoreTokenizationState,
  serializeTokenizationState,
  validTokenizationText,
  type TokenizationState,
} from '../../state/tokenization';
import { ChapterTrail } from './ChapterIndex';
import '../../styles/editorial.css';
import '../../styles/chapters.css';
import '../../styles/lesson.css';
import '../../styles/experiment.css';
import '../../styles/tokenization.css';

const title = T('文字怎样变成 token ID，再变成向量？');
const book = chapterUrl('第04章-Tokenization-文字如何变成数字/');
const hex = (bytes: number[]) =>
  bytes.map((byte) => byte.toString(16).toUpperCase().padStart(2, '0')).join(' ');
const lastStep = tokenizationSteps.length - 1;

function Tokens({ pieces, vocabulary }: { pieces: number[][]; vocabulary: number[][] }) {
  if (!pieces.length) return <p>{T('空文本，序列长度为 0。')}</p>;
  const scrollable = pieces.length > 10 || pieces.flat().length > 24;
  return (
    <ol
      className={scrollable ? 'bpe-pieces bpe-pieces-scroll' : 'bpe-pieces'}
      tabIndex={scrollable ? 0 : undefined}
    >
      {pieces.map((piece, i) => (
        <li key={i}>
          <span>{T('片段 {#0}', i + 1)}</span>
          <ul className="bpe-tokens">
            {piece.map((id, position) => {
              const bytes = vocabulary[id];
              const readable = tokenText(bytes);
              return (
                <li key={position}>
                  {readable !== null && <bdi>{JSON.stringify(readable)}</bdi>}
                  <code>{hex(bytes)}</code>
                  <small>ID {id}</small>
                </li>
              );
            })}
          </ul>
        </li>
      ))}
    </ol>
  );
}

export default function TokenizationChapter() {
  const [initial] = useState(() => restoreTokenizationState(location.hash));
  const [state, setState] = useState(initial.state);
  const [invalid, setInvalid] = useState(initial.invalid);
  const [playing, setPlaying] = useState(false);
  const [prediction, setPrediction] = useState<'more' | 'same' | null>(null);
  const [comparison, setComparison] = useState<ReturnType<typeof tokenize> | null>(null);
  const result = useMemo(
    () => tokenize(state.input.text, state.config.corpus, state.config.merges, state.seed),
    [state.input.text, state.config.corpus, state.config.merges, state.seed],
  );
  const roundIndex = Math.min(state.view.round, result.rounds.length);
  const round = result.rounds[roundIndex - 1];
  const step = tokenizationSteps[state.step];

  function change(next: TokenizationState) {
    setPlaying(false);
    setState(next);
    setInvalid(false);
    setPrediction(null);
    setComparison(null);
  }

  function jump(next: number) {
    setPlaying(false);
    setState((current) => ({ ...current, step: Math.max(0, Math.min(lastStep, next)) }));
  }

  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(() => {
      const next = Math.min(lastStep, state.step + 1);
      setState((current) => ({ ...current, step: next }));
      if (next === lastStep) setPlaying(false);
    }, 4000 / state.view.speed);
    return () => window.clearTimeout(timer);
  }, [playing, state.step, state.view.speed]);

  useEffect(() => {
    const restore = () => {
      const next = restoreTokenizationState(location.hash);
      change(next.state);
      setInvalid(next.invalid);
    };
    const stopInBackground = () => {
      if (document.hidden) setPlaying(false);
    };
    window.addEventListener('hashchange', restore);
    document.addEventListener('visibilitychange', stopInBackground);
    return () => {
      window.removeEventListener('hashchange', restore);
      document.removeEventListener('visibilitychange', stopInBackground);
    };
  }, []);

  const usedIds = [...new Set(result.ids)].sort((a, b) => a - b);
  const answers = [
    { value: 'more' as const, label: T('会更多') },
    { value: 'same' as const, label: T('一样多') },
  ];
  return (
    <AppShell
      active="chapters"
      title={title}
      crumb={T('Tokenization')}
      indicator={T('浏览器内计算')}
    >
      <article className="chapter-page bpe-page">
        <header className="page-heading">
          <div>
            <div className="eyebrow">{T('第 {#0} 章', 4)} · Tokenization</div>
            <h1>{title}</h1>
            <p>
              {T(
                '用一个字节级 BPE 小例子走完整条路径。改文本、换语料，再验证同一段文本能否有不同的 token 数。',
              )}
            </p>
          </div>
        </header>
        {invalid && (
          <p className="chapter-notice" role="status">
            {T('分享状态无效，已恢复默认示例。')}
          </p>
        )}
        <section className="lesson-panel" aria-labelledby="bpe-lab">
          <div className="lesson-panel-heading">
            <h2 id="bpe-lab">{T('从字节开始，每轮合并都能核对')}</h2>
            <span className="lesson-progress">
              {state.step + 1} / {tokenizationSteps.length}
            </span>
          </div>
          <div className="lesson-steps bpe-steps">
            {tokenizationSteps.map((item, i) => (
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
          <div className="chapter-control-panel bpe-inputs">
            <label className="bpe-input">
              {T('待编码文本')}
              <textarea
                rows={2}
                maxLength={MAX_TEXT_LENGTH}
                value={state.input.text}
                onChange={(event) => {
                  const text = event.target.value;
                  if (validTokenizationText(text))
                    change({ ...state, input: { text }, view: { ...state.view, round: 0 } });
                }}
              />
            </label>
            <p className="exp-caption">
              {T(
                '输入上限为 {#0} 个 UTF-16 代码单元。空白与非空白先分段，各段保留原字节。',
                MAX_TEXT_LENGTH,
              )}
            </p>
            <div className="chapter-inline-controls">
              <button
                className="button secondary"
                onClick={() => change(initialTokenizationState())}
              >
                {T('混合文字')}
              </button>
              <button
                className="button secondary"
                onClick={() =>
                  change({
                    ...initialTokenizationState(),
                    config: { corpus: 'repeats', merges: 3 },
                    input: { text: 'aaa aa a' },
                  })
                }
              >
                {T('重复字符')}
              </button>
            </div>
            <div className="chapter-inline-controls">
              <label>
                {T('教学语料')}
                <select
                  value={state.config.corpus}
                  onChange={(event) =>
                    change({
                      ...state,
                      config: {
                        ...state.config,
                        corpus: event.target.value === 'repeats' ? 'repeats' : 'pairs',
                      },
                      view: { ...state.view, round: 0 },
                    })
                  }
                >
                  <option value="pairs">{T('重复的 ab')}</option>
                  <option value="repeats">{T('重叠的 aa')}</option>
                </select>
              </label>
              <label>
                {T('合并预算')}
                <input
                  type="range"
                  min={0}
                  max={MAX_MERGES}
                  value={state.config.merges}
                  onChange={(event) =>
                    change({
                      ...state,
                      config: { ...state.config, merges: event.target.valueAsNumber },
                      view: { ...state.view, round: 0 },
                    })
                  }
                />
                <span>{state.config.merges}</span>
              </label>
            </div>
          </div>
          <div className="bpe-stage">
            <div className="lesson-reading-top">
              <h3>{step.name}</h3>
              <button
                onClick={() => {
                  setPlaying(false);
                  setState((current) => ({
                    ...current,
                    view: { ...current.view, explanations: !current.view.explanations },
                  }));
                }}
              >
                {state.view.explanations ? T('隐藏说明') : T('显示说明')}
              </button>
            </div>
            {state.view.explanations ? (
              <p className="lesson-explanation">{step.description}</p>
            ) : (
              <div className="lesson-recall">
                <strong>{T('先自己说一遍')}</strong>
                <p>{step.prompt}</p>
              </div>
            )}
            {state.step === 0 && <Tokens pieces={result.bytes} vocabulary={result.vocabulary} />}
            {state.step === 1 && (
              <>
                <p>
                  <bdi className="bpe-corpus">{CORPORA[state.config.corpus]}</bdi>
                </p>
                {result.rules.length ? (
                  <div className="exp-table-wrap">
                    <table className="exp-table">
                      <thead>
                        <tr>
                          <th scope="col">{T('合并优先级')}</th>
                          <th scope="col">{T('相邻对')}</th>
                          <th scope="col">{T('计数')}</th>
                          <th scope="col">{T('实际替换')}</th>
                          <th scope="col">{T('新 token ID')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.rules.map((rule, i) => (
                          <tr key={rule.id}>
                            <th scope="row">{i + 1}</th>
                            <td>{rule.pair.join(' + ')}</td>
                            <td>{rule.count}</td>
                            <td>{rule.replacements}</td>
                            <td>
                              {rule.id} · {hex(result.vocabulary[rule.id])}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p>{T('当前预算未生成合并规则。基础词表仍包含全部 256 个字节。')}</p>
                )}
              </>
            )}
            {state.step === 2 && (
              <>
                <label className="bpe-round">
                  {T('编码轮次')}
                  <select
                    value={roundIndex}
                    onChange={(event) => {
                      setPlaying(false);
                      const index = Number(event.target.value);
                      setState((current) => ({
                        ...current,
                        view: { ...current.view, round: index },
                      }));
                    }}
                  >
                    <option value={0}>{T('第 0 轮：基础字节')}</option>
                    {result.rounds.map(({ rule }, i) => (
                      <option key={rule.id} value={i + 1}>
                        {T('规则 {#0}：{#1} + {#2} → {#3}', i + 1, ...rule.pair, rule.id)}
                      </option>
                    ))}
                  </select>
                </label>
                {round && <p>{T('本轮实际替换 {#0} 次。', round.replacements)}</p>}
                <Tokens pieces={round?.pieces ?? result.bytes} vocabulary={result.vocabulary} />
                <p className="exp-caption">
                  {T('显示 token 的字节与 ID。只在字节能独立解码时显示文字。')}
                </p>
                <p className="exp-caption">
                  {T(
                    '输入共 {#0} 字节，当前规则生成 {#1} 个 token，词表共有 {#2} 行。',
                    result.bytes.flat().length,
                    result.ids.length,
                    result.vocabulary.length,
                  )}
                </p>
              </>
            )}
            {state.step === 3 && (
              <>
                <label className="bpe-round">
                  {T('随机种子')}
                  <input
                    type="number"
                    min={1}
                    max={9999}
                    value={state.seed}
                    onChange={(event) => {
                      const seed = event.target.valueAsNumber;
                      if (Number.isInteger(seed) && seed >= 1 && seed <= 9999)
                        change({ ...state, seed });
                    }}
                  />
                </label>
                <p className="exp-caption">
                  {T('随机种子只生成教学嵌入表，BPE 规则和 ID 序列保持不变。')}
                </p>
                <p>
                  {T(
                    '嵌入表形状 {#0}，查表结果形状 {#1}。下方只展示这段输入用到的表行。',
                    `[${result.embedding.length}, 3]`,
                    `[${result.hidden.length}, 3]`,
                  )}
                </p>
                {usedIds.length > 0 && (
                  <div className="bpe-matrices">
                    <Matrix
                      name={T('嵌入表的已用行')}
                      data={usedIds.map((id) => result.embedding[id])}
                      rowLabels={usedIds.map((id) => `ID ${id}`)}
                    />
                    <Matrix
                      name={T('序列嵌入')}
                      data={result.hidden}
                      rowLabels={result.ids.map((id, t) => `t${t} · ID ${id}`)}
                    />
                  </div>
                )}
              </>
            )}
            {state.step === 4 && (
              <div className="exp-experiment">
                <Prediction
                  question={T('把预算设为 0，token 数会比现在多吗？')}
                  options={answers}
                  choice={prediction}
                  onChoose={(answer) => {
                    setPrediction(answer);
                    setComparison(null);
                  }}
                />
                <button
                  className="button secondary"
                  disabled={!prediction}
                  onClick={() =>
                    setComparison(tokenize(state.input.text, state.config.corpus, 0, state.seed))
                  }
                >
                  {T('验证预测')}
                </button>
                {comparison && (
                  <div className="exp-verdict" role="status">
                    <p>
                      {T(
                        '当前合并预算得到 {#0} 个 token。预算为 0 时得到 {#1} 个。',
                        result.ids.length,
                        comparison.ids.length,
                      )}{' '}
                      {prediction === (comparison.ids.length > result.ids.length ? 'more' : 'same')
                        ? T('与计算一致。')
                        : T('预测与计算不一致。')}
                    </p>
                    <p>
                      {comparison.decoded === state.input.text &&
                      result.decoded === state.input.text
                        ? T('两种设置都精确还原了输入文本。')
                        : T('解码结果和输入不一致。')}
                    </p>
                    <pre dir="auto">{result.decoded}</pre>
                  </div>
                )}
              </div>
            )}
            <details className="lesson-formula" key={state.step}>
              <summary>{T('需要时再看公式')}</summary>
              <Formula
                expression={step.formula}
                symbols={step.symbols}
                showSymbols={state.view.explanations}
              />
            </details>
          </div>
          <div className="lesson-controls">
            <div>
              <button
                className="button secondary"
                disabled={!state.step}
                onClick={() => jump(state.step - 1)}
              >
                {T('← 上一步')}
              </button>
              <button
                className="button primary"
                disabled={state.step === lastStep}
                onClick={() => jump(state.step + 1)}
              >
                {T('下一步 →')}
              </button>
            </div>
            <div>
              <button
                className="lesson-text-button"
                onClick={() => change(initialTokenizationState())}
              >
                {T('恢复默认示例')}
              </button>
              <button
                className="lesson-text-button"
                aria-label={playing ? T('暂停自动逐步') : T('播放自动逐步')}
                onClick={() => {
                  if (!playing && state.step === lastStep) jump(0);
                  setPlaying(!playing);
                }}
              >
                {playing ? 'Ⅱ' : '▶'}
              </button>
              <label>
                {T('速度')}
                <select
                  aria-label={T('播放速度')}
                  value={state.view.speed}
                  onChange={(event) => {
                    const speed = Number(event.target.value);
                    change({ ...state, view: { ...state.view, speed } });
                  }}
                >
                  <option value={0.5}>0.5×</option>
                  <option value={1}>1×</option>
                  <option value={2}>2×</option>
                </select>
              </label>
            </div>
          </div>
        </section>
        <div className="chapter-section">
          <ShareState getHash={() => serializeTokenizationState(state)} />
          <p className="exp-caption">{T('输入会写入分享链接，请使用可以公开的示例。')}</p>
        </div>
        <details className="chapter-section">
          <summary>{T('适用条件和参考资料')}</summary>
          <p>
            {T(
              '初始词表覆盖 256 个字节。训练只在示例语料中计数，编码按固定规则合并，各阶段均保留字节顺序。空白分段、同频处理与种子嵌入表都是本例的教学设定。',
            )}
          </p>
          <p>
            {T(
              '真实 tokenizer 可能使用不同的预分词规则、词表和特殊 token。本页的合并次数和 ID 只适用于当前教学配置。嵌入向量由种子生成，用于检验查表关系。',
            )}
          </p>
          <ul>
            <li>
              <a href={book} target="_blank" rel="noreferrer">
                {T('原书第 4 章')}
              </a>
            </li>
            <li>
              <a href="https://aclanthology.org/P16-1162/" target="_blank" rel="noreferrer">
                {T('BPE 子词论文')}
              </a>
            </li>
            <li>
              <a
                href="https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py"
                target="_blank"
                rel="noreferrer"
              >
                {T('tiktoken 教学实现')}
              </a>
            </li>
          </ul>
          <p className="subtle">{T('来源核对日期：{#0}。', '2026-11-21')}</p>
        </details>
        <p>
          <a href={localePath('/chapters/15')}>{T('继续看完整前向')} →</a>
        </p>
        <ChapterTrail id="04" />
      </article>
    </AppShell>
  );
}
