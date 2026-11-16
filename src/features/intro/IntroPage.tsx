import { useMemo, useState } from 'react';
import { AppShell } from '../../components/AppShell';
import { Rich } from '../../components/Tex';
import { GLOSSARY, GROUP_STORY, HOW_TO, PROMPT, ROUNDS, STATIONS } from '../../content/intro';
import { TOPICS, TOPIC_GROUPS } from '../../content/topics';
import { softmax } from '../../math/attention';
import { LOCALE, localePath, T } from '../../i18n';
import { Tx } from '../../i18n/jsx';
import '../../styles/editorial.css';
import '../../styles/intro.css';

/** Chinese runs words together; English and German separate them, but not before punctuation. */
function joinTokens(tokens: string[]): string {
  if (LOCALE === 'zh') return tokens.join('');
  return tokens.reduce((text, token) =>
    /^[.,!?]/.test(token) ? text + token : `${text} ${token}`,
  );
}

const percent = (p: number) => `${(p * 100).toFixed(1)}%`;

/** “Guess the next word” three times. Scores are invented; probabilities are softmax of them. */
function NextWordDemo() {
  const [chosen, setChosen] = useState<number[]>([]);
  const finished =
    chosen.length === ROUNDS.length || ROUNDS.some((round, i) => round[chosen[i]]?.ends);
  const round = finished ? undefined : ROUNDS[chosen.length];
  const probabilities = useMemo(() => (round ? softmax(round.map((c) => c.score)) : []), [round]);
  const words = chosen.map((choice, i) => ROUNDS[i][choice].token);
  const best = probabilities.indexOf(Math.max(...probabilities));

  return (
    <section className="intro-demo" aria-labelledby="intro-demo-title">
      <div className="intro-demo-head">
        <h3 id="intro-demo-title">{T('动手试试：让“模型”接三次词')}</h3>
        <span>
          {Tx('第 {#0} / {#1} 轮', Math.min(chosen.length + 1, ROUNDS.length), ROUNDS.length)}
        </span>
      </div>
      <p className="intro-sentence" aria-live="polite">
        <span className="intro-sentence-label">{T('目前的文字')}</span>
        <span>
          {joinTokens([PROMPT, ...words])}
          {!finished && <span className="intro-blank"> ___</span>}
        </span>
      </p>
      {round ? (
        <>
          <p className="intro-demo-question">
            {T(
              '下一个词是什么？模型给每个候选打一个分，再用 Softmax 换成概率。点一个词，把它接上去：',
            )}
          </p>
          <ol className="intro-candidates">
            {round.map((candidate, i) => (
              <li key={candidate.token}>
                <button
                  onClick={() => setChosen((c) => [...c, i])}
                  aria-label={T(
                    '选择“{#0}”：分数 {#1}，概率 {#2}',
                    candidate.label ?? candidate.token,
                    candidate.score.toFixed(1),
                    percent(probabilities[i]),
                  )}
                >
                  <span className="intro-candidate-token">
                    {candidate.label ?? candidate.token}
                  </span>
                  <span className="intro-candidate-score">
                    {Tx('分数 {#0}', candidate.score.toFixed(1))}
                  </span>
                  <span className="intro-candidate-bar" aria-hidden="true">
                    <i style={{ width: percent(probabilities[i]) }} />
                  </span>
                  <strong>{percent(probabilities[i])}</strong>
                  {i === best && <em>{T('最可能')}</em>}
                </button>
              </li>
            ))}
          </ol>
        </>
      ) : (
        <p className="intro-demo-done">
          {T(
            '一句话就这样一个词一个词地接出来了。真实模型会继续往下猜，直到它选中一个表示“结束”的特殊 token。',
          )}
        </p>
      )}
      <div className="intro-demo-actions">
        {chosen.length > 0 && (
          <button className="button secondary" onClick={() => setChosen([])}>
            {T('重新开始')}
          </button>
        )}
      </div>
      <p className="intro-demo-note">
        {T(
          '分数是为了演示编的，百分比是用 Softmax 实际算出来的。真实模型每一步要在几万个候选里打分，而且会根据前面的全部文字重新打分；这里为了简单，每一轮的候选是固定的。',
        )}
      </p>
    </section>
  );
}

export default function IntroPage() {
  const byId = (id: string) => TOPICS.find((topic) => topic.id === id)!;
  const first = TOPICS[0];
  return (
    <AppShell active="intro" crumb={T('导读')} title={T('导读')} indicator={T('浏览器内计算')}>
      <article className="intro-page">
        <header className="page-heading intro-heading">
          <div>
            <div className="eyebrow">{T('导读 · 大约 5 分钟')}</div>
            <h1>{T('开始之前：大语言模型到底在做什么？')}</h1>
            <p>
              {T(
                '这个网站把 ChatGPT 这类模型背后的计算，拆成一个个可以亲手算的小实验。不需要先学线性代数，也不用安装任何东西。先花几分钟把整体图景聊清楚，之后每做一个实验，你都知道自己在拼哪一块。',
              )}
            </p>
          </div>
        </header>

        <aside className="intro-chapter-entry">
          <strong>{T('正在跟着原书学习？')}</strong>
          <p>{T('按章节查看核心问题、前置知识、已有实验与尚待补充的内容。')}</p>
          <a href={localePath('/chapters')}>{T('打开 32 章与 3 个附录的知识主线 →')}</a>
        </aside>

        <section className="intro-section" aria-labelledby="intro-guess">
          <h2 id="intro-guess">
            <span>1</span>
            {T('其实它只做一件事：猜下一个词')}
          </h2>
          <p>
            {T(
              '不管是写邮件、回答问题还是写代码，模型每一步做的事都一样：看着前面已经有的文字，给每个可能的“下一个词”打一个分，把分数换成概率，再从中挑一个写上去。然后把这个新词接到末尾，开始下一轮。一整段回答，就是这样一个词一个词接出来的。',
            )}
          </p>
          <NextWordDemo />
          <p>
            {T(
              '这里先认识两个后面会反复出现的词。Token 是模型处理文字的小块：可能是一个词、一个字，也可能只是半个词；为了好懂，这里把一个词当成一个 token。Softmax 是把一组分数变成概率的公式：分数越高，概率越大，所有概率加起来正好是 1。',
            )}
          </p>
        </section>

        <section className="intro-section" aria-labelledby="intro-pipeline">
          <h2 id="intro-pipeline">
            <span>2</span>
            {T('分数从哪里来？一条加工流水线')}
          </h2>
          <p>
            {T(
              '模型不认识字，它只会算数。所以文字要先变成数字，再经过很多层结构相同的计算，最后才得到上面那张候选分数表。整个过程大致分这么几站，后面的实验各自放大其中一站：',
            )}
          </p>
          <ol className="intro-stations">
            {STATIONS.map((station, i) => (
              <li key={station.english}>
                <span className="intro-station-number" aria-hidden="true">
                  {i + 1}
                </span>
                <div>
                  <h3>
                    {station.title} <small lang="en">{station.english}</small>
                  </h3>
                  <p>{station.text}</p>
                  {station.topics.length > 0 ? (
                    <p className="intro-station-links">
                      {station.topics.map((id) => {
                        const topic = byId(id);
                        return (
                          <a key={id} href={topic.href}>
                            {topic.number} · {topic.title} →
                          </a>
                        );
                      })}
                    </p>
                  ) : null}
                </div>
              </li>
            ))}
          </ol>
          <p>
            {T(
              '网站里的其他实验，大多在回答同一个问题：这条流水线怎样更快、更省、更大？KV 架构和 FlashAttention 省的是显存和数据搬运；MoE 让模型变得很大，但每个词只用到其中一小部分；LoRA 和量化让微调和部署更便宜；偏好学习讲的是模型怎样学会说人们想听的话。',
            )}
          </p>
        </section>

        <section className="intro-section" aria-labelledby="intro-words">
          <h2 id="intro-words">
            <span>3</span>
            {T('看懂实验，只需要这几个词')}
          </h2>
          <p>
            {T(
              '实验里所有的计算，说到底都是乘法和加法，只是数字排成了表格。遇到下面这些词时，可以回来看一眼：',
            )}
          </p>
          <dl className="intro-glossary">
            {GLOSSARY.map((entry) => (
              <div key={entry.english}>
                <dt>
                  {entry.term}
                  {/* English pages already name the term in English. */}
                  {LOCALE !== 'en' && <small lang="en"> {entry.english}</small>}
                </dt>
                <dd>
                  <Rich text={entry.text} />
                </dd>
              </div>
            ))}
          </dl>
        </section>

        <section className="intro-section" aria-labelledby="intro-how">
          <h2 id="intro-how">
            <span>4</span>
            {T('这些实验怎么用')}
          </h2>
          <p>
            {T(
              '每个实验都用很小的例子：通常只有三四个词，每个词只有两到四个数，小到可以拿纸笔核对。建议按这个节奏来：',
            )}
          </p>
          <ol className="intro-how">
            {HOW_TO.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ol>
          <p className="intro-note">
            {T(
              '示例里的数字是教学用的玩具，不是真实模型里的数据；两三维的小向量也不代表真实模型里的“语义空间”。所有计算都在你的浏览器里完成，不会上传任何数据。',
            )}
          </p>
        </section>

        <section className="intro-section" aria-labelledby="intro-start">
          <h2 id="intro-start">
            <span>5</span>
            {T('从哪里开始？')}
          </h2>
          <p>
            {Tx(
              '第一次来的话，建议从 {#0} 开始，点“跟着做一次”：几分钟就能带你完成一次预测和验证。之后按编号往下走就行。只想复习某个主题，也可以随时打开顶部的“学习目录”直接跳过去。',
              <strong>
                {first.number} · {first.title}
              </strong>,
            )}
          </p>
          <div className="intro-path">
            {TOPIC_GROUPS.map((group, index) => (
              <section key={group.id} aria-labelledby={`intro-group-${group.id}`}>
                <h3 id={`intro-group-${group.id}`}>
                  <span>{Tx('阶段 {#0}', index + 1)}</span>
                  {group.title}
                </h3>
                <p>{GROUP_STORY[group.id]}</p>
                <ul>
                  {TOPICS.filter((topic) => topic.group === group.id).map((topic) => (
                    <li key={topic.id}>
                      <a href={topic.href}>
                        <span aria-hidden="true">{topic.number}</span>
                        <strong>{topic.title}</strong>
                      </a>
                      <small>{topic.subtitle}</small>
                    </li>
                  ))}
                </ul>
              </section>
            ))}
          </div>
          <a className="button primary intro-start" href={first.href}>
            {Tx('从 {#0} 开始 →', first.title)}
          </a>
        </section>
      </article>
    </AppShell>
  );
}
