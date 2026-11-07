import { useId } from 'react';
import { Rich } from '../../components/Tex';
import { ARCHITECTURE_NODES, type ArchitectureNode } from '../../state/lesson';
import '../../styles/architecture.css';

const book = 'https://waylandz.com/llm-transformer-book/';
const chapters = {
  map: book + '第03章-Transformer全景图/',
  qkv: book + '第10章-QKV到底是什么-Attention的三个主角/',
  heads: book + '第11章-Multi-Head-Attention-多视角理解/',
  forward: book + '第15章-Transformer完整前向传播-从输入到输出/',
};
interface NodeContent {
  title: string;
  english: string;
  input: string;
  action: string;
  output: string;
  hint: string;
  chapter: string;
}
const content: Record<ArchitectureNode, NodeContent> = {
  text: {
    title: '先有一段文字',
    english: 'Inputs',
    input: '例如：“我喜欢学习”。',
    action: '文字先进入分词器。为了看清位置，本页用“我 / 喜欢 / 学习”三个词块举例。',
    output: '交给 Tokenization，把文本转换成编号。',
    hint: '这三个词块是人为教学切分，不是真实分词器的输出。',
    chapter: '第 3、4 章',
  },
  tokenization: {
    title: '给 token 找到编号',
    english: 'Tokenization',
    input: '输入的文字或字节。',
    action: '分词器把文本拆成 token，并查出各自的词表 ID。ID 是索引，不是大小有意义的语义分数。',
    output: '一串 token ID，交给 Embedding 查表。',
    hint: '一个 token 可以是字、词片段、标点或字节片段，并非固定一个汉字。',
    chapter: '第 4 章',
  },
  embedding: {
    title: '编号变成向量',
    english: 'Input Embeddings',
    input: '每个位置的 token ID。',
    action: '用 ID 查嵌入表，每个 token 取出一行向量。向量是后续计算使用的表示。',
    output: '每个位置都有一个向量，再加入位置信息。',
    hint: '到了后续 Block，Attention 接收的是上一层的表示，已经不只是原始查表向量。',
    chapter: '第 4、14 章',
  },
  position: {
    title: '把顺序加进表示',
    english: 'Positional Encoding · ⊕',
    input: 'token 向量，以及对应位置的向量。',
    action: '按第 3 章图的加法位置编码，把同一位置的两条向量相加。旁路箭头说明位置向量从哪里加入。',
    output: '带位置信息的表示，进入第一个 Block。',
    hint: '此处遵循原图的加法方案；RoPE 作用于 Q/K，不是这里的加法。',
    chapter: '第 5、14 章',
  },
  attention: {
    title: '在位置之间读取信息',
    english: 'Masked Multi-Head Attention',
    input: '每个位置当前的表示 X。',
    action:
      '每个位置先用 Q 与可见位置的 K 打分，得到取用比例；再按比例汇总 V。多个头各算各的，拼接后通过 $W_O$。',
    output: '各位置得到一份上下文信息，交给第一个残差加法。',
    hint: '本页接下来会放大一个头。这里得到的是向量，还没有预测出下一个词。',
    chapter: '第 9–12 章',
  },
  'attention-add': {
    title: '把读到的信息加回来',
    english: 'Residual Add · ⊕',
    input: 'Attention 的输入 X，以及完整 MHA 的输出。',
    action: '绕行箭头保留子层输入，与 MHA 输出逐元素相加。不是将两条向量拼长。',
    output: '相加后的表示交给 LayerNorm。',
    hint: '这里必须先完成多头拼接与 $W_O$，不能把单头 O 直接当作完整 MHA 输出。',
    chapter: '第 13 章',
  },
  'attention-norm': {
    title: '整理这个位置的特征尺度',
    english: 'LayerNorm',
    input: 'Attention 残差相加后的向量。',
    action: '分别对每个位置的特征维做归一化，使后续层更容易处理数值尺度。',
    output: '交给同一个 Block 中的前馈网络。',
    hint: '按本图的 Post-Norm 顺序，是先 Attention、残差相加，再 LayerNorm。',
    chapter: '第 6、13 章',
  },
  ffn: {
    title: '每个位置分别加工',
    english: 'Feed Forward',
    input: '每个位置经过归一化的向量。',
    action:
      '使用线性层和非线性函数加工特征。同一组 FFN 参数用于各位置，但不在此处让不同位置互相读取。',
    output: '加工后的向量交给第二个残差加法。',
    hint: 'Attention 负责跨位置读取；FFN 在这一层逐位置变换特征。',
    chapter: '第 7、13 章',
  },
  'ffn-add': {
    title: '把加工结果加回来',
    english: 'Residual Add · ⊕',
    input: 'FFN 输入的旁路副本，以及 FFN 输出。',
    action: '两条同形状的表示逐元素相加。右侧绕行线从 FFN 之前出发，在此汇合。',
    output: '交给第二个 LayerNorm。',
    hint: '残差加法保留一条直接的信息通路，不是重复执行一个 FFN。',
    chapter: '第 13 章',
  },
  'ffn-norm': {
    title: '完成这一个 Block',
    english: 'LayerNorm',
    input: 'FFN 残差相加后的表示。',
    action: '再做一次按位置的特征归一化。这个 Block 的输出将作为下一个 Block 的输入。',
    output: '重复经过 N 个 Block 后，交给词表映射层。',
    hint: 'N 是层数，各层通常有独立参数；不是把同一组权重反复执行，也不是头数。',
    chapter: '第 3、13 章',
  },
  linear: {
    title: '向量变成词表分数',
    english: 'Linear · LM Head',
    input: '最后一个 Block 输出的表示。',
    action:
      '映射到词表维度，为每个候选 token 产生一个 logit。生成下一个 token 时使用最后位置的 logits。',
    output: '词表分数交给末端 Softmax。',
    hint: '这一步与 Attention 内的 $W_O$ 不同：这里的输出维度对应整个词表。',
    chapter: '第 3、15 章',
  },
  softmax: {
    title: '给候选 token 分配概率',
    english: 'Vocabulary Softmax',
    input: '一个位置上所有候选 token 的 logits。',
    action: '沿词表轴归一化，得到下一个 token 的概率分布。',
    output: '交给选取或采样步骤。',
    hint: 'Attention 内的 Softmax 沿 Key 轴分配取用比例；这里沿词表轴分配预测概率，两者不是一张表。',
    chapter: '第 6、16 章',
  },
  output: {
    title: '选出下一个 token',
    english: 'Output Probabilities',
    input: '最后位置的词表概率分布。',
    action:
      '可以取概率最高的候选，也可以按解码规则采样。将得到的新 token 追加到上下文，再开始下一次生成。',
    output: '新的 token；继续生成时重新进入模型。',
    hint: '本页只演算 Attention，没有真实词表或语言模型，因此不会编造预测出的词及其概率。',
    chapter: '第 2、16 章',
  },
};
const positions: Record<ArchitectureNode, number> = {
  text: 768,
  tokenization: 714,
  embedding: 660,
  position: 606,
  attention: 530,
  'attention-add': 480,
  'attention-norm': 435,
  ffn: 385,
  'ffn-add': 340,
  'ffn-norm': 295,
  linear: 213,
  softmax: 155,
  output: 95,
};
const labels: Record<ArchitectureNode, string> = {
  text: '输入文字',
  tokenization: 'Tokenization',
  embedding: 'Input Embeddings',
  position: '⊕',
  attention: 'Masked Multi-Head Attention',
  'attention-add': '⊕',
  'attention-norm': 'LayerNorm',
  ffn: 'Feed Forward',
  'ffn-add': '⊕',
  'ffn-norm': 'LayerNorm',
  linear: 'Linear · 词表映射',
  softmax: 'Softmax · 词表概率',
  output: '下一个 token',
};

export function TransformerOverview({
  selected,
  onSelect,
  onExploreAttention,
}: {
  selected: ArchitectureNode;
  onSelect: (node: ArchitectureNode) => void;
  onExploreAttention: () => void;
}) {
  const arrow = useId().replace(/:/g, '') + '-overview';
  const current = content[selected];
  const index = ARCHITECTURE_NODES.indexOf(selected);
  const choose = (node: ArchitectureNode) => onSelect(node);
  return (
    <section className="architecture-panel" aria-label="Transformer 整体架构">
      <div className="architecture-panel-head">
        <div>
          <span className="lesson-eyebrow">先定位，再放大</span>
          <h2>Attention 在整张图的哪里？</h2>
        </div>
        <a href={chapters.map} target="_blank" rel="noreferrer">
          对照原书第 3 章 ↗
        </a>
      </div>
      <div className="architecture-layout">
        <div className="architecture-map">
          <div className="architecture-map-label">
            <span>↑ 从下往上读</span>
            <span>第 3 章 · Post-Norm</span>
          </div>
          <svg
            viewBox="0 0 440 816"
            role="group"
            aria-label="从下往上的 Transformer 架构图，点击模块查看解释"
          >
            <defs>
              <marker
                id={arrow}
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M0 0 L10 5 L0 10Z" fill="#6d6961" />
              </marker>
            </defs>
            <rect x="61" y="263" width="318" height="305" rx="14" fill="#d0dcd7" stroke="#a1b1a7" />
            <text x="75" y="284" className="architecture-block-caption">
              Block
            </text>
            <text x="328" y="555" className="architecture-block-caption">
              × N 层
            </text>
            {ARCHITECTURE_NODES.slice(0, -1).map((node, i) => {
              const next = ARCHITECTURE_NODES[i + 1];
              const round = node.includes('add') || node === 'position';
              const nextRound = next.includes('add') || next === 'position';
              return (
                <path
                  key={node}
                  d={`M220 ${positions[node] - (round ? 14 : 20)} V${positions[next] + (nextRound ? 15 : 21)}`}
                  className="architecture-wire"
                  markerEnd={`url(#${arrow})`}
                />
              );
            })}
            <path
              d="M220 559 H88 V480 H204"
              className="architecture-wire residual"
              markerEnd={`url(#${arrow})`}
            />
            <path
              d="M220 410 H350 V340 H236"
              className="architecture-wire residual"
              markerEnd={`url(#${arrow})`}
            />
            <text
              x="80"
              y="526"
              className="architecture-residual-label"
              transform="rotate(-90 80 526)"
            >
              保留输入 X
            </text>
            <text
              x="366"
              y="401"
              className="architecture-residual-label"
              transform="rotate(-90 366 401)"
            >
              保留 FFN 输入
            </text>
            <rect x="291" y="590" width="134" height="32" rx="5" fill="#ddd3c8" stroke="#e1c8a7" />
            <text x="358" y="610" textAnchor="middle" className="architecture-position-label">
              Positional Encoding
            </text>
            <path d="M291 606 H237" className="architecture-wire" markerEnd={`url(#${arrow})`} />
            {ARCHITECTURE_NODES.map((node) => {
              const y = positions[node];
              const round = node.includes('add') || node === 'position';
              return (
                <g
                  key={node}
                  className={`architecture-node ${selected === node ? 'selected' : ''} ${node === 'attention' ? 'attention-node' : ''}`}
                  role="button"
                  tabIndex={0}
                  aria-label={`架构：${content[node].english}，${content[node].title}`}
                  aria-pressed={selected === node}
                  onClick={() => choose(node)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      choose(node);
                    }
                  }}
                >
                  {round ? (
                    <circle cx="220" cy={y} r="14" />
                  ) : (
                    <rect
                      x={node === 'attention' ? 106 : 121}
                      y={y - 20}
                      width={node === 'attention' ? 228 : 198}
                      height="40"
                      rx="7"
                    />
                  )}
                  <text
                    x="220"
                    y={y + 5}
                    textAnchor="middle"
                    className={round ? 'architecture-plus' : ''}
                  >
                    {labels[node]}
                  </text>
                  {selected === node && !round && (
                    <text x="331" y={y + 5} className="architecture-you">
                      ◀
                    </text>
                  )}
                </g>
              );
            })}
            <text x="220" y="35" textAnchor="middle" className="architecture-reading-label">
              输入表示 → 反复加工 → 预测
            </text>
          </svg>
          <p className="architecture-diagram-note">
            结构导航，未执行完整模型。图按原书重绘；Block × N 表示层数。
          </p>
        </div>
        <div className="architecture-explanation" aria-live="polite">
          <div className="architecture-location">
            你正在看 <strong>{current.english}</strong>
          </div>
          <h3>{current.title}</h3>
          <dl>
            <div>
              <dt>进来什么</dt>
              <dd>{current.input}</dd>
            </div>
            <div>
              <dt>做什么</dt>
              <dd>
                <Rich text={current.action} />
              </dd>
            </div>
            <div>
              <dt>交给谁</dt>
              <dd>{current.output}</dd>
            </div>
          </dl>
          <p className="architecture-hint">
            <Rich text={current.hint} />
          </p>
          <div className="architecture-chapter">
            对应 {current.chapter} ·{' '}
            <a href={chapters.map} target="_blank" rel="noreferrer">
              查看总图 ↗
            </a>
          </div>
          <div className="architecture-navigation">
            <button disabled={index === 0} onClick={() => choose(ARCHITECTURE_NODES[index - 1])}>
              ← 上一站
            </button>
            <span>
              {index + 1} / {ARCHITECTURE_NODES.length}
            </span>
            <button
              disabled={index === ARCHITECTURE_NODES.length - 1}
              onClick={() => choose(ARCHITECTURE_NODES[index + 1])}
            >
              下一站 →
            </button>
          </div>
          <div className="architecture-explore">
            <span>本次动手学习的部分</span>
            <p>把图里的 Attention 方框打开，跟着一个位置看清 Q、K、V 的分工。</p>
            <button className="button primary" onClick={onExploreAttention}>
              拆开 Attention →
            </button>
          </div>
          <details className="architecture-difference">
            <summary>为什么和 GPT-2 的图顺序不同？</summary>
            <p>
              这里跟随第 3 章：子层 → 残差相加 → LayerNorm（Post-Norm）。第 15 章 GPT-2 采用
              LayerNorm → 子层 → 残差相加（Pre-Norm），所有 Block 后还有 final
              LayerNorm。两种顺序不混画。
            </p>
            <a href={chapters.forward} target="_blank" rel="noreferrer">
              对照第 15 章 ↗
            </a>
          </details>
        </div>
      </div>
    </section>
  );
}
