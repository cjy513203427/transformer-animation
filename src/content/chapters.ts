import { T, localePath } from '../i18n';
import { chapterUrl } from './book';
import type { TopicId } from './topics';

export type CoverageStatus = 'interactive' | 'explained' | 'reference' | 'missing';
export interface CoverageUnit {
  name: string;
  status: CoverageStatus;
  topics: TopicId[];
  href?: string;
}
export interface Chapter {
  id: string;
  part: string;
  title: string;
  question: string;
  summary: string;
  prerequisites: string[];
  units: CoverageUnit[];
  bookUrl: string;
}
export const CHAPTER_PARTS = [
  { id: 'intuition', title: T('建立直觉') },
  { id: 'components', title: T('核心组件') },
  { id: 'attention', title: T('Attention 机制') },
  { id: 'architecture', title: T('完整架构') },
  { id: 'code', title: T('代码实现') },
  { id: 'efficiency', title: T('生产优化') },
  { id: 'variants', title: T('架构变体') },
  { id: 'adaptation', title: T('部署与微调') },
  { id: 'frontier', title: T('前沿进展') },
  { id: 'appendices', title: T('附录') },
];
export const CHAPTERS: Chapter[] = [
  {
    id: '01',
    part: 'intuition',
    title: T('GPT 是什么'),
    question: T('语言模型在预测什么？'),
    summary: T(
      '自回归语言模型根据可见前缀预测下一个 token；模型的发展历史还涉及训练目标、数据与架构的变化。',
    ),
    prerequisites: [],
    bookUrl: chapterUrl('第01章-GPT是什么-LLM发展简史与核心思想/'),
    units: [
      { name: T('下一 token 的分布'), status: 'interactive', topics: [], href: localePath('/') },
      {
        name: T('发展历史与关键转折'),
        status: 'explained',
        topics: [],
        href: localePath('/chapters/15'),
      },
    ],
  },
  {
    id: '02',
    part: 'intuition',
    title: T('大模型的本质'),
    question: T('参数、程序和当前输入各自负责什么？'),
    summary: T(
      '参数保存学到的数值，前向程序规定怎样使用它们；当前输入产生的激活会变化，推理过程本身不更新权重。',
    ),
    prerequisites: ['01'],
    bookUrl: chapterUrl('第02章-大模型的本质-就是两个文件/'),
    units: [
      { name: T('参数与激活的分工'), status: 'explained', topics: [], href: localePath('/') },
      { name: T('前向时固定权重'), status: 'interactive', topics: ['attention'] },
    ],
  },
  {
    id: '03',
    part: 'intuition',
    title: T('Transformer 全景图'),
    question: T('一层里有哪些模块，信息怎样流动？'),
    summary: T(
      'Attention 混合不同位置的信息，FFN 加工每个位置；残差与归一化连接这些子层，多层堆叠后接词表输出头。',
    ),
    prerequisites: ['02'],
    bookUrl: chapterUrl('第03章-Transformer全景图/'),
    units: [
      { name: T('模块定位与残差路径'), status: 'explained', topics: ['attention'] },
      {
        name: T('完整计算链回放'),
        status: 'interactive',
        topics: [],
        href: localePath('/chapters/15'),
      },
    ],
  },
  {
    id: '04',
    part: 'components',
    title: T('Tokenization'),
    question: T('文字怎样变成编号，再变成向量？'),
    summary: T(
      'Tokenizer 先产生离散 ID，Embedding 再按 ID 查表；分词边界会改变序列长度，ID 的数值大小不代表语义距离。',
    ),
    prerequisites: ['02'],
    bookUrl: chapterUrl('第04章-Tokenization-文字如何变成数字/'),
    units: [
      {
        name: T('ID 与 Embedding 查表'),
        status: 'interactive',
        topics: [],
        href: localePath('/chapters/04'),
      },
      {
        name: T('字节级 BPE 与词表训练'),
        status: 'interactive',
        topics: [],
        href: localePath('/chapters/04'),
      },
      { name: T('字节、BPE 合并与查表的边界'), status: 'interactive', topics: ['bpe'] },
      { name: T('生产 tokenizer 的规则与词表'), status: 'reference', topics: [] },
      { name: T('Embedding 参数量 V×d'), status: 'interactive', topics: ['budget'] },
    ],
  },
  {
    id: '05',
    part: 'components',
    title: T('Positional Encoding'),
    question: T('顺序信息从哪里加入？'),
    summary: T(
      '同一 token 的内容嵌入可以相同，位置编码为不同位置提供不同的数值模式；加法方案在投影前改变输入。',
    ),
    prerequisites: ['04'],
    bookUrl: chapterUrl('第05章-Positional-Encoding-给文字加位置/'),
    units: [{ name: T('正弦频率、位置表与加法'), status: 'interactive', topics: ['position'] }],
  },
  {
    id: '06',
    part: 'components',
    title: T('LayerNorm 与 Softmax'),
    question: T('调整特征尺度与构造概率有什么区别？'),
    summary: T(
      'LayerNorm 在一个 token 的特征轴上统计均值和方差；Softmax 沿候选轴构造概率，减最大值只改变计算尺度。',
    ),
    prerequisites: ['04'],
    bookUrl: chapterUrl('第06章-LayerNorm与Softmax-数字的缩放与概率化/'),
    units: [
      {
        name: T('LayerNorm、RMSNorm 与仿射参数'),
        status: 'interactive',
        topics: ['normalization'],
      },
      { name: T('温度、Mask 与稳定 Softmax'), status: 'interactive', topics: ['normalization'] },
    ],
  },
  {
    id: '07',
    part: 'components',
    title: T('神经网络层'),
    question: T('线性层与非线性分别改变什么？'),
    summary: T(
      '矩阵投影改变特征表示，逐元素非线性让多层网络超出单次线性映射；FFN 对各 token 使用同一组参数。',
    ),
    prerequisites: ['06'],
    bookUrl: chapterUrl('第07章-神经网络层-不需要懂也能理解Transformer/'),
    units: [
      { name: T('扩维、激活与压回'), status: 'interactive', topics: ['ffn'] },
      { name: T('参数分布：Embedding、Attention、FFN 与 Norm'), status: 'interactive', topics: ['budget'] },
      {
        name: T('反向传播与优化器更新流程'),
        status: 'interactive',
        topics: [],
        href: localePath('/training-step'),
      },
      { name: T('完整模型的数值反向训练'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '08',
    part: 'attention',
    title: T('线性变换与矩阵乘法'),
    question: T('矩阵里的一个格子怎样算出来？'),
    summary: T(
      '矩阵乘法用左边的一行与右边的一列做点积；行向量约定下，参数矩阵把输入特征轴映射到输出特征轴。',
    ),
    prerequisites: ['04'],
    bookUrl: chapterUrl('第08章-线性变换的几何意义-矩阵乘法的本质/'),
    units: [
      { name: T('投影与逐格点积核对'), status: 'interactive', topics: ['attention'] },
      { name: T('二维线性变换的几何演示'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '09',
    part: 'attention',
    title: T('Attention 的几何逻辑'),
    question: T('点积、缩放和 Mask 怎样影响分数？'),
    summary: T(
      'Query 与 Key 的点积同时取决于方向和模长；缩放控制分数尺度，Mask 在 Softmax 前限定可见位置。',
    ),
    prerequisites: ['08'],
    bookUrl: chapterUrl('第09章-Attention的几何逻辑-为什么是点积/'),
    units: [{ name: T('点积、缩放与因果屏蔽'), status: 'interactive', topics: ['attention'] }],
  },
  {
    id: '10',
    part: 'attention',
    title: T('Q、K、V 的三个角色'),
    question: T('为什么修改 V 不会改变注意力权重？'),
    summary: T(
      '同一隐藏状态分别投影成 Q、K、V；Q/K 决定取用比例，V 提供内容，所以固定 Q/K 时两条支路可以分开验证。',
    ),
    prerequisites: ['09'],
    bookUrl: chapterUrl('第10章-QKV到底是什么-Attention的三个主角/'),
    units: [{ name: T('QKV 投影与 K/V 单变量对照'), status: 'interactive', topics: ['attention'] }],
  },
  {
    id: '11',
    part: 'attention',
    title: T('Multi-Head Attention'),
    question: T('切分头为什么不能切乱 token？'),
    summary: T(
      '投影后的特征列分配给不同头，每个头保留所有 token；reshape 与 transpose 改变轴的组织，不改变数值对应。',
    ),
    prerequisites: ['10'],
    bookUrl: chapterUrl('第11章-Multi-Head-Attention-多视角理解/'),
    units: [
      { name: T('投影、切头、转轴与独立 Attention'), status: 'interactive', topics: ['attention'] },
      { name: T('多头参数量与常见配置'), status: 'interactive', topics: ['budget'] },
    ],
  },
  {
    id: '12',
    part: 'attention',
    title: T('QKV 输出的本质'),
    question: T('各头怎样拼回去并相互混合？'),
    summary: T(
      '权重先对 Value 加权求和，各头沿特征维拼接；输出投影再读取所有头的列，产生新的隐藏表示。',
    ),
    prerequisites: ['11'],
    bookUrl: chapterUrl('第12章-QKV输出的本质/'),
    units: [{ name: T('加权和、拼接与输出投影'), status: 'interactive', topics: ['attention'] }],
  },
  {
    id: '13',
    part: 'architecture',
    title: T('残差连接与 Dropout'),
    question: T('信息和梯度为什么都需要一条直连路径？'),
    summary: T(
      '残差相加保留子层输入的直连项，但不保证信息无损；Dropout 在训练中随机屏蔽并缩放分支，推理时关闭。',
    ),
    prerequisites: ['07'],
    bookUrl: chapterUrl('第13章-残差连接与Dropout-训练稳定的秘密/'),
    units: [
      { name: T('残差、Jacobian 与训练/推理 Dropout'), status: 'interactive', topics: ['ffn'] },
    ],
  },
  {
    id: '14',
    part: 'architecture',
    title: T('词嵌入与位置相加'),
    question: T('相加后，内容与位置怎样共同影响分数？'),
    summary: T(
      '内容与位置同维相加，后续投影同时读取两者；展开点积可看到内容项、位置项及交叉项，不能把它们看成完全独立。',
    ),
    prerequisites: ['05', '10'],
    bookUrl: chapterUrl('第14章-词嵌入与位置信息的深层逻辑-为什么相加而不是拼接/'),
    units: [{ name: T('加法与点积四项分解'), status: 'interactive', topics: ['position'] }],
  },
  {
    id: '15',
    part: 'architecture',
    title: T('完整前向传播'),
    question: T('从 token ID 到下一个 token，所有模块怎样接起来？'),
    summary: T(
      '本章用小型 Pre-Norm Decoder 把查表、位置、多头 Attention、残差、GELU FFN 与词表输出串成一条真实计算链。',
    ),
    prerequisites: ['06', '11', '12', '13', '14'],
    bookUrl: chapterUrl('第15章-Transformer完整前向传播-从输入到输出/'),
    units: [
      {
        name: T('单层/双层完整数值回放'),
        status: 'interactive',
        topics: [],
        href: localePath('/chapters/15'),
      },
      {
        name: T('与真实 GPT-2 配置的差异'),
        status: 'explained',
        topics: [],
        href: localePath('/chapters/15#scope'),
      },
      { name: T('GPT-2 Small 参数量表'), status: 'interactive', topics: ['budget'] },
    ],
  },
  {
    id: '16',
    part: 'architecture',
    title: T('训练与推理'),
    question: T('训练可以并行，生成为什么仍有依赖？'),
    summary: T(
      '训练输入事先已知，因果 Mask 保证每个位置不偷看未来；生成的新 token 要等前一步选择后才成为输入。',
    ),
    prerequisites: ['15'],
    bookUrl: chapterUrl('第16章-训练与推理的异同-为什么推理要一个字一个字生成/'),
    units: [
      { name: T('右移目标、Prefill 与 Decode'), status: 'interactive', topics: ['inference'] },
      {
        name: T('损失、梯度与优化器的训练链'),
        status: 'interactive',
        topics: [],
        href: localePath('/training-step'),
      },
    ],
  },
  {
    id: '17',
    part: 'architecture',
    title: T('学习率'),
    question: T('同一个梯度，为什么步长不同会走向不同结果？'),
    summary: T(
      '学习率缩放优化器更新的步长；过大可能越过低损失区域，过小会减慢进展，Warmup 与衰减改变不同阶段的步长。',
    ),
    prerequisites: ['07', '16'],
    bookUrl: chapterUrl('第17章-学习率的理解-训练稳定的关键/'),
    units: [
      {
        name: T('优化轨迹、Warmup 与学习率调度'),
        status: 'interactive',
        topics: [],
        href: localePath('/training-step'),
      },
      { name: T('SGD、Adam 与 AdamW 的逐项更新'), status: 'interactive', topics: ['optimizer'] },
      { name: T('真实优化器与多维训练轨迹'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '18',
    part: 'code',
    title: T('手写 model.py'),
    question: T('公式中的张量对应代码里的哪些模块？'),
    summary: T(
      '模型定义把参数、张量形状与前向顺序组织成可重复调用的模块；先用数值链定位，再对照原书代码。',
    ),
    prerequisites: ['15'],
    bookUrl: chapterUrl('第18章-手写Model.py-模型定义/'),
    units: [
      {
        name: T('公式与前向模块对应'),
        status: 'explained',
        topics: [],
        href: localePath('/chapters/15'),
      },
      { name: T('PyTorch 实现与逐行代码联动'), status: 'reference', topics: [] },
      { name: T('权重绑定与不绑定的参数量'), status: 'interactive', topics: ['budget'] },
    ],
  },
  {
    id: '19',
    part: 'code',
    title: T('手写 train.py'),
    question: T('一次训练迭代怎样改变参数？'),
    summary: T(
      '训练循环依次取样本、前向算损失、反向求梯度、执行优化器更新；保存 checkpoint 还需考虑优化器和调度器状态。',
    ),
    prerequisites: ['16', '17'],
    bookUrl: chapterUrl('第19章-手写Train.py-训练循环/'),
    units: [
      { name: T('右移标签与因果前向'), status: 'interactive', topics: ['inference'] },
      { name: T('反向、优化器与 checkpoint 代码'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '20',
    part: 'code',
    title: T('手写 inference.py'),
    question: T('生成循环怎样把概率变成下一轮输入？'),
    summary: T(
      '每轮从最后位置的词表分布选择 token，追加到前缀，再继续计算；缓存路径还要维护每层的 K/V 与位置索引。',
    ),
    prerequisites: ['16', '22'],
    bookUrl: chapterUrl('第20章-手写Inference.py-推理逻辑/'),
    units: [
      { name: T('生成、采样与缓存机制'), status: 'interactive', topics: ['inference', 'decoding'] },
      { name: T('完整 Python 推理代码'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '21',
    part: 'efficiency',
    title: T('FlashAttention'),
    question: T('不保存完整分数矩阵，怎样得到同一个输出？'),
    summary: T(
      '分块读取 K/V，维护在线最大值、指数和与加权和；换尺度时一起重缩放，减少中间矩阵读写而保留完整 Attention。',
    ),
    prerequisites: ['10', '12'],
    bookUrl: chapterUrl('第21章-Flash-Attention-内存优化原理/'),
    units: [
      { name: T('分块、重缩放与等价核对'), status: 'interactive', topics: ['flash'] },
      { name: T('CUDA kernel、反向与真实性能测量'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '22',
    part: 'efficiency',
    title: T('KV Cache'),
    question: T('旧结果能复用哪些，什么时候失效？'),
    summary: T(
      '新位置用自己的 Query 读取历史 K/V，旧 Query 的输出已经完成；前缀、位置或权重改变时需要检查缓存依赖。',
    ),
    prerequisites: ['16'],
    bookUrl: chapterUrl('第22章-KV-Cache-推理加速/'),
    units: [
      { name: T('Prefill、Decode、重算对照与失效'), status: 'interactive', topics: ['inference'] },
      { name: T('缓存元素与字节账本'), status: 'interactive', topics: ['kv'] },
    ],
  },
  {
    id: '23',
    part: 'variants',
    title: T('MHA、MQA 与 GQA'),
    question: T('共享 KV 头会减少哪些量？'),
    summary: T(
      'Query 头可以保持不变，多个头改为读取共享 K/V；主要变化是 KV 投影与缓存，分组不会把整个模型同比缩小。',
    ),
    prerequisites: ['11', '22'],
    bookUrl: chapterUrl('第23章-MHA到MQA到GQA演进/'),
    units: [
      { name: T('分组映射、参数与缓存'), status: 'interactive', topics: ['kv'] },
      { name: T('投影参数随 KV 头数的变化'), status: 'interactive', topics: ['budget'] },
      { name: T('转换后的继续训练与质量评估'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '24',
    part: 'variants',
    title: T('Sparse 与 Infinite Attention'),
    question: T('少看历史与压缩历史有什么不同？'),
    summary: T(
      '稀疏连接限制可见位置，Linear Attention 改用特征核累积状态，Infini-attention 将局部 Attention 与跨段记忆组合。',
    ),
    prerequisites: ['09', '22'],
    bookUrl: chapterUrl('第24章-Sparse与Infinite-Attention/'),
    units: [
      { name: T('窗口、Linear 与 Infini 状态'), status: 'interactive', topics: ['memory'] },
      { name: T('更多稀疏拓扑与真实高效实现'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '25',
    part: 'variants',
    title: T('位置编码演进'),
    question: T('位置信息能加在输入、Q/K 还是分数上？'),
    summary: T(
      '加法位置编码改变输入，RoPE 按维度对旋转 Q/K，ALiBi 对分数加距离偏置；它们插入计算的位置不同。',
    ),
    prerequisites: ['05', '10'],
    bookUrl: chapterUrl('第25章-位置编码演进-Sinusoidal到RoPE到ALiBi/'),
    units: [
      { name: T('三处位置入口与共同平移核对'), status: 'interactive', topics: ['position'] },
      { name: T('PI、NTK-aware 与 YaRN 的上下文扩展'), status: 'interactive', topics: ['context'] },
    ],
  },
  {
    id: '26',
    part: 'adaptation',
    title: T('LoRA 与 QLoRA'),
    question: T('冻结基础权重后，低秩分支怎样改变输出？'),
    summary: T(
      'LoRA 学习缩放后的低秩权重增量；QLoRA 再把基础权重低比特存储，计算与适配器精度需分别说明。',
    ),
    prerequisites: ['08', '15'],
    bookUrl: chapterUrl('第26章-LoRA与QLoRA-高效微调/'),
    units: [
      { name: T('低秩分支、合并与存储口径'), status: 'interactive', topics: ['lora'] },
      { name: T('真实适配器训练与量化训练实现'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '27',
    part: 'adaptation',
    title: T('模型量化'),
    question: T('减少位数后，误差和额外元数据怎样计算？'),
    summary: T(
      '量化用步长与整数刻度近似浮点数，分组会改变误差和步长存储；GPTQ、AWQ 的校准方法还需要额外机制。',
    ),
    prerequisites: ['08'],
    bookUrl: chapterUrl('第27章-模型量化-GPTQ-AWQ-GGUF/'),
    units: [
      { name: T('对称量化、裁剪、分组与误差'), status: 'interactive', topics: ['quantization'] },
      { name: T('GPTQ、AWQ 算法与 GGUF 格式'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '28',
    part: 'frontier',
    title: T('Prompt Engineering'),
    question: T('改变提示如何改变条件分布与推理过程？'),
    summary: T(
      '提示提供模型本次使用的上下文，通常不改变权重；示例、采样投票与搜索分别改变条件信息或外部执行流程。',
    ),
    prerequisites: ['16'],
    bookUrl: chapterUrl('第28章-Prompt-Engineering-提示工程实战/'),
    units: [
      {
        name: T('提示、示例与上下文的作用'),
        status: 'explained',
        topics: [],
        href: localePath('/prompt-engineering'),
      },
      {
        name: T('提示链：提取、计算与核对'),
        status: 'interactive',
        topics: [],
        href: localePath('/prompt-engineering'),
      },
      {
        name: T('Self-Consistency 答案投票'),
        status: 'interactive',
        topics: [],
        href: localePath('/prompt-voting'),
      },
      {
        name: T('思路搜索与回退'),
        status: 'interactive',
        topics: ['prompt-search'],
      },
      {
        name: T('CoT、ToT、ReAct 与提示评估'),
        status: 'explained',
        topics: [],
        href: localePath('/prompt-search'),
      },
      { name: T('真实模型的采样、搜索与提示效果'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '29',
    part: 'frontier',
    title: T('RLHF 与偏好学习'),
    question: T('偏好数据怎样变成可优化的目标？'),
    summary: T(
      '偏好对可以训练奖励差或直接约束策略概率；PPO 的概率比、裁剪与参考约束分别控制更新的不同部分。',
    ),
    prerequisites: ['16'],
    bookUrl: chapterUrl('第29章-RLHF与偏好学习-让模型对齐人类/'),
    units: [
      { name: T('奖励差、DPO 与 PPO 裁剪'), status: 'interactive', topics: ['preference'] },
      {
        name: T('Rollout、价值函数与 RLHF 流程'),
        status: 'interactive',
        topics: [],
        href: localePath('/policy-update'),
      },
      {
        name: T('奖励偏差、RLAIF 与对齐评估'),
        status: 'explained',
        topics: [],
        href: localePath('/policy-update'),
      },
      { name: T('真实策略与奖励模型训练'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '30',
    part: 'frontier',
    title: T('Mixture of Experts'),
    question: T('选路、专家计算与合并怎样保留 token 身份？'),
    summary: T(
      'Router 给专家打分，Top-K 决定分支，门值合并专家输出；稀疏激活、权重存储、负载和通信要分别统计。',
    ),
    prerequisites: ['07', '12'],
    bookUrl: chapterUrl('第30章-Mixture-of-Experts-稀疏激活的秘密/'),
    units: [
      { name: T('Router、Top-K、合并与专家并行'), status: 'interactive', topics: ['moe'] },
      { name: T('MLA 压缩、解耦 RoPE 与矩阵吸收'), status: 'interactive', topics: ['kv'] },
      { name: T('共享专家、容量限制与辅助损失'), status: 'reference', topics: [] },
    ],
  },
  {
    id: '31',
    part: 'frontier',
    title: T('推理模型与推理预算'),
    question: T('训练目标与多花推理计算分别改变什么？'),
    summary: T(
      'GRPO 从同一 prompt 的多条回答构造组内优势；蒸馏学习教师分布，推理预算则属于生成时的计算分配。',
    ),
    prerequisites: ['28', '29'],
    bookUrl: chapterUrl('第31章-推理模型革命-从o1到R1/'),
    units: [
      { name: T('组内优势、KL 惩罚与蒸馏分布'), status: 'interactive', topics: ['preference'] },
      {
        name: T('推理预算、搜索与公开训练数据流'),
        status: 'explained',
        topics: [],
        href: localePath('/policy-update'),
      },
    ],
  },
  {
    id: '32',
    part: 'frontier',
    title: T('后 Transformer 架构'),
    question: T('读取历史、压缩状态与混合架构怎样联系？'),
    summary: T(
      '从完整 Attention 到状态递推，再到 Mamba Block、SSD 和混合层；区分序列混合器与 FFN/MoE 两个设计维度。',
    ),
    prerequisites: ['15', '23', '24', '30'],
    bookUrl: chapterUrl('第32章-后Transformer架构-Mamba与混合模型/'),
    units: [
      {
        name: T('架构全景、Mamba Block 与其他路线'),
        status: 'explained',
        topics: [],
        href: localePath('/chapters/32'),
      },
      { name: T('标量选择性 SSM'), status: 'interactive', topics: ['ssm'] },
      {
        name: T('结构化矩阵与递推等价'),
        status: 'interactive',
        topics: [],
        href: localePath('/chapters/32#ssd'),
      },
      {
        name: T('混合层的缓存与 FFN/MoE 参数账本'),
        status: 'interactive',
        topics: [],
        href: localePath('/chapters/32#hybrid'),
      },
      { name: T('完整架构训练与真实性能测量'), status: 'reference', topics: [] },
    ],
  },
  {
    id: 'A',
    part: 'appendices',
    title: T('Scaling Law 与计算估算'),
    question: T('经验拟合与确定性的计算账本怎样区分？'),
    summary: T(
      'Scaling Law 描述特定数据与训练设置下的经验关系；参数、token 与 FLOPs 估算必须明确假设、统计范围和单位。',
    ),
    prerequisites: ['17', '30'],
    bookUrl: chapterUrl('附录A-Scaling-Law与计算量估算/'),
    units: [
      {
        name: T('Scaling Law 的经验边界'),
        status: 'explained',
        topics: [],
        href: localePath('/model-budget'),
      },
      { name: T('真实模型损失曲线拟合'), status: 'reference', topics: [] },
      { name: T('参数量与 FLOPs：12Ld²、前向与 6ND'), status: 'interactive', topics: ['budget'] },
    ],
  },
  {
    id: 'B',
    part: 'appendices',
    title: T('解码策略详解'),
    question: T('过滤后的概率怎样重新归一化并采样？'),
    summary: T(
      '温度改变分布形状，Top-K/Top-P 改变候选集合，重新归一化后才能按有效概率采样；搜索使用序列概率。',
    ),
    prerequisites: ['16'],
    bookUrl: chapterUrl('附录B-解码策略详解/'),
    units: [
      {
        name: T('温度、Top-K/Top-P、采样与短序列搜索'),
        status: 'interactive',
        topics: ['decoding'],
      },
    ],
  },
  {
    id: 'C',
    part: 'appendices',
    title: T('常见问题与复述'),
    question: T('能否离开图，用自己的话解释原因？'),
    summary: T(
      '常见误区应回到一个可验证的例子；每个实验的隐藏说明、对照场景和复述提示用于检查理解。',
    ),
    prerequisites: [],
    bookUrl: chapterUrl('附录C-常见问题FAQ/'),
    units: [
      {
        name: T('各实验的边界、误区与复述提示'),
        status: 'explained',
        topics: ['attention', 'position', 'inference', 'ssm'],
      },
      { name: T('原书完整 FAQ 索引'), status: 'reference', topics: [] },
    ],
  },
];
const standaloneChapters = new Set(['04', '15', '32']);
export const chapterHref = (id: string): string =>
  standaloneChapters.has(id)
    ? localePath('/chapters/' + id)
    : localePath('/chapters') + '#chapter-' + id;
export const chapterLabel = (id: string): string =>
  /^\d+$/.test(id) ? T('第 {#0} 章', Number(id)) : T('附录 {#0}', id);
export const coverageLabel = (status: CoverageStatus): string =>
  ({
    interactive: T('可操作实验'),
    explained: T('已有讲解'),
    reference: T('原书阅读'),
    missing: T('尚待补充'),
  })[status];
export const CHAPTER_VERIFIED_DATE = '2026-11-15';
