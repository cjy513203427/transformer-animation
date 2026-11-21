import { T } from '../i18n';
import type { TopicId } from './topics';
import type { NoteId } from '../state/notebook';
export interface KnowledgeNote {
  id: NoteId;
  group: string;
  chapters: string[];
  topics: TopicId[];
  related?: string[];
  sources: string[];
  title: string;
  question: string;
  body: string[];
  example: string;
  boundary: string;
  recall: string;
}
export const NOTE_GROUPS = [
  { id: 'foundations', title: T('模型与计算') },
  { id: 'training', title: T('训练与规模') },
  { id: 'engineering', title: T('生成与架构选择') },
  { id: 'prompting', title: T('Prompt Engineering · 提示工程') },
  { id: 'alignment', title: T('RLHF · 对齐与偏好') },
  { id: 'reasoning', title: T('推理模型与复习') },
];
export const KNOWLEDGE_NOTES: KnowledgeNote[] = [
  {
    id: 'model',
    group: 'foundations',
    chapters: ['01', '02'],
    topics: ['inference'],
    sources: ['attention', 'gpt3'],
    title: T('GPT、参数与上下文'),
    question: T('一个会续写的模型，怎样变成聊天助手？'),
    body: [
      T(
        '从 RNN/LSTM 的递推，到 2017 年 Transformer 的 Attention，再到 GPT 的自回归预训练与指令微调，变化包含架构、训练目标和数据。GPT 是 Decoder-only 路线；原始 Transformer 还包含 Encoder 与 Cross-Attention。',
      ),
      T(
        '“程序 + 参数文件”是理解推理的好起点，实际还需要配置与匹配的 Tokenizer。权重是训练得到的长期数值；激活与 KV Cache 属于本次上下文。聊天格式把角色和历史编码成输入，不意味着模型在聊天时自动更新权重。',
      ),
    ],
    example: T(
      '对“我喜欢”计算下一个 token 的分布，追加选中的 token 后再计算。每轮输入不同，但同一轮生成期间参数固定。',
    ),
    boundary: T('下一 token 概率高不等于陈述为真；流畅、偏好和事实正确性需要分别检查。'),
    recall: T('解释参数、激活、上下文和缓存各保存什么，并说明聊天历史为什么会影响输出。'),
  },
  {
    id: 'tokenization',
    group: 'foundations',
    chapters: ['04'],
    topics: [],
    related: ['15'],
    sources: ['bpe'],
    title: T('Tokenization、BPE 与 Embedding'),
    question: T('文字变成 ID，与 ID 变成向量，是同一件事吗？'),
    body: [
      T(
        'Tokenizer 负责文本与离散 ID 之间的转换。BPE 在训练词表时统计相邻符号对，反复合并；编码时使用已经固定的合并规则。字符、字节和子词方案不同，一个 token 不固定对应一个字或一个单词。',
      ),
      T(
        'Embedding 是 E[id] 查表，E 的形状为 [词表大小, 隐藏维度]，表值通过训练学习。ID 只是行号，编号接近不表示语义接近。空格、Unicode、特殊 token 与聊天模板都可能改变编码；必须使用模型匹配的 Tokenizer。',
      ),
    ],
    example: T(
      '玩具合并：l o w → lo w → low。规则先合并 l+o，再合并 lo+w；这个例子没有模拟任何真实模型的词表。',
    ),
    boundary: T(
      '字节回退、预切分与正规化取决于具体 Tokenizer；这里提供 BPE 原理笔记，完整前向实验只接收预先给定的 ID。',
    ),
    recall: T('用“词表训练 → 编码 → 查表”三步解释 BPE 与 Embedding 的边界。'),
  },
  {
    id: 'architecture',
    group: 'foundations',
    chapters: ['03', '15', '18'],
    topics: ['attention', 'ffn'],
    related: ['15'],
    sources: ['attention', 'mingpt'],
    title: T('架构地图与 model.py'),
    question: T('公式、张量与代码怎样一一对应？'),
    body: [
      T(
        '查表与位置得到隐藏状态，重复 Block 后经最终 Norm、词表投影与 Softmax 得到分布。Attention 在位置间混合信息，FFN 在各位置共享参数处理特征。Self-Attention 的 Q/K/V 来自同一序列，Cross-Attention 的 K/V 来自另一序列。',
      ),
      T(
        'model.py 中参数注册对应权重，forward 对应激活计算，reshape/transpose 对应轴布局，Mask 对应可见性。固定行向量：X:[T,D]，XW 的输出特征维由 W 的列数决定。模块定义、训练循环与生成循环应分开读。',
      ),
    ],
    example: T(
      '完整前向实验可追踪 ID → E → 位置 → 两个头 → 拼接与 W_O → 残差与 FFN → logits → p；每一步都给出实际数值与形状。',
    ),
    boundary: T(
      '原始 Post-Norm 是 LN(x + SubLayer(x))；本实验采用 Pre-Norm：x + SubLayer(LN(x))。不能把两者顺序混写。',
    ),
    recall: T('画出一个 Block，并指出跨 token 交互、逐 token 处理、参数和激活分别在哪里。'),
  },
  {
    id: 'attention',
    group: 'foundations',
    chapters: ['08', '09', '10', '11', '12'],
    topics: ['attention'],
    sources: ['attention'],
    title: T('从矩阵乘法到多头 Attention'),
    question: T('分数、概率与搬运的内容，各由哪个量决定？'),
    body: [
      T(
        '矩阵乘法的一格等于左行与右列的点积；二维线性变换可旋转、缩放或剪切，多个无非线性的线性层仍可合并。Q=XW_Q、K=XW_K、V=XW_V；Q/K 决定分数，逐行 Softmax 得到 P，PV 对内容做加权和。',
      ),
      T(
        '点积同时依赖方向与模长，不等于余弦相似度。1/√d_k 的方差直觉假设分量近似独立、零均值且方差受控。多头切分特征轴，各头保留全部 token；拼接后由 W_O 混合，不是简单平均，也没有预先分配的“语法头”。',
      ),
    ],
    example: T(
      '固定 Q/K，只改 V：P 不变，输出改变。改一个 K：对应列分数改变，归一化可能让整行概率一起改变。',
    ),
    boundary: T(
      'Q/K/V 是激活，W_Q/W_K/W_V 才是参数；热力图行是 Query、列是 Key，因果 Mask 在 Softmax 前加入。',
    ),
    recall: T('从一个 Query 出发，说明它如何给各 Key 分配权重，再合并 Value。'),
  },
  {
    id: 'components',
    group: 'foundations',
    chapters: ['06', '07', '13'],
    topics: ['normalization', 'ffn'],
    sources: ['attention', 'mingpt'],
    title: T('Norm、非线性、残差与 Dropout'),
    question: T('这些模块都在“调整数字”，为什么不能互换？'),
    body: [
      T(
        'LayerNorm 在每个 token 的特征轴上减均值、除标准差，再做可学习仿射；RMSNorm 不减均值。Softmax 在候选轴上构造和为 1 的非负权重。三者的统计量、轴和用途不同，ε 用于数值稳定。',
      ),
      T(
        'FFN 通过升维、GELU/ReLU 等非线性与降维改变表示；SwiGLU 等门控变体使用逐元素乘法。残差 y=x+F(x) 的导数含直连项 I；Dropout 在训练时屏蔽并按 1/(1−p) 缩放保留项，推理时关闭。',
      ),
    ],
    example: T('当 p=0.5，训练时保留的分支值乘 2，使其期望保持；这不代表每次前向输出都相同。'),
    boundary: T('残差不保证无损或绝不梯度消失；FFN 的 4D 中间维是配置选择，不是所有架构的定律。'),
    recall: T('分别说明 Norm 的统计轴、Softmax 的候选轴、FFN 的非线性与残差的直连路径。'),
  },
  {
    id: 'position',
    group: 'foundations',
    chapters: ['05', '14', '25'],
    topics: ['position'],
    sources: ['attention', 'rope', 'alibi'],
    title: T('位置从哪里进入计算？'),
    question: T('加法、RoPE 和 ALiBi，改变的是同一处吗？'),
    body: [
      T(
        '正弦或学习式加法位置向量与 Embedding 同维，在投影前加入残差流。展开 (x_i+p_i)(x_j+p_j)^T 会出现内容、位置与两个交叉项；相加不是把两类信息装进互不影响的独立槽位。',
      ),
      T(
        'RoPE 对 Q/K 的二维维度对按频率旋转，使固定内容下的点积依赖相对位置；ALiBi 直接给分数加头相关的距离偏置。基础 RoPE 的共同位置平移性质要求内容与频率规则一致，点积不必随距离单调下降。',
      ),
    ],
    example: T(
      '固定 q/k，把位置 (2,5) 同时改为 (12,15)，基础 RoPE 点积不变；只移动一个位置则一般改变。',
    ),
    boundary: T(
      '位置插值、频率缩放和长上下文训练是额外选择；能够计算远处的位置，不等于模型可靠理解任意长输入。',
    ),
    recall: T('在前向链上标出三种方案的插入点，并说明哪种直接改 Attention 分数。'),
  },
  {
    id: 'training',
    group: 'training',
    chapters: ['16', '19'],
    topics: ['inference'],
    sources: ['mingpt', 'gpt3'],
    title: T('从交叉熵到 train.py 的更新闭环'),
    question: T('loss.backward() 与 optimizer.step() 各自改变什么？'),
    body: [
      T(
        '输入 [a,b,c] 对齐目标 [b,c,d]，因果 Mask 限定每个位置的可见前缀；交叉熵取正确目标的 −log 概率。训练标签事先已知，所以各位置前向可以并行，但标签不会作为未来输入泄漏到当前 Query。',
      ),
      T(
        '训练循环：zero_grad → forward/loss → backward 累积梯度 → 梯度裁剪（若采用）→ optimizer.step → scheduler。AdamW 还维护动量与二阶统计，权重衰减与学习率另有角色。梯度累积应按实际 token/样本数正确缩放；验证集用于检查泛化与过拟合。',
      ),
    ],
    example: T(
      '目标 token 概率为 0.25 时，单项交叉熵为 −ln(0.25)≈1.386。backward 求梯度，step 才按优化器规则更新参数；推理没有这两步。',
    ),
    boundary: T(
      '可恢复训练的 checkpoint 通常还要保存优化器、调度器、步数与随机状态。本站解释完整流程，未实现真实 Transformer 的反向训练。',
    ),
    recall: T('按顺序讲一次训练迭代，并区分前向激活、参数梯度与参数更新。'),
  },
  {
    id: 'learning-rate',
    group: 'training',
    chapters: ['17'],
    topics: [],
    sources: ['attention', 'adamw'],
    title: T('学习率、Warmup 与优化轨迹'),
    question: T('梯度方向没错，为什么 loss 仍会变大？'),
    body: [
      T(
        'SGD 用 w_next=w−η∇L 更新；η 控制步长，过大可能跨过低损失区域并发散。Momentum/AdamW 还会改变实际更新方向与尺度，因此同一个 η 在不同优化器或参数尺度下不能直接比较。',
      ),
      T(
        'Warmup 逐步提高学习率，随后可以线性或余弦衰减。本例只验证一维二次损失，固定 η 在 0<η<2 时收敛，η=2 等幅振荡，η>2 发散；真实神经网络的稳定区间不能照搬这个数字。',
      ),
    ],
    example: T('L(w)=w²/2，w₀=2。先预测 η=0.5 与 η=2.5 谁会靠近 0，再查看下方逐步数值。'),
    boundary: T(
      'Warmup 不是稳定性的保证；本例不含 minibatch 噪声、Adam 动量、混合精度或分布式训练。',
    ),
    recall: T('用一次越过最小值的更新说明方向正确为何仍可能发散。'),
  },
  {
    id: 'generation',
    group: 'engineering',
    chapters: ['16', '20', '22', 'B'],
    topics: ['inference', 'decoding'],
    sources: ['mingpt'],
    title: T('生成循环、KV Cache 与解码'),
    question: T('复用历史，为什么还要读取历史？'),
    body: [
      T(
        'Prefill 处理已有前缀，Decode 为新位置计算 Q/K/V 并追加各层 K/V。旧 Q 的输出已经完成，不必缓存它来供新位置查询。新 Q 仍要访问可见历史 K/V；改变前缀、权重、位置规则或运行模式可能使缓存失效。',
      ),
      T(
        'inference.py 从最后位置 logits 得到分布，再执行 Greedy、温度采样、Top-K/Top-P 或序列搜索，追加 token 直到 EOS/长度边界。Top-P 在排序后保留使累计概率首次达到阈值的 token，再归一化；Beam Search 比较累计 log 概率，并受长度处理影响。',
      ),
    ],
    example: T('概率 [0.5,0.3,0.2]，Top-P=0.6 时保留前两项，重新归一化为 [0.625,0.375]。'),
    boundary: T(
      '缓存、候选过滤和搜索解决不同问题；降低温度不能保证事实正确，缓存也没有通用固定加速倍数。',
    ),
    recall: T('从一个新 token 开始，讲清哪些量新算、哪些量复用、哪些步骤负责选择输出。'),
  },
  {
    id: 'efficiency',
    group: 'engineering',
    chapters: ['21', '23'],
    topics: ['flash', 'kv'],
    sources: ['flash', 'gqa'],
    title: T('FlashAttention 与 KV 共享各省什么？'),
    question: T('中间矩阵与历史缓存，是同一笔内存吗？'),
    body: [
      T(
        'FlashAttention 分块读取 Q/K/V，维护在线最大值、指数和与加权和；后续块出现更大最大值时重缩放旧累积量。它计算同条件下的完整 Softmax Attention，减少中间矩阵存储与 HBM I/O，配对计算仍是二次规模。',
      ),
      T(
        'MHA 每个 Query 头有自己的 KV 头，MQA 全部共享一个，GQA 按组共享。固定 Query 头数，只改变 KV 头数，主要改变 KV 投影与缓存。缓存字节=批量×层数×token数×KV头数×(d_k+d_v)×元素字节数。',
      ),
    ],
    example: T(
      '2 个 Query 头映射到 1 个 KV 头时，两个 Query 仍分别计算各自的权重，不能把两行 Query 合并成一行。',
    ),
    boundary: T(
      '浏览器账本不是 GPU 测量；KV 公式不含模型权重、其他激活与分配器。共享 KV 也不使整个模型同比缩小。',
    ),
    recall: T('分别解释重算减少、中间矩阵减少、KV 头减少，对应哪一种优化。'),
  },
  {
    id: 'memory',
    group: 'engineering',
    chapters: ['24', '32'],
    topics: ['memory', 'ssm'],
    related: ['32'],
    sources: ['mamba', 'ssd'],
    title: T('稀疏连接、压缩状态与混合架构'),
    question: T('不保存全部历史，还能怎样传递信息？'),
    body: [
      T(
        'Sparse Attention 限制连接，例如局部窗口、全局锚点或分块；跨层感受野可能扩大。Linear Attention 用特征映射与可累积状态改变核；Infini-attention 组合局部 Attention 和跨段压缩记忆。Mask 稀疏本身不会让通用稠密矩阵乘法自动变快。',
      ),
      T(
        'SSM 用状态递推传递历史，Mamba-1 让部分参数依赖当前输入；Mamba-2/SSD 展示受约束 SSM 与结构化矩阵的联系。RWKV、RetNet、Hyena 各有不同混合器；Hybrid/Jamba 可交替 Attention/SSM，并独立选择 Dense FFN 或 MoE。',
      ),
    ],
    example: T(
      '状态 h_t=a_t h_prev+b_t x_t 只保存固定数量的数值。把很久以前的一个 token 改掉，影响需要通过后续递推存活，而不是直接读取原始 KV。',
    ),
    boundary: T(
      '固定状态不等于无损保留任意历史；SSD 的受约束等价不等于任意 SSM 等价于标准 Softmax Attention。',
    ),
    recall: T('比较“少看一些位置”“压缩成状态”和“交替使用两种层”的信息路径。'),
  },
  {
    id: 'adaptation',
    group: 'engineering',
    chapters: ['26'],
    topics: ['lora'],
    sources: ['lora', 'qlora'],
    title: T('全量微调、LoRA 与 QLoRA'),
    question: T('冻结底座后，模型行为为什么仍能改变？'),
    body: [
      T(
        '全量微调更新基础权重；LoRA 固定 W，学习 ΔW=(α/r)AB，前向使用 XW+XΔW。rank r 控制更新分支的容量与参数数目，不是把已有 W 做无损压缩；可合并分支，但合并后的量化与舍入会影响等价精度。',
      ),
      T(
        'QLoRA 用低比特保存冻结基础权重，计算时反量化，适配器仍以更高精度训练；NF4、双重量化与分页优化器是论文具体设计。权重存储精度、乘法计算精度、梯度与优化器状态必须分开统计。',
      ),
    ],
    example: T(
      'D_in=D_out=4、r=1 时，两个低秩矩阵共 8 个参数，而完整增量矩阵有 16 个；这不包含仍需保存的基础权重。',
    ),
    boundary: T('低秩限制可能影响适配能力；少量训练参数不等于没有激活、梯度或基础权重的成本。'),
    recall: T('指出 LoRA 训练时更新什么、冻结什么，再区分 QLoRA 的三种精度。'),
  },
  {
    id: 'quantization',
    group: 'engineering',
    chapters: ['27'],
    topics: ['quantization'],
    sources: ['gptq', 'awq', 'gguf'],
    title: T('量化：GPTQ、AWQ 与 GGUF 的区别'),
    question: T('算法、数值格式和文件容器怎样区分？'),
    body: [
      T(
        '仿射量化先用 scale 与 zero-point 映射到整数，舍入、裁剪，再反量化。逐张量、逐通道、分组量化各有误差与元数据成本；异常值可能拉大 scale，混合精度与校准用于处理这些权衡。',
      ),
      T(
        'GPTQ 利用校准激活与近似二阶信息，逐步量化并补偿权重误差；AWQ 利用激活统计识别重要通道，通过等价缩放降低低比特误差。GGUF 是可携带张量和元数据的文件格式，包含多种量化类型，不是单一算法。',
      ),
    ],
    example: T(
      '4-bit 权重的理想载荷为每元素 0.5 字节；实际还需分组 scale、zero-point（若有）、对齐与未量化张量。',
    ),
    boundary: T(
      '本地实验验证均匀量化的刻度与误差，没有执行 GPTQ/AWQ 的校准优化，也不承诺特定模型的质量保持。',
    ),
    recall: T('说明“4-bit”“AWQ”“GGUF”各回答哪个问题，并列出权重载荷之外的成本。'),
  },
  {
    id: 'moe',
    group: 'engineering',
    chapters: ['30'],
    topics: ['moe', 'kv'],
    sources: ['deepseek'],
    title: T('MoE、专家并行与 MLA 的两条轴'),
    question: T('激活更少专家，与缓存更少 KV，是同一件事吗？'),
    body: [
      T(
        'Router 打分后选 Top-K，按当前变体的归一化规则加权合并专家输出。共享专家始终参与、容量限制可能丢弃或重路由；辅助负载损失可平衡使用率，但不自动保证速度。总参数、每 token 激活参数和通信必须分别计算。',
      ),
      T(
        '专家并行把 token 激活发到专家所在设备，计算后返回并按原 token 身份合并。MLA 则是 Attention 缓存的低秩联合压缩：KV latent、解耦 RoPE Key 分支与矩阵吸收另属一个设计轴；它可以与 MoE 同时使用。',
      ),
    ],
    example: T(
      '4 个专家只选 2 个，并不表示可以丢掉其余专家的权重；下一个 token 可能选中另外两个。MLA 的完整缓存还要计入位置 Key 分支。',
    ),
    boundary: T(
      '现有 MoE 实验不模拟共享专家、容量限制或辅助损失训练；通信字节是指定拓扑的估算，不是带宽测量。',
    ),
    recall: T('分别追踪一次专家 dispatch/combine，以及一次新 Query 对压缩 KV 的读取。'),
  },
  {
    id: 'prompt',
    group: 'prompting',
    chapters: ['28'],
    topics: [],
    sources: ['gpt3', 'react'],
    title: T('Prompt Engineering：把任务写成可检查的上下文'),
    question: T('角色、任务、资料与输出约束分别提供什么？'),
    body: [
      T(
        'Prompt 改变当前条件上下文，通常不更新权重。明确任务、输入边界、可用资料和输出标准，比只加“你是专家”更容易核对。角色可约束语气与视角，分隔符帮助区分指令与引用内容，结构化输出还需独立解析与验证。',
      ),
      T(
        '把文档或检索片段当作待处理数据，不因其中出现“忽略上文”就提升其指令权限。提示模板应配合代表性测试集，逐项看正确性、格式遵循、资料支持和成本；一次成功样例不能证明模板普遍更好。',
      ),
    ],
    example: T(
      '下方拼装器只展示真实拼出的 prompt：切换角色、示例、资料和 JSON 约束，观察上下文结构；没有调用模型或生成虚构回答。',
    ),
    boundary: T('分隔符和措辞不是安全隔离；输出格式、权限与工具执行仍需程序侧验证。'),
    recall: T('说明这份 prompt 的任务、数据与约束，并指出哪些效果必须用真实测试验证。'),
  },
  {
    id: 'few-shot',
    group: 'prompting',
    chapters: ['28'],
    topics: [],
    sources: ['gpt3'],
    title: T('Zero-shot、One-shot、Few-shot 与上下文学习'),
    question: T('在提示里给例子，等于训练了一次模型吗？'),
    body: [
      T(
        'Zero-shot 只描述任务，One-shot 给一个示例，Few-shot 给多个示例。示例可以展示标签定义、边界条件、格式和解题结构；上下文学习改变当次预测，不等于执行梯度下降。微调则会用训练过程更新参数。',
      ),
      T(
        '选择与目标任务相关、正确且覆盖不同情况的示例；顺序、类别偏置与上下文长度都可能影响输出。更多示例也会占用窗口和计算，应该把无示例模板作为基线做对照，而不是预设数量越多越好。',
      ),
    ],
    example: T(
      '把“正面/负面”分类任务改成“输出 0/1”，示例应同步改变标签。否则任务指令与演示会给出互相冲突的模式。',
    ),
    boundary: T('本页可以拼装示例并比较上下文，不在浏览器里训练或评测真实语言模型。'),
    recall: T('比较 Few-shot 与 SFT：数据放在哪里，参数是否更新，效果能保留多久？'),
  },
  {
    id: 'cot',
    group: 'prompting',
    chapters: ['28', '31'],
    topics: [],
    sources: ['cot'],
    title: T('Chain-of-Thought、分解与自检'),
    question: T('多写几个中间步骤，为什么可能有帮助，也可能出错？'),
    body: [
      T(
        'CoT 在示例或输出中加入中间解题步骤，后续 token 可利用这些已生成的上下文；任务分解把大问题拆成可检查的子问题。其收益依赖模型、任务与提示，不能给一个适用于所有模型的提升百分比。',
      ),
      T(
        '公开写出的解释可能事后合理化，并不能证明真实内部机制；自检也是一次可能出错的模型生成。把算术交给计算器、事实交给可追溯资料、格式交给解析器，能提供比“看起来有道理”更独立的检查。',
      ),
    ],
    example: T(
      '17−3×4：先核对乘法得到 12，再核对减法得到 5。若第一步误算为 14，后面解释再流畅也不能修复输入错误。',
    ),
    boundary: T('本站只呈现公开算法与手算例子，不声称知道闭源模型的隐藏思维或内部训练细节。'),
    recall: T('区分一个可验证的中间结果、一段解释文字与模型真正内部计算之间的关系。'),
  },
  {
    id: 'self-consistency',
    group: 'prompting',
    chapters: ['28'],
    topics: ['decoding'],
    sources: ['consistency'],
    title: T('Self-Consistency：多次采样与投票'),
    question: T('多数回答一致，是否就说明答案正确？'),
    body: [
      T(
        '对同一个问题独立采样多条路径，提取并规范化最终答案，再按答案计票。不同推导得到同一答案会合并；这是对完整候选回答做聚合，不是每个 token 逐位置投票，也不更新模型参数。',
      ),
      T(
        '提取失败要单独记录，平局要有明确处理。票数占比描述这一批候选的相互一致程度，不是经过校准的正确率。候选可能共享同一种错误；增加采样也增加推理开销，不能保证单调改善。',
      ),
    ],
    example: T(
      '下方使用手工给定的 5 个候选答案。先猜投票结果，再改其中一个；“多数答错”预设可直接展示一致性与正确性的差别。',
    ),
    boundary: T('本工具计算候选答案的投票，没有运行模型采样，也不把票数占比称作“置信度”。'),
    recall: T('说出采样、答案提取、规范化、聚合与独立验证五个环节。'),
  },
  {
    id: 'tot-react',
    group: 'prompting',
    chapters: ['28', '31'],
    topics: [],
    sources: ['tot', 'react', 'rag'],
    title: T('ToT、ReAct、RAG 与提示评估'),
    question: T('哪些是提示格式，哪些是模型之外的执行流程？'),
    body: [
      T(
        'Tree of Thoughts 将候选中间状态组成搜索树，交替生成、评估、剪枝或回溯。ReAct 交替产生动作与读取工具观察，真正执行动作的是外部程序。RAG 先检索资料再提供给生成器；它为答案提供资料，不自动保证检索完整或引文正确。',
      ),
      T(
        '提示优化需要固定代表性测试集与评分标准，每次只改少数因素，检查正确性、格式、引用和成本；把失败案例加入回归集。不要把搜索分支数、投票次数或更长文字直接当作性能改进。',
      ),
    ],
    example: T(
      '查询库存：生成查询动作 → 程序读取库存工具 → 把返回值写回上下文 → 根据观察作答。模型写出“已查询”不能替代真实工具调用。',
    ),
    boundary: T(
      '这一项是流程笔记；本站没有联网 Agent、检索服务或 ToT 模型搜索，RAG 作为与原书提示流程相连的扩展说明。',
    ),
    recall: T('用“上下文、控制器、工具、验证器”解释一次 ReAct，并与纯文本 CoT 对照。'),
  },
  {
    id: 'rlhf',
    group: 'alignment',
    chapters: ['29'],
    topics: ['preference'],
    sources: ['instruct', 'ppo'],
    title: T('RLHF 全流程：SFT → 奖励模型 → 策略优化'),
    question: T('人工比较两个回答，怎样最终改变生成模型？'),
    body: [
      T(
        '以 InstructGPT 式流程为例：先用人工示范做 SFT；同一 prompt 生成多个回答，由标注者排序形成偏好对；奖励模型用排序损失学习给回答打分；策略再生成新回答，用奖励与参考约束构造优化目标。RLHF 是利用人类反馈的训练范式，PPO 只是其中一种策略优化方法。',
      ),
      T(
        'Actor 是待更新策略；Reward Model 给分；Reference Policy 提供冻结的参照分布；Critic 估计前缀状态价值，帮助构造优势。旧策略概率记录本轮采样分布，与冻结参考策略用途不同。这些是逻辑角色，具体实现可以共享底座，并非必须部署四套完全独立模型。',
      ),
    ],
    example: T(
      '下方逐阶段查看一条数据怎样从示范、偏好对流向新 rollout、优势与更新；“谁更新、谁冻结”会随阶段改变。奖励差和 PPO 数值可进入已有偏好实验核对。',
    ),
    boundary: T(
      '流程图是公开训练方案的解释，没有在浏览器训练真实模型；奖励偏好分不等于事实真值或通用质量。',
    ),
    recall: T(
      '用一组 prompt/chosen/rejected 数据讲清 RM 怎么训练，再说明 PPO 如何使用新采样回答。',
    ),
  },
  {
    id: 'ppo-dpo',
    group: 'alignment',
    chapters: ['29'],
    topics: ['preference'],
    sources: ['ppo', 'dpo'],
    title: T('PPO、KL 与 DPO：三种约束别混在一起'),
    question: T('旧策略、参考策略和被拒绝回答，各在比较什么？'),
    body: [
      T(
        'PPO 的 ρ=π_current/π_old 比较当前与本轮采样时的策略，目标取 min(ρA,clip(ρ,1−ε,1+ε)A)。Critic 与 GAE 等方法估计优势；裁剪限制目标激励，不是硬性截断策略概率。KL 参考约束则限制相对固定参考策略的漂移。',
      ),
      T(
        '标准离线 DPO 直接用 chosen/rejected 的整段条件 log 概率，相对参考策略构造 log-ratio 差，再计算 logistic 损失；不单独训练 RM，也不需要 PPO 式在线 rollout。整段 log 概率是各 token 条件 log 概率之和，不是只看最后一个 token。',
      ),
    ],
    example: T(
      'π_current=0.3、π_old=0.2 时 ρ=1.5；A>0、ε=0.2 时裁剪项到 1.2A。参考策略可能是 0.1，那是另一条比较。',
    ),
    boundary: T(
      'PPO 裁剪不保证每次更新的真实 KL 都低于固定阈值；DPO 与 PPO 也不是按所有指标排序的优劣榜。',
    ),
    recall: T('分别说明 π_old 与 π_ref 的来源，以及 DPO 为什么不需要单独 RM。'),
  },
  {
    id: 'alignment-risks',
    group: 'alignment',
    chapters: ['29'],
    topics: ['preference'],
    sources: ['instruct', 'constitutional'],
    title: T('对齐目标、Reward Hacking 与 AI 反馈'),
    question: T('奖励更高，是否一定更有帮助、更诚实？'),
    body: [
      T(
        'Helpful、Honest、Harmless 是可讨论的目标，标注规范和样本分布决定“谁的偏好”。RM 可能学到长度、语气等捷径；策略持续优化代理分数，可能 exploit 这些偏差，出现 Reward Hacking 或迎合，而不是真正改善任务表现。',
      ),
      T(
        'RLAIF 用 AI 生成偏好或评审信号，Constitutional AI 还用明确原则指导批评与修订；规则验证器适合可判定任务。反馈来源变了，不代表评价自动客观。应保留独立验证、人工抽检和分布外评估，区分结果奖励与过程奖励。',
      ),
    ],
    example: T(
      '若 RM 偏爱长答案，模型可能增加重复段落来得分。检查真实任务是否完成，比只看 RM 平均分更能暴露这个问题。',
    ),
    boundary: T(
      'KL 能抑制漂移但不能消除奖励模型偏差；对齐训练也不保证所有事实或所有使用场景都正确。',
    ),
    recall: T('举一个代理奖励被利用的例子，并给出一个与该奖励相独立的检查。'),
  },
  {
    id: 'grpo',
    group: 'reasoning',
    chapters: ['31'],
    topics: ['preference'],
    sources: ['r1'],
    title: T('GRPO：同一问题的一组回答'),
    question: T('不训练 Critic，优势的参照从哪里来？'),
    body: [
      T(
        '在 R1 公开形式中，同一 prompt 采样 G 个回答，用各自奖励减组均值，再除组标准差得到优势。它以组内相对表现取代独立价值模型基线；组不是不同问题混在一起，也不是把别人的回答当上下文。',
      ),
      T(
        '策略目标仍包含新旧策略概率比、裁剪与参考约束，完整训练还需要 token 掩码、长度归一化与采样批次管理。若组内奖励全相同，本教学实现把标准化优势设为 0，避免除零，并明确没有组内区分信号。',
      ),
    ],
    example: T(
      '奖励 [0,0,1,1] 的均值是 0.5，标准差是 0.5，优势为 [−1,−1,1,1]；整体加 10 不改变优势。',
    ),
    boundary: T(
      '这里采用公开 R1 形式的教学简化；其他 GRPO 实现可能对长度、标准化和损失项采用不同约定。',
    ),
    recall: T('用四个奖励手算一次组内优势，并说明零方差时有什么信号。'),
  },
  {
    id: 'reasoning',
    group: 'reasoning',
    chapters: ['31'],
    topics: ['preference'],
    sources: ['r1', 'tot', 'distill'],
    title: T('R1 训练路线、蒸馏与 Test-time Compute'),
    question: T('训练得更会推理，与推理时多算一些，有何区别？'),
    body: [
      T(
        'R1-Zero 研究直接从基础模型做 RL；R1 公开路线加入冷启动示范、推理 RL、拒绝采样整理 SFT 数据，再进行后续对齐 RL。可验证任务用正确性与格式等信号，并不能据此推断闭源模型采用相同流程。',
      ),
      T(
        '蒸馏让学生学习教师生成的回答或软分布；R1 蒸馏实验主要使用教师样本做 SFT，不等于读取教师隐藏思维。Test-time Compute 在权重固定时分配更长生成、多次采样、验证或搜索；更多预算的收益依赖任务与控制策略，并非必定单调增加。',
      ),
    ],
    example: T(
      '串行生成 200 token 与并行生成 5×40 token 都有 200 个输出 token，但前缀开销、依赖、延迟、验证成本和答案聚合不同。',
    ),
    boundary: T('没有用浏览器示例推断真实 API 成本或模型成绩；更长解释不自动代表更强推理。'),
    recall: T('分别解释训练更新、教师样本蒸馏与固定权重下的推理预算，指出三者改变了什么。'),
  },
  {
    id: 'scaling',
    group: 'training',
    chapters: ['A'],
    topics: [],
    sources: ['chinchilla', 'scaling'],
    title: T('Scaling Law 与训练计算账本'),
    question: T('参数更多、数据更多与计算更多，是什么关系？'),
    body: [
      T(
        'Scaling Law 是在特定数据、模型族与训练范围中拟合的经验关系，常用幂律描述损失随参数 N、训练 token D 或计算 C 的变化。Chinchilla 讨论固定训练预算下的模型与数据分配；约 20 token/参数只是特定拟合的粗略起点，部署时也需考虑后续推理成本。',
      ),
      T(
        '稠密 Transformer 的粗略训练估算 C≈6ND，把前向与反向主要矩阵计算合并。参数量与计算量页面按这个口径和更细的逐层算式计算，不预测 loss 或训练时间；Attention 的序列长度项、Embedding、硬件利用率与通信没有在 6ND 中精确展开。',
      ),
    ],
    example: T(
      '7B 参数、140B 训练 token：D/N=20，C≈5.88×10²¹ FLOPs。参数或 token 单独翻倍，按这个估算计算量都翻倍。',
    ),
    boundary: T(
      '此估算不直接适用 MoE 总参数；FLOPs 不是显存、秒数或费用，经验拟合不能推出通用最优配置。',
    ),
    recall: T('说明一个确定性的账本与一个经验损失拟合的区别，并列出 6ND 的假设。'),
  },
  {
    id: 'faq',
    group: 'reasoning',
    chapters: ['C'],
    topics: ['attention', 'inference', 'normalization'],
    sources: ['attention', 'mingpt'],
    title: T('常见误区与排查顺序'),
    question: T('输出异常、loss 变化、显存不足，该先检查什么？'),
    body: [
      T(
        '乱码先检查 Tokenizer、聊天模板、checkpoint 与运行精度是否匹配；重复可检查数据、解码参数与停止条件，不要预设调高温度一定解决。训练 loss 下降而验证 loss 停滞或上升提示泛化问题；数据泄漏也会让验证分数虚高。',
      ),
      T(
        '训练内存包括权重、梯度、优化器状态和激活；推理还需 KV Cache 与工作区。Batch、序列长度、精度、梯度检查点与参数冻结分别影响不同成本。先固定配置、记录口径并做单变量对照，再判断问题来自数据、公式、状态还是实现。',
      ),
    ],
    example: T(
      '多头输出通常是 concat(heads)W_O，不是各头平均；Post-Norm 是 LN(x+F(x))；Cache Decode 仍访问历史。这些结论都可回到对应实验验证。',
    ),
    boundary: T(
      '这里按机制合并 FAQ，并修正原书部分简化；不是对原书每一问逐条复刻。真实部署故障仍需日志与可复现配置。',
    ),
    recall: T('关掉解释，任选一个常见误区，给出能证伪它的输入变化与预期结果。'),
  },
];
export const RLHF_STAGES = [
  {
    name: 'SFT',
    explanation: T(
      '用 prompt 与人工示范 answer 做监督微调。只对约定的目标 token 计算损失，训练策略学会任务与聊天格式。',
    ),
    data: 'prompt → demonstration → −log π(answer | prompt)',
    updates: T('更新：SFT 策略参数。这里还没有偏好奖励或 PPO。'),
  },
  {
    name: T('收集偏好'),
    explanation: T(
      '对同一个 prompt 生成多个回答，让人按明确标准比较。得到 chosen/rejected，比较的是整段回答而非给每个 token 人工打分。',
    ),
    data: '(prompt, answer_A, answer_B) → chosen / rejected',
    updates: T('不更新模型；生成偏好数据。标注分歧与规范会影响后续奖励。'),
  },
  {
    name: T('训练 RM'),
    explanation: T(
      '奖励模型输出标量 r(prompt,answer)，用 −log σ(r_chosen−r_rejected) 学习排序。分数的共同平移不改变这个损失，奖励零点不是绝对质量刻度。',
    ),
    data: '(prompt, chosen, rejected) → ranking loss → RM',
    updates: T('更新：RM 参数。策略在这个阶段不因 RM 的反向传播而自动更新。'),
  },
  {
    name: 'Rollout',
    explanation: T(
      '冻结本轮采样策略 π_old，对训练 prompt 生成新回答，记录 token、Mask 与旧 log 概率。PPO 后续多次更新都要和这批数据的旧策略比较。',
    ),
    data: 'prompt → π_old → sampled tokens + log π_old',
    updates: T('采样时不更新策略；π_old 是当前 rollout 的快照，π_ref 是参考锚点，两者不同。'),
  },
  {
    name: T('奖励与价值'),
    explanation: T(
      '冻结 RM 给完整回答打分；参考策略为相同生成 token 提供 log 概率，用于参考约束。Critic 为前缀状态预测价值，优势估计还需考虑 token 时间轴与终止条件。',
    ),
    data: 'RM score + reference log probabilities + V(prefix)',
    updates: T('此处是评估数据；RM 与参考策略保持冻结。价值估计不是人工真实答案。'),
  },
  {
    name: T('PPO 更新'),
    explanation: T(
      '由奖励与价值构造 return/advantage，更新 Actor 的裁剪目标，并用 return 拟合 Critic。一次采样可做有限轮小批更新，再重新 rollout，避免把旧数据无限重复当成当前策略数据。',
    ),
    data: 'rewards + values → returns / advantages → actor + critic update',
    updates: T('更新：Actor 与 Critic（或各自参数头）；RM、π_ref 与本轮 π_old 保持固定。'),
  },
  {
    name: T('独立评估'),
    explanation: T(
      '在保留问题上检查任务成功、事实、偏好与副作用。RM 分数上涨而真实质量下降可能是奖励被利用；根据结果调整数据或目标，再进入下一轮采样。',
    ),
    data: 'updated policy → held-out tasks + human / verifier checks → next rollout',
    updates: T('评估不直接反向更新；是否继续训练由独立指标与约定的停止条件决定。'),
  },
];
export const NOTE_SOURCES: Record<string, [string, string]> = {
  attention: ['Attention Is All You Need', 'https://arxiv.org/abs/1706.03762'],
  gpt3: ['Language Models are Few-Shot Learners', 'https://arxiv.org/abs/2005.14165'],
  bpe: [
    'Neural Machine Translation of Rare Words with Subword Units',
    'https://arxiv.org/abs/1508.07909',
  ],
  mingpt: ['minGPT · model.py / trainer.py', 'https://github.com/karpathy/minGPT'],
  rope: ['RoFormer', 'https://arxiv.org/abs/2104.09864'],
  alibi: ['ALiBi', 'https://arxiv.org/abs/2108.12409'],
  adamw: ['Decoupled Weight Decay Regularization', 'https://arxiv.org/abs/1711.05101'],
  flash: ['FlashAttention', 'https://arxiv.org/abs/2205.14135'],
  gqa: ['GQA', 'https://arxiv.org/abs/2305.13245'],
  mamba: ['Mamba', 'https://arxiv.org/abs/2312.00752'],
  ssd: ['Transformers are SSMs · SSD', 'https://arxiv.org/abs/2405.21060'],
  lora: ['LoRA', 'https://arxiv.org/abs/2106.09685'],
  qlora: ['QLoRA', 'https://arxiv.org/abs/2305.14314'],
  gptq: ['GPTQ', 'https://arxiv.org/abs/2210.17323'],
  awq: ['AWQ', 'https://arxiv.org/abs/2306.00978'],
  gguf: ['GGUF specification', 'https://github.com/ggml-org/ggml/blob/master/docs/gguf.md'],
  deepseek: ['DeepSeek-V2', 'https://arxiv.org/abs/2405.04434'],
  cot: ['Chain-of-Thought Prompting', 'https://arxiv.org/abs/2201.11903'],
  consistency: ['Self-Consistency', 'https://arxiv.org/abs/2203.11171'],
  tot: ['Tree of Thoughts', 'https://arxiv.org/abs/2305.10601'],
  react: ['ReAct', 'https://arxiv.org/abs/2210.03629'],
  rag: ['Retrieval-Augmented Generation', 'https://arxiv.org/abs/2005.11401'],
  instruct: ['InstructGPT', 'https://arxiv.org/abs/2203.02155'],
  ppo: [
    'PPO · author documentation',
    'https://spinningup.openai.com/en/latest/algorithms/ppo.html',
  ],
  dpo: ['Direct Preference Optimization', 'https://arxiv.org/abs/2305.18290'],
  constitutional: ['Constitutional AI', 'https://arxiv.org/abs/2212.08073'],
  r1: ['DeepSeek-R1 · public training recipe', 'https://arxiv.org/html/2501.12948v1'],
  distill: ['Distilling the Knowledge in a Neural Network', 'https://arxiv.org/abs/1503.02531'],
  chinchilla: [
    'Training Compute-Optimal Large Language Models',
    'https://arxiv.org/abs/2203.15556',
  ],
  scaling: ['Scaling Laws for Neural Language Models', 'https://arxiv.org/abs/2001.08361'],
};
export const NOTEBOOK_VERIFIED_DATE = '2026-11-15';
