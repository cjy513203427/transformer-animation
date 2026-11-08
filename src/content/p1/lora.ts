import { lora } from '../../math/p1/lora';
import { choice, field, lesson, type TopicDefinition } from './types';
export default {
  id: 'lora',
  path: '/lora',
  title: 'LoRA：冻结底座，更新从哪里进入？',
  chapter: '第 26 章',
  intro:
    '用两个二维输入核对完整底座与低秩支路。改变 rank、更新强度，再比较合并权重和量化底座路径。',
  input: [1, 2, -1, 1],
  fields: [
    choice('rank', 'rank r', 1, [
      { value: 1, label: '1' },
      { value: 2, label: '2' },
    ]),
    field('alpha', '更新强度 α', 1, 0, 4, 0.25),
  ],
  steps: [
    lesson(
      '冻结底座',
      String.raw`Y_0=XW_0`,
      '底座权重 W₀ 保持固定，输入仍经过完整线性投影。冻结只影响哪些参数被优化，不自动省掉前向或存储。',
      '为什么冻结后还需要保存并读取 W₀？',
      '底座保持，新增支路学习更新。',
    ),
    lesson(
      '低秩支路',
      String.raw`A\in\mathbb R^{D\times r},\quad B\in\mathbb R^{r\times K},\quad\Delta W=\frac\alpha r AB`,
      '采用行向量记法，先用 A 进入 r 维，再用 B 返回输出维度。本例使用非零教学权重，展示已存在的更新。',
      '低秩约束限制的是 W₀ 还是 ΔW？',
      'rank 限制更新矩阵的秩，不能解释成底座的无损压缩。',
    ),
    lesson(
      '输出相加',
      String.raw`Y=XW_0+\frac\alpha r(XA)B`,
      '底座输出与适配器输出形状相同，逐项相加。α 为 0 时适配器贡献为零。',
      '只改 α，底座输出会变吗？',
      '控制更新支路即可改变输出，底座数值没有被训练。',
    ),
    lesson(
      'rank 与合并',
      String.raw`XW_0+X\Delta W=X(W_0+\Delta W)`,
      '将更新加到权重后，可与双支路输出核对。参数数目 r(D+K) 随 r 变化；本例维度太小，rank=2 已没有节省。',
      '合并为什么成立？更大的 rank 是否保证更好的任务效果？',
      '代数合并可验证，任务质量需要训练与评估。',
    ),
    lesson(
      'QLoRA 数据流',
      String.raw`Y=X\operatorname{dequant}(W_{0,q})+\frac\alpha r(XA)B`,
      '底座以低比特存储，计算前还原到计算精度；适配器使用独立精度。这里以对称均匀 4bit 作简化，不冒充 NF4。',
      '存储位宽、计算精度和适配器训练精度为什么不能混为一谈？',
      'QLoRA 的量化底座仍冻结，适配器才参与训练。',
    ),
    lesson(
      '账本与对照',
      String.raw`N_{adapter}=r(D+K)`,
      '先预测只改变 α 后输出是否保持，再执行并核对。底座存储、适配器存储和前向工作量分别列出。',
      'α=0 时是什么模型？适配器的零初始化怎样保留原输出？',
      '底座和更新分别核算；低参数训练不意味着全模型无需存储。',
    ),
  ],
  examples: [
    { name: '两行输入', input: [1, 2, -1, 1] },
    { name: '零更新', input: [1, 2, -1, 1], parameters: { alpha: 0 } },
  ],
  compute: lora,
  prediction: {
    key: 'alpha',
    target: 2,
    question: '只改变 α，当前 LoRA 输出会保持不变吗？',
    label: 'LoRA 输出',
  },
  sources: [
    { title: 'LoRA §4.1', url: 'https://arxiv.org/html/2106.09685v2' },
    { title: 'QLoRA §3 与附录E', url: 'https://arxiv.org/html/2305.14314v1' },
  ],
  limits: [
    '没有反向、优化器或真实微调数据。常见初始化可将 B 置零，使初始更新为零。',
    '量化分支没有 NF4、Double Quantization 或 Paged Optimizers，因此只展示 QLoRA 的简化数据流。',
    'JS float64 计算与账本中的假想 float32 存储明确分开。',
  ],
} satisfies TopicDefinition;
