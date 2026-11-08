import { quantization } from '../../math/p1/quantization';
import { choice, field, lesson, type TopicDefinition } from './types';
export default {
  id: 'quantization',
  path: '/quantization',
  title: '量化：少存几个比特，会丢掉什么？',
  chapter: '第 27 章',
  intro:
    '编辑八个数，查看对称刻度、舍入、裁剪、分组和逐项误差。把异常值留在同一组或拆开，比较误差和元数据账单。',
  input: [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6],
  fields: [
    choice('bits', '整数位数', 4, [
      { value: 2, label: '2bit' },
      { value: 4, label: '4bit' },
      { value: 8, label: '8bit' },
    ]),
    choice('group', '组大小', 8, [
      { value: 2, label: '2' },
      { value: 4, label: '4' },
      { value: 8, label: '全向量8' },
    ]),
    field('cap', '校准绝对上限', 8, 0.5, 8, 0.5),
  ],
  steps: [
    lesson(
      '离散刻度',
      String.raw`q_{max}=2^{b-1}-1,\quad s_g=\min(cap,\max|x_g|)/q_{max}`,
      '每组共享一个尺度。零点取 0，正负对称，整数范围为 ±qmax。全零组用 scale=1、q=0 避免除零。',
      '同样位数下，一个大异常值怎样改变格点间距？',
      '位数与校准范围一起决定刻度。',
    ),
    lesson(
      '舍入',
      String.raw`q_i=\operatorname{clip}(round(x_i/s_g),-q_{max},q_{max}),\quad\hat x_i=s_gq_i`,
      '数值落到最近整数格点，再乘回尺度。明确规定恰好半格时远离零，防止与其他库的舍入约定混淆。',
      '还原值等于原值需要满足什么条件？',
      '有损来自舍入与裁剪，解量化不能恢复被丢弃的细节。',
    ),
    lesson(
      '裁剪',
      String.raw`x_{clip}=\max(-a_g,\min(a_g,x))`,
      '缩小校准范围会让内部格点更密，同时截断超出范围的值。逐项误差让两种影响可见。',
      '为什么更密的刻度仍可能导致更大的总误差？',
      '校准是在分辨率和异常值截断之间做取舍。',
    ),
    lesson(
      '分组',
      String.raw`B_{scale}=4\lceil N/G\rceil`,
      '各组分别取尺度，减少一个异常值对其他组的影响。本例用每组一个 float32 scale；组越小，元数据越多。',
      '更小组一定让文件按相同比例变小吗？',
      '分组精度收益需要与元数据一起计算。',
    ),
    lesson(
      '异常值',
      String.raw`MSE=\frac1N\sum_i(\hat x_i-x_i)^2`,
      '比较普通小值和最后的异常值，核对每项误差。本例只做均匀对称量化，没有声称所有模型权重都服从同样分布。',
      '异常值在最后一组时，哪些尺度会被影响？',
      '看局部误差，才能解释总 MSE 的来源。',
    ),
    lesson(
      '存储与误差',
      String.raw`B=\lceil Nb/8\rceil+4\lceil N/G\rceil`,
      '先预测提高位数后 MSE 是否保持不变，再运行。小向量的元数据占比可能很高，不能只按位数比例估算文件。',
      '为什么理论整数载荷不等于模型文件大小？',
      '账本必须写明粒度、scale、格式头和对齐是否计入。',
    ),
  ],
  examples: [
    { name: '含异常值', input: [-0.9, -0.4, 0.2, 0.8, -1, 0.1, 0.5, 6] },
    { name: '全零边界', input: Array(8).fill(0) },
    { name: '均匀小值', input: [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1] },
  ],
  compute: quantization,
  prediction: {
    key: 'bits',
    target: 8,
    question: '只提高整数位数，当前 MSE 会保持不变吗？',
    label: '还原均方误差',
  },
  sources: [
    {
      title: 'PyTorch fake quantize 定义',
      url: 'https://docs.pytorch.org/docs/2.15/generated/torch.fake_quantize_per_tensor_affine.html',
    },
    { title: 'GGUF 官方格式', url: 'https://github.com/ggml-org/ggml/blob/master/docs/gguf.md' },
  ],
  limits: [
    '对称有符号编码有一个补码值未使用；不是所有 INT4 或 GGUF 量化的规则。',
    '理论字节数假设整数紧密打包，不含对齐、文件头或其他元数据；浏览器数组没有实际压缩。',
    '不模拟 GPU kernel、推理速度或量化后的真实任务质量。',
  ],
} satisfies TopicDefinition;
