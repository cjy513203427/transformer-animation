import { matmul } from '../attention';
import { quantize } from './quantization';
import { check, flat, frame, pairRows, tensor, type Computation, type Parameters } from './shared';
export const BASE_WEIGHT = [
  [1, 0.5],
  [-0.5, 2],
];
export function lora(x: number[], p: Parameters): Computation {
  const X = pairRows(x),
    r = p.rank,
    A = [
      [1, 0.5],
      [0.5, -1],
    ].map((row) => row.slice(0, r)),
    B = [
      [0.5, -1],
      [1, 0.5],
    ].slice(0, r),
    s = p.alpha / r;
  const base = matmul(X, BASE_WEIGHT),
    bottleneck = matmul(X, A),
    delta = matmul(bottleneck, B).map((row) => row.map((v) => s * v)),
    combined = base.map((row, i) => row.map((v, j) => v + delta[i][j]));
  const deltaW = matmul(A, B).map((row) => row.map((v) => s * v)),
    mergedWeight = BASE_WEIGHT.map((row, i) => row.map((v, j) => v + deltaW[i][j])),
    merged = matmul(X, mergedWeight);
  const quant = quantize(BASE_WEIGHT.flat(), 4, 4, 8),
    quantWeight = pairRows(quant.restored),
    quantBase = matmul(X, quantWeight),
    quantCombined = quantBase.map((row, i) => row.map((v, j) => v + delta[i][j]));
  return {
    frames: [
      frame(
        '冻结的底座仍做完整线性投影',
        [tensor('X [2, 2]', X), tensor('W₀ [2, 2]（冻结）', BASE_WEIGHT), tensor('XW₀', base)],
        [],
        '冻结表示不更新参数；不表示省掉它的前向计算或权重存储。',
      ),
      frame(
        '低秩支路先压到 r 维，再投影回来',
        [tensor('A [2, r]', A), tensor('XA [2, r]', bottleneck), tensor('B [r, 2]', B)],
        [
          ['rank r', r],
          ['缩放 α/r', s],
        ],
        '行向量约定：ΔW = (α/r)AB。这里用非零教学参数演示已存在的更新，没有训练循环。',
      ),
      frame('把更新支路加到原输出', [
        tensor('XW₀', base),
        tensor('(α/r)XAB', delta),
        tensor('LoRA 输出', combined),
      ]),
      frame(
        'rank 约束的是更新，不是底座的无损压缩',
        [tensor('ΔW', deltaW), tensor('W₀ + ΔW', mergedWeight), tensor('合并权重后的输出', merged)],
        [
          ['底座参数', 4],
          ['适配器参数', 4 * r],
        ],
        '二维玩具的 rank=2 已无参数节省；真实 d 很大、r 很小时才体现低秩参数优势。改变 r 也改变本例更新，质量不由 rank 单独决定。',
      ),
      frame(
        '量化底座加高精度适配器：QLoRA 的数据流',
        [
          tensor('对称 4bit 还原底座', quantWeight),
          tensor('量化底座输出', quantBase),
          tensor('量化底座 + LoRA', quantCombined),
        ],
        [],
        '真实 QLoRA 用 NF4 存储底座，通常用 BF16 计算与适配器训练；本页用均匀4bit与 JS float64 替代，只解释分工。没有复现双重量化、分页优化器或完整训练。',
      ),
      frame(
        '分别统计底座存储、适配器与计算',
        [tensor('完整精度 LoRA 输出', combined), tensor('量化底座 LoRA 输出', quantCombined)],
        [
          ['底座 float32 B', 16],
          ['假想量化底座 B（含 scale）', quant.bytes],
          ['适配器 float32 B', 4 * r * 4],
        ],
        '底座冻结，A/B 可训练；示例所有运算使用 JS float64，账本假设 4B 适配器，不是真实 GPU dtype 或显存实测。',
      ),
    ].map((item, index) =>
      index === 2
        ? { ...item, branch: { base, delta, total: combined } }
        : index === 3
          ? {
              ...item,
              branch: { base, delta, total: merged, totalLabel: '合并后 X(W₀ + ΔW)' },
            }
          : index === 4
            ? {
                ...item,
                branch: {
                  base: quantBase,
                  delta,
                  total: quantCombined,
                  baseLabel: '量化底座 X·deq(W₀)',
                  totalLabel: 'QLoRA 输出 Y',
                },
              }
            : item,
    ),
    checks: [check('双支路与合并权重输出等价', flat(combined), flat(merged))],
    signature: flat(combined),
  };
}
