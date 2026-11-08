import { attention, matmul, type Matrix } from './attention';

/** Head counts offered by the lesson. d_model = 4 is divisible by each of them. */
export const HEAD_OPTIONS = [1, 2, 4] as const;
export type HeadCount = (typeof HEAD_OPTIONS)[number];

function checkGrid(matrix: Matrix, name: string): [number, number] {
  const columns = matrix[0]?.length ?? 0;
  if (
    !matrix.length ||
    !columns ||
    matrix.some((row) => row.length !== columns || row.some((x) => !Number.isFinite(x)))
  )
    throw new Error(`${name} 必须是形状规则、只含有限数值的非空矩阵。`);
  return [matrix.length, columns];
}

/**
 * 切分多头：[T, d_model] → [num_heads][T][d_key].
 * Same data layout as PyTorch `view(T, num_heads, d_key).transpose(0, 1)`: head h owns the
 * contiguous columns [h·d_key, (h+1)·d_key). Token rows keep their order inside every head.
 */
export function splitHeads(matrix: Matrix, numHeads: number): Matrix[] {
  const [, columns] = checkGrid(matrix, '待切分的矩阵');
  if (!Number.isInteger(numHeads) || numHeads < 1 || columns % numHeads !== 0)
    throw new Error('d_model 必须能被 num_heads 整除。');
  const dKey = columns / numHeads;
  return Array.from({ length: numHeads }, (_, h) =>
    matrix.map((row) => row.slice(h * dKey, (h + 1) * dKey)),
  );
}

/**
 * view(T, num_heads, d_key)：[T, d_model] → [T][num_heads][d_key]. Row t is cut into num_heads
 * consecutive pieces; no number moves. Transposing the first two axes gives `splitHeads`.
 */
export function viewHeads(matrix: Matrix, numHeads: number): Matrix[] {
  const [, columns] = checkGrid(matrix, '待变形的矩阵');
  if (!Number.isInteger(numHeads) || numHeads < 1 || columns % numHeads !== 0)
    throw new Error('d_model 必须能被 num_heads 整除。');
  const dKey = columns / numHeads;
  return matrix.map((row) =>
    Array.from({ length: numHeads }, (_, h) => row.slice(h * dKey, (h + 1) * dKey)),
  );
}

/** Concatenate：[num_heads][T][d_key] → [T, num_heads·d_key]，即转置回去再 view。 */
export function mergeHeads(heads: Matrix[]): Matrix {
  if (!heads.length) throw new Error('至少需要一个头。');
  const [tokens, dKey] = checkGrid(heads[0], '头输出');
  heads.forEach((head) => {
    const [rows, columns] = checkGrid(head, '头输出');
    if (rows !== tokens || columns !== dKey) throw new Error('所有头的形状必须相同。');
  });
  return Array.from({ length: tokens }, (_, t) => heads.flatMap((head) => head[t]));
}

/** Columns [start, end) of a weight matrix: W_Q 的第 h 组列就是“方式一”里的 Wq_h。 */
export function columnBlock(matrix: Matrix, start: number, end: number): Matrix {
  checkGrid(matrix, '权重');
  return matrix.map((row) => row.slice(start, end));
}

/** Rows [start, end) of a weight matrix: Wo 中与第 h 个头相乘的那几行。 */
export function rowBlock(matrix: Matrix, start: number, end: number): Matrix {
  checkGrid(matrix, '权重');
  return matrix.slice(start, end).map((row) => [...row]);
}

export interface MultiHeadWeights {
  wq: Matrix;
  wk: Matrix;
  wv: Matrix;
  wo: Matrix;
}

/** A deliberate activation intervention after projection; the weights stay fixed. */
export interface KeyIntervention {
  token: number;
  column: number;
  delta: number;
}

export function multiHeadAttention(
  x: Matrix,
  weights: MultiHeadWeights,
  numHeads: number,
  causal: boolean,
  intervention?: KeyIntervention,
) {
  const [, dModel] = checkGrid(x, 'X');
  for (const [name, w] of Object.entries(weights)) {
    const [rows, columns] = checkGrid(w, name);
    if (rows !== dModel || columns !== dModel)
      throw new Error('本教学实现的 Wq、Wk、Wv、Wo 均为 [d_model, d_model]。');
  }
  const q = matmul(x, weights.wq);
  const k = matmul(x, weights.wk);
  const v = matmul(x, weights.wv);
  if (intervention && intervention.delta !== 0) {
    const { token, column, delta } = intervention;
    if (!k[token] || k[token][column] === undefined) throw new Error('干预位置超出 K 的范围。');
    k[token][column] += delta;
  }
  const qHeads = splitHeads(q, numHeads);
  const kHeads = splitHeads(k, numHeads);
  const vHeads = splitHeads(v, numHeads);
  const heads = qHeads.map((qHead, h) => ({
    q: qHead,
    k: kHeads[h],
    v: vHeads[h],
    // Each head scales by its own √d_key, exactly as `scores / (d_key ** 0.5)` in chapter 11.
    ...attention(qHead, kHeads[h], vHeads[h], causal, true),
  }));
  const concat = mergeHeads(heads.map((head) => head.output));
  return {
    dKey: dModel / numHeads,
    q,
    k,
    v,
    heads,
    concat,
    output: matmul(concat, weights.wo),
  };
}

export type MultiHeadResult = ReturnType<typeof multiHeadAttention>;

/** Multiply-adds in all Q·Kᵀ products: num_heads × T × T × d_key = T² × d_model. */
export function scoreMultiplyAdds(tokens: number, dModel: number, numHeads: number): number {
  return numHeads * tokens * tokens * (dModel / numHeads);
}

/** Wq, Wk, Wv, Wo, each [d_model, d_model]; biases are ignored. */
export function attentionWeightCount(dModel: number): number {
  return 4 * dModel * dModel;
}

// Fixed teaching values for d_model = 4, not weights from a trained model.
export const MH_X: Matrix = [
  [1, 0, 1, 0],
  [0, 1, 0, 1],
  [1, 1, 0, 1],
];
export const MH_WEIGHTS: MultiHeadWeights = {
  // Every head's column block reads both halves of X: heads split outputs, never input features.
  wq: [
    [1, 1, 0, 1],
    [0, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
  ],
  wk: [
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
  ],
  // Distinct V entries per column, so a change of weights always moves a weighted average.
  wv: [
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
  ],
  wo: [
    [1, 0.5, -0.5, 1],
    [0.5, 1, 1, -0.5],
    [-0.5, 1, 0.5, 1],
    [1, -0.5, 1, 0.5],
  ],
};
