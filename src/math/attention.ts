export type Matrix = number[][];

function shape(matrix: Matrix): [number, number] {
  const columns = matrix[0]?.length ?? 0;
  if (
    !matrix.length ||
    !columns ||
    matrix.some((row) => row.length !== columns || row.some((x) => !Number.isFinite(x)))
  ) {
    throw new Error('矩阵必须非空、形状规则，并且只包含有限数值。');
  }
  return [matrix.length, columns];
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const [rows, inner] = shape(a);
  const [otherInner, columns] = shape(b);
  if (inner !== otherInner) throw new Error('矩阵乘法的内维必须相同。');
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: columns }, (_, j) =>
      a[i].reduce((sum, value, k) => sum + value * b[k][j], 0),
    ),
  );
}

export function transpose(matrix: Matrix): Matrix {
  const [rows, columns] = shape(matrix);
  return Array.from({ length: columns }, (_, j) =>
    Array.from({ length: rows }, (_, i) => matrix[i][j]),
  );
}

/** Negative infinity represents a masked key. An entirely masked row returns zeros. */
export function softmax(row: number[]): number[] {
  if (row.some((x) => Number.isNaN(x) || x === Infinity))
    throw new Error('Softmax 输入必须是有限值或负无穷。');
  const max = Math.max(...row);
  if (max === -Infinity) return row.map(() => 0);
  const weights = row.map((value) => Math.exp(value - max));
  const sum = weights.reduce((a, b) => a + b, 0);
  return weights.map((value) => value / sum);
}

export function attention(q: Matrix, k: Matrix, v: Matrix, causal = false, scaled = true) {
  const [queries, dimensions] = shape(q);
  const [keys, keyDimensions] = shape(k);
  const [values] = shape(v);
  if (dimensions !== keyDimensions || keys !== values)
    throw new Error('Q/K 维度与 K/V 长度必须匹配。');
  if (causal && queries !== keys) throw new Error('此教学实现的因果 Mask 要求 Q/K 位置一一对应。');
  const divisor = scaled ? Math.sqrt(dimensions) : 1;
  const rawScores = matmul(q, transpose(k));
  const scores = rawScores.map((row) => row.map((x) => x / divisor));
  const masked = scores.map((row, i) => row.map((x, j) => (causal && j > i ? -Infinity : x)));
  const probabilities = masked.map(softmax);
  return { rawScores, scores, masked, probabilities, output: matmul(probabilities, v) };
}

// Fixed teaching parameters, not weights from a trained model.
export const W_Q: Matrix = [
  [1, 0],
  [0, 1],
];
export const W_K: Matrix = [
  [1, 0],
  [0.5, 1],
];
export const W_V: Matrix = [
  [1, 2],
  [2, 0],
];

export function computeExperiment(
  x: Matrix,
  keyDelta: number,
  valueDelta: number,
  causal: boolean,
  scaled: boolean,
) {
  const q = matmul(x, W_Q);
  const k = matmul(x, W_K);
  const v = matmul(x, W_V);
  // A deliberate activation intervention after projection. The parameters stay fixed.
  k[1][0] += keyDelta;
  v[1][0] += valueDelta;
  return { q, k, v, ...attention(q, k, v, causal, scaled) };
}

export function seededInput(seed: number): Matrix {
  let value = seed >>> 0;
  const next = () => {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    return ((value % 9) - 4) / 2;
  };
  return Array.from({ length: 3 }, () => [next(), next()]);
}

export function formatNumber(value: number, digits = 3): string {
  if (value === -Infinity) return '−∞';
  return Number(value.toFixed(digits)).toString();
}

/** Numeric annotations for the guide; renderers consume these instead of recomputing Attention. */
export function explainRow(result: ReturnType<typeof computeExperiment>, query: number) {
  const maximum = Math.max(...result.masked[query]);
  const exponentials = result.masked[query].map((score) =>
    maximum === -Infinity ? 0 : Math.exp(score - maximum),
  );
  return {
    maximum,
    probabilitySum: result.probabilities[query].reduce((a, b) => a + b, 0),
    denominator: exponentials.reduce((a, b) => a + b, 0),
    keys: result.k.map((key, j) => ({
      products: key.map((value, d) => result.q[query][d] * value),
      rawScore: result.rawScores[query][j],
      score: result.scores[query][j],
      maskedScore: result.masked[query][j],
      exponential: exponentials[j],
      probability: result.probabilities[query][j],
      contribution: result.v[j].map((value) => value * result.probabilities[query][j]),
    })),
  };
}
