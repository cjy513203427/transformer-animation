import { attention, type Matrix } from './attention';

export const FLASH_TOKENS = 4;
export const FLASH_D_KEY = 2;
export const FLASH_D_VALUE = 2;

export interface FlashConfig {
  blockSize: 1 | 2 | 3 | 4;
  causal: boolean;
}

export interface FlashInput {
  preset: 'rising' | 'book' | 'mixed';
  query: number;
  keyDelta: number;
  block: number;
}

export interface FlashOptions {
  config: FlashConfig;
  input: FlashInput;
  seed: number;
}

/** One completed Key/Value tile for one Query row. a is the unnormalized weighted sum. */
export interface FlashBlock {
  index: number;
  keys: number[];
  scores: number[];
  previousM: number;
  previousL: number;
  previousA: number[];
  blockM: number;
  m: number;
  alpha: number;
  scaledPreviousL: number;
  scaledPreviousA: number[];
  exponentials: number[];
  blockL: number;
  blockA: number[];
  l: number;
  a: number[];
  output: number[];
  validCount: number;
  rescaled: boolean;
}

function matrixShape(matrix: Matrix): [number, number] {
  const width = matrix[0]?.length ?? 0;
  if (
    !matrix.length ||
    !width ||
    matrix.some((row) => row.length !== width || row.some((x) => !Number.isFinite(x)))
  ) {
    throw new Error('矩阵必须非空、形状规则，并且只包含有限数值。');
  }
  return [matrix.length, width];
}

function checkBlockSize(blockSize: number) {
  if (!Number.isInteger(blockSize) || blockSize < 1) throw new Error('块大小必须是正整数。');
}

/**
 * Stream one score row, including a wholly masked row. Negative infinity means masked.
 * The teaching trace retains snapshots for inspection; a production kernel discards them.
 */
export function traceOnlineRow(scores: number[], v: Matrix, blockSize: number): FlashBlock[] {
  const [keys, dValue] = matrixShape(v);
  checkBlockSize(blockSize);
  if (scores.length !== keys || scores.some((x) => !Number.isFinite(x) && x !== -Infinity)) {
    throw new Error('分数必须与 V 行数一致，且只包含有限值或表示 Mask 的负无穷。');
  }
  const blocks: FlashBlock[] = [];
  let m = -Infinity;
  let l = 0;
  let a = Array<number>(dValue).fill(0);
  for (let start = 0; start < keys; start += blockSize) {
    const indices = Array.from({ length: Math.min(blockSize, keys - start) }, (_, j) => start + j);
    const currentScores = indices.map((j) => scores[j]);
    const previousM = m;
    const previousL = l;
    const previousA = a.slice();
    const blockM = Math.max(...currentScores);
    m = Math.max(previousM, blockM);
    // exp(-Infinity - -Infinity) is NaN: skip empty tiles explicitly.
    const alpha = blockM === -Infinity ? 1 : previousM === -Infinity ? 0 : Math.exp(previousM - m);
    const exponentials = currentScores.map((s) => (s === -Infinity ? 0 : Math.exp(s - m)));
    const blockL = exponentials.reduce((sum, value) => sum + value, 0);
    const blockA = Array.from({ length: dValue }, (_, d) =>
      indices.reduce((sum, j, offset) => sum + exponentials[offset] * v[j][d], 0),
    );
    l = alpha * previousL + blockL;
    a = blockA.map((value, d) => alpha * previousA[d] + value);
    blocks.push({
      index: blocks.length,
      keys: indices,
      scores: currentScores,
      previousM,
      previousL,
      previousA,
      blockM,
      m,
      alpha,
      scaledPreviousL: alpha * previousL,
      scaledPreviousA: previousA.map((value) => alpha * value),
      exponentials,
      blockL,
      blockA,
      l,
      a: a.slice(),
      output: a.map((value) => (l === 0 ? 0 : value / l)),
      validCount: currentScores.filter((s) => s !== -Infinity).length,
      rescaled: previousL > 0 && m > previousM,
    });
  }
  return blocks;
}

/** Standard scaled dot-product Attention, streamed one Query row and one K/V tile at a time. */
export function onlineAttention(
  q: Matrix,
  k: Matrix,
  v: Matrix,
  blockSize: number,
  causal = false,
) {
  const [queries, dimensions] = matrixShape(q);
  const [keys, keyDimensions] = matrixShape(k);
  const [values] = matrixShape(v);
  checkBlockSize(blockSize);
  if (dimensions !== keyDimensions || keys !== values || (causal && queries !== keys)) {
    throw new Error('Q/K 维度、K/V 长度必须匹配；本例因果 Mask 要求 Q/K 位置一一对应。');
  }
  const divisor = Math.sqrt(dimensions);
  const rows = q.map((query, i) => {
    const scores = k.map((key, j) =>
      causal && j > i ? -Infinity : query.reduce((sum, x, d) => sum + x * key[d], 0) / divisor,
    );
    return traceOnlineRow(scores, v, blockSize);
  });
  return { rows, output: rows.map((blocks) => blocks[blocks.length - 1].output.slice()) };
}

/**
 * Illustrative logical slots / intermediate transfers, not total GPU HBM I/O or timing.
 * Dense baseline materializes S and P; streaming keeps one Q row, one K/V tile,
 * one score tile, two dv accumulators, and three scalar slots. Inputs/outputs are excluded
 * from the dense intermediate count. Actual kernels have different schedules and reuse.
 */
export function flashLedger(
  tokens: number,
  dKey: number,
  dValue: number,
  blockSize: number,
  causal: boolean,
) {
  if (![tokens, dKey, dValue].every((n) => Number.isInteger(n) && n > 0))
    throw new Error('账本维度必须是正整数。');
  checkBlockSize(blockSize);
  const actualBlock = Math.min(blockSize, tokens);
  return {
    tokens,
    dKey,
    dValue,
    blockSize: actualBlock,
    pairCount: tokens * tokens,
    validPairs: causal ? (tokens * (tokens + 1)) / 2 : tokens * tokens,
    naiveIntermediateElements: 2 * tokens * tokens,
    tiledScratchElements: actualBlock * (dKey + dValue + 1) + dKey + 2 * dValue + 3,
    naiveIntermediateTransfers: 4 * tokens * tokens,
    tiledIntermediateTransfers: 0,
    bytesPerElement: 4,
  };
}

/** The score row of the book's §21.4.4 hand calculation (and §6.3.4 Softmax example). */
export const BOOK_SCORE_ROW = [3.01, 0.09, 2.48, 1.95];

export function makeFlashInputs(preset: FlashInput['preset'], seed: number) {
  if (preset === 'book') {
    // q = [√2, 0] makes q·k / √d_k equal to the first coordinate of k for every row.
    return {
      q: Array.from({ length: FLASH_TOKENS }, () => [Math.SQRT2, 0]),
      k: BOOK_SCORE_ROW.map((score) => [score, 0]),
      v: [
        [1, 0],
        [0, 2],
        [2, 1],
        [-1, 3],
      ],
    };
  }
  if (preset === 'rising') {
    return {
      q: [
        [Math.SQRT2, 0],
        [Math.SQRT2, 0.5],
        [Math.SQRT2, -0.5],
        [Math.SQRT2, 1],
      ],
      k: [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
      ],
      v: [
        [1, 0],
        [0, 2],
        [2, 1],
        [-1, 3],
      ],
    };
  }
  let current = seed >>> 0;
  const next = () => {
    current = (Math.imul(current, 1664525) + 1013904223) >>> 0;
    return ((current % 17) - 8) / 4;
  };
  const matrix = () => Array.from({ length: FLASH_TOKENS }, () => [next(), next()]);
  return { q: matrix(), k: matrix(), v: matrix() };
}

export function computeFlash({ config, input, seed }: FlashOptions) {
  const { q, k, v } = makeFlashInputs(input.preset, seed);
  k[3][0] += input.keyDelta;
  const reference = attention(q, k, v, config.causal, true);
  const { rows, output } = onlineAttention(q, k, v, config.blockSize, config.causal);
  const maxError = Math.max(
    ...output.flatMap((row, i) => row.map((value, d) => Math.abs(value - reference.output[i][d]))),
  );
  return {
    q,
    k,
    v,
    reference,
    rows,
    output,
    maxError,
    ledger: flashLedger(FLASH_TOKENS, FLASH_D_KEY, FLASH_D_VALUE, config.blockSize, config.causal),
  };
}

export type FlashResult = ReturnType<typeof computeFlash>;
