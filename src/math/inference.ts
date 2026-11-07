import { matmul, softmax, type Matrix } from './attention';
import { sinusoidalPE } from './position';

/**
 * A toy decoder-only model for the "training, generation and cache" experiment.
 *
 * Simplifications (all deliberate): one head, two layers, d_model = d_k = d_v = 2, no W_O,
 * no FFN, no LayerNorm, no bias, no Dropout. Each layer is H' = H + softmax(causal(QK^T/√d)) V.
 * Positions come from additive sinusoidal encodings. The output head reuses the embedding table
 * (logits = H E^T). Weights are fixed teaching numbers, not a trained model.
 */
export const VOCAB = ['a', 'b', 'c', 'd', 'e', 'f'] as const;
export const D_MODEL = 2;
export const NUM_LAYERS = 2;

export interface LayerWeights {
  wq: Matrix;
  wk: Matrix;
  wv: Matrix;
}
export interface ToyModel {
  embed: Matrix;
  layers: LayerWeights[];
}

export const TOY_MODEL: ToyModel = {
  // One row per token, spread around the unit circle so that the output head (logits = H E^T)
  // prefers different tokens for different hidden states.
  embed: [
    [1, 0],
    [0.5, 0.866],
    [-0.5, 0.866],
    [-1, 0],
    [-0.5, -0.866],
    [0.5, -0.866],
  ],
  layers: [
    {
      wq: [
        [1, 0.5],
        [-0.5, 1],
      ],
      wk: [
        [1, -0.5],
        [0.5, 1],
      ],
      wv: [
        [0.8, 0.6],
        [-0.6, 0.8],
      ],
    },
    {
      wq: [
        [0.9, 0.4],
        [-0.4, 0.9],
      ],
      wk: [
        [0.9, -0.4],
        [0.4, 0.9],
      ],
      wv: [
        [0.2, 1],
        [-1, 0.2],
      ],
    },
  ],
};

/** Multiply-add style counters, incremented where the work is actually done. */
export interface Workload {
  /** (position, layer) pairs for which K and V were projected. */
  kvProjections: number;
  /** Query-key dot products (one per visible pair, per layer). */
  scorePairs: number;
}
const zeroWork = (): Workload => ({ kvProjections: 0, scorePairs: 0 });

function checkIds(ids: number[]) {
  if (!ids.length) throw new Error('序列不能为空。');
  if (ids.some((id) => !Number.isInteger(id) || id < 0 || id >= VOCAB.length))
    throw new Error('token 编号超出词表范围。');
}

/** X[t] = E[token_t] + PE[start + t]. */
export function embedTokens(model: ToyModel, ids: number[], start = 0): Matrix {
  checkIds(ids);
  return ids.map((id, t) => {
    const pe = sinusoidalPE(start + t, D_MODEL);
    return model.embed[id].map((x, d) => x + pe[d]);
  });
}

function rowTimes(row: number[], weights: Matrix): number[] {
  return matmul([row], weights)[0];
}
function dot(a: number[], b: number[]): number {
  return a.reduce((sum, x, i) => sum + x * b[i], 0);
}

/** Output distribution for one hidden row: softmax(h E^T). */
export function nextTokenProbs(
  model: ToyModel,
  hidden: number[],
): { logits: number[]; probs: number[] } {
  const logits = model.embed.map((row) => dot(hidden, row));
  return { logits, probs: softmax(logits) };
}

/** Greedy choice: the largest probability, the lowest token id on an exact tie. */
export function greedyToken(probs: number[]): number {
  let best = 0;
  probs.forEach((p, i) => {
    if (p > probs[best]) best = i;
  });
  return best;
}

export interface LayerTrace {
  input: Matrix;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  /** Scaled scores with −∞ above the diagonal. */
  scores: Matrix;
  weights: Matrix;
  context: Matrix;
  output: Matrix;
}
export interface FullForward {
  layers: LayerTrace[];
  hidden: Matrix;
  logits: Matrix;
  probs: Matrix;
  work: Workload;
}

/** Causal forward pass over a whole sequence: what training and Prefill do, and the no-cache baseline. */
export function forwardFull(model: ToyModel, ids: number[]): FullForward {
  let x = embedTokens(model, ids);
  const work = zeroWork();
  const root = Math.sqrt(D_MODEL);
  const layers: LayerTrace[] = [];
  for (const { wq, wk, wv } of model.layers) {
    const q = matmul(x, wq);
    const k = matmul(x, wk);
    const v = matmul(x, wv);
    work.kvProjections += x.length;
    const scores: Matrix = [];
    const weights: Matrix = [];
    for (let i = 0; i < x.length; i++) {
      const row: number[] = x.map(() => -Infinity);
      for (let j = 0; j <= i; j++) {
        row[j] = dot(q[i], k[j]) / root;
        work.scorePairs += 1;
      }
      scores.push(row);
      weights.push(softmax(row));
    }
    const context = matmul(weights, v);
    const output = x.map((row, i) => row.map((value, d) => value + context[i][d]));
    layers.push({ input: x, q, k, v, scores, weights, context, output });
    x = output;
  }
  const heads = x.map((h) => nextTokenProbs(model, h));
  return {
    layers,
    hidden: x,
    logits: heads.map((h) => h.logits),
    probs: heads.map((h) => h.probs),
    work,
  };
}

/** Cached keys and values: `k[layer][position]` and `v[layer][position]`. */
export interface KVCache {
  k: Matrix[];
  v: Matrix[];
}
export const cacheLength = (cache: KVCache) => cache.k[0]?.length ?? 0;

/** Prefill writes every position's K and V of every layer into the cache. */
export function cacheFromForward(forward: FullForward): KVCache {
  return {
    k: forward.layers.map((layer) => layer.k.map((row) => [...row])),
    v: forward.layers.map((layer) => layer.v.map((row) => [...row])),
  };
}

export interface DecodeLayerTrace {
  input: number[];
  q: number[];
  /** The new position's K and V, appended to this layer's cache. */
  k: number[];
  v: number[];
  /** Scaled scores of the new query against the n + 1 keys (old keys came from the cache). */
  scores: number[];
  weights: number[];
  context: number[];
  output: number[];
}
export interface DecodeResult {
  cache: KVCache;
  layers: DecodeLayerTrace[];
  logits: number[];
  probs: number[];
  position: number;
  work: Workload;
}

/**
 * One Decode step. Only the new token is projected; earlier K and V come from the cache, and
 * earlier queries are not needed because their outputs were finished in earlier steps.
 */
export function decodeStep(model: ToyModel, cache: KVCache, tokenId: number): DecodeResult {
  const position = cacheLength(cache);
  if (!position) throw new Error('Decode 之前必须先 Prefill。');
  checkIds([tokenId]);
  const work = zeroWork();
  const root = Math.sqrt(D_MODEL);
  let x = embedTokens(model, [tokenId], position)[0];
  const next: KVCache = { k: [], v: [] };
  const layers: DecodeLayerTrace[] = [];
  model.layers.forEach(({ wq, wk, wv }, layer) => {
    const q = rowTimes(x, wq);
    const k = rowTimes(x, wk);
    const v = rowTimes(x, wv);
    work.kvProjections += 1;
    const keys = [...cache.k[layer], k];
    const values = [...cache.v[layer], v];
    const scores = keys.map((key) => {
      work.scorePairs += 1;
      return dot(q, key) / root;
    });
    const weights = softmax(scores);
    const context = values[0].map((_, d) =>
      weights.reduce((sum, w, j) => sum + w * values[j][d], 0),
    );
    const output = x.map((value, d) => value + context[d]);
    next.k.push(keys);
    next.v.push(values);
    layers.push({ input: x, q, k, v, scores, weights, context, output });
    x = output;
  });
  const { logits, probs } = nextTokenProbs(model, x);
  return { cache: next, layers, logits, probs, position, work };
}

export const MAX_DECODE_STEPS = 5;

export interface GenerationStep {
  /** Token fed into this Decode step (the one produced by the previous step). */
  input: number;
  decode: DecodeResult;
  /** Token produced by this step. */
  produced: number;
  /** The same step done the slow way: a full causal pass over every token so far. */
  recompute: FullForward;
  /** Largest absolute difference between the two logits vectors for the newest position. */
  logitsDiff: number;
}
export interface Generation {
  prompt: number[];
  prefill: FullForward;
  /** First token, chosen from the prefill output at the last prompt position. */
  first: number;
  steps: GenerationStep[];
  /** Prompt followed by every token produced so far. */
  tokens: number[];
}

/** Greedy generation. Every Decode step is checked against a full recompute of the same context. */
export function generate(model: ToyModel, prompt: number[], decodeSteps: number): Generation {
  checkIds(prompt);
  if (!Number.isInteger(decodeSteps) || decodeSteps < 0 || decodeSteps > MAX_DECODE_STEPS)
    throw new Error('Decode 步数超出范围。');
  const prefill = forwardFull(model, prompt);
  let cache = cacheFromForward(prefill);
  const first = greedyToken(prefill.probs[prompt.length - 1]);
  const context = [...prompt];
  let feed = first;
  const steps: GenerationStep[] = [];
  for (let s = 0; s < decodeSteps; s++) {
    context.push(feed);
    const decode = decodeStep(model, cache, feed);
    const recompute = forwardFull(model, context);
    const last = recompute.logits[context.length - 1];
    const logitsDiff = Math.max(...decode.logits.map((value, i) => Math.abs(value - last[i])));
    const produced = greedyToken(decode.probs);
    steps.push({ input: feed, decode, produced, recompute, logitsDiff });
    cache = decode.cache;
    feed = produced;
  }
  return { prompt, prefill, first, steps, tokens: [...context, feed] };
}

/** Whether the cache entry (layer, position) differs between two caches, with an explicit tolerance. */
export function entryChanged(
  a: KVCache,
  b: KVCache,
  layer: number,
  posA: number,
  posB: number,
  tolerance = 1e-12,
): boolean {
  const pairs: [number[], number[]][] = [
    [a.k[layer][posA], b.k[layer][posB]],
    [a.v[layer][posA], b.v[layer][posB]],
  ];
  return pairs.some(([x, y]) => x.some((value, d) => Math.abs(value - y[d]) > tolerance));
}

/**
 * Which cache entries are no longer valid after the token at `position` is replaced, with all
 * other tokens kept. Rows are layers, columns are positions.
 */
export function invalidatedByEdit(
  model: ToyModel,
  ids: number[],
  position: number,
  token: number,
): boolean[][] {
  const edited = ids.map((id, i) => (i === position ? token : id));
  const before = cacheFromForward(forwardFull(model, ids));
  const after = cacheFromForward(forwardFull(model, edited));
  return before.k.map((_, layer) => ids.map((_, p) => entryChanged(before, after, layer, p, p)));
}

/**
 * Which entries of the old cache still match when the oldest token is dropped and the rest is
 * renumbered from position 0 (a sliding window). Column 0 is the dropped token.
 */
export function invalidatedByWindowSlide(model: ToyModel, ids: number[]): (boolean | null)[][] {
  const before = cacheFromForward(forwardFull(model, ids));
  const after = cacheFromForward(forwardFull(model, ids.slice(1)));
  return before.k.map((_, layer) =>
    ids.map((_, p) => (p === 0 ? null : entryChanged(before, after, layer, p, p - 1))),
  );
}

export interface TrainingRow {
  position: number;
  input: number;
  target: number;
  /** Input positions the output at this row is allowed to depend on. */
  visible: number[];
  probs: number[];
  targetProb: number;
  loss: number;
}

/** Teacher forcing: inputs x[0..T−1], targets x[1..T], one causal pass gives every row at once. */
export function trainingRows(model: ToyModel, sequence: number[]): TrainingRow[] {
  if (sequence.length < 2) throw new Error('训练样本至少需要两个 token。');
  const inputs = sequence.slice(0, -1);
  const targets = sequence.slice(1);
  const { probs } = forwardFull(model, inputs);
  return inputs.map((input, position) => {
    const targetProb = probs[position][targets[position]];
    return {
      position,
      input,
      target: targets[position],
      visible: Array.from({ length: position + 1 }, (_, i) => i),
      probs: probs[position],
      targetProb,
      loss: -Math.log(targetProb),
    };
  });
}

/** Counterfactual for the training scene: replace input x[position] but keep every target fixed. */
export function trainingWithEditedInput(
  model: ToyModel,
  sequence: number[],
  position: number,
  token: number,
): { rows: TrainingRow[]; changedRows: boolean[] } {
  const base = trainingRows(model, sequence);
  const targets = sequence.slice(1);
  const edited = sequence.map((id, i) => (i === position ? token : id));
  // Keep the targets of the original sample: only the model's inputs are perturbed.
  const { probs } = forwardFull(model, edited.slice(0, -1));
  const rows = base.map((row, i) => {
    const targetProb = probs[i][targets[i]];
    return { ...row, input: edited[i], probs: probs[i], targetProb, loss: -Math.log(targetProb) };
  });
  const changedRows = rows.map((row, i) =>
    row.probs.some((p, k) => Math.abs(p - base[i].probs[k]) > 0),
  );
  return { rows, changedRows };
}

/** Closed-form workload of a full causal pass over n tokens (used to cross-check the counters). */
export function fullPassWork(n: number, layers = NUM_LAYERS): Workload {
  return { kvProjections: layers * n, scorePairs: (layers * n * (n + 1)) / 2 };
}
/** Closed-form workload of one cached Decode step when the cache already holds n − 1 tokens. */
export function cachedStepWork(n: number, layers = NUM_LAYERS): Workload {
  return { kvProjections: layers, scorePairs: layers * n };
}
