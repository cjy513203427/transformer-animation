import { matmul, softmax, transpose, type Matrix } from './attention';
import { multiHeadAttention, type MultiHeadWeights } from './multihead';
import { normalize } from './p1/normalization';

export const DECODER_WIDTH = 4;
export const DECODER_VOCAB = ['a', 'b', 'c', 'd'] as const;
export const DECODER_EPSILON = 1e-5;
export const DECODER_EMBEDDING: Matrix = [
  [1, 0, 0.5, -0.5],
  [0, 1, -0.5, 0.5],
  [0.5, -0.5, 1, 0],
  [-0.5, 0.5, 0, 1],
];

export type ForwardKind =
  | 'tokens'
  | 'embedding'
  | 'position'
  | 'attention-norm'
  | 'qkv'
  | 'attention'
  | 'attention-add'
  | 'ffn-norm'
  | 'ffn-expand'
  | 'ffn-contract'
  | 'ffn-add'
  | 'final-norm'
  | 'logits'
  | 'probability'
  | 'selection';

export interface ForwardTensor {
  name: string;
  data: Matrix;
}
export interface ForwardFrame {
  id: string;
  kind: ForwardKind;
  /** One-based block index; absent outside the block stack. */
  layer?: number;
  tensors: ForwardTensor[];
  parameters?: ForwardTensor[];
}
export interface DecoderLayer {
  attention: MultiHeadWeights;
  w1: Matrix;
  w2: Matrix;
}

/** Deterministic teaching parameters, never pretrained weights. */
export function decoderLayers(seed: number): DecoderLayer[] {
  let state = seed >>> 0;
  const matrix = (rows: number, columns: number): Matrix =>
    Array.from({ length: rows }, () =>
      Array.from({ length: columns }, () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return ((state % 9) - 4) / 8;
      }),
    );
  return Array.from({ length: 2 }, () => ({
    attention: { wq: matrix(4, 4), wk: matrix(4, 4), wv: matrix(4, 4), wo: matrix(4, 4) },
    w1: matrix(4, 6),
    w2: matrix(6, 4),
  }));
}

export const gelu = (x: number): number =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const add = (a: Matrix, b: Matrix): Matrix => a.map((row, t) => row.map((v, d) => v + b[t][d]));
export const normalizeRows = (x: Matrix): Matrix =>
  x.map((row) => normalize(row, DECODER_EPSILON).ln);
const tensor = (name: string, data: Matrix): ForwardTensor => ({ name, data });

/** A complete small Pre-Norm decoder forward, with no batch, bias, dropout or optimizer. */
export function decoderForward(
  tokens: readonly number[],
  depth: 1 | 2,
  causal: boolean,
  seed = 42,
) {
  if (
    !tokens.length ||
    tokens.length > 8 ||
    tokens.some((id) => !Number.isInteger(id) || id < 0 || id >= DECODER_VOCAB.length)
  )
    throw new Error('Invalid token sequence.');
  if (depth !== 1 && depth !== 2) throw new Error('Invalid decoder depth.');
  const embedding = tokens.map((id) => [...DECODER_EMBEDDING[id]]);
  const position = tokens.map((_, t) =>
    Array.from({ length: DECODER_WIDTH }, (_, d) => {
      const angle = t / 10000 ** ((2 * Math.floor(d / 2)) / DECODER_WIDTH);
      return d % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }),
  );
  let hidden = add(embedding, position);
  const frames: ForwardFrame[] = [
    {
      id: 'tokens',
      kind: 'tokens',
      tensors: [
        tensor(
          'token_ids',
          tokens.map((id) => [id]),
        ),
      ],
    },
    {
      id: 'embedding',
      kind: 'embedding',
      tensors: [tensor('E[token]', embedding)],
      parameters: [tensor('E', DECODER_EMBEDDING)],
    },
    { id: 'position', kind: 'position', tensors: [tensor('PE', position), tensor('H_0', hidden)] },
  ];
  const layers = decoderLayers(seed).slice(0, depth);
  const outputs: Matrix[] = [];
  for (const [i, weights] of layers.entries()) {
    const layer = i + 1;
    const push = (kind: ForwardKind, tensors: ForwardTensor[], parameters?: ForwardTensor[]) =>
      frames.push({ id: `block-${layer}-${kind}`, layer, kind, tensors, parameters });
    const u = normalizeRows(hidden);
    push('attention-norm', [tensor(`U_${layer}`, u)]);
    const attention = multiHeadAttention(u, weights.attention, 2, causal);
    push(
      'qkv',
      [tensor('Q', attention.q), tensor('K', attention.k), tensor('V', attention.v)],
      [
        tensor('W_Q', weights.attention.wq),
        tensor('W_K', weights.attention.wk),
        tensor('W_V', weights.attention.wv),
      ],
    );
    push(
      'attention',
      [
        tensor('P_0', attention.heads[0].probabilities),
        tensor('P_1', attention.heads[1].probabilities),
        tensor('A', attention.output),
      ],
      [tensor('W_O', weights.attention.wo)],
    );
    const residual = add(hidden, attention.output);
    push('attention-add', [
      tensor(`H_${i}`, hidden),
      tensor('A', attention.output),
      tensor(`R_${layer}`, residual),
    ]);
    const z = normalizeRows(residual);
    push('ffn-norm', [tensor(`Z_${layer}`, z)]);
    const preactivation = matmul(z, weights.w1);
    const g = preactivation.map((row) => row.map(gelu));
    push(
      'ffn-expand',
      [tensor('Z W_1', preactivation), tensor('G', g)],
      [tensor('W_1', weights.w1)],
    );
    const f = matmul(g, weights.w2);
    push('ffn-contract', [tensor('F', f)], [tensor('W_2', weights.w2)]);
    hidden = add(residual, f);
    push('ffn-add', [tensor('R', residual), tensor('F', f), tensor(`H_${layer}`, hidden)]);
    outputs.push(hidden);
  }
  const final = normalizeRows(hidden);
  const logits = matmul(final, transpose(DECODER_EMBEDDING));
  const probabilities = logits.map(softmax);
  const last = probabilities.at(-1)!;
  const selected = last.indexOf(Math.max(...last));
  frames.push(
    { id: 'final-norm', kind: 'final-norm', tensors: [tensor('H_f', final)] },
    {
      id: 'logits',
      kind: 'logits',
      tensors: [tensor('logits', logits)],
      parameters: [tensor('E^T', transpose(DECODER_EMBEDDING))],
    },
    { id: 'probability', kind: 'probability', tensors: [tensor('p', probabilities)] },
    { id: 'selection', kind: 'selection', tensors: [tensor('p_last', [last])] },
  );
  return { frames, embedding, position, outputs, final, logits, probabilities, selected };
}

export const forwardStepCount = (depth: 1 | 2): number => 3 + 8 * depth + 4;
