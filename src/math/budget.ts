/**
 * Parameter and FLOP ledger for a dense decoder-only Transformer (row-vector convention).
 * Counts weights exactly from the configuration; FLOPs count one multiply-add as 2 and only
 * matrix products. Norms, Softmax, activations, bias additions and the embedding lookup are
 * omitted, as in Appendix A.6 of the book. MoE, MLA and encoder–decoder models are out of scope.
 */
export type FFNKind = 'gelu' | 'swiglu';
export type NormKind = 'layer' | 'rms';
/** none: no biases; ffn: FFN biases only (book chapter 18); all: Q/K/V/O and FFN (GPT-2). */
export type BiasKind = 'none' | 'ffn' | 'all';

export interface BudgetConfig {
  vocab: number;
  dModel: number;
  layers: number;
  heads: number;
  kvHeads: number;
  dFF: number;
  ffn: FFNKind;
  norm: NormKind;
  bias: BiasKind;
  /** Input embedding and output projection share one matrix. */
  tied: boolean;
  /** Rows of a learned absolute position table; 0 for sinusoidal, RoPE or ALiBi. */
  positions: number;
  /** Sequence length s used for the FLOP counts. */
  context: number;
  /** Training tokens D, in billions. */
  tokensB: number;
}

export function budgetLedger(c: BudgetConfig) {
  const d = c.dModel,
    headDim = d / c.heads,
    kvWidth = c.kvHeads * headDim,
    s = c.context;
  const embedding = c.vocab * d;
  const position = c.positions * d;
  // W_Q [d, d], W_K and W_V [d, H_kv·d_h], W_O [d, d].
  const attentionWeights = 2 * d * d + 2 * d * kvWidth;
  const attentionBias = c.bias === 'all' ? 2 * d + 2 * kvWidth : 0;
  // GELU: W_1 [d, d_ff], W_2 [d_ff, d]. SwiGLU adds the gate W_3 [d, d_ff].
  const ffnWeights = (c.ffn === 'swiglu' ? 3 : 2) * d * c.dFF;
  const ffnBias = c.bias === 'none' ? 0 : (c.ffn === 'swiglu' ? 2 : 1) * c.dFF + d;
  const normEach = c.norm === 'layer' ? 2 * d : d;
  const perLayer = attentionWeights + attentionBias + ffnWeights + ffnBias + 2 * normEach;
  const head = c.tied ? 0 : c.vocab * d;
  const parts = {
    embedding,
    position,
    attention: c.layers * (attentionWeights + attentionBias),
    ffn: c.layers * (ffnWeights + ffnBias),
    norm: (2 * c.layers + 1) * normEach,
    head,
  };
  const total = Object.values(parts).reduce((a, b) => a + b, 0);
  const nonEmbedding = total - embedding - position - head;

  // Forward FLOPs for one sequence of s tokens.
  const linearPerLayer = 2 * s * (attentionWeights + ffnWeights);
  // QKᵀ and PV each take s·s·d_h multiply-adds per head; summed over H heads that is s²·d.
  const pairsPerLayer = 4 * s * s * d;
  const logits = 2 * s * d * c.vocab;
  const forward = c.layers * (linearPerLayer + pairsPerLayer) + logits;
  const forwardPerToken = forward / s;
  // Backward ≈ 2 × forward, the usual estimate behind C ≈ 6ND.
  const trainingPerToken = 3 * forwardPerToken;
  const tokens = c.tokensB * 1e9;
  // One new token at context length s: projections for one row, plus reading s cached keys/values.
  const decodeToken =
    2 * (c.layers * (attentionWeights + ffnWeights) + d * c.vocab) + c.layers * 4 * s * d;
  return {
    headDim,
    kvWidth,
    parts,
    perLayer,
    total,
    nonEmbedding,
    approximation: 12 * c.layers * d * d,
    flops: {
      linear: c.layers * linearPerLayer,
      pairs: c.layers * pairsPerLayer,
      logits,
      forward,
      forwardPerToken,
      trainingPerToken,
      training: trainingPerToken * tokens,
      sixND: 6 * total * tokens,
      /** Training tokens per parameter, D/N. */
      tokensPerParameter: tokens / total,
      decodeToken,
      /** Length where the s² pair term equals the per-layer linear term. */
      crossover: (attentionWeights + ffnWeights) / (2 * d),
    },
  };
}
