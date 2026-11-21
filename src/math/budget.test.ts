import { describe, expect, it } from 'vitest';
import { budgetLedger, type BudgetConfig } from './budget';
import { BUDGET_PRESETS } from '../state/budget';

/** Independent reference: list every tensor shape of a Llama-style block and multiply out. */
function llamaShapes(c: BudgetConfig) {
  const d = c.dModel,
    kv = (c.kvHeads * d) / c.heads;
  const block = [
    [d, d], // q_proj
    [d, kv], // k_proj
    [d, kv], // v_proj
    [d, d], // o_proj
    [d, c.dFF], // gate_proj
    [d, c.dFF], // up_proj
    [c.dFF, d], // down_proj
    [d], // input_layernorm (RMSNorm)
    [d], // post_attention_layernorm
  ];
  const tensors = [
    [c.vocab, d], // embed_tokens
    ...Array.from({ length: c.layers }, () => block).flat(),
    [d], // final norm
    [c.vocab, d], // lm_head (untied)
  ];
  return tensors.reduce((sum, shape) => sum + shape.reduce((a, b) => a * b, 1), 0);
}

describe('参数量账本', () => {
  it('公开配置的总数与逐张量形状清单一致', () => {
    expect(budgetLedger(BUDGET_PRESETS.llama7b).total).toBe(6_738_415_616);
    expect(budgetLedger(BUDGET_PRESETS.llama7b).total).toBe(llamaShapes(BUDGET_PRESETS.llama7b));
    // GQA: 64 Query heads share 8 KV heads; K/V projections are 8× narrower than W_Q.
    expect(budgetLedger(BUDGET_PRESETS.llama70b).total).toBe(68_976_648_192);
    expect(budgetLedger(BUDGET_PRESETS.llama70b).total).toBe(llamaShapes(BUDGET_PRESETS.llama70b));
    expect(budgetLedger(BUDGET_PRESETS.gpt2).total).toBe(124_439_808);
  });

  it('复现原书第 15.7 与 18.11 节的分项表', () => {
    const gpt2 = budgetLedger(BUDGET_PRESETS.gpt2).parts;
    expect(gpt2).toEqual({
      embedding: 38_597_376,
      position: 786_432,
      attention: 28_348_416,
      ffn: 56_669_184,
      norm: 38_400,
      head: 0,
    });
    const book = budgetLedger(BUDGET_PRESETS.book18);
    expect([book.parts.embedding, book.parts.attention, book.parts.ffn, book.parts.norm]).toEqual([
      25_600_000, 6_291_456, 12_598_272, 13_312,
    ]);
    expect(book.total).toBe(44_503_040);
    expect(budgetLedger({ ...BUDGET_PRESETS.book18, tied: false }).total).toBe(70_103_040);
  });

  it('12Ld² 只在 MHA、d_ff = 4d、GELU、无偏置时等于逐层精确值', () => {
    const plain = { ...BUDGET_PRESETS.gpt2, bias: 'none' as const };
    const r = budgetLedger(plain);
    // Norm γ/β are the only per-layer parameters the approximation drops.
    expect(r.nonEmbedding - (2 * plain.layers + 1) * 2 * plain.dModel).toBe(r.approximation);
    const swiglu = budgetLedger(BUDGET_PRESETS.llama7b);
    expect(swiglu.nonEmbedding).not.toBe(swiglu.approximation);
  });

  it('权重绑定减少参数但不减少前向 FLOPs；词表大小不改变每个 Block', () => {
    const tied = budgetLedger(BUDGET_PRESETS.book18),
      untied = budgetLedger({ ...BUDGET_PRESETS.book18, tied: false });
    expect(untied.total - tied.total).toBe(50000 * 512);
    expect(untied.flops.forward).toBe(tied.flops.forward);
    const bigVocab = budgetLedger({ ...BUDGET_PRESETS.llama7b, vocab: 128_000 });
    expect(bigVocab.perLayer).toBe(budgetLedger(BUDGET_PRESETS.llama7b).perLayer);
  });

  it('序列长度到 s* 时配对项与线性项相等，之前线性项更大', () => {
    const c = BUDGET_PRESETS.gpt2,
      star = budgetLedger(c).flops.crossover;
    expect(star).toBe(6 * c.dModel); // 12d² / 2d for MHA with d_ff = 4d
    const at = budgetLedger({ ...c, context: star }).flops;
    expect(at.pairs).toBe(at.linear);
    const short = budgetLedger({ ...c, context: 512 }).flops;
    expect(short.pairs).toBeLessThan(short.linear);
  });

  it('训练总量与 6ND 都随 D 线性增长；D/N 只是比值', () => {
    const base = budgetLedger(BUDGET_PRESETS.llama7b),
      twice = budgetLedger({ ...BUDGET_PRESETS.llama7b, tokensB: 4000 });
    expect(twice.flops.training).toBe(2 * base.flops.training);
    expect(twice.flops.sixND).toBe(2 * base.flops.sixND);
    expect(base.flops.tokensPerParameter).toBeCloseTo(2e12 / 6_738_415_616, 12);
  });

  it('每 token 训练约为前向的 3 倍；短序列时与 6ND 同量级', () => {
    const r = budgetLedger({ ...BUDGET_PRESETS.llama7b, context: 1 });
    expect(r.flops.trainingPerToken).toBe(3 * r.flops.forwardPerToken);
    // At s = 1, 6N per token counts the embedding lookup and norms (no matrix products here)
    // but misses the attention pair term 3 · 4sd per layer.
    expect(6 * r.total - r.flops.trainingPerToken).toBe(
      6 * (r.parts.embedding + r.parts.norm) - 3 * 4 * 32 * 4096,
    );
  });
});
