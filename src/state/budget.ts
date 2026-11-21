import type { BiasKind, BudgetConfig, FFNKind, NormKind } from '../math/budget';
import { T } from '../i18n';

export type BudgetPreset = 'gpt2' | 'book18' | 'llama7b' | 'llama70b';
/** Published configurations; parameter totals are computed, then checked in tests. */
export const BUDGET_PRESETS: Record<BudgetPreset, BudgetConfig> = {
  gpt2: {
    vocab: 50257,
    dModel: 768,
    layers: 12,
    heads: 12,
    kvHeads: 12,
    dFF: 3072,
    ffn: 'gelu',
    norm: 'layer',
    bias: 'all',
    tied: true,
    positions: 1024,
    context: 1024,
    tokensB: 10,
  },
  book18: {
    vocab: 50000,
    dModel: 512,
    layers: 6,
    heads: 8,
    kvHeads: 8,
    dFF: 2048,
    ffn: 'gelu',
    norm: 'layer',
    bias: 'ffn',
    tied: true,
    positions: 0,
    context: 256,
    tokensB: 1,
  },
  llama7b: {
    vocab: 32000,
    dModel: 4096,
    layers: 32,
    heads: 32,
    kvHeads: 32,
    dFF: 11008,
    ffn: 'swiglu',
    norm: 'rms',
    bias: 'none',
    tied: false,
    positions: 0,
    context: 4096,
    tokensB: 2000,
  },
  llama70b: {
    vocab: 32000,
    dModel: 8192,
    layers: 80,
    heads: 64,
    kvHeads: 8,
    dFF: 28672,
    ffn: 'swiglu',
    norm: 'rms',
    bias: 'none',
    tied: false,
    positions: 0,
    context: 4096,
    tokensB: 2000,
  },
};

export interface BudgetState {
  version: 1;
  config: BudgetConfig;
  view: { recall: boolean };
}
export const initialBudgetState = (): BudgetState => ({
  version: 1,
  config: { ...BUDGET_PRESETS.gpt2 },
  view: { recall: false },
});

/** Integer limits for each numeric field: [min, max]. */
export const BUDGET_LIMITS = {
  vocab: [2, 1_000_000],
  dModel: [8, 65_536],
  layers: [1, 256],
  heads: [1, 512],
  kvHeads: [1, 512],
  dFF: [1, 262_144],
  positions: [0, 1_048_576],
  context: [1, 1_048_576],
} as const;
export type BudgetIntegerKey = keyof typeof BUDGET_LIMITS;

const integer = (v: unknown, [min, max]: readonly [number, number]): v is number =>
  typeof v === 'number' && Number.isInteger(v) && v >= min && v <= max;

/** Why a configuration cannot describe a real model, or null when it can. */
export function budgetProblem(c: BudgetConfig): string | null {
  for (const key of Object.keys(BUDGET_LIMITS) as BudgetIntegerKey[])
    if (!integer(c[key], BUDGET_LIMITS[key])) return T('{#0} 超出允许范围。', key);
  if (!(Number.isFinite(c.tokensB) && c.tokensB >= 0.001 && c.tokensB <= 100_000))
    return T('{#0} 超出允许范围。', 'tokensB');
  if (c.dModel % c.heads !== 0)
    return T('d_model = {#0} 不能被 Query 头数 H = {#1} 整除，每个头的宽度 d_h 不是整数。', c.dModel, c.heads);
  if (c.kvHeads > c.heads || c.heads % c.kvHeads !== 0)
    return T('Query 头数 H = {#0} 必须是 KV 头数 H_kv = {#1} 的整数倍，才能按组共享 K/V。', c.heads, c.kvHeads);
  return null;
}

export const serializeBudgetState = (s: BudgetState) =>
  '#budget=' + encodeURIComponent(JSON.stringify(s));

const FFN: FFNKind[] = ['gelu', 'swiglu'],
  NORM: NormKind[] = ['layer', 'rms'],
  BIAS: BiasKind[] = ['none', 'ffn', 'all'];

export function restoreBudgetState(hash: string): { state: BudgetState; invalid: boolean } {
  const fallback = { state: initialBudgetState(), invalid: !!hash };
  if (!hash) return { state: initialBudgetState(), invalid: false };
  if (!hash.startsWith('#budget=') || hash.length > 1200) return fallback;
  try {
    const x: unknown = JSON.parse(decodeURIComponent(hash.slice(8)));
    if (!x || typeof x !== 'object' || Array.isArray(x)) return fallback;
    const { version, config: c, view } = x as Record<string, unknown>;
    if (version !== 1 || !c || typeof c !== 'object' || !view || typeof view !== 'object')
      return fallback;
    const r = c as Record<string, unknown>;
    if (
      !FFN.includes(r.ffn as FFNKind) ||
      !NORM.includes(r.norm as NormKind) ||
      !BIAS.includes(r.bias as BiasKind) ||
      typeof r.tied !== 'boolean' ||
      typeof (view as Record<string, unknown>).recall !== 'boolean' ||
      typeof r.tokensB !== 'number'
    )
      return fallback;
    // Copy only known fields; anything else in the link is dropped.
    const config = {
      ffn: r.ffn,
      norm: r.norm,
      bias: r.bias,
      tied: r.tied,
      tokensB: r.tokensB,
      ...Object.fromEntries(Object.keys(BUDGET_LIMITS).map((k) => [k, r[k]])),
    } as BudgetConfig;
    if (budgetProblem(config)) return fallback;
    return {
      state: { version: 1, config, view: { recall: (view as { recall: boolean }).recall } },
      invalid: false,
    };
  } catch {
    return fallback;
  }
}
