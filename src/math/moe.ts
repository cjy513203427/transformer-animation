import { matmul, softmax, type Matrix } from './attention';

export const TOKENS = 4;
export const D_MODEL = 2;
export const D_FF = 3;
export const EXPERTS = 4;
/** Communication ledger assumes FP32 payloads, while browser arithmetic is float64. */
export const ELEMENT_BYTES = 4;

export interface MoEConfig {
  topK: 1 | 2;
  topology: 'single' | 'split';
}
export interface MoEInput {
  preset: 'balanced' | 'concentrated';
  token: number;
  delta: number;
  expertScale: number;
}
export interface MoEOptions {
  seed: number;
  config: MoEConfig;
  input: MoEInput;
}
export interface MoEWeights {
  router: Matrix;
  w1: Matrix[];
  w2: Matrix[];
}
export interface DispatchPacket {
  token: number;
  expert: number;
  homeDevice: number;
  expertDevice: number;
  remote: boolean;
  gate: number;
  input: number[];
  preactivation: number[];
  hidden: number[];
  output: number[];
  contribution: number[];
}

/** Fixed teaching weights; changing the seed rebuilds parameters, never trains them. */
export function makeMoEWeights(seed: number): MoEWeights {
  let current = seed >>> 0;
  const random = () => {
    current = (Math.imul(current, 1664525) + 1013904223) >>> 0;
    return current / 4294967296;
  };
  const router = [
    [1, 0, -1, 0],
    [0, 1, 0, -1],
  ].map((row) => row.map((v) => v + (Math.floor(random() * 5) - 2) / 32));
  const matrix = (rows: number, columns: number): Matrix =>
    Array.from({ length: rows }, () =>
      Array.from({ length: columns }, () => (Math.floor(random() * 9) - 4) / 4),
    );
  return {
    router,
    w1: Array.from({ length: EXPERTS }, () => matrix(D_MODEL, D_FF)),
    w2: Array.from({ length: EXPERTS }, () => matrix(D_FF, D_MODEL)),
  };
}

export function makeMoEInput(input: MoEInput): Matrix {
  const x =
    input.preset === 'balanced'
      ? [
          [1, 0.25],
          [0.25, 1],
          [-1, -0.25],
          [-0.25, -1],
        ]
      : [
          [1, 0.25],
          [1.25, 0.375],
          [1.5, 0.5],
          [1.75, 0.625],
        ];
  x[input.token][0] += input.delta;
  return x;
}

/** Token-choice Top-K, then renormalize over the selected set. Not a universal MoE rule. */
export function selectExperts(logits: number[], topK: 1 | 2) {
  if (logits.length !== EXPERTS || logits.some((v) => !Number.isFinite(v)))
    throw new Error('Router 需要 4 个有限分数。');
  if (topK !== 1 && topK !== 2) throw new Error('本实验的 Top-K 只能取 1 或 2。');
  const probabilities = softmax(logits);
  const selected = Array.from({ length: EXPERTS }, (_, e) => e)
    .sort((a, b) => logits[b] - logits[a] || a - b)
    .slice(0, topK);
  // Stable even if the full probabilities underflow for the unselected experts.
  const selectedGates = softmax(selected.map((e) => logits[e]));
  const gates = Array<number>(EXPERTS).fill(0);
  selected.forEach((e, i) => (gates[e] = selectedGates[i]));
  return { probabilities, selected, gates };
}

function validateOptions({ seed, config, input }: MoEOptions) {
  if (!Number.isInteger(seed) || seed < 1 || seed > 9999)
    throw new Error('随机种子应为 1–9999 的整数。');
  if (![1, 2].includes(config.topK) || !['single', 'split'].includes(config.topology))
    throw new Error('不支持的 MoE 配置。');
  if (
    !['balanced', 'concentrated'].includes(input.preset) ||
    !Number.isInteger(input.token) ||
    input.token < 0 ||
    input.token >= TOKENS ||
    !Number.isFinite(input.delta) ||
    input.delta < -2 ||
    input.delta > 2 ||
    !Number.isFinite(input.expertScale) ||
    input.expertScale < 0 ||
    input.expertScale > 2
  )
    throw new Error('MoE 输入超出教学范围。');
}

/** Dispatch by expert, compute only selected FFNs, then scatter-add to original token rows. */
export function computeMoE(options: MoEOptions) {
  validateOptions(options);
  const { seed, config, input } = options;
  const x = makeMoEInput(input);
  const weights = makeMoEWeights(seed);
  const logits = matmul(x, weights.router);
  const routes = logits.map((row) => selectExperts(row, config.topK));
  const probabilities = routes.map((r) => r.probabilities);
  const selected = routes.map((r) => r.selected);
  const gates = routes.map((r) => r.gates);
  const packets: DispatchPacket[] = [];
  const load = Array<number>(EXPERTS).fill(0);
  const output: Matrix = Array.from({ length: TOKENS }, () => Array<number>(D_MODEL).fill(0));
  const expertOutputs: (number[] | null)[][] = Array.from({ length: TOKENS }, () =>
    Array<number[] | null>(EXPERTS).fill(null),
  );
  const contributions: (number[] | null)[][] = Array.from({ length: TOKENS }, () =>
    Array<number[] | null>(EXPERTS).fill(null),
  );

  for (let expert = 0; expert < EXPERTS; expert++) {
    for (let token = 0; token < TOKENS; token++) {
      if (!selected[token].includes(expert)) continue;
      const homeDevice = config.topology === 'single' ? 0 : token % 2;
      const expertDevice = config.topology === 'single' ? 0 : Math.floor(expert / 2);
      const preactivation = matmul([x[token]], weights.w1[expert])[0];
      const hidden = preactivation.map((v) => Math.max(0, v));
      const expertOutput = matmul([hidden], weights.w2[expert])[0].map(
        (v) => v * (expert === 0 ? input.expertScale : 1),
      );
      const gate = gates[token][expert];
      const contribution = expertOutput.map((v) => v * gate);
      packets.push({
        token,
        expert,
        homeDevice,
        expertDevice,
        remote: homeDevice !== expertDevice,
        gate,
        input: x[token].slice(),
        preactivation,
        hidden,
        output: expertOutput,
        contribution,
      });
      load[expert]++;
      expertOutputs[token][expert] = expertOutput;
      contributions[token][expert] = contribution;
      contribution.forEach((v, d) => (output[token][d] += v));
    }
  }
  const remoteAssignments = packets.filter((p) => p.remote).length;
  const dispatchBytes = remoteAssignments * D_MODEL * ELEMENT_BYTES;
  const returnBytes = remoteAssignments * D_MODEL * ELEMENT_BYTES;
  const routerParams = D_MODEL * EXPERTS;
  const expertParams = 2 * D_MODEL * D_FF;
  return {
    x,
    weights,
    logits,
    probabilities,
    selected,
    gates,
    expertOutputs,
    contributions,
    output,
    packets,
    load,
    traffic: {
      remoteAssignments,
      dispatchBytes,
      returnBytes,
      totalBytes: dispatchBytes + returnBytes,
    },
    counts: {
      routerParams,
      expertParams,
      totalParams: routerParams + EXPERTS * expertParams,
      activeParamsPerToken: routerParams + config.topK * expertParams,
      expertMACsPerToken: config.topK * expertParams,
      routerMACsPerToken: routerParams,
      totalExpertMACs: TOKENS * config.topK * expertParams,
      totalRouterMACs: TOKENS * routerParams,
    },
  };
}

export type MoEResult = ReturnType<typeof computeMoE>;

/** Independent token-first baseline; no dispatch list or production matmul/gating helpers. */
export function directReference(options: MoEOptions): Matrix {
  validateOptions(options);
  const { config, input, seed } = options;
  const x = makeMoEInput(input);
  const weights = makeMoEWeights(seed);
  return x.map((row) => {
    const logits = Array.from({ length: EXPERTS }, (_, e) =>
      row.reduce((sum, v, d) => sum + v * weights.router[d][e], 0),
    );
    const chosen = [0, 1, 2, 3]
      .sort((a, b) => logits[b] - logits[a] || a - b)
      .slice(0, config.topK);
    const max = Math.max(...chosen.map((e) => logits[e]));
    const exps = chosen.map((e) => Math.exp(logits[e] - max));
    const denominator = exps.reduce((a, b) => a + b, 0);
    const y = Array<number>(D_MODEL).fill(0);
    chosen.forEach((e, index) => {
      const hidden = Array.from({ length: D_FF }, (_, h) =>
        Math.max(
          0,
          row.reduce((sum, v, d) => sum + v * weights.w1[e][d][h], 0),
        ),
      );
      for (let d = 0; d < D_MODEL; d++) {
        const raw = hidden.reduce((sum, v, h) => sum + v * weights.w2[e][h][d], 0);
        y[d] += (exps[index] / denominator) * raw * (e === 0 ? input.expertScale : 1);
      }
    });
    return y;
  });
}
