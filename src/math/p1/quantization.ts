import {
  check,
  dot,
  frame,
  mean,
  tensor,
  type Computation,
  type NumberLine,
  type Parameters,
} from './shared';
import { T } from '../../i18n';

/** Fixed calibration activation u used to evaluate linear-layer output error u·x̂ − u·x. */
export const QUANT_ACTIVATION = [1, -1, 0.5, 0, 1, -0.5, 2, -1];

/**
 * Small deterministic calibration batch X_cal [8, 8] (row 0 = QUANT_ACTIVATION) with correlated
 * channels and full rank so H = 2 X^T X + λI is well-conditioned for teaching GPTQ compensation.
 */
export const GPTQ_CALIBRATION: number[][] = [
  QUANT_ACTIVATION,
  [1, -0.75, 0.5, 0.25, 0.75, -0.5, 1.75, -0.75],
  [0.75, -0.5, 0.25, -0.25, 1, -0.25, 1.5, -1],
  [0.5, -1, 0.75, 0.5, 0.5, -0.75, 1.25, -0.5],
  [-0.5, 0.5, 0.5, 0.25, -0.5, 0.5, -1, 0.5],
  [0.25, -0.25, -0.5, 0.75, 0.25, 0.25, 0.75, -0.25],
  [-0.25, 0.5, 0.25, -0.5, 0.5, -0.25, 0.5, 0.25],
  [0.5, 0.25, -0.25, 0.25, -0.25, 0.5, 0.25, -0.5],
];

const GPTQ_DAMP = 0.05;

export function quantize(
  x: number[],
  bits: number,
  groupSize: number,
  cap: number,
  asymmetric = false,
) {
  const qmin = asymmetric ? 0 : -(2 ** (bits - 1) - 1);
  const qmax = asymmetric ? 2 ** bits - 1 : 2 ** (bits - 1) - 1;
  const q: number[] = [];
  const restored: number[] = [];
  const scales: number[] = [];
  const zeroPoints: number[] = [];
  const clipped: number[] = [];
  const lowBounds: number[] = [];
  const highBounds: number[] = [];

  for (let start = 0; start < x.length; start += groupSize) {
    const group = x.slice(start, start + groupSize);
    if (!asymmetric) {
      const extent = Math.min(cap, Math.max(...group.map(Math.abs)));
      const scale = extent === 0 ? 1 : extent / qmax;
      scales.push(scale);
      zeroPoints.push(0);
      lowBounds.push(-extent);
      highBounds.push(extent);
      for (const value of group) {
        const limited = Math.max(-extent, Math.min(extent, value));
        const signed = Math.sign(limited) * Math.floor(Math.abs(limited / scale) + 0.5);
        q.push(signed);
        restored.push(signed * scale);
        clipped.push(limited);
      }
    } else {
      const preClipped = group.map((v) => Math.max(-cap, Math.min(cap, v)));
      // Include 0 in [lo, hi] so 0 maps to an exact integer zero-point z ∈ [0, 2^b - 1].
      const lo = Math.min(0, ...preClipped);
      const hi = Math.max(0, ...preClipped);
      const span = hi - lo;
      const scale = span === 0 ? 1 : span / qmax;
      const zp = span === 0 ? 0 : Math.max(0, Math.min(qmax, Math.round(-lo / scale)));
      scales.push(scale);
      zeroPoints.push(zp);
      lowBounds.push(lo);
      highBounds.push(hi);
      for (const limited of preClipped) {
        const raw = Math.round(limited / scale) + zp;
        const code = Math.max(qmin, Math.min(qmax, raw));
        q.push(code);
        restored.push((code - zp) * scale);
        clipped.push(limited);
      }
    }
  }

  const difference = x.map((v, i) => restored[i] - v);
  const mse = mean(difference.map((v) => v * v));
  const payloadBytes = Math.ceil((x.length * bits) / 8);
  const scaleBytes = scales.length * 4;
  const zeroPointBytes = asymmetric ? scales.length : 0;
  return {
    asymmetric,
    qmin,
    qmax,
    q,
    restored,
    scales,
    zeroPoints,
    clipped,
    lowBounds,
    highBounds,
    difference,
    mse,
    payloadBytes,
    scaleBytes,
    zeroPointBytes,
    bytes: payloadBytes + scaleBytes + zeroPointBytes,
  };
}

/** AWQ (Lin et al., 2023): scale salient weight channels up by s_i = (|u_i|/mean|u|)^α before RTN. */
export function quantizeAWQ(
  x: number[],
  activation: number[],
  bits: number,
  groupSize: number,
  cap: number,
  asymmetric = false,
  alpha = 0.5,
) {
  const meanAbs = mean(activation.map(Math.abs));
  const channelScales = activation.map((u) => {
    if (meanAbs === 0) return 1;
    const ratio = Math.max(Math.abs(u), 0.15) / meanAbs;
    return Math.max(0.25, Math.min(4, Math.pow(ratio, alpha)));
  });
  const scaledWeights = x.map((w, i) => w * channelScales[i]);
  const scaledActivation = activation.map((u, i) => u / channelScales[i]);
  const maxScale = Math.max(1, ...channelScales);
  const inner = quantize(scaledWeights, bits, groupSize, cap * maxScale, asymmetric);
  const restored = inner.restored.map((w, i) => w / channelScales[i]);
  const difference = x.map((v, i) => restored[i] - v);
  return {
    ...inner,
    channelScales,
    scaledWeights,
    scaledActivation,
    scaledRestored: inner.restored,
    restored,
    difference,
    mse: mean(difference.map((v) => v * v)),
    scaledOutput: dot(scaledActivation, inner.restored),
    directOutput: dot(activation, restored),
  };
}

/** Exact Gauss-Jordan inverse for small SPD Hessian H (n <= 8). */
export function invertMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  const aug = matrix.map((row, i) => [
    ...row,
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  ]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r;
    }
    if (pivot !== col) [aug[col], aug[pivot]] = [aug[pivot], aug[col]];
    const diag = aug[col][col];
    if (Math.abs(diag) < 1e-14) throw new Error('Singular matrix in GPTQ Hessian inversion.');
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= diag;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r][col];
      for (let j = 0; j < 2 * n; j++) aug[r][j] -= factor * aug[col][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

export function buildHessian(calibration: number[][], damp = GPTQ_DAMP): number[][] {
  const n = calibration[0].length;
  const h = Array.from({ length: n }, () => Array<number>(n).fill(0));
  for (const row of calibration) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) h[i][j] += 2 * row[i] * row[j];
    }
  }
  for (let i = 0; i < n; i++) h[i][i] += damp;
  return h;
}

export function quadraticForm(diff: number[], h: number[][]): number {
  let total = 0;
  for (let i = 0; i < diff.length; i++) {
    for (let j = 0; j < diff.length; j++) total += diff[i] * h[i][j] * diff[j];
  }
  return total;
}

/** GPTQ (Frantar et al., 2022): sequential column quantization + Schur-complement OBS compensation. */
export function quantizeGPTQ(
  x: number[],
  bits: number,
  groupSize: number,
  cap: number,
  asymmetric = false,
  calibration: number[][] = GPTQ_CALIBRATION,
) {
  const base = quantize(x, bits, groupSize, cap, asymmetric);
  const h = buildHessian(calibration);
  const hInv = invertMatrix(h);
  // Working inverse updated by Schur complement after each column is fixed (exact OBS / GPTQ).
  const workInv = hInv.map((row) => row.slice());
  const w = x.slice();
  const q: number[] = [];
  const restored: number[] = [];
  const compensationApplied = Array<number>(x.length).fill(0);

  for (let j = 0; j < x.length; j++) {
    const g = Math.floor(j / groupSize);
    const scale = base.scales[g];
    const zp = base.zeroPoints[g];
    const lo = base.lowBounds[g];
    const hi = base.highBounds[g];
    const limited = Math.max(lo, Math.min(hi, w[j]));
    let code: number;
    if (!asymmetric) {
      code = Math.sign(limited) * Math.floor(Math.abs(limited / scale) + 0.5);
      code = Math.max(base.qmin, Math.min(base.qmax, code));
    } else {
      code = Math.max(base.qmin, Math.min(base.qmax, Math.round(limited / scale) + zp));
    }
    const wHat = asymmetric ? (code - zp) * scale : code * scale;
    q.push(code);
    restored.push(wHat);
    const err = w[j] - wHat;
    const invDiag = workInv[j][j];
    for (let k = j + 1; k < x.length; k++) {
      const delta = -err * (workInv[j][k] / invDiag);
      w[k] += delta;
      compensationApplied[k] += delta;
    }
    // Eliminate column j from the remaining inverse Hessian via Schur complement.
    for (let r = j + 1; r < x.length; r++) {
      const colFactor = workInv[r][j] / invDiag;
      for (let c = j + 1; c < x.length; c++) {
        workInv[r][c] -= colFactor * workInv[j][c];
      }
    }
  }

  const difference = x.map((v, i) => restored[i] - v);
  const identityError = Math.max(
    0,
    ...h.flatMap((row, i) =>
      h[0].map((_, j) => {
        const prod = row.reduce((s, v, k) => s + v * hInv[k][j], 0);
        return Math.abs(prod - (i === j ? 1 : 0));
      }),
    ),
  );

  return {
    ...base,
    q,
    restored,
    difference,
    compensationApplied,
    mse: mean(difference.map((v) => v * v)),
    hessianLoss: quadraticForm(difference, h),
    rtnHessianLoss: quadraticForm(base.difference, h),
    identityError,
  };
}

export function quantization(x: number[], p: Parameters): Computation {
  const asymmetric = p.scheme === 1;
  const method = p.method ?? 0; // 0: RTN, 1: AWQ, 2: GPTQ
  const q = quantize(x, p.bits, p.group, p.cap, asymmetric);
  const global = quantize(x, p.bits, x.length, p.cap, asymmetric);
  const awq = quantizeAWQ(x, QUANT_ACTIVATION, p.bits, p.group, p.cap, asymmetric);
  const gptq = quantizeGPTQ(x, p.bits, p.group, p.cap, asymmetric);

  const active = method === 1 ? awq : method === 2 ? gptq : q;
  const methodLabel =
    method === 1 ? T('AWQ 激活缩放') : method === 2 ? T('GPTQ 逐列补偿') : T('均匀 RTN');

  const fullOutput = dot(QUANT_ACTIVATION, x);
  const rtnOutput = dot(QUANT_ACTIVATION, q.restored);
  const activeOutput = dot(QUANT_ACTIVATION, active.restored);

  // One group has one grid; several groups have different grids, so none is drawn for them.
  const single = q.scales.length === 1;
  const lo0 = q.lowBounds[0];
  const hi0 = q.highBounds[0];
  const ticks = !single
    ? undefined
    : hi0 - lo0 === 0
      ? [0]
      : asymmetric
        ? Array.from({ length: q.qmax + 1 }, (_, k) => (k - q.zeroPoints[0]) * q.scales[0])
        : Array.from({ length: 2 * q.qmax + 1 }, (_, k) => (k - q.qmax) * q.scales[0]);
  const bounds = (row: number) =>
    single && hi0 - lo0 > 0
      ? [
          { row, value: lo0, label: T('−边界') },
          { row, value: hi0, label: T('+边界') },
        ]
      : [];
  const names = x.map((_, i) => String(i));
  const gridNote = single
    ? T('短竖线是能表示的全部格点')
    : T('{#0} 组各有自己的格点间距，图中不画格点', q.scales.length);

  const lines: (NumberLine | undefined)[] = [
    {
      rows: [{ label: T('原 x'), values: x }],
      names,
      shared: true,
      ticks,
      markers: bounds(0),
      caption: T('{#0}；每个数只能落到其中一个格点上', gridNote),
    },
    {
      rows: [
        { label: T('裁剪后 x'), values: q.clipped },
        { label: T('还原 x̂'), values: q.restored },
      ],
      names,
      shared: true,
      ticks,
      caption: T('{#0}；连线从原位置指向最近的格点', gridNote),
    },
    {
      rows: [
        { label: T('原 x'), values: x },
        { label: T('还原 x̂'), values: q.restored },
      ],
      names,
      shared: true,
      ticks,
      markers: bounds(0),
      caption: T('{#0}；虚线是校准边界，超出的数被压到边界上', gridNote),
    },
    {
      rows: [
        { label: T('原 x'), values: x },
        { label: T('全向量一组'), values: global.restored },
        { label: T('当前分组'), values: q.restored },
      ],
      names,
      shared: true,
      caption: T('同一把尺：比较一组共用尺度与分组各用尺度，小数值离原位置有多远'),
    },
    {
      rows: [
        { label: T('原 x'), values: x },
        { label: T('均匀 RTN'), values: q.restored },
        ...(method === 0 ? [] : [{ label: methodLabel, values: active.restored }]),
      ],
      names,
      shared: true,
      ticks: method === 0 ? ticks : undefined,
      caption:
        method === 0
          ? T('{#0}；连线越斜，这一项的误差越大', gridNote)
          : T('对照均匀 RTN 与当前方法还原后的有效权重；观察高激活通道 6 的落点'),
    },
  ];

  const zeroTest = quantize([0, 0], p.bits, 2, p.cap, asymmetric);

  return {
    frames: [
      frame(
        asymmetric
          ? T('非对称整数格点：独立尺度与零点 z')
          : T('对称整数格点：共享尺度与零点 0'),
        [
          tensor(T('原 x'), x),
          tensor(T('每组 scale'), q.scales),
          ...(asymmetric ? [tensor(T('每组零点 z'), q.zeroPoints)] : []),
        ],
        [
          [T('整数范围下界'), q.qmin],
          [T('整数范围 ±qmax'), q.qmax],
          [T('第 0 组零点 z'), q.zeroPoints[0]],
          [T('位数'), p.bits],
        ],
        asymmetric
          ? T('非对称量化使用 [0, 2^b−1]，把零点 z 映射到实数 0，偏态分布不再浪费一半编码。')
          : T('采用 [−qmax, qmax]，有一个补码编码未使用；不是所有量化格式的通用规则。'),
      ),
      frame(
        T('先裁剪，再按尺度舍入到整数'),
        [
          tensor(T('裁剪后的 x'), q.clipped),
          tensor(T('整数 q'), q.q),
          tensor(
            asymmetric ? T('还原 x̂ = (q − z) × scale') : T('还原 x̂ = q × scale'),
            q.restored,
          ),
        ],
        [
          [T('第 0 组 scale'), q.scales[0]],
          [T('第 0 组零点 z'), q.zeroPoints[0]],
        ],
        T('本例对称半整数采用远离零舍入，非对称采用标准四舍五入；不同实现可能差一格。'),
      ),
      frame(
        T('校准上限变小，格点更密，也更容易截断异常值'),
        [tensor(T('原 x'), x), tensor(T('裁剪后的 x'), q.clipped), tensor(T('还原 x̂'), q.restored)],
        [[T('校准上限 cap'), p.cap]],
        T('cap 限制每组最大绝对值；这是可控的教学校准规则。'),
      ),
      frame(
        T('分组让不同区间拥有不同尺度'),
        [
          tensor(T('全向量共用尺度的还原'), global.restored),
          tensor(T('当前分组还原'), q.restored),
          tensor(T('当前分组 scale'), q.scales),
        ],
        [
          [T('组大小'), p.group],
          [T('组数'), q.scales.length],
        ],
        T('组更小需要更多尺度元数据；不是无代价提高精度。'),
      ),
      frame(
        method === 1
          ? T('AWQ：放大高激活通道再量化，保护关键输出')
          : method === 2
            ? T('GPTQ：逐列量化，并用逆 Hessian 补偿后续列')
            : T('一个异常大值，会拉宽所在组的格点'),
        [
          tensor(T('原 x'), x),
          tensor(T('均匀 RTN 误差'), q.difference),
          ...(method === 1
            ? [
                tensor(T('AWQ 通道缩放 s'), awq.channelScales),
                tensor(T('AWQ 有效误差'), awq.difference),
              ]
            : method === 2
              ? [
                  tensor(T('GPTQ 累积补偿 Δw'), gptq.compensationApplied),
                  tensor(T('GPTQ 有效误差'), gptq.difference),
                ]
              : [tensor(T('还原 x̂'), q.restored), tensor(T('逐项误差 x̂ − x'), q.difference)]),
        ],
        [
          [T('均方误差 MSE'), active.mse],
          [T('RTN 输出误差'), Math.abs(rtnOutput - fullOutput)],
          [T('AWQ 输出误差'), Math.abs(awq.directOutput - fullOutput)],
          [T('GPTQ 二次损失 ΔᵀHΔ'), gptq.hessianLoss],
        ],
        method === 1
          ? T('AWQ 按激活幅值 |u| 构造缩放 s_i：权重乘 s_i 再量化、激活除 s_i，总线性输出误差明显下降。')
          : method === 2
            ? T('GPTQ 依次量化第 j 列，把误差按 −(w_j−ŵ_j)·[H⁻¹]_{j,k}/[H⁻¹]_{j,j} 补偿到尚未量化的列 k>j。')
            : T('可切换上方“量化方法”对比 AWQ 激活缩放与 GPTQ 逐列误差补偿。'),
      ),
      frame(
        T('存储账本与误差同时检查'),
        [
          tensor(T('固定激活 u'), QUANT_ACTIVATION),
          tensor(T('线性输出 [u·x, RTN, 当前方法]'), [fullOutput, rtnOutput, activeOutput]),
          tensor(T('逐项误差'), active.difference),
        ],
        [
          [T('打包整数载荷 B'), active.payloadBytes],
          [T('float32 尺度 B'), active.scaleBytes],
          [T('零点元数据 B'), active.zeroPointBytes],
          [T('合计 B'), active.bytes],
          [T('float32 基线 B'), x.length * 4],
          ['MSE', active.mse],
          [T('线性输出误差'), activeOutput - fullOutput],
          [T('RTN 二次损失 ΔᵀHΔ'), gptq.rtnHessianLoss],
          [T('GPTQ 二次损失 ΔᵀHΔ'), gptq.hessianLoss],
        ],
        T(
          '将当前八个数看作线性层一行权重，用固定激活 u 核对输出误差。账本假设整数紧密打包、每组一个 4B scale（非对称另加 1B zero-point），不含对齐和格式头；GGUF 是文件格式，非单一量化算法。',
        ),
      ),
    ].map((item, index) => (lines[index] ? { ...item, line: lines[index] } : item)),
    checks: [
      check(
        T('整数在有效编码范围内'),
        active.q.map((v) => Math.max(0, v - active.qmax, active.qmin - v)),
        active.q.map(() => 0),
      ),
      check(T('零输入可还原'), [zeroTest.restored[0]], [0]),
      check(T('AWQ 缩放等价 (u/s)·ŵ\' = u·(ŵ\'/s)'), [awq.scaledOutput], [awq.directOutput]),
      check(T('GPTQ Hessian 逆矩阵 H·H⁻¹ = I'), [gptq.identityError], [0], 1e-10),
    ],
    signature: [active.mse],
  };
}
