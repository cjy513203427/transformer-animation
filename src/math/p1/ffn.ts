import { matmul } from '../attention';
import {
  check,
  flat,
  frame,
  pairRows,
  tensor,
  uniforms,
  type Computation,
  type Parameters,
} from './shared';
import { T } from '../../i18n';
export const FFN_W1 = [
    [1, -1, 0.5],
    [0.5, 1, -1],
  ],
  FFN_W2 = [
    [1, 0],
    [0, 1],
    [1, -1],
  ];
export function ffn(x: number[], p: Parameters, seed: number): Computation {
  const X = pairRows(x),
    h = matmul(X, FFN_W1),
    relu = h.map((r) => r.map((v) => Math.max(0, v))),
    nonlinear = p.nonlinear ? relu : h,
    f = matmul(nonlinear, FFN_W2),
    residual = f.map((r, i) => r.map((v, j) => X[i][j] + v));
  const u = pairRows(uniforms(seed, x.length)),
    mask = u.map((r) => r.map((v) => (v >= p.dropout ? 1 : 0))),
    dropped = f.map((r, i) =>
      r.map((v, j) => (p.training ? (v * mask[i][j]) / (1 - p.dropout) : v)),
    ),
    y = dropped.map((r, i) => r.map((v, j) => X[i][j] + v));
  const collapsed = matmul(X, matmul(FFN_W1, FFN_W2));
  // ReLU's input derivative is 0 at the non-differentiable origin by this explicit convention.
  const derivative = X.map((_, i) =>
    [0, 1].map((a) =>
      [0, 1].map(
        (b) =>
          (a === b ? 1 : 0) +
          FFN_W1[a].reduce(
            (s, w, k) => s + w * (p.nonlinear ? (h[i][k] > 0 ? 1 : 0) : 1) * FFN_W2[k][b],
            0,
          ),
      ),
    ),
  );
  return {
    frames: [
      frame(T('逐 token 升维：二维输入变成三维特征'), [
        tensor('X [4, 2]', X),
        tensor('W₁ [2, 3]', FFN_W1),
        tensor('H = XW₁ [4, 3]', h),
      ]),
      frame(
        T('非线性决定哪些坐标继续传递'),
        [tensor('H', h), tensor(p.nonlinear ? 'ReLU(H)' : T('旁路：H'), nonlinear)],
        [[T('ReLU 开关'), p.nonlinear ? T('启用') : T('关闭')]],
        T('关闭 ReLU 后，两次线性变换可合并；启用后一般不成立。'),
      ),
      frame(T('独立的降维投影回到残差维度'), [
        tensor(T('激活 [4, 3]'), nonlinear),
        tensor('W₂ [3, 2]', FFN_W2),
        tensor('F(X) [4, 2]', f),
      ]),
      frame(
        T('残差把原输入逐项加回来'),
        [
          tensor(T('原 X'), X),
          tensor(T('分支 F(X)'), f),
          tensor('X + F(X)', residual),
          tensor(T('t0 的 Jacobian I + J_F'), derivative[0]),
        ],
        [],
        T(
          '梯度包含恒等支路 I；这不保证总梯度永不消失，其他支路可能抵消。此 Jacobian 针对未加 Dropout 的残差。',
        ),
      ),
      frame(
        T('训练随机屏蔽并补偿；推理直接通过'),
        [
          tensor(T('固定种子的 Mask'), mask),
          tensor('Dropout(F)', dropped),
          tensor(T('最终 X + Dropout(F)'), y),
        ],
        [
          [T('丢弃率 p'), p.dropout],
          [T('模式'), p.training ? T('训练') : T('推理')],
        ],
        T('采用 inverted Dropout：训练保留项除以 1−p，残差支路不丢弃。固定种子只为复现。'),
      ),
      frame(
        T('两层线性可以合并，非线性通常不能'),
        [tensor('X(W₁W₂)', collapsed), tensor(T('当前 F(X)'), f), tensor(T('当前残差输出'), y)],
        [['ReLU', p.nonlinear ? T('启用') : T('旁路')]],
        T('切换 ReLU 与运行模式；不是所有输入都一定能暴露非线性差异。'),
      ),
    ].map((item, index) =>
      index >= 2
        ? { ...item, points: { before: X, after: index === 2 ? f : index === 3 ? residual : y } }
        : item,
    ),
    checks: [
      check(T('线性支路的两种计算次序等价'), flat(matmul(h, FFN_W2)), flat(collapsed)),
      ...(!p.training ? [check(T('推理 Dropout 为恒等'), flat(dropped), flat(f))] : []),
    ],
    signature: flat(y),
  };
}
