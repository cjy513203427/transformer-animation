import { T } from '../../i18n';
import { commonSymbols, createNotation, type FormulaSymbol } from './shared';

const terms = {
  ...commonSymbols,
  bytes: ['\\mathrm{bytes},b', T('bytes 是 KV 缓存总字节数；b 是每个缓存元素占用的字节数。')],
  counts: ['B,L,T,H_{KV}', T('依次是批大小、层数、token 数、KV 头数；缓存公式将这些数量相乘。')],
  dimensions: ['d_k,d_v', T('每个 KV 头的 Key、Value 维数；相加因为两种向量都要保存。')],
  groups: [
    'h,H,G,g(h)',
    T('h 是从 0 开始的 Query 头编号，H 是 Query 头数，G 是 KV 组数；g(h) 给出该头所属的组。'),
  ],
  groupVectors: [
    'Q_h,K_{g(h)},V_{g(h)},\\mathrm{head}_h',
    T('头 h 使用自己的 Q，以及所属组共享的 K、V；headₕ 是该头的输出。'),
  ],
  floor: ['\\lfloor x\\rfloor,d', T('⌊x⌋ 向下取整得到组号；d 是每头 Query / Key 维数。')],
  latent: [
    'h_t,c_t,W_{DKV}',
    T('hₜ 是位置 t 的隐藏行向量；乘下投影参数 W_DKV 得到联合 KV 压缩向量 cₜ（latent）。'),
  ],
  up: [
    'W_{UK}^{(i)},W_{UV}^{(i)}',
    T('第 i 个头的 Key、Value 上投影参数，把同一个 latent 展开为各头自己的内容向量。'),
  ],
  content: [
    'k^C_{t,i},v^C_{t,i}',
    T('位置 t、第 i 个头的内容 Key、Value；右上角 C 表示内容分支。'),
  ],
  branches: [
    'q^C,q^R,k^C,k^R',
    T('C 表示内容分支，R 表示带 RoPE 的位置分支；[·;·] 在此沿特征维拼接。'),
  ],
  indices: [
    't,j,i',
    T('t 是当前 Query 位置，j 是历史 Key 位置，i 是头编号；共享的位置 Key 没有头下标。'),
  ],
  ropeKey: [
    '\\operatorname{RoPE}_j,h_j,W_{KR}',
    T('隐藏向量 hⱼ 乘共享投影 W_KR，再按位置 j 做 RoPE；结果 kᴿⱼ 也要缓存。'),
  ],
  absorbed: [
    'W_{QC}^{(i)},W_O^{(i)},h_t',
    T(
      'hₜ 是当前位置的输入隐藏行向量；W_QC 是该头的内容 Query 投影，W_O 是对应的输出投影行块，可按乘法结合律提前合并。',
    ),
  ],
  weightedLatent: [
    'p_j,c_j',
    T('pⱼ 是当前头分给 Key j 的权重；cⱼ 是缓存中的 latent，先加权汇总再上投影。'),
  ],
  comparison: [
    'H,G,d_h,d_c,d_h^R',
    T(
      'H 是 Query 头数，G 是 KV 组数，dₕ 是每头内容维数，d_c 是 latent 维数，dₕᴿ 是共享位置 Key 维数。',
    ),
  ],
} satisfies Record<string, FormulaSymbol>;

/** One local glossary for each calculation step, in the same order as the lesson. */
export const kvSymbols = createNotation(terms, [
  ['bytes', 'counts', 'dimensions'],
  ['groups', 'groupVectors', 'floor', 'mask', 'softmax', 'transpose'],
  ['latent', 'up', 'content'],
  ['branches', 'indices', 'ropeKey'],
  ['indices', 'branches', 'absorbed', 'up', 'weightedLatent', 'transpose', 'sum'],
  ['comparison'],
]);
