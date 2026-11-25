import type { WhyContent } from '../components/WhyIntro';
import { chapterUrl } from './book';
import type { FormulaSymbol } from './notation/shared';
import { T } from '../i18n';

const CH08 = chapterUrl('第08章-线性变换的几何意义-矩阵乘法的本质/');
const CH09 = chapterUrl('第09章-Attention的几何逻辑-为什么是点积/');

export const linearVerifiedDate = '2026-11-25';

export const linearWhy: WhyContent = {
  question: T('为什么大模型里几乎每一步都在做矩阵乘法？'),
  points: [
    T(
      'Attention 里的 $Q=XW_Q$、$K=XW_K$、$V=XW_V$ 和 FFN 里的升维降维，全都是线性变换。背“行乘列再相加”很容易忘，但把它看成对空间的旋转、拉伸、剪切或压扁，就能一眼看懂矩阵在做什么。',
    ),
    T(
      '全站采用行向量约定 $y = xW$：$W$ 的第 0 行决定原基向量 $e_0=[1,0]$ 落到哪里，第 1 行决定 $e_1=[0,1]$ 落到哪里。拖动或调节矩阵里的四个数，整个坐标网格会跟着一起动。',
    ),
  ],
  analogy: {
    title: T('把网格当成弹性画布（第 8 章）'),
    text: T(
      '线性变换像握住画布的两个基准方向去拉动、旋转或错切它：原点固定不动，原本平行的网格线变换后仍然平行且等距。只要知道两个基向量的新落点，平面上每个点的新坐标就唯一确定了。',
    ),
  },
  book: [T('第 8 章 · 线性变换的几何意义'), CH08],
};

export const linearSteps: {
  name: string;
  title: string;
  explanation: string;
  takeaway: string;
  prompt: string;
  formula: string;
  symbols: FormulaSymbol[];
  book: [string, string];
}[] = [
  {
    name: T('基向量与网格'),
    title: T('矩阵的两行，就是两个基向量的新落点'),
    explanation: T(
      '在行向量约定下，$x = [x_0, x_1] = x_0 e_0 + x_1 e_1$。乘以 $2\\times 2$ 矩阵 $W$ 后，输出 $y = x_0 e_0\' + x_1 e_1\'$，其中 $e_0\'$ 就是 $W$ 的第 0 行，$e_1\'$ 是第 1 行。拖动图中蓝色或橙色端点，或者切换预设（旋转、剪切、缩放、镜像），观察整个网格如何被这两行唯一决定。',
    ),
    takeaway: T('行向量乘矩阵 $xW$，等于用 $x$ 的两个分量去线性组合 $W$ 的两行。'),
    prompt: T('如果不看公式，只看变换后的两个基向量，怎样找出任意输入 $x$ 的输出 $y$？'),
    formula: String.raw`y = xW = x_0\,W_{[0,:]} + x_1\,W_{[1,:]} = x_0\,e_0' + x_1\,e_1'`,
    symbols: [
      ['x,y', T('x 是 [1, 2] 输入行向量，y 是变换后的 [1, 2] 输出行向量。')],
      ['W', T('2×2 线性变换矩阵；第 0 行是 e₀\'，第 1 行是 e₁\'。')],
      ["e_0',e_1'", T('标准基向量 [1, 0] 与 [0, 1] 经 W 变换后的新位置。')],
    ],
    book: [T('第 8.2 节 · 线性变换的几何直觉'), CH08],
  },
  {
    name: T('面积与降维'),
    title: T('行列式：空间被放大、翻转，还是压扁了？'),
    explanation: T(
      '原单位正方形面积为 1，经 $W$ 变换后成为以 $e_0\'$、$e_1\'$ 为邻边的平行四边形。行列式 $\\det(W)=w_{00}w_{11}-w_{01}w_{10}$ 的绝对值是面积缩放倍数：正值保持定向，负值表示镜像翻转；若 $\\det(W)=0$（试试“降维压扁”预设），两行共线，整个二维平面被压成一条直线，不同输入会撞到同一点。',
    ),
    takeaway: T('行列式衡量面积缩放与定向；行列式为 0 意味着降维，信息无法唯一还原。'),
    prompt: T('什么情况下两个不同的输入向量 $x$ 会被同一个矩阵 $W$ 映射到同一个输出？'),
    formula: String.raw`\det(W)=w_{00}w_{11}-w_{01}w_{10},\qquad \operatorname{Area}=|\det(W)|`,
    symbols: [
      ['\\det(W)', T('2×2 矩阵的行列式：绝对值是面积缩放倍率，符号表示是否翻转定向。')],
      ['w_{ij}', T('矩阵 W 第 i 行第 j 列的元素（行索引与列索引均从 0 开始）。')],
    ],
    book: [T('第 8.3 节 · 矩阵的几何效果'), CH08],
  },
  {
    name: T('点积几何'),
    title: T('点积 = 投影 × 长度，不等于余弦相似度'),
    explanation: T(
      '把变换后的输出记作 $q = xW$，再取一个参考向量 $k$。点积 $q \\cdot k = q_0k_0 + q_1k_1$ 在几何上等于 $\\|q\\|\\|k\\|\\cos\\theta$：先将 $q$ 投影到 $k$ 的方向，再乘以 $k$ 的长度。正值表示夹角小于 90°，负值表示方向相反，0 表示垂直。下方可以验证：只拉长 $k$ 不改方向，点积会变而余弦不变。',
    ),
    takeaway: T('Attention 打分用点积：它同时受夹角和向量长度影响，不等同于余弦相似度。'),
    prompt: T('保持 $q$ 和 $k$ 的夹角不变，只把 $k$ 拉长到 2 倍，点积和余弦相似度各自怎样变化？'),
    formula: String.raw`q\cdot k = q_0k_0 + q_1k_1 = \|q\|\,\|k\|\cos\theta`,
    symbols: [
      ['q,k', T('两个二维行向量；这里取 q = xW，k 为可编辑的参考向量。')],
      ['\\|q\\|,\\|k\\|', T('向量的欧氏长度（L2 范数）。')],
      ['\\theta', T('两向量之间的夹角；任一向量为零时夹角与余弦未定义。')],
    ],
    book: [T('第 9 章 · Attention 的几何逻辑：为什么是点积'), CH09],
  },
  {
    name: T('连续变换'),
    title: T('连续做两次线性变换，等于一次矩阵乘法'),
    explanation: T(
      '先用 $W_1$ 把 $x$ 映到 $xW_1$，再用固定的 $W_2$ 映到 $(xW_1)W_2$，结果与先算出复合矩阵 $W_{12}=W_1W_2$ 再算 $xW_{12}$ 完全一致。但把顺序调换为 $W_2W_1$，结果通常不同——矩阵乘法满足结合律，却一般不满足交换律。这也是为什么多层纯线性网络可以合并成单层，必须夹入非线性激活。',
    ),
    takeaway: T('矩阵乘法代表连续线性变换的复合：满足结合律，一般不满足交换律。'),
    prompt: T('为什么两次线性投影之间如果不加 ReLU 或 GELU，堆再多层也等价于一层？'),
    formula: String.raw`(xW_1)W_2 = x(W_1W_2),\qquad W_1W_2 \neq W_2W_1`,
    symbols: [
      ['W_1,W_2', T('当前可编辑的 2×2 矩阵 W₁ 与本步固定的第二个矩阵 W₂。')],
      ['W_1W_2', T('先施加 W₁、再施加 W₂ 对应的单个 2×2 复合矩阵。')],
    ],
    book: [T('第 8.4 节 · 矩阵乘法的本质'), CH08],
  },
  {
    name: T('连回 Transformer'),
    title: T('从二维画布回到 Q、K、V 与 FFN'),
    explanation: T(
      '在 Transformer 里，同一个隐藏状态 $X$ 分别乘以 $W_Q, W_K, W_V$，相当于把同一组点同时投到三个不同的线性视角里：一个负责提问、一个负责被匹配、一个负责提供内容。而在 FFN 里，两次线性变换之间必须夹入 ReLU / GELU，才能折弯网格、跳出单个线性矩阵的表达上限。',
    ),
    takeaway: T('线性投影负责切换视角与维度，非线性负责打破“多层合并成一层”的限制。'),
    prompt: T('为什么 Transformer 要用三个矩阵 $W_Q, W_K, W_V$ 而不是直接拿原输入 $X$ 做点积？'),
    formula: String.raw`Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad \operatorname{FFN}(x)=\sigma(xW_1)W_2`,
    symbols: [
      ['W_Q,W_K,W_V', T('把同一隐藏状态投影到 Query、Key、Value 空间的三组线性参数。')],
      ['\\sigma', T('逐元素非线性激活（如 ReLU、GELU），阻止 W₁ 与 W₂ 合并为单个矩阵。')],
    ],
    book: [T('第 8.5 节 · 线性变换在 Transformer 中的位置'), CH08],
  },
];

export const linearSources = [
  { title: T('原书第 8 章 · 线性变换的几何意义'), url: CH08 },
  { title: T('原书第 9 章 · Attention 的几何逻辑'), url: CH09 },
  {
    title: 'Attention Is All You Need · §3.2.1 Scaled Dot-Product Attention',
    url: 'https://arxiv.org/abs/1706.03762',
  },
];

export const linearLimits = [
  T('全页使用行向量约定 y = xW；若改用列向量约定 y = Wx，矩阵会转置，基向量对应的是列而非行。'),
  T('二维平面仅用于建立线性映射与点积的几何直觉；真实模型的隐藏维度通常在数百到数千，不能把二维坐标当作语义空间。'),
  T('拖动端点与滑杆限制在 −2 到 2、步长 0.25，便于手算核对；内部运算保持 float64 精度。'),
  T('任意向量长度为 0 时，夹角与余弦相似度未定义，界面明确显示“未定义”而不输出 NaN。'),
];
