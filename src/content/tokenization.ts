import { T } from '../i18n';

export const tokenizationSteps = [
  {
    name: T('字节'),
    description: T(
      '先找出这一段文本的 UTF-8 字节。非 ASCII 字符可能由多个字节组成，表中每组保持原有顺序。',
    ),
    prompt: T('为什么一个汉字或表情符号可能占多个字节？'),
    formula: String.raw`\mathrm{text}\xrightarrow{\mathrm{UTF\text{-}8}}[b_0,\ldots,b_{B-1}],\quad 0\leq b_i<256`,
    symbols: [[String.raw`B,\ b_i`, T('字节总数与第 i 个字节')]],
  },
  {
    name: T('学习合并'),
    description: T(
      '在训练语料的每个片段内统计相邻对。取频次最高的一对，从左到右做不重叠替换，再重新计数。同频时按两个 ID 依次比较，较小者优先。',
    ),
    prompt: T('在 aaa 中，为什么 aa 能被数到两次，却只能替换一次？'),
    formula: String.raw`(a,b)=\operatorname*{argmax}_{(u,v)}\operatorname{count}(u,v)`,
    symbols: [
      [String.raw`(a,b)`, T('被选中的相邻 token ID 对')],
      [String.raw`\operatorname{count}(u,v)`, T('相邻对在当前语料中的出现次数')],
    ],
  },
  {
    name: T('应用规则'),
    description: T(
      '输入沿用语料中学到的规则，按学习顺序依次应用。切换当前轮可以看到哪些相邻字节或子词合在一起。输入文本不会参与重新计数。',
    ),
    prompt: T('有一条训练规则不出现在这段文本里时，这一轮会发生什么？'),
    formula: String.raw`(a,b)\longmapsto k,\qquad \mathrm{bytes}(k)=\mathrm{bytes}(a)\,\Vert\,\mathrm{bytes}(b)`,
    symbols: [
      [String.raw`(a,b)`, T('被选中的相邻 token ID 对')],
      [String.raw`k`, T('合并后的新 token ID')],
      [String.raw`\Vert`, T('依次连接字节序列')],
    ],
  },
  {
    name: T('嵌入查表'),
    description: T(
      '用每个 token ID 单独取出嵌入表的一行。位置是序列的行号，ID 是表的行号。重复 ID 在固定表中会取出同一个向量。',
    ),
    prompt: T('两个位置的 ID 相同，加入位置编码之前，它们的 Embedding 行是否相同？'),
    formula: String.raw`X_{t,:}=E[\mathrm{id}_t,:],\qquad E\in\mathbb{R}^{V\times D},\ X\in\mathbb{R}^{T\times D}`,
    symbols: [
      [String.raw`V,D,T`, T('词表大小、嵌入维度、序列长度')],
      [String.raw`t,\ \mathrm{id}_t`, T('序列位置与该位置的 token ID')],
      [String.raw`E,X`, T('嵌入表和查表结果')],
    ],
  },
  {
    name: T('预测与核对'),
    description: T(
      '保持同一段文本，对照当前设置和只使用基础字节的设置。先预测 token 数是否改变，再重新编码并解码两次。',
    ),
    prompt: T('为什么 token 个数可以变，而解码后仍能逐字还原输入？'),
    formula: String.raw`\operatorname{decode}(\operatorname{encode}(s))=s`,
    symbols: [[String.raw`s`, T('输入文本')]],
  },
] as const;
