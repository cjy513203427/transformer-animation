import type { ForwardKind } from '../math/forward';
import { chapterUrl } from './book';
import { T } from '../i18n';

export interface CodeHighlight {
  /** 1-based inclusive line numbers highlighted for the current step. */
  lines: number[];
  /** Short localized explanation tying this step's math/tensors to those code lines. */
  note: string;
}

export interface LinkedCodeSpec {
  file: string;
  chapterLabel: string;
  chapterHref: string;
  scopeNote: string;
  lines: string[];
}

const CH18 = chapterUrl('第18章-手写Model.py-模型定义/');
const CH19 = chapterUrl('第19章-手写Train.py-训练循环/');
const CH20 = chapterUrl('第20章-手写Inference.py-推理逻辑/');

/** Chapter 18: Pre-Norm GPT Decoder matching ForwardChapter's exact computation order. */
export const MODEL_PY_CODE: LinkedCodeSpec = {
  file: 'model.py',
  chapterLabel: T('第 18 章 · 手写 model.py'),
  chapterHref: CH18,
  scopeNote: T(
    '教学精简版 PyTorch 代码（省略 batch 维、偏置、Dropout 与 KV Cache），行顺序与本页前向步骤一一对应。',
  ),
  lines: [
    'class ToyGPT(nn.Module):',
    '    def forward(self, token_ids):              # [T]',
    '        h_tok = self.embed(token_ids)          # [T, D]',
    '        h = h_tok + self.pe[:len(token_ids)]   # [T, D]',
    '        for block in self.blocks:',
    '            u = block.ln1(h)                   # [T, D]  Pre-Norm',
    '            q, k, v = block.q(u), block.k(u), block.v(u)  # [T, D]',
    '            a = multi_head_attn(q, k, v, mask) @ block.wo # [T, D]',
    '            r = h + a                          # [T, D]  residual 1',
    '            z = block.ln2(r)                   # [T, D]',
    '            g = F.gelu(z @ block.w1, approximate="tanh")  # [T, d_ff]',
    '            f = g @ block.w2                   # [T, D]',
    '            h = r + f                          # [T, D]  residual 2',
    '        h_f = self.ln_f(h)                     # [T, D]',
    '        logits = h_f @ self.embed.weight.T     # [T, V]  weight tying',
    '        probs = F.softmax(logits, dim=-1)      # [T, V]',
    '        next_id = probs[-1].argmax(dim=-1)     # scalar',
    '        return logits, probs, next_id',
  ],
};

export const MODEL_PY_HIGHLIGHTS: Record<ForwardKind, CodeHighlight> = {
  tokens: {
    lines: [2],
    note: T('输入是一维整数下标 token_ids [T]，尚未进入浮点矩阵乘法。'),
  },
  embedding: {
    lines: [3],
    note: T('nn.Embedding 按整数行号查表，把 [T] 变成 [T, D] 的内容向量。'),
  },
  position: {
    lines: [4],
    note: T('截出前 T 行位置编码与内容向量逐元素相加，作为进入第 1 个 Block 的 H₀。'),
  },
  'attention-norm': {
    lines: [5, 6],
    note: T('Pre-Norm 在送入 Attention 前对每个 token 的特征轴做 LayerNorm，残差主路保持原 h。'),
  },
  qkv: {
    lines: [7],
    note: T('同一归一化张量 u 分别经过三组线性投影得到激活 Q、K、V；参数本身在前向中不更新。'),
  },
  attention: {
    lines: [8],
    note: T('切分多头后逐头计算因果 Attention，沿特征维拼接再乘输出投影 W_O。'),
  },
  'attention-add': {
    lines: [9],
    note: T('第一次残差连接把 Attention 输出加回原 Block 输入 h。'),
  },
  'ffn-norm': {
    lines: [10],
    note: T('第二次 LayerNorm 作用在残差和 r 上，为逐 token 的 FFN 做准备。'),
  },
  'ffn-expand': {
    lines: [11],
    note: T('FFN 先用 W₁ 升维到 d_ff，再逐元素施加 tanh 近似的 GELU 非线性。'),
  },
  'ffn-contract': {
    lines: [12],
    note: T('W₂ 把中间激活从 d_ff 压回隐藏维 D，以便与残差支路相加。'),
  },
  'ffn-add': {
    lines: [13],
    note: T('第二次残差相加完成本 Block，输出 h 直接进入下一层或最终归一化。'),
  },
  'final-norm': {
    lines: [14],
    note: T('全部 Block 结束后执行最终 LayerNorm，输出轴仍是特征维 D。'),
  },
  logits: {
    lines: [15],
    note: T('乘以嵌入表转置 weight.T 实现权重绑定，把 [T, D] 映射到词表分数 [T, V]。'),
  },
  probability: {
    lines: [16],
    note: T('沿最后一个维度 dim=-1（词表轴）做 Softmax，得到各位置的候选概率分布。'),
  },
  selection: {
    lines: [17, 18],
    note: T('生成只读取最后一个位置 probs[-1]，按 argmax 选出下一个 token ID。'),
  },
};

/** Chapter 19: Minimal training step matching /training-step's 6 steps. */
export const TRAIN_PY_CODE: LinkedCodeSpec = {
  file: 'train.py',
  chapterLabel: T('第 19 章 · 手写 train.py'),
  chapterHref: CH19,
  scopeNote: T(
    '对应单样本输出层的最小训练闭环；完整模型会在 forward 中运行全部 Block，并用 optimizer.zero_grad() / step() 管理全部参数。',
  ),
  lines: [
    'optimizer.zero_grad(set_to_none=True)',
    'logits = h @ W                                # [1, 3]',
    'probs = F.softmax(logits, dim=-1)             # [1, 3]',
    'loss = F.cross_entropy(logits, target_id)     # -log probs[target_id]',
    'loss.backward()                               # W.grad = h.T @ (probs - onehot)',
    'with torch.no_grad():',
    '    W -= lr * W.grad                          # SGD parameter update',
    'next_probs = F.softmax(h @ W, dim=-1)         # re-evaluate with updated W',
  ],
};

export const TRAIN_PY_HIGHLIGHTS: CodeHighlight[] = [
  {
    lines: [1, 2],
    note: T('每轮先清零旧梯度，再用当前固定权重 W 计算三个候选的分数 logits = h @ W。'),
  },
  {
    lines: [3],
    note: T('沿词表轴做稳定 Softmax 得到预测概率 probs；此时权重 W 尚未改动。'),
  },
  {
    lines: [4],
    note: T('交叉熵只取出正确标签 target_id 对应的负对数概率；PyTorch 在内部合并 log_softmax 保证数值稳定。'),
  },
  {
    lines: [5],
    note: T('loss.backward() 按链式法则算出分数梯度 p - onehot(y) 与权重梯度 h.T @ g_z，存入 W.grad。'),
  },
  {
    lines: [6, 7],
    note: T('参数更新在 no_grad 块（或 optimizer.step()）内沿负梯度方向按学习率 lr 修改权重 W。'),
  },
  {
    lines: [8],
    note: T('用更新后的 W 对同一个 h 再次前向，核对目标 token 的概率是否升高、损失是否下降。'),
  },
];

/** Chapter 20: Autoregressive generation loop with and without KV Cache matching /inference-cache. */
export const INFERENCE_PY_CODE: LinkedCodeSpec = {
  file: 'inference.py',
  chapterLabel: T('第 20 章 · 手写 inference.py'),
  chapterHref: CH20,
  scopeNote: T(
    '并列展示训练前向、无缓存重算与 Prefill + Decode 缓存循环；实际推理需固定 eval() 模式与 inference_mode()。',
  ),
  lines: [
    '# Training: shifted targets, one parallel causal pass',
    'logits = model(seq[:-1], causal=True)         # targets = seq[1:]',
    '# Generation without KV Cache (recomputes whole prefix each step)',
    'with torch.inference_mode():',
    '    for _ in range(max_new_tokens):',
    '        logits = model(ids, causal=True)      # recomputes 0..n-1',
    '        next_id = logits[:, -1].argmax(-1, keepdim=True)',
    '        ids = torch.cat([ids, next_id], dim=1)',
    '# Generation with KV Cache: Prefill once, then Decode 1 token/step',
    'with torch.inference_mode():',
    '    logits, kv_cache = model(prompt, kv_cache=None)     # Prefill',
    '    next_id = logits[:, -1].argmax(-1, keepdim=True)',
    '    for _ in range(max_new_tokens - 1):',
    '        logits, kv_cache = model(next_id, kv_cache=kv_cache)  # Decode',
    '        next_id = logits[:, -1].argmax(-1, keepdim=True)',
    '        # Prefix edit or shifted positions -> invalidate affected kv_cache',
  ],
};

export const INFERENCE_PY_HIGHLIGHTS: CodeHighlight[] = [
  {
    lines: [1, 2],
    note: T('训练把完整序列错位切片成输入 seq[:-1] 与目标 seq[1:]，一次因果前向算出全部位置。'),
  },
  {
    lines: [3, 4, 5, 6, 7, 8],
    note: T('无缓存循环每步把越变越长的整段 ids 重新喂给模型，只取最后一行 logits[:, -1] 选词。'),
  },
  {
    lines: [9, 10, 11, 12],
    note: T('Prefill 用整段 prompt 做一次前向，初始化每层的 kv_cache，并选出第一个新 token。'),
  },
  {
    lines: [13, 14, 15],
    note: T('Decode 循环每步只喂入刚生成的单个 next_id，模型只算它的 q/k/v 并追加进 kv_cache。'),
  },
  {
    lines: [6, 11, 14],
    note: T('两条路径在相同权重、位置索引与 Mask 下输出一致，差别在于第 14 行跳过了历史 K/V 重算。'),
  },
  {
    lines: [16],
    note: T('改动前缀或滑动窗口平移位置编号后，依赖旧前缀/旧位置的缓存条目必须失效重算。'),
  },
];
