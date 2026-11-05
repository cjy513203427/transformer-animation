# LLM 原理交互实验室

用可手算的小例子理解计算过程：修改输入、先预测、逐步核对，再隐藏说明复述原因。

## 本地开发

推荐 Node.js **24.22+（24.x）**，使用 npm；版本记录在 `.nvmrc`。当前 Cloudflare CLI 也支持 Node 22.23+（22.x）与 26.12+（26.x）。依赖版本由 `package-lock.json` 锁定。

```powershell
npm ci
npm run dev
```

打开 <http://127.0.0.1:5173/attention>，根路径也显示同一实验。

```powershell
npm run check             # 类型检查、数值与状态测试、生产构建
npm run test:e2e          # 本机 Chrome：桌面与手机尺寸交互测试
npm run preview          # 预览 dist，http://127.0.0.1:4173
npm run build:cloudflare  # 生成并校验 Cloudflare Build Output，不发布
npm run format           # 格式化项目源码
```

浏览器测试默认使用已安装的 Google Chrome。若环境没有 Chrome，可先运行 `npx playwright install chrome`。CI 包含该安装步骤。移动测试为 Chrome 触摸/窄屏模拟，并启用 reduced-motion，不代表真实 iOS Safari 测试。

## 学习入口

页面按原书的图组织成三层，默认进入“整体架构”：

1. **整体架构**：按第 3 章自下向上的 Post-Norm 主图，点选模块查看“进来什么、做什么、交给谁”，保留位置加法、两条残差支路与 Block × N。
2. **拆开 Attention**：显示 Q/K 打分支路与 V 绕行支路。先观察一个 token，按生成 QKV、打分/缩放、Mask、Softmax、汇总 V 五步解释；公式按需展开。
3. **动手核对**：保留完整小矩阵、输入编辑、对照实验和分享链接。

默认开启因果 Mask 并观察中间位置 t1“喜欢”，让未来的 t2“学习”被明确排除。三个词块是人为教学切分，不是真实 tokenizer 的输出。关闭 Mask 明确标为双向读取对照。

第 3 章的 Post-Norm 主图和第 15 章 GPT-2 的 Pre-Norm 顺序分别说明。整体架构为结构导航；只有单头 Attention 执行数值演算。单头 O 对应书中的 H_head，之后的 Concat 与 W_O 仅作定位说明，不能当作完整 MHA 已算完。

## 已实现的范围

- Vite + TypeScript + React，KaTeX 排版，原生 SVG 与矩阵表格，无模型服务。
- `/attention` 单头计算：X 投影 → 点积与缩放 → Mask → 行 Softmax → 加权输出。
- 播放、暂停、前进、后退、跳步、速度、重置步骤与恢复基础示例。
- 基础、只改 V、因果 Mask、相同分数、双向读取五种预设。
- 编辑 X、投影后干预 K/V、选择 Query/Key 格子、稳定数值计算与种子生成。
- 先预测再验证 V 的影响，隐藏说明与复述提示。
- 分享链接记录已执行状态、学习层次和选中架构模块；v2 状态校验版本、结构、范围和长度，兼容 v1 数值视图链接；坏链接恢复入门示例。
- 主题延迟加载、后台暂停、键盘控件、窄屏布局与减少动态效果支持。

**尚未实现：**多头投影后的 reshape/transpose、拼接与输出投影，以及位置编码、训练/生成/缓存、KV 架构、MoE、FlashAttention。导航将后续主题明确标为未实现，没有空实验页面。初始化只完成单头基础场景，不代表 Attention 主题所有 P0 条件均已覆盖。

## 目录

```text
src/
  features/attention/   实验界面与交互
  math/                 纯数值计算及数学性质测试
  components/           矩阵与公式视图
  state/                可分享状态、预设、版本校验与测试
  content/              步骤解释与参考来源
  styles/               视觉规则与响应式布局
tests/                  Playwright 浏览器交互测试
docs/                   数值约定与 Cloudflare 准备说明
```

## 数值与状态约定

固定行向量记法、T = 3、D = d_k = d_v = 2，忽略 batch、偏置、位置编码、Dropout、残差及输出投影。所有结果由浏览器实际运算得到；显示时舍入，内部使用 JavaScript float64。词语仅为位置标签，小向量不是模型语义表示。

修改实际输入会暂停播放并重建计算。ΔV 滑杆为待执行输入，点击“验证预测”后才生效；分享仅包含已经计算的输入。修改 Mask/缩放跳到相应步骤，修改 X 回到投影。重置步骤保留参数，“恢复基础示例”恢复所有默认状态。离开页面到后台会暂停，返回后不自动恢复播放。

普通浏览器分享链接只能在相应站点可访问时使用；本地 `127.0.0.1` 链接需要接收者也运行此项目，不等于已发布网站。当前不收集笔记、进度或个人数据，不接入分析服务。

更多内容见 [数学说明](docs/math.md) 与 [Cloudflare 准备](docs/cloudflare.md)。

## 参考

- [Attention Is All You Need，§3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)
- [Transformer 架构：从直觉到实现](https://waylandz.com/llm-transformer-book/)：第 3 章为主架构图，第 6、9–12 章为 Attention 解释主线。

来源核对日期：2026-11-05。说明和图形自行编写，没有复制书中的文本或图片。
