import { test, expect } from '@playwright/test';
import { openDirectory } from './helpers/navigation';
import { initialFlashState, serializeFlashState } from '../src/state/flash';

const orderedTitles = [
  '跟图理解 Attention',
  '位置编码',
  '归一化',
  'FFN 与残差',
  '训练、生成与缓存',
  '解码策略',
  'KV 架构对照',
  'FlashAttention',
  '稀疏与压缩记忆',
  'MoE 与专家并行',
  'SSM / Mamba',
  'LoRA / QLoRA',
  '量化',
  '偏好学习 / GRPO',
];

test('学习目录分四阶段，14个实验顺序明确，当前项与键盘关闭可用', async ({ page }, info) => {
  await page.goto('/moe');
  const toggle = page.getByRole('button', { name: '学习目录', exact: true });
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(page.locator('.course-location')).toContainText('MoE 与专家并行');
  await expect(page.locator('.course-location')).toContainText('架构扩展');
  const position = await page.evaluate(() => ({ x: scrollX, y: scrollY }));
  if (info.project.name === 'mobile') await toggle.tap();
  else await toggle.click();
  const directory = page.getByRole('navigation', { name: '实验目录', exact: true });
  await expect(directory).toBeVisible();
  for (const group of ['基础计算', '生成与效率', '架构扩展', '训练与部署']) {
    await expect(directory.getByRole('heading', { name: new RegExp(group) })).toBeVisible();
  }
  const links = directory.getByRole('link');
  // 14 experiments, then the introduction linked from the note below them.
  await expect(links).toHaveCount(15);
  await expect(links.nth(14)).toHaveAccessibleName('00 导读');
  await expect(links.nth(14)).toHaveAttribute('href', '/');
  for (const [i, title] of orderedTitles.entries()) {
    await expect(links.nth(i)).toHaveAccessibleName(title);
    await expect(links.nth(i)).toContainText(String(i + 1).padStart(2, '0'));
  }
  await expect(directory.locator('[aria-current="page"]')).toHaveAccessibleName('MoE 与专家并行');
  expect(await page.evaluate(() => ({ x: scrollX, y: scrollY }))).toEqual(position);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  await page.locator('.site-header').screenshot({
    path: `test-results/${info.project.name}-ordered-directory.png`,
    scale: 'css',
  });
  await page.keyboard.press('Escape');
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(directory).toBeHidden();
  await expect(toggle).toBeFocused();
});

test('上一与下一实验跨阶段衔接，首尾边界明确，目录可直接跳转', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/ffn-residual');
  const header = page.locator('.site-header');
  const continued = page.getByRole('navigation', { name: '按顺序继续学习', exact: true });
  await expect(header.getByRole('link', { name: '上一实验：归一化', exact: true })).toBeVisible();
  await expect(
    continued.getByRole('link', { name: '下一实验：训练、生成与缓存', exact: true }),
  ).toBeVisible();
  await header.getByRole('link', { name: '下一实验：训练、生成与缓存', exact: true }).click();
  await expect(page).toHaveURL(/\/inference-cache$/);
  await expect(page.locator('.course-location')).toContainText('生成与效率');
  await header.getByRole('link', { name: '上一实验：FFN 与残差', exact: true }).click();
  await expect(page).toHaveURL(/\/ffn-residual$/);

  const directory = await openDirectory(page);
  await directory.getByRole('link', { name: '跟图理解 Attention', exact: true }).click();
  await expect(page).toHaveURL(/\/attention$/);
  await expect(header.getByRole('link', { name: /^上一实验：/ })).toHaveCount(0);
  await expect(header.getByRole('link', { name: /^回到导读：/ })).toHaveAttribute('href', '/');
  await expect(header.getByRole('link', { name: '下一实验：位置编码', exact: true })).toBeVisible();

  const lastDirectory = await openDirectory(page);
  await lastDirectory.getByRole('link', { name: '偏好学习 / GRPO', exact: true }).click();
  await expect(page).toHaveURL(/\/preference$/);
  await expect(header.getByRole('link', { name: /^下一实验：/ })).toHaveCount(0);
  await expect(header.getByRole('link', { name: '上一实验：量化', exact: true })).toBeVisible();
  await expect(page.locator('.course-location')).toContainText('训练与部署');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});

test('目录选择当前实验只收起目录，保留已分享的参数和步骤', async ({ page }) => {
  const state = initialFlashState();
  state.step = 3;
  state.input.keyDelta = 1;
  const hash = serializeFlashState(state);
  await page.goto('/flash-attention' + hash);
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 6');
  const directory = await openDirectory(page);
  await directory.getByRole('link', { name: 'FlashAttention', exact: true }).click();
  await expect(page.getByRole('button', { name: '学习目录', exact: true })).toBeFocused();
  await expect(directory).toBeHidden();
  expect(new URL(page.url()).hash).toBe(hash);
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 6');
  await expect(page.locator('.flash-inputs')).toContainText('当前 ΔK[3,0] = 1');
});

test('根路径是导读：接词演示用 Softmax 实际计算，学习路径链接到各实验', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('h1')).toHaveText('开始之前：大语言模型到底在做什么？');
  await expect(page.locator('.course-location')).toContainText('导读');
  await expect(page.locator('.site-brand')).toHaveAttribute('href', '/');

  // Scores 2.1, 1.5, 0.4, −0.3 → softmax ≈ 54.9%, 30.1%, 10.0%, 5.0%; the shares sum to 100%.
  const demo = page.locator('.intro-demo');
  const shares = await demo.locator('.intro-candidates strong').allTextContents();
  expect(shares).toEqual(['54.9%', '30.1%', '10.0%', '5.0%']);
  await demo.getByRole('button', { name: /^选择“咖啡”/ }).click();
  await expect(demo.locator('.intro-sentence')).toContainText('早上我想来一杯咖啡');
  // Choosing the full stop ends the sentence before the third round.
  await demo.getByRole('button', { name: /^选择“。（句号）”/ }).click();
  await expect(demo.locator('.intro-sentence')).toContainText('早上我想来一杯咖啡。');
  await expect(demo.locator('.intro-candidates')).toHaveCount(0);
  await demo.getByRole('button', { name: '重新开始', exact: true }).click();
  await expect(demo.locator('.intro-candidates button')).toHaveCount(4);

  await expect(page.locator('.intro-path a')).toHaveCount(14);
  await expect(page.getByRole('link', { name: /^从 跟图理解 Attention 开始/ })).toHaveAttribute(
    'href',
    '/attention',
  );
  const continued = page.getByRole('navigation', { name: '按顺序继续学习', exact: true });
  await expect(
    continued.getByRole('link', { name: '下一实验：跟图理解 Attention', exact: true }),
  ).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});
