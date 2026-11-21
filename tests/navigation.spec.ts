import { test, expect } from '@playwright/test';
import { openDirectory } from './helpers/navigation';
import { initialFlashState, serializeFlashState } from '../src/state/flash';

const orderedTitles = [
  '跟图理解 Attention',
  '分词与 BPE',
  '位置编码',
  '归一化',
  'FFN 与残差',
  '训练、生成与缓存',
  '解码策略',
  'KV 架构对照',
  'FlashAttention',
  '提示链与核对',
  '答案投票',
  '思路搜索',
  '上下文扩展',
  '稀疏与压缩记忆',
  'MoE 与专家并行',
  'SSM / Mamba',
  '前向、损失与参数更新',
  '优化器：SGD、Adam 与 AdamW',
  '参数量与计算量',
  'RLHF 与策略更新',
  'LoRA / QLoRA',
  '量化',
  'RLHF / 偏好学习 / GRPO',
];

test('学习目录分四阶段，23个实验顺序明确，当前项与键盘关闭可用', async ({ page }, info) => {
  await page.goto('/moe');
  const toggle = page.getByRole('button', { name: '学习目录', exact: true });
  await expect(toggle).toHaveAttribute(
    'aria-expanded',
    info.project.name === 'desktop' ? 'true' : 'false',
  );
  await expect(page.locator('.course-location')).toContainText('MoE 与专家并行');
  await expect(page.locator('.course-location')).toContainText('架构扩展');
  const position = await page.evaluate(() => ({ x: scrollX, y: scrollY }));
  if (info.project.name === 'mobile') await toggle.tap();
  const directory = page.getByRole('navigation', { name: '实验目录', exact: true });
  await expect(directory).toBeVisible();
  await expect(page.locator('.study-views')).toHaveCount(0);
  await expect(page.locator('.course-sidebar a[href$="/chapters"]')).toHaveCount(0);
  for (const group of ['基础计算', '生成与效率', '架构扩展', '训练与部署']) {
    await expect(directory.getByRole('heading', { name: new RegExp(group) })).toBeVisible();
  }
  const links = directory.locator('ol a');
  await expect(links).toHaveCount(23);
  await expect(directory.getByRole('link')).toHaveCount(24);
  await expect(directory.locator('.course-note-guide,.course-architecture-guide')).toHaveCount(0);
  await expect(directory.getByRole('link', { name: '00 导读', exact: true })).toHaveAttribute(
    'href',
    '/',
  );
  for (const [i, title] of orderedTitles.entries()) {
    await expect(links.nth(i)).toHaveAccessibleName(title);
    await expect(links.nth(i)).toContainText(String(i + 1).padStart(2, '0'));
  }
  await expect(directory.locator('[aria-current="page"]')).toHaveAccessibleName('MoE 与专家并行');
  expect(await page.evaluate(() => ({ x: scrollX, y: scrollY }))).toEqual(position);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  await page.screenshot({
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
  await expect(header.getByRole('link', { name: '下一实验：分词与 BPE', exact: true })).toBeVisible();

  const lastDirectory = await openDirectory(page);
  await lastDirectory.getByRole('link', { name: 'RLHF / 偏好学习 / GRPO', exact: true }).click();
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
  expect(new URL(page.url()).hash).toBe(hash);
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 6');
  const directory = await openDirectory(page);
  await directory.getByRole('link', { name: 'FlashAttention', exact: true }).click();
  await expect(page.getByRole('button', { name: '学习目录', exact: true })).toBeFocused();
  await expect(directory).toBeHidden();
  expect(new URL(page.url()).hash).toBe(hash);
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 6');
  await expect(page.locator('.flash-inputs')).toContainText('当前 ΔK[3,0] = 1');
});

test('目录固定在左侧，可拖动和用键盘调整宽度，收放后实验状态不变', async ({ page }, info) => {
  test.skip(info.project.name !== 'desktop', 'The resizable outline is a desktop control.');
  const state = initialFlashState();
  state.step = 3;
  const hash = serializeFlashState(state);
  await page.goto('/flash-attention' + hash);
  const sidebar = page.locator('.course-sidebar');
  const separator = page.getByRole('separator', { name: '调整目录宽度' });
  const initial = await sidebar.boundingBox();
  expect(initial?.x).toBe(0);
  expect(initial?.width).toBe(280);
  const handle = (await separator.boundingBox())!;
  await page.mouse.move(handle.x + handle.width / 2, handle.y + 100);
  await page.mouse.down();
  await page.mouse.move(handle.x + handle.width / 2 + 60, handle.y + 100, { steps: 6 });
  await page.mouse.up();
  await expect(separator).toHaveAttribute('aria-valuenow', '340');
  await separator.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(separator).toHaveAttribute('aria-valuenow', '330');
  await page.keyboard.press('Home');
  await expect(separator).toHaveAttribute('aria-valuenow', '220');
  await page.keyboard.press('End');
  await expect(separator).toHaveAttribute('aria-valuenow', '400');
  await separator.dblclick({ position: { x: 4, y: 100 } });
  await expect(separator).toHaveAttribute('aria-valuenow', '280');
  await page.keyboard.press('Escape');
  await expect(sidebar).toBeHidden();
  expect((await page.locator('.main-shell').boundingBox())?.x).toBe(0);
  await openDirectory(page);
  await page.evaluate(() => window.scrollTo(0, 500));
  expect((await sidebar.boundingBox())?.y).toBe(0);
  expect(new URL(page.url()).hash).toBe(hash);
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 6');
});

test('阅读缩放有边界、可重置并跨页面记忆，不改变分享状态', async ({ page }) => {
  const state = initialFlashState();
  state.step = 3;
  const hash = serializeFlashState(state);
  await page.goto('/flash-attention' + hash);
  const increase = page.getByRole('button', { name: '放大阅读内容' });
  const decrease = page.getByRole('button', { name: '缩小阅读内容' });
  const reset = page.getByRole('button', { name: '恢复 100% 缩放' });
  const headingHeight = (await page.locator('h1').boundingBox())!.height;
  for (let i = 0; i < 3; i++) await increase.click();
  await expect(reset).toHaveText('130%');
  await expect(increase).toBeDisabled();
  expect((await page.locator('h1').boundingBox())!.height).toBeGreaterThan(headingHeight);
  expect(new URL(page.url()).hash).toBe(hash);
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 6');
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
  ).toBeLessThanOrEqual(1);
  await page.reload();
  await expect(reset).toHaveText('130%');
  await page.goto('/position');
  await expect(reset).toHaveText('130%');
  for (let i = 0; i < 5; i++) await decrease.click();
  await expect(reset).toHaveText('80%');
  await expect(decrease).toBeDisabled();
  await reset.click();
  await expect(reset).toHaveText('100%');
});

test('手机目录锁定背景、管理焦点并保留阅读位置，可用遮罩和 Escape 关闭', async ({ page }, info) => {
  test.skip(info.project.name !== 'mobile', 'The modal drawer is a mobile control.');
  await page.goto('/position');
  await page.evaluate(() => window.scrollTo(0, 400));
  const before = await page.evaluate(() => scrollY);
  const toggle = page.getByRole('button', { name: '学习目录', exact: true });
  await toggle.tap();
  const drawer = page.getByRole('dialog', { name: '学习目录', exact: true });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByRole('button', { name: '收起目录' })).toBeFocused();
  expect(await page.locator('.main-shell').evaluate((el) => (el as HTMLElement).inert)).toBe(true);
  await drawer.getByRole('link', { name: 'Deutsch' }).focus();
  await page.keyboard.press('Tab');
  await expect(drawer.locator('.site-brand')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(drawer).toBeHidden();
  await expect(toggle).toBeFocused();
  expect(await page.evaluate(() => scrollY)).toBe(before);
  await toggle.tap();
  await page.locator('.directory-backdrop').tap({ position: { x: 375, y: 200 } });
  await expect(drawer).toBeHidden();
  expect(await page.locator('.main-shell').evaluate((el) => (el as HTMLElement).inert)).toBe(false);
});

test('失效的阅读偏好回退到安全范围，各页面共用机制实验目录', async ({ page }, info) => {
  await page.addInitScript(() => {
    localStorage.setItem('llm-reading-layout-v1', JSON.stringify({ width: -500, zoom: 'broken' }));
  });
  await page.goto('/chapters');
  await expect(page.locator('.reading-zoom-reset')).toHaveText('100%');
  const toggle = page.getByRole('button', { name: '学习目录', exact: true });
  if (info.project.name === 'mobile') await toggle.tap();
  const directory = page.getByRole('navigation', { name: '实验目录', exact: true });
  await expect(directory.locator('ol a')).toHaveCount(23);
  if (info.project.name === 'desktop')
    await expect(page.getByRole('separator', { name: '调整目录宽度' })).toHaveAttribute(
      'aria-valuenow',
      '220',
    );
  await directory.getByRole('link', { name: '位置编码', exact: true }).click();
  await expect(page).toHaveURL(/\/position$/);
  if (info.project.name === 'mobile') {
    await expect(page.locator('.course-sidebar')).toBeHidden();
    expect(await page.locator('.main-shell').evaluate((el) => (el as HTMLElement).inert)).toBe(
      false,
    );
  }
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
  ).toBeLessThanOrEqual(1);
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

  await expect(page.locator('.intro-path a')).toHaveCount(23);
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
