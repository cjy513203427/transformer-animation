import { test, expect, type Page } from '@playwright/test';
import { initialFlashState, serializeFlashState } from '../src/state/flash';

const scene = (page: Page, index: number) => page.locator('.flash-steps button').nth(index);
const size = (page: Page, value: number) =>
  page.getByRole('group', { name: '每块 Key 数 B', exact: true }).getByRole('button', {
    name: 'B = ' + value,
    exact: true,
  });

test('从完整基线到尾块、在线累积和全屏蔽块，图表同步且可回退', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/flash-attention');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('FlashAttention');
  const baseline = page.getByRole('table', { name: '完整行 Attention 数值', exact: true });
  await expect(baseline.locator('tbody tr td:first-of-type')).toHaveText(['0', '1', '2', '3']);
  const probabilities = await baseline.locator('tbody tr td:nth-child(3)').allTextContents();
  expect(probabilities.reduce((sum, value) => sum + Number(value), 0)).toBeCloseTo(1, 4);
  await page.getByText('展开同一份 Q、K、V 激活', { exact: true }).click();
  await expect(page.getByRole('table', { name: 'Q 矩阵', exact: true })).toBeVisible();

  await page.getByRole('button', { name: '下一步：分块读取 →', exact: true }).click();
  await size(page, 3).click();
  const tile = page.getByRole('table', { name: '当前 K V 块数值', exact: true });
  await expect(tile.locator('tbody tr')).toHaveCount(3);
  await page.getByRole('button', { name: '读下一块 →', exact: true }).click();
  await expect(tile.locator('tbody tr')).toHaveCount(1);
  await expect(tile.locator('tbody th')).toHaveText('t3');
  await expect(tile.locator('caption')).toContainText('实际 1 行');
  await expect(page.getByRole('button', { name: '读下一块 →', exact: true })).toBeDisabled();
  await expect(page.getByRole('img', { name: /分块图：第2块含Key t3/ })).toBeVisible();

  await scene(page, 2).click();
  await expect(page.getByRole('table', { name: '在线累积量更新' })).toBeVisible();
  await expect(page.locator('.lesson-reading [role="status"]')).toContainText(
    '旧 l 与旧 a 都已重缩放',
  );
  await page.getByRole('button', { name: '← 前一块', exact: true }).click();
  await expect(page.locator('.lesson-reading [role="status"]')).toContainText('前缀暂时输出');
  await page.getByRole('button', { name: 't0', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 't0', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  const beforeMasked = await page.getByLabel('当前块状态', { exact: true }).textContent();
  await page.getByRole('button', { name: '读下一块 →', exact: true }).click();
  await expect(page.locator('.lesson-reading [role="status"]')).toContainText('全屏蔽块：α = 1');
  await expect(page.getByLabel('当前块状态', { exact: true })).toContainText('m = 0 · l = 1');
  await expect(page.getByLabel('当前块状态', { exact: true })).toContainText('[1, 0]');
  expect(beforeMasked).toContain('m = 0 · l = 1');
  await expect(page.locator('.flash-panel')).not.toContainText('NaN');
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-flash-masked.png`,
    fullPage: true,
    scale: 'css',
  });
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});

test('先预测再修改 Key，再以不同块大小核对输出和存储账本', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/flash-attention');
  await scene(page, 3).click();
  const experiment = page.locator('.exp-experiment');
  await expect(experiment.getByRole('button', { name: '验证预测并执行 →' })).toBeDisabled();
  await expect(page.locator('.flash-inputs')).toContainText('当前 ΔK[3,0] = 0');
  await expect(page.getByRole('slider', { name: '待执行 Key 偏移' })).toHaveValue('2');
  await page.getByRole('button', { name: 'm 变大', exact: true }).click();
  await page.getByRole('button', { name: '验证预测并执行 →', exact: true }).click();
  await expect(experiment.locator('[role="status"]')).toContainText('最终 m：3 → 5');
  await expect(experiment.locator('[role="status"]')).toContainText('与计算一致');
  await expect(page.locator('.flash-inputs')).toContainText('当前 ΔK[3,0] = 2');
  await expect(page.locator('.lesson-reading .flash-equation')).toContainText('α = 0.018316');
  await page.getByRole('button', { name: 't0', exact: true }).click();
  await expect(experiment.locator('[role="status"]')).toHaveCount(0);
  await page.getByRole('button', { name: 'm 不变', exact: true }).click();
  await page.getByRole('button', { name: '验证预测并执行 →', exact: true }).click();
  await expect(experiment.locator('[role="status"]')).toContainText('最终 m：0 → 0');
  await expect(experiment.locator('[role="status"]')).toContainText('与计算一致');

  await page.getByRole('button', { name: '下一步：等价核对 →', exact: true }).click();
  const outputs = page.getByRole('table', { name: '分块与全量输出核对', exact: true });
  const before = await outputs.textContent();
  await expect(outputs.locator('tbody tr')).toHaveCount(4);
  await expect(page.locator('.lesson-reading .exp-verdict.ok')).toContainText('容差 1e-12');
  await page.getByRole('button', { name: '切为 3', exact: true }).click();
  await expect(size(page, 2)).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: '容差内不变', exact: true }).click();
  await page.getByRole('button', { name: '验证分块预测 →', exact: true }).click();
  await expect(size(page, 3)).toHaveAttribute('aria-pressed', 'true');
  await expect(experiment.locator('[role="status"]')).toContainText('与计算一致');
  await expect(outputs).toHaveText(before ?? '');
  await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
  await expect(page.getByText('先自己说一遍', { exact: true })).toBeVisible();
  await expect(page.locator('.lesson-explanation')).toHaveCount(0);
  await page.getByText('需要时再看公式', { exact: true }).click();
  await expect(page.locator('.katex-error')).toHaveCount(0);
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-flash-equality.png`,
    fullPage: true,
    scale: 'css',
  });

  await page.getByRole('button', { name: '下一步：存储与 I/O →', exact: true }).click();
  const ledger = page.getByRole('table', { name: '中间存储与 I O 账本', exact: true });
  await expect(ledger.locator('tbody tr').nth(0).locator('td')).toHaveText(['32', '0']);
  await expect(ledger.locator('tbody tr').nth(2).locator('td')).toHaveText(['256 B', '0 B']);
  await expect(ledger.locator('tbody tr').nth(3).locator('td')).toHaveText(['10', '10']);
  await page.getByRole('checkbox', { name: '因果 Mask', exact: true }).uncheck();
  await expect(ledger.locator('tbody tr').nth(3).locator('td')).toHaveText(['16', '16']);
  await expect(page.locator('.lesson-reading .exp-verdict')).toContainText('0 B 只指完整 S / P');
  await page.getByText('适用条件与来源', { exact: true }).click();
  await expect(page.getByText('来源核对日期：2026-11-08。图形与说明为本项目编写。')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});

test('分享恢复全部参数，坏链接回退，待执行预测不进入链接', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const state = initialFlashState();
  state.step = 3;
  state.seed = 123;
  state.config = { blockSize: 3, causal: false };
  state.input = { preset: 'mixed', query: 1, keyDelta: -1, block: 1 };
  state.view = { explanations: false, speed: 2 };
  await page.goto('/flash-attention' + serializeFlashState(state));
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 6');
  await expect(page.getByRole('spinbutton', { name: '混合输入种子' })).toHaveValue('123');
  await expect(size(page, 3)).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: 't1', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByRole('checkbox', { name: '因果 Mask', exact: true })).not.toBeChecked();
  await expect(page.getByRole('button', { name: '块 2 · t3', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByText('先自己说一遍', { exact: true })).toBeVisible();
  await page.getByRole('slider', { name: '待执行 Key 偏移' }).focus();
  await page.keyboard.press('ArrowRight');
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const url = await page.evaluate(() => navigator.clipboard.readText());
  expect(url).toContain('/flash-attention#flash=');
  const shared = JSON.parse(decodeURIComponent(new URL(url).hash.slice(7)));
  expect(shared).toEqual(state);
  await page.goto(url);
  await expect(page.locator('.flash-inputs')).toContainText('当前 ΔK[3,0] = -1');
  await expect(page.locator('.exp-experiment [role="status"]')).toHaveCount(0);
  await page.getByRole('button', { name: '恢复默认示例', exact: true }).click();
  await expect(page.locator('.lesson-progress')).toHaveText('1 / 6');
  await expect(size(page, 2)).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: 't3', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByRole('spinbutton', { name: '混合输入种子' })).toBeDisabled();
  await page.goto('/flash-attention#flash=%7Bbroken');
  await expect(page.getByRole('status').first()).toContainText('分享链接无效');
  await expect(page.locator('.lesson-progress')).toHaveText('1 / 6');
  await page.getByRole('link', { name: 'MoE 与专家并行', exact: true }).click();
  await expect(page).toHaveURL(/\/moe$/);
  await page.getByRole('link', { name: 'FlashAttention', exact: true }).click();
  await expect(page).toHaveURL(/\/flash-attention$/);
});

test('播放、暂停、编辑和外部恢复清除旧预测，触摸与减少动态效果可操作', async ({
  page,
}, testInfo) => {
  const state = initialFlashState();
  state.view.speed = 2;
  await page.goto('/flash-attention' + serializeFlashState(state));
  await page.clock.install();
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.clock.fastForward(4100);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('button', { name: '暂停自动逐步', exact: true }).click();
  await page.clock.fastForward(5000);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.getByRole('button', { name: '混合输入', exact: true }).click();
  await expect(page.getByRole('button', { name: '播放自动逐步', exact: true })).toBeVisible();
  await page.clock.fastForward(5000);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('spinbutton', { name: '混合输入种子' }).fill('77');
  await expect(page.getByRole('spinbutton', { name: '混合输入种子' })).toHaveValue('77');
  await page.getByRole('button', { name: '从头看', exact: true }).click();
  await expect(page.locator('.lesson-progress')).toHaveText('1 / 6');
  await scene(page, 4).click();
  await page.getByRole('button', { name: '容差内不变', exact: true }).click();
  await page.getByRole('button', { name: '验证分块预测 →', exact: true }).click();
  await expect(page.locator('.exp-experiment [role="status"]')).toContainText('与计算一致');
  state.step = 4;
  state.input.keyDelta = 1;
  await page.evaluate((hash) => {
    location.hash = hash;
  }, serializeFlashState(state));
  await expect(page.locator('.flash-inputs')).toContainText('当前 ΔK[3,0] = 1');
  await expect(page.locator('.exp-experiment [role="status"]')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '验证分块预测 →', exact: true })).toBeDisabled();
  if (testInfo.project.name === 'mobile') {
    await page.getByRole('button', { name: 't2', exact: true }).tap();
    await expect(page.getByRole('button', { name: 't2', exact: true })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    await expect(
      page.getByText('已遵循减少动态效果偏好；图表直接切换，不做过渡动画。'),
    ).toBeVisible();
  }
  await page.getByRole('button', { name: '查看参考解释', exact: true }).click();
  await expect(page.locator('.recall-answer')).toContainText('exp(m − m′)');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
});
