import { test, expect } from '@playwright/test';
import { openDirectory } from './helpers/navigation';
import { initialState, serializeState } from '../src/state/experiment';

test('按书的架构定位模块，再进入分支图逐步解释', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/attention');
  await expect(
    page.getByRole('region', { name: 'Transformer 整体架构', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: '在位置之间读取信息' })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Q 矩阵' })).toHaveCount(0);
  await page
    .getByRole('button', { name: '架构：Feed Forward，每个位置分别加工', exact: true })
    .click();
  await expect(page.getByRole('heading', { name: '每个位置分别加工' })).toBeVisible();
  await page
    .getByRole('button', { name: '架构：Vocabulary Softmax，给候选 token 分配概率', exact: true })
    .focus();
  await page.keyboard.press('Enter');
  await expect(page.getByText(/这里沿词表轴分配预测概率/)).toBeVisible();
  await page.getByText('为什么和 GPT-2 的图顺序不同？').click();
  await expect(page.getByText(/所有 Block 后还有 final LayerNorm/)).toBeVisible();
  await page.getByRole('button', { name: '拆开 Attention →', exact: true }).click();
  await expect(page.getByRole('region', { name: '跟图理解 Attention', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '同一个位置，分出三种用途' })).toBeVisible();
  await expect(page.getByRole('button', { name: 't1 喜欢 正在观察', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.locator('[data-testid="value-bypass"]')).toBeAttached();
  await page.getByRole('button', { name: '下一步：算匹配分数 →', exact: true }).click();
  await expect(page.getByRole('table', { name: '逐位置点积', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '下一步：挡住未来 →', exact: true }).click();
  await expect(page.locator('.lesson-visibility .blocked')).toContainText('学习');
  await page.getByRole('checkbox', { name: '开启因果 Mask', exact: true }).uncheck();
  await expect(page.locator('.lesson-visibility .blocked')).toHaveCount(0);
  await expect(page.getByText(/当前是双向读取对照/)).toBeVisible();
  await page.getByRole('checkbox', { name: '开启因果 Mask', exact: true }).check();
  await page.getByRole('button', { name: '下一步：分配比例 →', exact: true }).click();
  await expect(page.locator('.lesson-proportion').last()).toContainText('0.0%');
  await page.getByRole('button', { name: '下一步：汇总内容 →', exact: true }).click();
  await expect(page.getByRole('table', { name: 'Value 加权贡献', exact: true })).toBeVisible();
  await expect(page.locator('.lesson-result')).toContainText('O[1]');
  await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
  await expect(page.locator('.lesson-explanation')).toHaveCount(0);
  await page.getByRole('button', { name: '显示说明', exact: true }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(
    false,
  );
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-guide.png`,
    fullPage: true,
    scale: 'css',
  });
  await page.getByRole('button', { name: '改动 V，验证理解 →', exact: true }).click();
  await expect(page.getByRole('table', { name: 'P 矩阵', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('导读可回退、播放、暂停并通过链接恢复当前层次', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/attention');
  await page.getByRole('button', { name: '拆开 Attention', exact: true }).click();
  await page
    .getByRole('button', { name: '图中步骤：Softmax → P，分配取用比例', exact: true })
    .click();
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 5');
  await page.getByRole('button', { name: '← 上一步', exact: true }).click();
  await expect(page.locator('.lesson-visibility .blocked')).toContainText('学习');
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const url = await page.evaluate(() => navigator.clipboard.readText());
  await page.goto(url);
  await expect(page.getByRole('region', { name: '跟图理解 Attention', exact: true })).toBeVisible();
  await expect(page.locator('.lesson-progress')).toHaveText('3 / 5');
  await page.getByRole('button', { name: '从头看', exact: true }).click();
  await page.getByRole('combobox', { name: '导读速度' }).selectOption('2');
  await page.getByRole('button', { name: '播放导读' }).click();
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 5', { timeout: 6000 });
  await page.getByRole('button', { name: '暂停导读' }).click();
  await page.waitForTimeout(3700);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 5');
  await page.getByRole('button', { name: '整体架构', exact: true }).click();
  await expect(
    page.getByRole('region', { name: 'Transformer 整体架构', exact: true }),
  ).toBeVisible();
});

test('当前学习位置保持可见，展开目录不横向滚动，跳到正文保留分享状态', async ({ page }) => {
  await page.goto('/moe');
  const toggle = page.getByRole('button', { name: '学习目录', exact: true });
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(page.locator('.course-location')).toContainText('MoE 与专家并行');
  const initialScroll = await page.evaluate(() => ({ x: scrollX, y: scrollY }));
  const navigation = await openDirectory(page);
  await expect(navigation.locator('[aria-current="page"]')).toHaveAccessibleName('MoE 与专家并行');
  expect(await page.evaluate(() => ({ x: scrollX, y: scrollY }))).toEqual(initialScroll);
  await page.keyboard.press('Escape');
  await expect(toggle).toHaveAttribute('aria-expanded', 'false');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(
    false,
  );

  const state = initialState();
  state.view.lesson = 'guide';
  state.step = 3;
  await page.goto('/attention' + serializeState(state));
  const guide = page.getByRole('region', { name: '跟图理解 Attention', exact: true });
  await expect(guide).toBeVisible();
  await expect(guide.locator('.lesson-progress')).toHaveText('4 / 5');
  const sharedUrl = page.url();
  const sharedHash = new URL(sharedUrl).hash;

  await page.keyboard.press('Tab');
  const skipLink = page.getByRole('link', { name: '跳到实验内容', exact: true });
  await expect(skipLink).toBeFocused();
  await page.keyboard.press('Enter');

  await expect(page.getByRole('main', { name: 'Attention', exact: true })).toBeFocused();
  await expect(page).toHaveURL(sharedUrl);
  expect(await page.evaluate(() => window.location.hash)).toBe(sharedHash);
  await expect(page.getByRole('button', { name: '拆开 Attention', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(guide).toBeVisible();
  await expect(guide.locator('.lesson-progress')).toHaveText('4 / 5');
});
