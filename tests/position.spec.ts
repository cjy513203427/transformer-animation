import { test, expect } from '@playwright/test';
import { initialPositionState, serializePositionState } from '../src/state/position';

test('位置编码：位置盲、正弦表与书中数值、相加、位置播放', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/position');
  await expect(page.getByRole('heading', { name: '位置编码：顺序从哪里进入计算' })).toBeVisible();
  const verdict = page.locator('.pos-verdict');
  await expect(verdict).toContainText('两句话里“我”的输出完全相同');
  await page.getByRole('checkbox', { name: '加入正弦位置编码（d_model = 4）' }).check();
  await expect(verdict).toContainText('输出不同');
  await page.getByRole('checkbox', { name: '加入正弦位置编码（d_model = 4）' }).uncheck();
  await page.getByRole('button', { name: '加因果 Mask：我爱你 / 爱我你', exact: true }).click();
  await expect(verdict).toContainText('两句话里“你”的输出完全相同');

  await page.getByRole('button', { name: '下一步：正弦表 →', exact: true }).click();
  await expect(page.getByRole('button', { name: /^① 加法位置编码/ })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  const table = page.getByRole('table', { name: '正弦位置编码表，前 4 维', exact: true });
  await expect(table).toContainText('0.841');
  await expect(table).toContainText('0.682');
  await expect(page.locator('.pos-book')).toContainText('0.84、0.54、0.68、0.73');
  await page.getByRole('slider', { name: '位置 pos', exact: true }).fill('5');
  await expect(page.locator('.pos-book')).toContainText('-0.95、0.28、-0.57、-0.82');
  await page.getByRole('slider', { name: '位置 pos', exact: true }).fill('0');
  await page.getByRole('button', { name: '播放位置逐个 +1', exact: true }).click();
  await expect(page.locator('.pos-range output').first()).toHaveText('2', { timeout: 4000 });
  await page.getByRole('button', { name: '暂停位置逐个 +1', exact: true }).click();
  const paused = await page.locator('.pos-range output').first().textContent();
  await page.waitForTimeout(1500);
  await expect(page.locator('.pos-range output').first()).toHaveText(paused ?? '');

  await page.getByRole('button', { name: /加到输入$/ }).click();
  await page.getByRole('slider', { name: '第二个 L 的位置', exact: true }).fill('1');
  await expect(page.getByRole('table', { name: '嵌入加位置编码', exact: true })).toContainText(
    '1.461',
  );
  await expect(page.locator('.pos-verdict')).toContainText('嵌入相同、位置不同');
  await page.getByRole('slider', { name: '第二个 L 的位置', exact: true }).fill('0');
  await expect(page.locator('.pos-verdict')).toContainText('X 也完全相同');
  expect(errors).toEqual([]);
});

test('RoPE 平移不变、ALiBi 斜率预测、对照跳转与键盘', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/position');
  const progress = page.locator('.pos-panel .lesson-progress');
  await page.getByRole('button', { name: /RoPE 旋转$/ }).click();
  await expect(progress).toHaveText('4 / 7');
  await page.getByRole('slider', { name: 'q 的位置 m', exact: true }).fill('17');
  const vectors = page.getByRole('table', { name: 'RoPE 旋转前后的 q 与 k', exact: true });
  await expect(vectors).toContainText('R_17 q');
  await expect(vectors.locator('tbody tr').nth(1).locator('td').last()).toHaveText('2.236');

  await page.getByRole('button', { name: /相对距离$/ }).click();
  await page.getByRole('button', { name: '分数不变', exact: true }).click();
  await page.getByRole('button', { name: '执行平移并核对 →', exact: true }).click();
  await expect(page.locator('.pos-predict .pos-verdict')).toContainText('预测正确');
  await expect(page.locator('.pos-now')).toContainText('距离 n − m = -12');

  await page.getByRole('button', { name: /ALiBi$/ }).click();
  await page.getByRole('button', { name: 'head 8，斜率 1/256', exact: true }).click();
  await expect(page.locator('.pos-chip')).toContainText('1/256');
  await page.getByRole('button', { name: 'head 1', exact: true }).click();
  await page.getByRole('button', { name: '对比两个头 →', exact: true }).click();
  await expect(page.locator('.pos-predict .pos-verdict')).toContainText('预测正确');
  await expect(page.locator('.pos-predict .pos-verdict')).toContainText('head 1 给“师”自己 41.4%');

  await page.getByRole('button', { name: /三处对照$/ }).click();
  await expect(page.getByRole('table', { name: '三种位置编码对照', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'RoPE（第 25 章） ↗', exact: true }).click();
  await expect(progress).toHaveText('4 / 7');
  const alibi = page.getByRole('button', { name: /^③ ALiBi/ });
  await alibi.focus();
  await page.keyboard.press('Enter');
  await expect(progress).toHaveText('6 / 7');
  await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
  await expect(page.locator('.lesson-explanation')).toHaveCount(0);
  await expect(page.locator('.lesson-recall')).toContainText('ALiBi 改的是哪个矩阵');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(
    false,
  );
  expect(errors).toEqual([]);
});

test('位置编码链接恢复、坏链接回退与目录导航', async ({ page, context }, testInfo) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const state = initialPositionState();
  state.step = 5;
  state.alibi = { head: 3, query: 2, content: 'example' };
  await page.goto('/position' + serializePositionState(state));
  await expect(page.locator('.pos-panel .lesson-progress')).toHaveText('6 / 7');
  await expect(page.locator('.pos-chip')).toContainText('1/16');
  await expect(page.locator('.pos-row-detail')).toContainText('Query 2（M）这一行');
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  await expect(page.getByText('已复制当前实验状态的链接。')).toBeVisible();
  const link = await page.evaluate(() => navigator.clipboard.readText());
  expect(link).toContain('/position#pos=');
  await page.goto('/position#pos=%bad');
  await expect(page.getByText('分享链接无效或版本不受支持，已恢复默认示例。')).toBeVisible();
  await expect(page.locator('.pos-panel .lesson-progress')).toHaveText('1 / 7');
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-position.png`,
    fullPage: false,
    scale: 'css',
  });
  await page.getByRole('link', { name: /跟图理解 Attention/ }).click();
  await expect(
    page.getByRole('heading', { name: 'Attention：信息怎样在词之间流动' }),
  ).toBeVisible();
  await page.getByRole('link', { name: /位置编码/ }).click();
  await expect(page.getByRole('heading', { name: '位置编码：顺序从哪里进入计算' })).toBeVisible();
});
