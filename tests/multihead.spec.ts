import { test, expect } from '@playwright/test';

test('多头：切分步骤、头数变化、形状对照与预测验证', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/attention');
  await page.getByRole('button', { name: /04 多头注意力/ }).click();
  const panel = page.getByRole('region', { name: '多头注意力：切分与合并', exact: true });
  await expect(panel).toBeVisible();
  await expect(panel.locator('.lesson-progress')).toHaveText('1 / 6');
  await expect(page.getByText(/与“方式二”切出的结果完全相同/)).toBeVisible();
  await page.getByRole('button', { name: '下一步：切分多头 →', exact: true }).click();
  await expect(page.getByRole('table', { name: '一行 Q、K、V 的切分', exact: true })).toBeVisible();
  await expect(page.locator('.mh-shapes tr.current')).toContainText('[4,16,512] → [4,16,4,128]');

  const heads = page.getByRole('group', { name: '头数 num_heads', exact: true });
  await heads.getByRole('button', { name: '4', exact: true }).click();
  await expect(page.locator('.mh-dkey')).toContainText('d_key = 4 ÷ 4 = 1');
  await page.getByRole('button', { name: /每头 Attention$/ }).click();
  await expect(page.getByRole('table', { name: /^head \d 的打分与权重$/ })).toHaveCount(4);
  await heads.getByRole('button', { name: '2', exact: true }).click();
  await expect(page.getByRole('table', { name: /^head \d 的打分与权重$/ })).toHaveCount(2);

  await page
    .getByRole('button', {
      name: '只有 head 2 的 P 和 A 变，但 Output 每一列都可能变',
      exact: true,
    })
    .click();
  await page.getByRole('button', { name: '验证预测 →', exact: true }).click();
  const feedback = page.locator('.mh-experiment .prediction-feedback');
  await expect(feedback).toContainText('预测正确');
  await expect(feedback).toContainText('P 发生变化的头：head 2');
  await expect(feedback).toContainText('A 变化的列：d2、d3');
  await expect(feedback).toContainText('Output 变化的列：d0、d1、d2、d3');
  await expect(panel.locator('.lesson-progress')).toHaveText('6 / 6');
  await expect(
    page.getByRole('table', { name: 'Output 每一列由各头贡献相加', exact: true }),
  ).toContainText('已变');
  await page.getByRole('button', { name: '撤销干预（恢复 ΔK = 0）', exact: true }).click();
  await expect(
    page.getByRole('table', { name: 'Output 每一列由各头贡献相加', exact: true }),
  ).not.toContainText('已变');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(
    false,
  );
  expect(errors).toEqual([]);
});

test('多头：播放、暂停、键盘、链接恢复与单头入口', async ({ page, context }, testInfo) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/attention');
  await page.getByRole('button', { name: /02 拆开 Attention/ }).click();
  await page.getByRole('button', { name: '看多头怎样切分与合并 →', exact: true }).click();
  const progress = page.locator('.mh-panel .lesson-progress');
  await expect(progress).toHaveText('1 / 6');
  await page.getByRole('combobox', { name: '多头演示速度' }).selectOption('2');
  await page.getByRole('button', { name: '播放多头演示' }).click();
  await expect(progress).toHaveText('2 / 6', { timeout: 6000 });
  await page.getByRole('button', { name: '暂停多头演示' }).click();
  await page.waitForTimeout(3800);
  await expect(progress).toHaveText('2 / 6');
  const next = page.getByRole('button', { name: '下一步：矩阵变形 →', exact: true });
  await next.focus();
  await page.keyboard.press('Enter');
  await expect(progress).toHaveText('3 / 6');
  await page
    .getByRole('group', { name: '头数 num_heads' })
    .getByRole('button', { name: '1', exact: true })
    .click();
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const link = await page.evaluate(() => navigator.clipboard.readText());
  await page.goto(link);
  await expect(
    page.getByRole('region', { name: '多头注意力：切分与合并', exact: true }),
  ).toBeVisible();
  await expect(progress).toHaveText('3 / 6');
  await expect(page.locator('.mh-dkey')).toContainText('d_key = 4 ÷ 1 = 4');
  await expect(
    page.getByText('只有一个头时，所有维度都属于同一个头。', { exact: false }),
  ).toBeVisible();
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-multihead.png`,
    fullPage: false,
    scale: 'css',
  });
});
