import { test, expect } from '@playwright/test';

test('符号随计算步骤和缩放设置更新，数值展开式也有解释', async ({ page }) => {
  await page.goto('/attention');
  await page.getByRole('button', { name: '动手核对', exact: true }).click();
  const symbols = page.locator('.step-formula .formula-symbols');
  await expect(symbols).toContainText('查询（Query）矩阵');
  await expect(page.locator('.formula-symbols').first()).toContainText('下标中的数字');

  await page.getByRole('button', { name: /打分 S/ }).click();
  await expect(symbols).toContainText('除以其平方根');
  await expect(symbols).not.toContainText('投影为');
  await page.getByRole('switch', { name: /缩放 1/ }).uncheck();
  await expect(symbols).not.toContainText('除以其平方根');
  await expect(symbols).toContainText('转置');

  await page.getByRole('button', { name: /归一化 P/ }).click();
  await expect(symbols).toContainText('最大分数');
  await expect(symbols).toContainText('k 在分母中遍历');
  await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
  await expect(symbols).toHaveCount(0);
  await page.getByRole('button', { name: '显示说明', exact: true }).click();
  await expect(symbols).toContainText('最大分数');
});

for (const locale of ['', '/en', '/de'])
  test(`${locale || '中文'} 的公式可用键盘展开，在线状态符号在窄屏也能阅读`, async ({
    page,
  }, testInfo) => {
    await page.goto(`${locale}/flash-attention`);
    const details = page
      .locator('details.lesson-formula')
      .filter({ has: page.locator('.formula-block') });
    await details.locator('summary').focus();
    await page.keyboard.press('Enter');
    await expect(details.locator('.formula-symbols')).toBeVisible();
    const next = page.locator('.lesson-controls .button.primary');
    await next.click();
    await next.click();
    await expect(details.locator('.formula-symbols')).toBeVisible();
    await expect(details.locator('.katex-error')).toHaveCount(0);
    expect(await details.locator('.formula-symbol').count()).toBeGreaterThan(3);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
    ).toBeLessThanOrEqual(1);
    await details.locator('.formula-symbols').scrollIntoViewIfNeeded();
    await page.screenshot({
      path: `test-results/${testInfo.project.name}-formula-symbols-${locale.slice(1) || 'zh'}.png`,
      fullPage: true,
      scale: 'css',
    });
  });
