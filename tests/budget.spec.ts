import { expect, test } from '@playwright/test';

test('预设复现公开参数量，无效头数被拒绝并说明，预测与分享可恢复', async ({ page, context }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/model-budget');
  const total = page.getByTestId('budget-total');
  await expect(total).toHaveText('124,439,808');
  for (const [name, value] of [
    ['原书第 18 章模型', '44,503,040'],
    ['Llama 2 7B', '6,738,415,616'],
    ['Llama 2 70B（GQA）', '68,976,648,192'],
  ]) {
    await page.getByRole('button', { name, exact: true }).click();
    await expect(total).toHaveText(value);
  }

  // 64 Query heads cannot be split into 7 KV groups; the last valid result stays.
  const kv = page.getByRole('spinbutton', { name: 'KV 头数 H_kv', exact: true });
  await kv.fill('7');
  await expect(page.getByRole('alert')).toContainText('整数倍');
  await expect(total).toHaveText('68,976,648,192');
  await kv.fill('64');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(total).not.toHaveText('68,976,648,192');

  // Tying removes V·d parameters but no forward FLOPs.
  await page.getByRole('button', { name: '不变', exact: true }).click();
  await page.getByRole('button', { name: '计算两种设置', exact: true }).click();
  await expect(page.getByTestId('budget-verdict')).toContainText('262,144,000');
  await expect(page.getByTestId('budget-verdict')).toContainText('FLOPs 变化 0');
  await expect(page.getByTestId('budget-verdict')).toContainText('与计算一致');

  await page.getByRole('spinbutton', { name: '序列长度 s', exact: true }).fill('8192');
  const forward = await page.getByTestId('budget-forward').innerText();
  await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
  await expect(page.locator('.budget-explain')).toHaveCount(0);
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const url = await page.evaluate(() => navigator.clipboard.readText());
  expect(url).toContain('#budget=');
  await page.goto('about:blank');
  await page.goto(url);
  await expect(page.getByTestId('budget-forward')).toHaveText(forward);
  await expect(page.getByRole('spinbutton', { name: 'KV 头数 H_kv', exact: true })).toHaveValue('64');
  await expect(page.locator('.budget-explain')).toHaveCount(0);

  await page.goto('/model-budget#budget=%7B%22version%22%3A1%7D');
  await expect(page.getByText('分享状态无效，已恢复默认示例。', { exact: true })).toBeVisible();
  await expect(total).toHaveText('124,439,808');
  expect(errors).toEqual([]);
});

for (const locale of ['en', 'de'] as const)
  test(`${locale} 计算器完整翻译且窄屏不横向溢出`, async ({ page }) => {
    const problems: string[] = [];
    page.on('pageerror', (e) => problems.push(e.message));
    page.on('console', (m) => {
      if (m.text().includes('[i18n] missing')) problems.push(m.text());
    });
    await page.goto(`/${locale}/model-budget`);
    await expect(page.getByTestId('budget-total')).toHaveText('124,439,808');
    await page.locator('.why-intro').evaluate((d) => ((d as HTMLDetailsElement).open = true));
    await page.locator('#review details').evaluate((d) => ((d as HTMLDetailsElement).open = true));
    expect(await page.locator('main').innerText()).not.toMatch(/[一-鿿]/);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
    ).toBeLessThanOrEqual(1);
    expect(problems).toEqual([]);
  });
