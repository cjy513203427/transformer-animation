import { expect, test } from '@playwright/test';

test('逐次合并、预测 token 减少量、切分新词、回退与分享恢复', async ({ page, context }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/tokenization');
  const total = page.getByTestId('bpe-total');
  await expect(total).toHaveText('95');
  await expect(page.getByTestId('bpe-next')).toContainText('e + s → es');
  await expect(page.getByTestId('bpe-encoded').locator('.bpe-token')).toHaveCount(7);

  await page.getByRole('button', { name: '减少这一对的加权次数', exact: true }).click();
  await page.getByRole('button', { name: '执行合并并核对', exact: true }).click();
  await expect(page.getByTestId('bpe-verdict')).toContainText('实际减少 9');
  await expect(page.getByTestId('bpe-verdict')).toContainText('与计算一致');
  await expect(total).toHaveText('86');

  for (let i = 0; i < 4; i++) await page.getByRole('button', { name: '下一步 →', exact: true }).click();
  await expect(page.getByText('已合并 5 / 12 次', { exact: true })).toBeVisible();
  await expect(page.getByTestId('bpe-encoded').locator('.bpe-token')).toHaveText(['low', 'est</w>']);

  await page.getByRole('button', { name: '← 上一步', exact: true }).click();
  await expect(page.getByTestId('bpe-encoded').locator('.bpe-token')).toHaveText(['lo', 'w', 'est</w>']);

  const probe = page.getByRole('textbox', { name: '新词（小写字母 a–z）', exact: true });
  await probe.fill('Low');
  await expect(page.getByRole('alert')).toContainText('只允许');
  await probe.fill('newer');
  await expect(page.getByRole('alert')).toHaveCount(0);

  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const url = await page.evaluate(() => navigator.clipboard.readText());
  expect(url).toContain('#bpe=');
  await page.goto('about:blank');
  await page.goto(url);
  await expect(page.getByText('已合并 4 / 12 次', { exact: true })).toBeVisible();
  await expect(probe).toHaveValue('newer');

  await page.getByRole('button', { name: '重置', exact: true }).click();
  await expect(total).toHaveText('95');
  await page.goto('/tokenization#bpe=%7B%7D');
  await expect(page.getByText('分享状态无效，已恢复默认示例。', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

for (const locale of ['en', 'de'] as const)
  test(`${locale} 分词页完整翻译且窄屏不横向溢出`, async ({ page }) => {
    const problems: string[] = [];
    page.on('pageerror', (e) => problems.push(e.message));
    page.on('console', (m) => {
      if (m.text().includes('[i18n] missing')) problems.push(m.text());
    });
    await page.goto(`/${locale}/tokenization`);
    await expect(page.getByTestId('bpe-total')).toHaveText('95');
    await page.locator('#review details').evaluate((d) => ((d as HTMLDetailsElement).open = true));
    // The byte demo's default sample text contains one Chinese character on purpose.
    const text = (await page.locator('main').innerText()).replace(/低/g, '');
    expect(text).not.toMatch(/[一-鿿]/);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
    ).toBeLessThanOrEqual(1);
    expect(problems).toEqual([]);
  });
