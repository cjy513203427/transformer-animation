import { expect, test } from '@playwright/test';

test('全局模糊搜索：快捷键打开、容错/子序列检索、分类筛选与键盘跳转', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/attention');
  const trigger = page.getByRole('button', { name: '站内搜索', exact: true });
  await expect(trigger).toBeVisible();
  // Open via "/" shortcut
  await page.keyboard.press('/');
  const dialog = page.getByRole('dialog', { name: '站内搜索', exact: true });
  await expect(dialog).toBeVisible();

  const input = dialog.getByRole('searchbox', { name: '搜索实验、章节或概念', exact: true });
  await expect(input).toBeFocused();

  // Typo-tolerant search: "attnetion" -> finds Attention
  await input.fill('attnetion');
  await expect(dialog.locator('.site-search-results li a').first()).toContainText('Attention');

  // Chinese subsequence fuzzy search: "线变" -> finds 线性变换与几何
  await input.fill('线变');
  await expect(dialog.locator('.site-search-results li a').first()).toContainText('线性变换与几何');

  // Filter by chapter & search "model.py" -> Ch 18
  await dialog.getByRole('button', { name: '原书章节', exact: true }).click();
  await input.fill('model.py');
  await expect(dialog.locator('.site-search-results li a').first()).toContainText('第 18 章');

  // Keyboard navigation + Enter jump to quantization via "gptq 量化"
  await dialog.getByRole('button', { name: '实验', exact: true }).click();
  await input.fill('gptq 量化');
  await expect(dialog.locator('.site-search-results li a').first()).toContainText('量化');
  await input.press('Enter');
  await expect(page).toHaveURL(/\/quantization$/);

  // Trigger button + Escape closes and restores focus without overflow
  const nextTrigger = page.getByRole('button', { name: '站内搜索', exact: true });
  await nextTrigger.click();
  await expect(dialog).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(nextTrigger).toBeFocused();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
  ).toBeLessThanOrEqual(1);
  expect(errors).toEqual([]);
});

test('章节页支持模糊搜索，英德页面搜索弹窗翻译完整且不溢出', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/chapters');
  const chapterSearch = page.getByRole('searchbox', { name: '查找章节或概念', exact: true });
  await chapterSearch.fill('flsh');
  await expect(page.locator('#chapter-21')).toBeVisible();

  for (const [locale, label] of [
    ['en', 'Site search'],
    ['de', 'Seitensuche'],
  ] as const) {
    await page.goto(`/${locale}/attention`);
    await page.getByRole('button', { name: label, exact: true }).click();
    const dialog = page.getByRole('dialog', { name: label, exact: true });
    await expect(dialog).toBeVisible();
    expect(await dialog.innerText()).not.toMatch(/[一-鿿]/);
    await dialog.getByRole('searchbox').fill('qunatize');
    await expect(dialog.locator('.site-search-results li a').first()).toHaveAttribute(
      'href',
      new RegExp(`^/${locale}/quantization`),
    );
    await page.keyboard.press('Escape');
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
    ).toBeLessThanOrEqual(1);
  }
  expect(errors).toEqual([]);
});
