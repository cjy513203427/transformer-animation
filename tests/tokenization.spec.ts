import { expect, test } from '@playwright/test';
import { initialTokenizationState, serializeTokenizationState } from '../src/state/tokenization';

test('第 4 章可从目录进入，观察合并、查表和预测', async ({ page }) => {
  await page.goto('/chapters#chapter-04');
  const chapter = page.locator('#chapter-04');
  await chapter
    .getByText('字节级 BPE 与词表训练', { exact: true })
    .locator('..')
    .getByRole('link', { name: '打开计算 →' })
    .click();
  await expect(page).toHaveURL(/\/chapters\/04$/);
  await page.getByRole('button', { name: '重复字符', exact: true }).click();
  await page.getByRole('button', { name: /学习合并/ }).click();
  await expect(page.getByRole('table').locator('tbody tr').first()).toContainText('97 + 97');
  await expect(page.getByRole('table').locator('tbody tr').first().locator('td')).toHaveText([
    '97 + 97',
    '7',
    '4',
    '256 · 61 61',
  ]);
  await page.getByRole('button', { name: /应用规则/ }).click();
  await page.getByRole('combobox', { name: '编码轮次' }).selectOption('1');
  await expect(page.locator('.bpe-stage')).toContainText('本轮实际替换 2 次');
  await page.getByRole('textbox', { name: '待编码文本' }).fill('aa aa 猫');
  await page.getByRole('button', { name: /嵌入查表/ }).click();
  const rows = page.getByRole('table', { name: '序列嵌入 矩阵' }).locator('tbody tr');
  expect(await rows.nth(0).locator('td').allTextContents()).toEqual(
    await rows.nth(2).locator('td').allTextContents(),
  );
  await page.getByRole('button', { name: /预测与核对/ }).click();
  await page.getByRole('button', { name: '会更多', exact: true }).click();
  await page.getByRole('button', { name: '验证预测', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('两种设置都精确还原了输入文本');
  await expect(page.getByRole('status')).toContainText(
    '当前合并预算得到 7 个 token。预算为 0 时得到 9 个。',
  );
  await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
  await expect(page.getByText('先自己说一遍', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('英德页面恢复同一编码状态，支持编辑暂停和坏链接回退', async ({ page }) => {
  const state = initialTokenizationState();
  state.input.text = '猫 🐈';
  state.step = 2;
  state.view.round = 1;
  const hash = serializeTokenizationState(state);
  for (const [code, textLabel, roundLabel] of [
    ['en', 'Text to encode', 'Encoding round'],
    ['de', 'Zu kodierender Text', 'Kodierrunde'],
  ]) {
    await page.goto('/' + code + '/chapters/04' + hash);
    await expect(page.getByRole('textbox', { name: textLabel, exact: true })).toHaveValue('猫 🐈');
    await expect(page.getByRole('combobox', { name: roundLabel, exact: true })).toHaveValue('1');
    await expect(page.locator('.lesson-progress')).toHaveText('3 / 5');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
      true,
    );
  }
  await page.goto('/chapters/04#bpe=%ZZ');
  await expect(page.getByRole('status')).toContainText('分享状态无效');
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.getByRole('textbox', { name: '待编码文本' }).fill('');
  await expect(page.getByRole('button', { name: '播放自动逐步', exact: true })).toBeVisible();
  await expect(page.getByText('空文本，序列长度为 0。', { exact: true })).toBeVisible();
  await page.getByRole('textbox', { name: '待编码文本' }).fill('猫'.repeat(48));
  const byteList = page.locator('.bpe-pieces-scroll');
  await expect(byteList.locator('.bpe-tokens li')).toHaveCount(144);
  await expect(byteList).toHaveAttribute('tabindex', '0');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
