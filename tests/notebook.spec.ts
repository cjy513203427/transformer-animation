import { expect, test } from '@playwright/test';

test('RLHF从导航可发现，阶段、回退、复述与原书链接可用', async ({ page }) => {
  await page.goto('/preference');
  await page.getByRole('button', { name: '学习目录', exact: true }).click();
  await page.getByRole('link', { name: /RLHF 全流程笔记/ }).click();
  await expect(page).toHaveURL(/\/notes#rlhf$/);
  await expect(page.locator('#note-title')).toContainText('RLHF');
  const stage = page.getByLabel('RLHF 当前阶段', { exact: true });
  await expect(stage).toHaveText('1 / 7');
  await page.getByRole('button', { name: '下一步', exact: true }).click();
  await expect(stage).toHaveText('2 / 7');
  await page.getByRole('button', { name: '上一步', exact: true }).click();
  await expect(stage).toHaveText('1 / 7');
  await page.locator('.note-pipeline button').nth(5).click();
  await expect(page.locator('.note-stage')).toContainText('Actor 与 Critic');
  await expect(page.locator('.note-stage')).toContainText('π_old');
  await page.getByRole('button', { name: '隐藏说明，试着复述', exact: true }).click();
  await expect(page.locator('.note-stage')).toHaveCount(0);
  await expect(page.locator('.note-recall')).toBeVisible();
  await page.getByRole('button', { name: '显示说明', exact: true }).click();
  await expect(stage).toHaveText('6 / 7');
  await page.getByText('对应章节与原始来源', { exact: true }).click();
  await expect(page.locator('.note-sources a')).toHaveCount(3);
});

test('提示拼装、投票反例和分享状态可恢复，锚点切换保留输入', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/notes#prompt');
  await page.getByRole('checkbox', { name: '加入角色', exact: true }).check();
  await page.getByRole('combobox', { name: '示例数量', exact: true }).selectOption('2');
  await expect(page.getByTestId('composed-prompt')).toContainText('示例 2');
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const link = await page.evaluate(() => navigator.clipboard.readText());
  await page.goto(link);
  await expect(page.getByRole('checkbox', { name: '加入角色', exact: true })).toBeChecked();
  await expect(page.getByRole('combobox', { name: '示例数量', exact: true })).toHaveValue('2');
  await page.evaluate(() => {
    window.location.hash = '#self-consistency';
  });
  await page.getByRole('button', { name: '对照：多数答错', exact: true }).click();
  await page.getByRole('button', { name: '3', exact: true }).click();
  await page.getByRole('button', { name: '计算投票', exact: true }).click();
  await expect(page.getByTestId('vote-result')).toContainText('投票答案：3');
  await expect(page.getByTestId('vote-result')).toContainText('60%');
  await page.getByRole('button', { name: '对照：平局与提取失败', exact: true }).click();
  await expect(page.getByTestId('vote-result')).toHaveCount(0);
  await page.getByRole('button', { name: '无法选出唯一答案', exact: true }).click();
  await page.getByRole('button', { name: '计算投票', exact: true }).click();
  await expect(page.getByTestId('vote-result')).toContainText('提取失败 3');
  await expect(page.getByTestId('vote-result')).toContainText('无法选出唯一答案');
  await page.evaluate(() => {
    window.location.hash = '#prompt';
  });
  await expect(page.getByRole('checkbox', { name: '加入角色', exact: true })).toBeChecked();
});

test('学习率真实发散，训练账本联动，错误链接回退且搜索可清除', async ({ page }) => {
  await page.goto('/notes#learning-rate');
  await page.getByRole('button', { name: 'η=2.5', exact: true }).click();
  await page.getByRole('button', { name: '不更小', exact: true }).click();
  await page.getByRole('button', { name: '执行 12 次更新', exact: true }).click();
  await expect(page.getByTestId('learning-result')).toContainText('预测与计算一致');
  await expect(page.locator('.note-table-scroll tbody tr')).toHaveCount(13);
  await page.evaluate(() => {
    window.location.hash = '#scaling';
  });
  await expect(page.getByTestId('training-budget')).toContainText('5.880e+21');
  await page.getByRole('spinbutton', { name: '参数 N（十亿）', exact: true }).fill('14');
  await expect(page.getByTestId('training-budget')).toContainText('1.176e+22');
  await expect(page.getByTestId('training-budget')).toContainText('D/N = 10');
  await page.getByRole('searchbox', { name: '查找知识点', exact: true }).fill('DPO');
  const links = page.getByRole('navigation', { name: '选择知识点', exact: true }).getByRole('link');
  await expect(links).toHaveCount(1);
  await page.goto('/notes#notes=bad');
  await expect(page.getByText('分享状态无效，已恢复默认示例。', { exact: true })).toBeVisible();
  await expect(page.locator('#note-title')).toContainText('RLHF');
});

for (const locale of ['en', 'de'])
  test(`${locale} 所有笔记与工具完整翻译，窄屏可读且无异常`, async ({ page }, info) => {
    test.setTimeout(90000);
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => {
      if (m.text().includes('[i18n] missing')) errors.push(m.text());
    });
    await page.goto(`/${locale}/notes`);
    await expect(page.locator('.notebook-directory nav a')).toHaveCount(26);
    const ids = await page
      .locator('.notebook-directory nav a')
      .evaluateAll((links) => links.map((a) => a.getAttribute('href')!));
    expect(ids).toHaveLength(26);
    for (const id of ids) {
      await page.evaluate((hash) => {
        window.location.hash = hash;
      }, id);
      await expect(page.locator('#note-reading')).toHaveAttribute('data-note', id.slice(1));
      await expect(page.locator('#note-title')).toBeVisible();
      await expect(page.locator('.katex-error')).toHaveCount(0);
      if (id === '#learning-rate') {
        await page.locator('.exp-predict button').first().click();
        await page.locator('.note-tool > button').last().click();
      }
      if (id === '#self-consistency') {
        await page.locator('.exp-predict button').first().click();
        await page.locator('.note-tool > button').last().click();
      }
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
      ).toBeLessThanOrEqual(1);
      expect(await page.locator('#note-reading').innerText()).not.toMatch(/[一-鿿]/);
    }
    await page.evaluate(() => {
      window.location.hash = '#rlhf';
    });
    await page
      .locator('.note-reading')
      .screenshot({
        path: `test-results/${info.project.name}-notebook-${locale}.png`,
        scale: 'css',
      });
    expect(errors).toEqual([]);
  });
