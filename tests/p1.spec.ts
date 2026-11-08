import { expect, test } from '@playwright/test';
const topics = [
  { id: 'normalization', path: '/normalization', prediction: '会改变' },
  { id: 'ffn', path: '/ffn-residual', prediction: '会改变' },
  { id: 'lora', path: '/lora', prediction: '会改变' },
  { id: 'quantization', path: '/quantization', prediction: '会改变' },
  { id: 'decoding', path: '/decoding', prediction: '会改变' },
  { id: 'preference', path: '/preference', prediction: '容差内不变' },
  { id: 'memory', path: '/attention-variants', prediction: '会改变' },
  { id: 'ssm', path: '/ssm-mamba', prediction: '会改变' },
];
test('Top-P=1保留极小正概率，显示与零屏蔽可区分', async ({ page }) => {
  await page.goto('/decoding');
  await page.getByText('编辑输入向量', { exact: false }).click();
  for (const [i, value] of [8, 0, -8].entries())
    await page.getByRole('spinbutton', { name: `输入 x${i}`, exact: true }).fill(String(value));
  await page.getByRole('slider', { name: '温度 τ', exact: true }).focus();
  await page.keyboard.press('Home');
  await page.locator('.p1-steps button').nth(3).click();
  await expect(page.locator('.p1-metrics')).toContainText('a、b、c');
  await expect(
    page.getByRole('table', { name: '筛选并重新归一化', exact: true }).locator('tbody td').nth(2),
  ).toContainText('e-28');
});
for (const topic of topics)
  test(`${topic.id} 六步、预测、输入编辑与分享恢复`, async ({ page, context }, info) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.goto(topic.path);
    await expect(page.locator('.lesson-progress')).toHaveText('1 / 6');
    const firstTable = page.locator('.lesson-reading table').first(),
      before = await firstTable.textContent();
    await page.getByText('编辑输入向量', { exact: false }).click();
    await page.getByRole('spinbutton', { name: '输入 x0', exact: true }).fill('0.75');
    await expect(firstTable).not.toHaveText(before ?? '');
    await page.getByRole('button', { name: '恢复默认示例', exact: true }).click();
    await expect(firstTable).toHaveText(before ?? '');
    for (let i = 0; i < 6; i++) {
      await page.locator('.p1-steps button').nth(i).click();
      await expect(page.locator('.lesson-progress')).toHaveText(`${i + 1} / 6`);
      await expect(page.locator('.lesson-reading table').first()).toBeVisible();
      await expect(page.locator('.diagram-panel svg')).toBeVisible();
      await page.getByText('需要时再看公式', { exact: true }).click();
      await expect(page.locator('.katex-error')).toHaveCount(0);
      await page.getByText('需要时再看公式', { exact: true }).click();
      expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(
        false,
      );
    }
    await expect(
      page.getByRole('button', { name: '验证预测并执行 →', exact: true }),
    ).toBeDisabled();
    await page.getByRole('button', { name: topic.prediction, exact: true }).click();
    await page.getByRole('button', { name: '验证预测并执行 →', exact: true }).click();
    await expect(page.locator('.exp-experiment [role="status"]')).toContainText('与计算一致');
    await page.getByRole('button', { name: '复制实验链接', exact: false }).click();
    const url = await page.evaluate(() => navigator.clipboard.readText());
    expect(url).toContain('#lab=');
    const shared = JSON.parse(decodeURIComponent(new URL(url).hash.slice(5)));
    expect(shared.topic).toBe(topic.id);
    expect(shared.step).toBe(5);
    await page.goto(url);
    await expect(page.locator('.lesson-progress')).toHaveText('6 / 6');
    await expect(page.locator('.exp-experiment [role="status"]')).toHaveCount(0);
    await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
    await expect(page.getByText('先自己说一遍', { exact: true })).toBeVisible();
    await expect(page.locator('.lesson-explanation')).toHaveCount(0);
    await page.getByText('数值性质核对', { exact: false }).click();
    await expect(page.locator('.p1-checks')).not.toContainText('超出容差');
    await page.getByRole('button', { name: '查看参考解释', exact: true }).click();
    await expect(page.locator('.recall-answer')).toBeVisible();
    await page.getByText('适用条件与来源', { exact: true }).click();
    await expect(
      page.getByText('来源核对日期：2026-11-08。解释、图形和教学数值由本项目编写。'),
    ).toBeVisible();
    if (info.project.name === 'mobile') await expect(page.locator('.lesson-motion')).toBeVisible();
    await page.screenshot({
      path: `test-results/${info.project.name}-p1-${topic.id}.png`,
      fullPage: true,
      scale: 'css',
    });
    expect(errors).toEqual([]);
  });
test('播放、参数编辑、坏链接与触摸键盘', async ({ page }, info) => {
  await page.goto('/normalization');
  await page.clock.install();
  await page.getByRole('combobox', { name: '播放速度', exact: true }).selectOption('2');
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.clock.fastForward(4100);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('button', { name: '暂停自动逐步', exact: true }).click();
  await page.clock.fastForward(5000);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.getByRole('combobox', { name: 'ε', exact: true }).selectOption('1');
  await expect(page.getByRole('button', { name: '播放自动逐步', exact: true })).toBeVisible();
  await page.clock.fastForward(5000);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.locator('.p1-steps button').nth(4).click();
  await page.getByRole('combobox', { name: '候选屏蔽', exact: true }).selectOption('2');
  const p = page.getByRole('table', { name: '概率 P', exact: true });
  await expect(p.locator('tbody td')).toHaveText(['0', '0', '0', '0']);
  await expect(page.locator('.lesson-reading')).toContainText('没有合法分布');
  await expect(page.locator('.p1-panel')).not.toContainText('NaN');
  await page.goto('/normalization#lab=%ZZ');
  await expect(page.getByRole('status').first()).toContainText('分享链接无效');
  await expect(page.locator('.lesson-progress')).toHaveText('1 / 6');
  const next = page.locator('.p1-steps button').nth(2);
  if (info.project.name === 'mobile') await next.tap();
  else {
    await next.focus();
    await page.keyboard.press('Enter');
  }
  await expect(page.locator('.lesson-progress')).toHaveText('3 / 6');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
});
test('专门的二维图随参数实时重算：Top-P 截断、PPO 裁剪、SSM 状态链', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/decoding');
  await page.locator('.p1-steps button').nth(3).click();
  const ranking = page.getByRole('img', { name: /^按概率排序/ });
  await expect(ranking).toHaveAccessibleName(/c 0\.14 → 0\.14/);
  const topP = page.getByRole('slider', { name: 'Top-P', exact: true });
  await topP.focus();
  for (let i = 0; i < 8; i++) await page.keyboard.press('ArrowLeft');
  await expect(topP).toHaveValue('0.6');
  await expect(ranking).toHaveAccessibleName(/a 0\.629 → 1，b 0\.231 → 删去，c 0\.14 → 删去/);
  await page.locator('.p1-steps button').nth(4).click();
  await expect(page.getByRole('img', { name: /选中 a$/ })).toBeVisible();

  await page.goto('/preference');
  await page.locator('.p1-steps button').nth(2).click();
  const ppo = page.getByRole('img', { name: /^单项目标随概率比 ρ变化的图：PPO 目标、不裁剪 ρA/ });
  await expect(ppo).toHaveAccessibleName(/A = 1 > 0/);
  await page.getByRole('slider', { name: '单项 PPO 优势 A', exact: true }).focus();
  await page.keyboard.press('Home');
  await expect(ppo).toHaveAccessibleName(/A = -2 < 0/);
  await page.locator('.p1-steps button').nth(3).click();
  await expect(page.getByRole('img', { name: /^数轴：组内奖励 r，标准化优势 A/ })).toBeVisible();

  await page.goto('/ssm-mamba');
  await page.locator('.p1-steps button').nth(3).click();
  const chain = page.getByRole('img', { name: /^输入相关：初始 1/ });
  await expect(chain).not.toHaveAccessibleName(/待计算/);
  await page.getByRole('slider', { name: '查看时间步 t', exact: true }).focus();
  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('ArrowLeft');
  await expect(chain).toHaveAccessibleName(/t1 保留 .*；t2 待计算；t3 待计算$/);
  await page.locator('.p1-steps button').nth(5).click();
  await expect(page.getByRole('img', { name: /^历史 A：初始 1/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});
