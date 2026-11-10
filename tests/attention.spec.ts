import { test, expect } from '@playwright/test';
import { initialState, serializeState } from '../src/state/experiment';

test.beforeEach(async ({ page }) => {
  await page.goto('/attention');
  await expect(
    page.getByRole('heading', { name: 'Attention：信息怎样在词之间流动' }),
  ).toBeVisible();
  await page.getByRole('button', { name: '动手核对', exact: true }).click();
});

test('播放、暂停、回退、跳步、重置与条件变化', async ({ page }) => {
  await expect(page.getByRole('button', { name: '上一步', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: '下一步', exact: true }).click();
  await expect(page.getByRole('table', { name: 'S 矩阵', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '上一步', exact: true }).click();
  await expect(page.getByRole('table', { name: 'Q 矩阵', exact: true })).toBeVisible();
  await page.getByRole('combobox', { name: '播放速度' }).selectOption('2');
  await page.getByRole('button', { name: '播放', exact: true }).click();
  await expect(page.locator('[aria-current="step"]')).toContainText('打分');
  await page.getByRole('button', { name: '暂停', exact: true }).click();
  const paused = await page.locator('[aria-current="step"]').textContent();
  await page.waitForTimeout(900);
  expect(await page.locator('[aria-current="step"]').textContent()).toBe(paused);
  await page.getByRole('button', { name: /加权求和 O/ }).click();
  await expect(page.getByRole('button', { name: '下一步', exact: true })).toBeDisabled();
  await page.getByRole('switch', { name: /因果 Mask/ }).uncheck();
  await expect(page.locator('[aria-current="step"]')).toContainText('屏蔽');
  await expect(page.getByRole('button', { name: '播放', exact: true })).toBeVisible();
  await page.getByRole('switch', { name: /缩放 1/ }).uncheck();
  await expect(page.getByText(/当前为关闭缩放的对照/)).toBeVisible();
  await page.getByRole('button', { name: '恢复基础示例', exact: true }).click();
  await expect(page.getByRole('switch', { name: /因果 Mask/ })).toBeChecked();
  await expect(page.locator('[aria-current="step"]')).toContainText('投影');
});

test('预测后验证只改 V，并检查因果边界', async ({ page }) => {
  await page.getByRole('button', { name: '只改变 V', exact: false }).click();
  const probabilities = await page
    .getByRole('table', { name: 'P 矩阵', exact: true })
    .textContent();
  const output = await page.locator('.output-values').textContent();
  await page.getByRole('slider', { name: 'V[1,0] 的偏移 ΔV', exact: true }).fill('4');
  await expect(page.getByText('有待验证的改动；上方仍显示上一次计算结果。')).toBeVisible();
  expect(await page.locator('.output-values').textContent()).toBe(output);
  await page.getByRole('button', { name: 'P 不会变', exact: true }).click();
  await page.getByRole('button', { name: '验证预测 →', exact: true }).click();
  expect(await page.getByRole('table', { name: 'P 矩阵', exact: true }).textContent()).toBe(
    probabilities,
  );
  expect(await page.locator('.output-values').textContent()).not.toBe(output);
  await expect(page.getByText(/预测正确/)).toBeVisible();
  await page.getByRole('switch', { name: /因果 Mask/ }).check();
  await page.getByRole('button', { name: /^t0/ }).click();
  await page.getByRole('button', { name: '验证预测 →', exact: true }).click();
  await expect(page.getByText(/t1 被屏蔽，因此对当前输出没有影响/)).toBeVisible();
});

test('链接恢复、坏链接回退、输入重算、复述和窄屏', async ({ page }, testInfo) => {
  const state = initialState();
  state.view.lesson = 'numbers';
  state.step = 4;
  state.config.causal = true;
  state.input.valueDelta = 3;
  state.view.query = 1;
  state.view.explanations = false;
  await page.goto('/attention' + serializeState(state));
  await expect(page.getByRole('button', { name: '显示说明', exact: true })).toBeVisible();
  await expect(page.locator('.output-caption')).toContainText('O[1]');
  await expect(page.getByRole('switch', { name: /因果 Mask/ })).toBeChecked();
  await page.getByText('输入与参数', { exact: true }).click();
  await page.getByRole('spinbutton', { name: 'X[0,0]', exact: true }).fill('2');
  await expect(page.locator('[aria-current="step"]')).toContainText('投影');
  await page.getByRole('button', { name: '查看参考解释', exact: true }).click();
  await expect(page.locator('.recall-answer')).toBeVisible();
  await page.goto('/attention#lab=%invalid');
  await expect(page.getByText('分享链接无效或版本不受支持，已恢复基础示例。')).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  if (testInfo.project.name === 'mobile') {
    await page.getByRole('button', { name: '拆开 Attention', exact: true }).click();
    await expect(page.getByText(/已遵循减少动态效果偏好/)).toBeVisible();
  }
  await page.goto('/attention');
  await expect(
    page.getByRole('heading', { name: 'Attention：信息怎样在词之间流动' }),
  ).toBeVisible();
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-attention.png`,
    fullPage: false,
    scale: 'css',
  });
});

test('分享生成可恢复链接，键盘可操作且无页面异常', async ({ page, context }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.getByRole('button', { name: '相同分数', exact: false }).click();
  await page.getByRole('button', { name: '复制实验链接', exact: false }).click();
  await expect(page.getByText('已复制当前已计算状态的链接。')).toBeVisible();
  const link = await page.evaluate(() => navigator.clipboard.readText());
  await page.goto(link);
  await expect(page.getByRole('button', { name: 'P[2,1] = 0.333', exact: true })).toBeVisible();
  const next = page.getByRole('button', { name: '下一步', exact: true });
  await next.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('.output-values')).toContainText('0');
  expect(errors).toEqual([]);
});
