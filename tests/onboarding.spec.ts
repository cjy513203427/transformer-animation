import { test, expect, type Locator, type Page } from '@playwright/test';
import { initialState, restoreState, serializeState } from '../src/state/experiment';

const regionName = 'Attention 操作引导';
const sliderName = '引导：V[1,0] 的偏移 ΔV';
// In the basic example, Query t1 has visible scores [0, 1/√2].
// This independent, hand-computable expectation does not call the UI's Attention implementation.
const selfWeight = 1 / (1 + Math.exp(-1 / Math.sqrt(2)));

function vectorFrom(text: string | null): number[] {
  const groups = [...(text ?? '').matchAll(/\[([^\[\]]+)\]/g)];
  const vector = groups.at(-1)?.[1];
  expect(vector, 'The result exposes a numeric vector').toBeDefined();
  return vector!.split(',').map((value) => Number(value.trim()));
}

function expectVector(actual: number[], expected: number[]) {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, i) => expect(value).toBeCloseTo(expected[i], 3));
}

async function startGuide(page: Page) {
  await page.getByRole('button', { name: '跟着做一次', exact: true }).click();
  const guide = page.getByRole('region', { name: regionName, exact: true });
  await expect(guide).toBeVisible();
  await expect(guide.locator('.guided-progress-count')).toHaveText('1 / 4');
  return guide;
}

async function chooseQueryAndPrediction(guide: Locator, prediction = 'P 不会变') {
  await guide.getByRole('button', { name: '观察“喜欢”', exact: true }).click();
  await expect(guide.locator('.guided-progress-count')).toHaveText('2 / 4');
  await guide.getByRole('button', { name: prediction, exact: true }).click();
  await expect(guide.locator('.guided-progress-count')).toHaveText('3 / 4');
}

test('实际操作完成短任务，权重不变而输出改变，并保留结果进入探索', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/attention');
  await expect(page.getByRole('region', { name: regionName, exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '自由探索', exact: true })).toBeVisible();
  const guide = await startGuide(page);
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-attention-onboarding-start.png`,
    fullPage: false,
    scale: 'css',
  });

  // The task advances only after the requested action; another token is still selectable.
  await guide.getByRole('button', { name: '观察“学习”', exact: true }).click();
  await expect(guide.locator('.guided-progress-count')).toHaveText('1 / 4');
  await chooseQueryAndPrediction(guide);
  const verify = guide.getByRole('button', { name: '验证预测', exact: true });
  await expect(verify).toBeDisabled();
  await guide.getByRole('slider', { name: sliderName, exact: true }).fill('3');
  await expect(guide.locator('.guided-progress-count')).toHaveText('3 / 4');
  await verify.click();
  await expect(guide.locator('.guided-progress-count')).toHaveText('4 / 4');

  const probabilitiesBefore = guide.getByTestId('guided-probability-before');
  const probabilitiesAfter = guide.getByTestId('guided-probability-after');
  expect(await probabilitiesAfter.textContent()).toBe(await probabilitiesBefore.textContent());
  expectVector(vectorFrom(await probabilitiesAfter.textContent()), [1 - selfWeight, selfWeight, 0]);
  expectVector(vectorFrom(await guide.getByTestId('guided-output-before').textContent()), [
    1 + selfWeight,
    2 * (1 - selfWeight),
  ]);
  const outputAfter = vectorFrom(await guide.getByTestId('guided-output-after').textContent());
  expectVector(outputAfter, [1 + 4 * selfWeight, 2 * (1 - selfWeight)]);
  await expect(guide.getByLabel('本式符号说明')).toContainText('Δ 表示改动前后的差值');
  await expect(guide.getByTestId('guided-applied-delta')).toHaveText('3');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  await page.screenshot({
    path: `test-results/${testInfo.project.name}-attention-onboarding.png`,
    fullPage: true,
    scale: 'css',
  });

  await guide.getByRole('button', { name: '保留结果，自由探索', exact: true }).click();
  await expect(guide).toHaveCount(0);
  await expect(page.getByRole('table', { name: 'P 矩阵', exact: true })).toBeVisible();
  await expect(page.locator('.output-caption')).toContainText('O[1]');
  await expect(page.getByRole('slider', { name: 'V[1,0] 的偏移 ΔV', exact: true })).toHaveValue(
    '3',
  );
  const freeOutput = await page.locator('.output-values strong').allTextContents();
  expectVector(freeOutput.map(Number), outputAfter);
  expect(errors).toEqual([]);
});

test('回退可修改预测，退出保留实验，重走恢复教学示例', async ({ page }) => {
  await page.goto('/attention');
  const guide = await startGuide(page);
  await chooseQueryAndPrediction(guide, 'P 会变');
  await guide.getByRole('slider', { name: sliderName, exact: true }).fill('2');
  await guide.getByRole('button', { name: '验证预测', exact: true }).click();
  await expect(guide.locator('.guided-progress-count')).toHaveText('4 / 4');
  await expect(guide.getByText(/实际结果和你的预测不同/)).toBeVisible();

  await guide.getByRole('button', { name: '← 上一步', exact: true }).click();
  await expect(guide.locator('.guided-progress-count')).toHaveText('3 / 4');
  await expect(guide.getByText(/实际结果和你的预测不同/)).toHaveCount(0);
  await guide.getByRole('button', { name: '← 上一步', exact: true }).click();
  await expect(guide.locator('.guided-progress-count')).toHaveText('2 / 4');
  await guide.getByRole('button', { name: 'P 不会变', exact: true }).click();
  await guide.getByRole('slider', { name: sliderName, exact: true }).fill('-2');
  await guide.getByRole('button', { name: '验证预测', exact: true }).click();
  await expect(guide.locator('.guided-progress-count')).toHaveText('4 / 4');
  await expect(guide.getByText(/预测正确/)).toBeVisible();
  await expect(guide.getByText(/实际结果和你的预测不同/)).toHaveCount(0);
  expectVector(vectorFrom(await guide.getByTestId('guided-output-after').textContent()), [
    1 - selfWeight,
    2 * (1 - selfWeight),
  ]);
  await expect(guide.getByTestId('guided-applied-delta')).toHaveText('-2');

  await guide.getByRole('button', { name: '退出引导，自由探索', exact: true }).click();
  await expect(guide).toHaveCount(0);
  await expect(page.getByRole('slider', { name: 'V[1,0] 的偏移 ΔV', exact: true })).toHaveValue(
    '-2',
  );

  const restarted = await startGuide(page);
  await chooseQueryAndPrediction(restarted);
  await expect(restarted.getByRole('slider', { name: sliderName, exact: true })).toHaveValue('0');
  await restarted.getByRole('button', { name: '从头再做', exact: true }).click();
  await expect(restarted.locator('.guided-progress-count')).toHaveText('1 / 4');
  await chooseQueryAndPrediction(restarted);
  await expect(restarted.getByRole('slider', { name: sliderName, exact: true })).toHaveValue('0');
});

test('分享链接直接恢复实验，草稿不冒充已计算状态，hash 切换会退出引导', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const shared = initialState();
  shared.view.lesson = 'numbers';
  shared.view.query = 2;
  shared.step = 4;
  shared.input.valueDelta = -1.5;
  await page.goto('/attention' + serializeState(shared));
  const guide = page.getByRole('region', { name: regionName, exact: true });
  await expect(guide).toHaveCount(0);
  await expect(page.locator('.output-caption')).toContainText('O[2]');
  await expect(page.getByRole('slider', { name: 'V[1,0] 的偏移 ΔV', exact: true })).toHaveValue(
    '-1.5',
  );

  await startGuide(page);
  await chooseQueryAndPrediction(guide);
  await guide.getByRole('slider', { name: sliderName, exact: true }).fill('4');
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const pendingUrl = await page.evaluate(() => navigator.clipboard.readText());
  const pendingState = restoreState(new URL(pendingUrl).hash);
  expect(pendingState.warning).toBeNull();
  expect(pendingState.state.input.valueDelta).toBe(0);

  const next = initialState();
  next.view.lesson = 'guide';
  next.step = 3;
  next.view.query = 0;
  await page.evaluate((hash) => {
    window.location.hash = hash;
  }, serializeState(next));
  await expect(guide).toHaveCount(0);
  const lesson = page.getByRole('region', { name: '跟图理解 Attention', exact: true });
  await expect(lesson).toBeVisible();
  await expect(lesson.locator('.lesson-progress')).toHaveText('4 / 5');
  await expect(lesson.getByRole('button', { name: 't0 我 正在观察', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
});

test('键盘可完成主要任务，本地存储不可用时仍能进入和退出引导', async ({ page }) => {
  await page.addInitScript(() => {
    Storage.prototype.getItem = () => {
      throw new DOMException('Storage unavailable', 'SecurityError');
    };
    Storage.prototype.setItem = () => {
      throw new DOMException('Storage unavailable', 'SecurityError');
    };
  });
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/attention');
  await page.getByRole('button', { name: '自由探索', exact: true }).click();
  await expect(page.getByRole('region', { name: regionName, exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: '跟着做一次', exact: true }).focus();
  await page.keyboard.press('Enter');
  const guide = page.getByRole('region', { name: regionName, exact: true });
  await expect(guide).toBeVisible();
  await guide.getByRole('button', { name: '观察“喜欢”', exact: true }).focus();
  await page.keyboard.press('Enter');
  await guide.getByRole('button', { name: 'P 不会变', exact: true }).focus();
  await page.keyboard.press('Enter');
  const slider = guide.getByRole('slider', { name: sliderName, exact: true });
  await slider.focus();
  await page.keyboard.press('ArrowRight');
  await expect(slider).toHaveValue('0.5');
  await guide.getByRole('button', { name: '验证预测', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(guide.locator('.guided-progress-count')).toHaveText('4 / 4');
  await guide.getByRole('button', { name: '保留结果，自由探索', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(guide).toHaveCount(0);
  await expect(page.getByRole('slider', { name: 'V[1,0] 的偏移 ΔV', exact: true })).toHaveValue(
    '0.5',
  );
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});

test('探索入口明确标出可点图和矩阵，编辑输入会实际重算', async ({ page }) => {
  await page.goto('/attention');
  await expect(page.getByText('点选模块，查看它的输入与输出', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '拆开 Attention', exact: true }).click();
  await expect(page.getByText('点选计算节点，跳到对应步骤', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '动手核对', exact: true }).click();
  for (const name of ['Q', 'K', 'V']) {
    const table = page.getByRole('table', { name: `${name} 矩阵`, exact: true });
    await expect(table).toBeVisible();
    await expect(table.getByRole('button')).toHaveCount(0);
  }

  await page.getByRole('button', { name: /打分 S/ }).click();
  await expect(page.getByText('点选格子，核对这一项', { exact: true })).toBeVisible();
  const score = page.getByRole('button', { name: 'S[0,1] = 0.354', exact: true });
  await score.focus();
  await page.keyboard.press('Enter');
  await expect(score).toHaveAttribute('aria-pressed', 'true');

  await page.getByRole('button', { name: /归一化 P/ }).click();
  await expect(page.getByText('点选格子，核对这一项', { exact: true })).toBeVisible();
  const probability = page.getByRole('button', { name: 'P[1,0] = 0.33', exact: true });
  await probability.focus();
  await page.keyboard.press('Enter');
  await expect(probability).toHaveAttribute('aria-pressed', 'true');

  await page.getByRole('button', { name: '修改 X / K ↓', exact: true }).click();
  const input = page.getByRole('spinbutton', { name: 'X[0,0]', exact: true });
  await expect(input).toBeVisible();
  await expect(input).toBeFocused();
  await input.fill('2');
  const query = page.getByRole('table', { name: 'Q 矩阵', exact: true });
  await expect(query).toBeVisible();
  await expect(query.locator('tbody tr').first().locator('td').first()).toHaveText('2');
  await page.getByRole('button', { name: /打分 S/ }).click();
  await expect(page.getByRole('button', { name: 'S[0,1] = 0.707', exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);

  await page.goto('/attention' + serializeState(initialState()));
  await expect(page.getByRole('region', { name: regionName, exact: true })).toHaveCount(0);
  await expect(page.getByText('点选模块，查看它的输入与输出', { exact: true })).toBeVisible();
});
