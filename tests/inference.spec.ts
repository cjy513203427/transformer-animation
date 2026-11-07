import { test, expect } from '@playwright/test';
import { initialInferenceState, serializeInferenceState } from '../src/state/inference';

test('训练并行：反事实只改输入，之前的行逐位不变', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/inference-cache');
  await expect(
    page.getByRole('heading', { name: '训练、生成与缓存：哪些结果可以复用' }),
  ).toBeVisible();
  const table = page.getByRole('table', { name: '每个位置的预测与 loss', exact: true });
  await expect(table.locator('tbody tr')).toHaveCount(4);
  await expect(table).toContainText('a b c d');

  await page.getByRole('button', { name: '被改的位置及之后', exact: true }).click();
  await page.getByRole('button', { name: '应用改动并核对 →', exact: true }).click();
  const verdict = page.locator('.exp-experiment .exp-verdict');
  await expect(verdict).toContainText('输出分布逐位不变的行：0、1；发生变化的行：2、3');
  await expect(verdict).toContainText('与计算一致');
  await expect(
    page.getByRole('table', { name: '训练时的因果 Mask：行是输出位置，列是输入', exact: true }),
  ).toContainText('已变');

  // Editing the counterfactual withdraws the applied result until it is checked again.
  await page
    .getByRole('group', { name: '改哪个输入', exact: true })
    .getByRole('button', { name: 'x0' })
    .click();
  await expect(verdict).toHaveCount(0);
  await page.getByRole('button', { name: '应用改动并核对 →', exact: true }).click();
  await expect(verdict).toContainText('发生变化的行：0、1、2、3');
  expect(errors).toEqual([]);
});

test('无缓存基线、Prefill、Decode 与缓存对照', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/inference-cache');
  await page.getByRole('button', { name: '下一步：无缓存基线 →', exact: true }).click();
  const baseline = page.getByRole('table', { name: '不用缓存时每一步的工作', exact: true });
  await expect(baseline.locator('tbody tr')).toHaveCount(4);
  await expect(page.locator('.exp-widget > .exp-verdict')).toContainText('最大差为 0');
  await page.getByRole('slider', { name: '生成步数', exact: true }).fill('5');
  await expect(baseline.locator('tbody tr')).toHaveCount(6);

  await page.getByRole('button', { name: '下一步：Prefill →', exact: true }).click();
  await expect(
    page.getByRole('table', { name: '第 2 层写入缓存的 K、V', exact: true }).locator('tbody tr'),
  ).toHaveCount(3);
  await expect(page.locator('.exp-widget > .exp-verdict')).toContainText('第一个新 token 是 c');

  await page.getByRole('button', { name: '下一步：Decode 一步 →', exact: true }).click();
  await page.getByRole('slider', { name: '看第几次 Decode', exact: true }).fill('1');
  const layer = page.getByRole('table', { name: '第 1 层：新 Query 读缓存', exact: true });
  await expect(layer.locator('tbody tr')).toHaveCount(4);
  await expect(layer.locator('tbody tr').last()).toContainText('本步新算');
  await expect(page.locator('.exp-widget > .exp-verdict')).toContainText(
    '新算 1 组 Q/K/V，从缓存读 3 组 K/V，打分 4 次',
  );

  // In-scene playback advances the Decode step and stops at the last one.
  await page.getByRole('button', { name: '播放逐步 Decode', exact: true }).click();
  await expect(page.locator('.exp-range output')).toHaveText('3', { timeout: 6000 });
  await page.getByRole('button', { name: '暂停逐步 Decode', exact: true }).click();
  const paused = await page.locator('.exp-range output').textContent();
  await page.waitForTimeout(1800);
  await expect(page.locator('.exp-range output')).toHaveText(paused ?? '');

  await page.getByRole('button', { name: '下一步：缓存对照重算 →', exact: true }).click();
  await page.getByRole('slider', { name: 'Decode 步数', exact: true }).fill('3');
  const compare = page.getByRole('table', { name: '缓存与重算的工作量对照', exact: true });
  const total = compare.locator('tr.exp-total');
  // 2 layers: recompute 8 + 10 + 12 projections and 20 + 30 + 42 scores; cache 2 + 2 + 2 and 8 + 10 + 12.
  await expect(total).toContainText('30');
  await expect(total).toContainText('92');
  await expect(page.locator('.exp-widget > .exp-verdict')).toContainText(
    '选出的 token 完全相同，logits 最大差为 0',
  );
  expect(errors).toEqual([]);
});

test('缓存失效：预测后核对，滑动窗口与键盘操作', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/inference-cache');
  await page.getByRole('button', { name: /缓存失效$/ }).click();
  const grid = page.getByRole('table', { name: '改动前缀后的缓存有效性', exact: true });
  await expect(grid).toContainText('?');
  await expect(grid).not.toContainText('失效');

  await page.getByRole('button', { name: '被改的位置及之后', exact: true }).click();
  await page.getByRole('button', { name: '重算并核对 →', exact: true }).click();
  const verdict = page.locator('.exp-widget > .exp-verdict');
  await expect(verdict).toContainText('第 1 层失效的位置：1；第 2 层失效的位置：1、2、3、4、5');
  await expect(verdict).toContainText('与计算一致');
  await expect(grid).toContainText('失效');

  // Any edit hides the old answer again.
  await page
    .getByRole('group', { name: '改哪个位置', exact: true })
    .getByRole('button', { name: 'pos 2' })
    .click();
  await expect(grid).toContainText('?');
  await page.getByRole('button', { name: '重算并核对 →', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(verdict).toContainText('第 1 层失效的位置：2；第 2 层失效的位置：2、3、4、5');

  await page.getByRole('button', { name: '滑动窗口', exact: true }).click();
  await page.getByRole('button', { name: '重算并核对 →', exact: true }).click();
  const slide = page.getByRole('table', { name: '滑动窗口后的缓存有效性', exact: true });
  await expect(slide).toContainText('丢弃');
  await expect(verdict).toContainText('第 1 层失效的位置：1、2、3、4、5');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(
    false,
  );
  expect(errors).toEqual([]);
});

test('分享链接恢复、坏链接回退与目录导航', async ({ page }) => {
  const state = initialInferenceState();
  state.step = 3;
  state.decode = { steps: 4 };
  state.view = { explanations: false, speed: 2 };
  await page.goto('/inference-cache' + serializeInferenceState(state));
  await expect(page.locator('.inf-panel .lesson-progress')).toHaveText('4 / 6');
  await expect(page.locator('.exp-range output')).toHaveText('4');
  await expect(page.getByText('先自己说一遍')).toBeVisible();
  await expect(page.getByRole('button', { name: '显示说明', exact: true })).toBeVisible();

  await page.goto('/inference-cache#inf=%7Bbroken');
  await expect(page.getByRole('status').first()).toContainText('分享链接无效');
  await expect(page.locator('.inf-panel .lesson-progress')).toHaveText('1 / 6');

  await page
    .getByRole('link', { name: /位置编码/ })
    .first()
    .click();
  await expect(page).toHaveURL(/\/position$/);
  await page
    .getByRole('link', { name: /训练、生成与缓存/ })
    .first()
    .click();
  await expect(page).toHaveURL(/\/inference-cache$/);
});
