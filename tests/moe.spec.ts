import { test, expect } from '@playwright/test';
import { goToExperiment } from './helpers/navigation';
import { initialMoEState, serializeMoEState } from '../src/state/moe';

test('路由、Top-K、专家干预与按 token 合并形成完整计算链', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/moe');
  await expect(
    page.getByRole('heading', { name: 'MoE：一个 token 怎样经过多个专家' }),
  ).toBeVisible();
  const router = page.getByRole('table', { name: 'Router 打分与概率' });
  const before = (await router.textContent()) ?? '';
  await page.getByRole('slider', { name: '输入 x0 偏移' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(router).not.toHaveText(before);
  await page.getByRole('button', { name: '恢复默认示例', exact: true }).click();
  await expect(router).toHaveText(before);

  await page.getByRole('button', { name: '下一步：Top-K 门控 →', exact: true }).click();
  await page.getByRole('button', { name: 'Top-1', exact: true }).click();
  const gates = page.getByRole('table', { name: 'Top-K 与归一化门控' });
  await expect(gates.locator('tbody tr td:last-child')).toHaveText(['1', '0', '0', '0']);
  await page.getByRole('button', { name: 'Top-2', exact: true }).click();
  await page.getByRole('button', { name: '下一步：专家计算 →', exact: true }).click();
  await expect(page.getByText('当前 E0 输出倍率为 1', { exact: false })).toBeVisible();
  await expect(page.getByRole('button', { name: '验证预测并执行 →' })).toBeDisabled();
  await page.getByRole('button', { name: '路由不变', exact: true }).click();
  await page.getByRole('button', { name: '验证预测并执行 →' }).click();
  await expect(page.locator('.exp-experiment [role="status"]')).toContainText('门控最大差 = 0');
  await expect(page.locator('.exp-experiment [role="status"]')).toContainText('与计算一致');
  await expect(page.locator('.exp-experiment [role="status"]')).not.toContainText(
    '输出最大差 = 0。',
  );
  await expect(page.getByText('当前 E0 输出倍率为 2', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: '下一步：加权合并 →', exact: true }).click();
  await expect(
    page.getByRole('table', { name: '按原 token 顺序返回的 Y', exact: true }).locator('tbody tr'),
  ).toHaveCount(4);
  await expect(page.locator('.lesson-reading .exp-verdict.ok')).toContainText('容差 1e-12');
  await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
  await expect(page.getByText('先自己说一遍')).toBeVisible();
  await expect(page.locator('.lesson-explanation')).toHaveCount(0);
  await page.getByText('公式与符号说明', { exact: true }).click();
  await expect(page.locator('.lesson-formula .katex-error')).toHaveCount(0);
  await page.screenshot({
    path: 'test-results/' + testInfo.project.name + '-moe-combine.png',
    fullPage: true,
    scale: 'css',
  });
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});

test('集中负载、参数账本、跨设备传输及单设备对照', async ({ page }, testInfo) => {
  await page.goto('/moe');
  await page.getByRole('button', { name: /^(5\s*)?负载与参数$/ }).click();
  const load = page.getByRole('list', { name: '四个专家的 token 分配负载' });
  await expect(load.locator('strong')).toHaveText(['2 次', '2 次', '2 次', '2 次']);
  const costs = page.getByRole('table', { name: 'Top-1 与 Top-2 参数和计算对照' });
  await expect(costs.locator('tbody tr').nth(0).locator('td')).toHaveText(['56', '56']);
  await expect(costs.locator('tbody tr').nth(1).locator('td')).toHaveText(['20', '32']);
  await page.getByRole('button', { name: '集中输入', exact: true }).click();
  await expect(load.locator('strong')).toHaveText(['4 次', '4 次', '0 次', '0 次']);
  await page.getByRole('button', { name: 'Top-1', exact: true }).click();
  await expect(load.locator('strong')).toHaveText(['4 次', '0 次', '0 次', '0 次']);
  await page.getByRole('button', { name: 'Top-2', exact: true }).click();
  await page.getByRole('button', { name: '下一步：专家并行 →', exact: true }).click();
  const traffic = page.getByLabel('跨设备通信账本', { exact: true });
  await expect(traffic).toContainText('64 B');
  await page.getByRole('button', { name: '单设备', exact: true }).click();
  await expect(traffic.locator('strong')).toHaveText(['0 份', '0 B', '0 B', '0 B']);
  await expect(page.locator('.lesson-reading .exp-verdict.ok')).toContainText('最大差 0');
  await page.getByRole('button', { name: '双设备专家并行', exact: true }).click();
  await expect(
    page.getByRole('table', { name: '专家分发与返回记录' }).locator('tbody tr'),
  ).toHaveCount(8);
  await page.getByRole('button', { name: 't3', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 't3', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await page.getByText('适用条件与来源', { exact: true }).click();
  await expect(page.getByText('来源核对日期：2026-11-07。图形与说明为本项目编写。')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  await page.screenshot({
    path: 'test-results/' + testInfo.project.name + '-moe-parallel.png',
    fullPage: true,
    scale: 'css',
  });
});

test('分享恢复、坏链接、回退、播放暂停与改输入暂停', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const state = initialMoEState();
  state.step = 2;
  state.input.token = 2;
  state.input.delta = -0.5;
  state.seed = 53;
  state.view = { explanations: false, speed: 2 };
  await page.goto('/moe' + serializeMoEState(state));
  await expect(page.locator('.lesson-progress')).toHaveText('3 / 6');
  await expect(page.getByRole('spinbutton', { name: '权重种子' })).toHaveValue('53');
  await expect(page.getByRole('slider', { name: '输入 x0 偏移' })).toHaveValue('-0.5');
  await expect(page.getByText('先自己说一遍')).toBeVisible();
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const url = await page.evaluate(() => navigator.clipboard.readText());
  expect(url).toContain('#moe=');
  await page.goto(url);
  await expect(page.locator('.lesson-progress')).toHaveText('3 / 6');
  await page.getByRole('button', { name: '← 上一步', exact: true }).click();
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('button', { name: '从头看', exact: true }).click();
  await page.clock.install();
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.clock.fastForward(4100);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('button', { name: '暂停自动逐步', exact: true }).click();
  await page.clock.fastForward(5000);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.getByRole('button', { name: '集中输入', exact: true }).click();
  await expect(page.getByRole('button', { name: '播放自动逐步', exact: true })).toBeVisible();
  await page.clock.fastForward(5000);
  await expect(page.locator('.lesson-progress')).toHaveText('2 / 6');
  await page.getByRole('button', { name: '恢复默认示例', exact: true }).click();
  await expect(page.locator('.lesson-progress')).toHaveText('1 / 6');
  await page.goto('/moe#moe=%7Bbroken');
  await expect(page.getByRole('status').first()).toContainText('分享链接无效');
  await expect(page.locator('.lesson-progress')).toHaveText('1 / 6');
  await goToExperiment(page, 'KV 架构对照');
  await expect(page).toHaveURL(/\/kv-architectures$/);
  await goToExperiment(page, 'MoE 与专家并行');
  await expect(page).toHaveURL(/\/moe$/);
});

test('预测控件暂停播放，恢复预设与外部链接不会残留旧核对结果', async ({ page }) => {
  const state = initialMoEState();
  state.step = 2;
  state.view.speed = 2;
  await page.goto('/moe' + serializeMoEState(state));
  await page.clock.install();
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.getByRole('button', { name: '路由不变', exact: true }).click();
  await expect(page.getByRole('button', { name: '播放自动逐步', exact: true })).toBeVisible();
  await page.clock.fastForward(5000);
  await expect(page.locator('.lesson-progress')).toHaveText('3 / 6');
  await page.getByRole('button', { name: '播放自动逐步', exact: true }).click();
  await page.getByRole('slider', { name: 'E0 待执行输出倍率' }).focus();
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByRole('button', { name: '播放自动逐步', exact: true })).toBeVisible();
  await expect(page.getByText('当前 E0 输出倍率为 1', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: '验证预测并执行 →' }).click();
  await expect(page.locator('.exp-experiment [role="status"]')).toContainText('与计算一致');
  await page.getByRole('button', { name: '分散输入', exact: true }).click();
  await expect(page.locator('.exp-experiment [role="status"]')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '验证预测并执行 →' })).toBeDisabled();
  await expect(page.getByRole('slider', { name: 'E0 待执行输出倍率' })).toHaveValue('2');
  await page.getByRole('button', { name: '路由不变', exact: true }).click();
  await page.getByRole('button', { name: '验证预测并执行 →' }).click();
  state.input.expertScale = 0;
  await page.evaluate((hash) => {
    location.hash = hash;
  }, serializeMoEState(state));
  await expect(page.getByText('当前 E0 输出倍率为 0', { exact: false })).toBeVisible();
  await expect(page.locator('.exp-experiment [role="status"]')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '验证预测并执行 →' })).toBeDisabled();
});
