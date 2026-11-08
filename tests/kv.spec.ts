import { test, expect } from '@playwright/test';
import { goToExperiment } from './helpers/navigation';
import { initialKVState, serializeKVState } from '../src/state/kv';

test('缓存账本：原书例子、改 KV 头数与输入参数', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/kv-architectures');
  await expect(page.getByRole('heading', { name: 'KV 架构对照：缓存里到底存什么' })).toBeVisible();
  const result = page.locator('.kv-result');
  await expect(result).toContainText('= 512 MiB');

  const kvHeads = page.getByRole('group', { name: 'KV 头数', exact: true });
  await kvHeads.getByRole('button', { name: '8（GQA-8）' }).click();
  await expect(result).toContainText('= 128 MiB');
  await expect(result).toContainText('25.0%');
  await kvHeads.getByRole('button', { name: '1（MQA）' }).click();
  await expect(result).toContainText('= 16 MiB');

  await page.getByRole('button', { name: '原书第 22 章：Llama 2 7B、4096 token' }).click();
  await expect(result).toContainText('= 2 GiB');
  await expect(result).toContainText('每个 token 512 KiB');
  await page.getByRole('spinbutton', { name: '层数 L', exact: true }).fill('64');
  await expect(result).toContainText('= 4 GiB');

  // Changing the Query head count keeps the grouping valid.
  await page.getByRole('spinbutton', { name: 'Query 头数', exact: true }).fill('12');
  await expect(kvHeads.getByRole('button', { name: '12（MHA）' })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByRole('list', { name: '不同 KV 头数的 KV 缓存字节数' })).toContainText(
    'MQA',
  );
  expect(errors).toEqual([]);
});

test('共享 KV 头与 MLA：预测打分次数、还原 K、解耦 RoPE、矩阵吸收', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/kv-architectures');
  await page.getByRole('button', { name: '下一步：共享 KV 头 →', exact: true }).click();
  const table = page.getByRole('table', { name: 'KV 头数改变了什么', exact: true });
  await expect(table).toContainText('?');
  await page.getByRole('button', { name: '不变', exact: true }).click();
  await page.getByRole('button', { name: '核对打分次数 →', exact: true }).click();
  await expect(page.locator('.exp-widget .exp-verdict.ok')).toContainText('= 40 次');
  await expect(page.locator('.exp-widget .exp-verdict.ok')).toContainText('与计算一致');
  await expect(page.getByText(/输出最大差为 (0|\d(\.\d)?e-\d+)/)).toBeVisible();
  await page
    .getByRole('group', { name: 'KV 头数 G', exact: true })
    .getByRole('button', { name: '1（MQA）' })
    .click();
  await expect(page.getByRole('img', { name: /Q3 读第 0 组/ })).toBeVisible();

  await page.getByRole('button', { name: '下一步：MLA 压缩 →', exact: true }).click();
  await expect(
    page.getByRole('table', { name: '由 c_3 还原的各头 K、V', exact: true }).locator('tbody tr'),
  ).toHaveCount(4);
  await expect(page.locator('.exp-verdict.ok')).toContainText('MLA 存 3 + 2 = 5 个');
  const blocks = page.getByRole('group', { name: '一层里 4 个 token 的 KV 缓存块', exact: true });
  await expect(blocks.getByRole('img')).toHaveCount(4);
  await expect(blocks).toContainText('每 token 16 个 × 4 = 64');
  await expect(
    blocks.getByRole('img', { name: 'MLA：4 个 token × 5 个数 = 20 个数', exact: true }),
  ).toBeVisible();

  await page.getByRole('button', { name: '下一步：解耦 RoPE →', exact: true }).click();
  const scores = page.getByRole('table', { name: '精确分数与吸收后分数的对照', exact: true });
  await expect(scores).toContainText('?');
  await page.getByRole('button', { name: '只有 j = t', exact: true }).click();
  await page.getByRole('button', { name: '核对 →', exact: true }).click();
  await expect(page.locator('.exp-verdict.ok')).toContainText('误差为 0 的位置：t3');
  await expect(page.locator('.exp-verdict.ok')).toContainText('与计算一致');
  await page.getByRole('button', { name: '单独一支（解耦）', exact: true }).click();
  await expect(scores.locator('tbody tr td:last-child')).toHaveText([
    '≤ 1e-12',
    '≤ 1e-12',
    '≤ 1e-12',
    '≤ 1e-12',
  ]);

  await page.getByRole('button', { name: '下一步：矩阵吸收 →', exact: true }).click();
  await expect(page.locator('.exp-verdict.ok')).toContainText('两条路径的最大差为');
  await expect(page.locator('.exp-verdict.ok')).toContainText('容差 1e-12');
  expect(errors).toEqual([]);
});

test('对照：DeepSeek-V2 的头数与头维，MLA 的缓存包括 RoPE Key', async ({ page }) => {
  await page.goto('/kv-architectures');
  // On narrow screens the step number is hidden, so the button is named only “对照”.
  await page.getByRole('button', { name: /^(6\s*)?对照$/ }).click();
  await page
    .getByRole('group', { name: 'token 数', exact: true })
    .getByRole('button', { name: '128K' })
    .click();
  const table = page.getByRole('table', { name: '四种设计的 KV 缓存对照', exact: true });
  await expect(table).toContainText('480 GiB');
  await expect(table).toContainText('8.44 GiB');
  await expect(table).toContainText('576（512 + 64）');
  await expect(page.locator('.exp-verdict.ok')).toContainText('2.25 组 GQA');
  await expect(page.locator('.exp-verdict.ok')).toContainText('11.1%');
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(
    false,
  );
});

test('分享链接恢复、坏链接回退、键盘与目录导航', async ({ page }) => {
  const state = initialKVState();
  state.step = 3;
  state.mla = { head: 2, query: 1, coupled: false };
  state.view = { explanations: false, speed: 2 };
  await page.goto('/kv-architectures' + serializeKVState(state));
  await expect(page.locator('.kv-panel .lesson-progress')).toHaveText('4 / 6');
  await expect(page.getByRole('button', { name: '单独一支（解耦）' })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByText('先自己说一遍')).toBeVisible();

  await page.getByRole('button', { name: '头 3', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: '头 3', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );

  await page.goto('/kv-architectures#kv=%7Bbroken');
  await expect(page.getByRole('status').first()).toContainText('分享链接无效');
  await expect(page.locator('.kv-panel .lesson-progress')).toHaveText('1 / 6');

  await goToExperiment(page, '训练、生成与缓存');
  await expect(page).toHaveURL(/\/inference-cache$/);
  await goToExperiment(page, 'KV 架构对照');
  await expect(page).toHaveURL(/\/kv-architectures$/);
});
