import { expect, test } from '@playwright/test';

test('第 8 章二维线性变换：预设、点积预测、分享恢复与英德无溢出', async ({ page, context }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);

  await page.goto('/linear-transform');
  await expect(page.locator('.lesson-progress')).toHaveText('1 / 5');
  await page.getByRole('button', { name: '降维压扁 (det=0)', exact: true }).click();
  await page.locator('.lin-steps button').nth(1).click();
  await expect(page.locator('.lin-metrics')).toContainText('降维压扁（det = 0）');

  await page.getByRole('button', { name: '水平剪切', exact: true }).click();
  await page.locator('.lin-steps button').nth(2).click();
  await page.getByRole('button', { name: '点积变 2 倍，余弦不变', exact: true }).click();
  await page.getByRole('button', { name: '计算对照 2k →', exact: true }).click();
  await expect(page.locator('.exp-experiment [role="status"]')).toContainText('与计算一致');

  await page.locator('.lin-steps button').nth(3).click();
  await expect(page.locator('.exp-verdict')).toContainText('结合律核对');

  await page.getByRole('button', { name: /复制实验链接/ }).click();
  const url = await page.evaluate(() => navigator.clipboard.readText());
  expect(url).toContain('#lin=');
  await page.goto('about:blank');
  await page.goto(url);
  await expect(page.locator('.lesson-progress')).toHaveText('4 / 5');

  for (const locale of ['en', 'de']) {
    await page.goto(`/${locale}/linear-transform`);
    await expect(page.locator('.lesson-progress')).toHaveText('1 / 5');
    expect(await page.locator('.lesson-reading').innerText()).not.toMatch(/[一-鿿]/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  }
  expect(errors).toEqual([]);
});

test('第 18–20 章联动 PyTorch 代码随步骤高亮更新', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/forward-pass');
  const modelCode = page.getByTestId('linked-code-model.py');
  await modelCode.locator('summary').click();
  await expect(modelCode.locator('.code-line-active')).toHaveCount(1);
  await page.getByRole('button', { name: '下一步', exact: true }).click();
  await expect(modelCode.locator('summary')).toContainText('高亮第 3 行');

  await page.goto('/training-step');
  const trainCode = page.getByTestId('linked-code-train.py');
  await trainCode.locator('summary').click();
  await expect(trainCode.locator('summary')).toContainText('高亮第 1–2 行');
  await page.locator('.p1-steps button').nth(3).click();
  await expect(trainCode.locator('summary')).toContainText('高亮第 5 行');

  await page.goto('/inference-cache');
  const infCode = page.getByTestId('linked-code-inference.py');
  await infCode.locator('summary').click();
  await expect(infCode.locator('summary')).toContainText('高亮第 1–2 行');
  await page.locator('.inf-steps button').nth(3).click();
  await expect(infCode.locator('summary')).toContainText('高亮第 13–15 行');
  expect(errors).toEqual([]);
});

test('第 27 章非对称量化、AWQ 与 GPTQ 预设及第 30 章共享专家与容量丢弃可交互验证', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/quantization');
  await page.getByRole('button', { name: '偏态正值（非对称）', exact: true }).click();
  await expect(page.locator('.p1-metrics')).toContainText('第 0 组零点 z');
  await page.getByRole('button', { name: 'GPTQ 误差补偿', exact: true }).click();
  await page.locator('.p1-steps button').nth(4).click();
  await expect(page.locator('.lesson-reading')).toContainText('GPTQ 累积补偿 Δw');
  await page.getByText('数值性质核对', { exact: false }).click();
  await expect(page.locator('.p1-checks')).not.toContainText('超出容差');

  await page.goto('/moe');
  await page.getByRole('button', { name: '集中输入', exact: true }).click();
  await page.getByRole('button', { name: '开启（+1 共享）', exact: true }).click();
  await page.getByRole('button', { name: '2', exact: true }).click();
  await page.locator('.moe-steps button').nth(4).click();
  await expect(page.locator('.moe-ledger')).toContainText('4 份');
  await expect(page.locator('.moe-ledger')).toContainText('辅助损失 L_aux');
  expect(errors).toEqual([]);
});
