import { test, expect, type Page } from '@playwright/test';

async function layoutAndLanguage(page: Page, locale: 'en' | 'de') {
  await expect(page.locator('html')).toHaveAttribute('lang', locale);
  await expect(page.locator('.katex-error')).toHaveCount(0);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth - innerWidth),
  ).toBeLessThanOrEqual(1);
  const problems = await page.evaluate(() => {
    const found: string[] = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (
        !n.parentElement?.closest('[lang="zh-CN"],script,style') &&
        /[　-〿一-鿿＀-￯]/.test(n.textContent ?? '')
      )
        found.push((n.textContent ?? '').trim().slice(0, 70));
    }
    for (const el of document.querySelectorAll('main *')) {
      if (
        el.closest('.katex,svg') ||
        !el.getClientRects().length ||
        ['auto', 'scroll'].includes(getComputedStyle(el).overflowX)
      )
        continue;
      if (
        [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent?.trim()) &&
        el.clientWidth > 0 &&
        el.scrollWidth > el.clientWidth + 2
      )
        found.push('overflow: ' + el.textContent?.trim().slice(0, 70));
    }
    return found;
  });
  expect(problems).toEqual([]);
}

test('旧章节索引链接仍可使用搜索、缺口筛选和锚点', async ({ page }) => {
  await page.goto('/chapters');
  await expect(page).toHaveURL(/\/chapters$/);
  await expect(page.locator('.chapter-entry')).toHaveCount(35);
  const search = page.getByRole('searchbox', { name: '查找章节或概念' });
  await search.fill('Mamba');
  await expect(page.locator('.chapter-entry')).toHaveCount(1);
  await expect(page.locator('#chapter-32')).toBeVisible();
  await search.fill('');
  await page.getByRole('button', { name: '仍有覆盖缺口', exact: true }).click();
  await expect(page.locator('#chapter-17')).toBeVisible();
  await expect(page.locator('#chapter-06')).toHaveCount(0);
  await page.goto('/chapters#chapter-17');
  const learningRate = page.locator('#chapter-17');
  await expect(learningRate).toHaveAttribute('open', '');
  await expect(learningRate.locator('.coverage-status')).toHaveText(['可操作实验', '原书阅读']);
  await expect(learningRate.getByRole('link', { name: '打开计算 →' })).toHaveAttribute(
    'href',
    '/notes#learning-rate',
  );
  await page.goto('/chapters');
  await page.locator('#chapter-32 > summary').focus();
  await page.keyboard.press('Enter');
  await page.locator('#chapter-32').getByRole('link', { name: '打开讲解 →' }).click();
  await expect(page).toHaveURL(/\/chapters\/32$/);
  await expect(page.locator('h1')).toContainText('历史怎样被读取');
});

test('完整前向可跳步、播放暂停、回退、改参重建并分享恢复', async ({ page, context }, info) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/chapters/15');
  const progress = page.getByTestId('forward-progress');
  const step = page.getByRole('combobox', { name: '前向计算步骤', exact: true });
  await expect(progress).toHaveText('1 / 23');
  await page.getByRole('button', { name: '下一步', exact: true }).click();
  await expect(progress).toHaveText('2 / 23');
  await expect(page.getByRole('table', { name: 'E[token] 矩阵', exact: true })).toBeVisible();
  await step.selectOption('21');
  const p = page.getByRole('table', { name: 'p 矩阵', exact: true });
  await expect(p.getByRole('columnheader')).toHaveText(['位置 ↓ 词表 →', 'a', 'b', 'c', 'd']);
  await page.getByRole('button', { name: '隐藏说明', exact: true }).click();
  await expect(page.locator('.forward-current .formula-symbols')).toHaveCount(0);
  await page.getByRole('button', { name: '显示说明', exact: true }).click();
  await expect(page.locator('.forward-current .formula-symbols')).toBeVisible();
  await page.getByRole('button', { name: '下一步', exact: true }).click();
  await expect(page.getByTestId('forward-selected')).toBeVisible();
  await page.getByRole('button', { name: '上一步', exact: true }).click();
  await step.selectOption('0');
  await page.getByRole('combobox', { name: '前向回放速度' }).selectOption('2');
  await page.getByRole('button', { name: '播放', exact: true }).click();
  await expect(progress).toHaveText('2 / 23', { timeout: 4000 });
  await page.getByRole('button', { name: '暂停', exact: true }).click();
  const paused = await progress.textContent();
  await page.waitForTimeout(1100);
  await expect(progress).toHaveText(paused!);
  await page.getByRole('combobox', { name: 'Block 数', exact: true }).selectOption('1');
  await expect(progress).toHaveText('1 / 15');
  await page.getByRole('combobox', { name: '位置 2 的 token', exact: true }).selectOption('3');
  await step.selectOption('13');
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const link = await page.evaluate(() => navigator.clipboard.readText());
  expect(link).toContain('#forward=');
  await page.goto(link);
  await expect(progress).toHaveText('14 / 15');
  await expect(page.getByRole('combobox', { name: '位置 2 的 token', exact: true })).toHaveValue(
    '3',
  );
  await page
    .locator('.forward-current')
    .screenshot({ path: `test-results/${info.project.name}-chapter15.png`, scale: 'css' });
  await page.goto('/chapters/15#forward=bad');
  await expect(page.getByText('分享状态无效，已恢复默认示例。', { exact: true })).toBeVisible();
  await expect(progress).toHaveText('1 / 23');
});

test('完整前向的因果对照通过所有Block，双向对照确实改变早期输出', async ({ page }) => {
  await page.goto('/chapters/15');
  await page.getByRole('button', { name: '不变', exact: true }).click();
  await page.getByRole('button', { name: '替换 t2 并核对整条前向', exact: true }).click();
  expect(await page.getByTestId('forward-before').textContent()).toBe(
    await page.getByTestId('forward-after').textContent(),
  );
  await page.getByRole('checkbox', { name: '启用因果 Mask', exact: true }).uncheck();
  await expect(page.getByTestId('forward-after')).toHaveCount(0);
  await page.getByRole('button', { name: '会变化', exact: true }).click();
  await page.getByRole('button', { name: '替换 t2 并核对整条前向', exact: true }).click();
  expect(await page.getByTestId('forward-before').textContent()).not.toBe(
    await page.getByTestId('forward-after').textContent(),
  );
  await expect(page.getByText('预测与计算一致。', { exact: true })).toBeVisible();
});

test('Mamba模块、SSD输入、锚点和分享状态联动，复述模式可恢复', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/chapters/32');
  await expect(page.locator('.mamba-flow button')).toHaveCount(7);
  for (const button of await page.locator('.mamba-flow button').all()) {
    await button.click();
    await expect(button).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('.chapter-module-reading .formula-symbols')).toBeVisible();
  }
  const slider = page.getByRole('slider', { name: '结构化递推保留比例', exact: true });
  await slider.fill('1');
  const table = page.getByRole('table', { name: '递推与结构化矩阵核对', exact: true });
  const before = (await table.locator('tbody tr').allTextContents()).slice(0, 3);
  await page.getByRole('spinbutton', { name: '结构化递推输入 x3', exact: true }).fill('-4');
  expect((await table.locator('tbody tr').allTextContents()).slice(0, 3)).toEqual(before);
  await expect(page.getByTestId('ssd-error')).toContainText('0.0e+0');
  await page
    .getByRole('navigation', { name: '本章小节' })
    .getByRole('link', { name: '混合架构账本', exact: true })
    .click();
  await expect(slider).toHaveValue('1');
  await page.getByRole('button', { name: '复制实验链接', exact: true }).click();
  const link = await page.evaluate(() => navigator.clipboard.readText());
  expect(JSON.parse(decodeURIComponent(new URL(link).hash.slice(14))).view.node).toBe(6);
  await page.goto(link);
  await expect(slider).toHaveValue('1');
  await expect(
    page.getByRole('spinbutton', { name: '结构化递推输入 x3', exact: true }),
  ).toHaveValue('-4');
  await expect(page.locator('.mamba-flow button').last()).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: '隐藏图与说明', exact: true }).click();
  await expect(page.locator('#mamba')).toBeHidden();
  await page.getByRole('button', { name: '显示图与说明', exact: true }).click();
  await expect(page.locator('#mamba')).toBeVisible();
});

for (const locale of ['', 'en', 'de'] as const)
  test(`${locale || 'zh'} 机制目录可直达并展开RWKV，章节锚点保留已编辑输入`, async ({ page }) => {
    const prefix = locale ? `/${locale}` : '';
    await page.goto(prefix + '/ssm-mamba');
    const toggle = page.locator('.course-directory-toggle');
    if ((await toggle.getAttribute('aria-expanded')) !== 'true') await toggle.click();
    const link = page.locator('#course-directory .course-architecture-guide');
    await expect(link).toContainText('RWKV');
    await expect(link).toHaveAttribute('href', prefix + '/chapters/32#family-rwkv');
    await link.click();
    await expect(page).toHaveURL(/\/chapters\/32#family-rwkv$/);
    const rwkv = page.locator('#family-rwkv');
    await expect(rwkv).toHaveAttribute('open', '');
    await expect(rwkv.locator('summary')).toBeInViewport();
    await expect(rwkv).toContainText(/time-mixing/i);
    await expect(page.locator('.chapter-page > .chapter-notice')).toHaveCount(0);
    await page.reload();
    await expect(rwkv).toHaveAttribute('open', '');
    const input = page.locator('#ssd input[type="number"]').last();
    await input.fill('-3');
    await page.evaluate(() => (window.location.hash = '#family-retnet'));
    await expect(page.locator('#family-retnet')).toHaveAttribute('open', '');
    await expect(input).toHaveValue('-3');
    if (locale) await layoutAndLanguage(page, locale);
  });

test('混合账本区分KV、状态与MoE，短序列可暴露总量反例', async ({ page }, info) => {
  await page.goto('/chapters/32#hybrid');
  const kv = page.getByTestId('hybrid-kv'),
    history = page.getByTestId('hybrid-history');
  await expect(kv).toHaveText('16,384 B');
  await expect(history).toHaveText('17,168 B');
  const params = await page.getByTestId('hybrid-all-parameters').textContent();
  await page.getByRole('checkbox', { name: '第 1 层使用 MoE', exact: true }).check();
  await expect(history).toHaveText('17,168 B');
  await expect(page.getByTestId('hybrid-all-parameters')).not.toHaveText(params!);
  await page.getByRole('spinbutton', { name: '已有 token 数 T', exact: true }).fill('2048');
  await expect(kv).toHaveText('32,768 B');
  await page.getByRole('button', { name: '全部 SSM', exact: true }).click();
  await expect(kv).toHaveText('0 B');
  await expect(history).toHaveText('896 B');
  await page.getByRole('button', { name: '恢复 1:7 教学预设', exact: true }).click();
  await page.getByRole('button', { name: '短序列对照：T=1', exact: true }).click();
  await expect(history).toHaveText('800 B');
  await page.getByRole('button', { name: '减少', exact: true }).click();
  await page.getByRole('button', { name: '替换一个 Attention 层并核对', exact: true }).click();
  await expect(kv).toHaveText('0 B');
  await expect(history).toHaveText('896 B');
  await expect(page.getByText(/短序列下，新增 SSM 与卷积状态可能比省下的 KV 更大/)).toBeVisible();
  await page
    .locator('.hybrid-layer-grid')
    .screenshot({ path: `test-results/${info.project.name}-chapter32-layers.png`, scale: 'css' });
});

for (const locale of ['en', 'de'] as const)
  for (const route of ['/chapters', '/chapters/15', '/chapters/32'])
    test(`${locale}${route}: 全部讲解与数值步骤翻译完整且无横向溢出`, async ({ page }, info) => {
      test.setTimeout(90000);
      const problems: string[] = [];
      page.on('pageerror', (e) => problems.push(e.message));
      page.on('console', (m) => {
        if (m.text().includes('[i18n] missing')) problems.push(m.text());
      });
      await page.goto(`/${locale}${route}`);
      await expect(page.locator('h1')).toBeVisible();
      if (route === '/chapters') {
        await page
          .locator('.chapter-entry')
          .evaluateAll((elements) =>
            elements.forEach((el) => ((el as HTMLDetailsElement).open = true)),
          );
        await expect(page.locator('.chapter-entry[open]')).toHaveCount(35);
        await layoutAndLanguage(page, locale);
        await page.screenshot({
          path: `test-results/${info.project.name}-chapters-${locale}.png`,
          scale: 'css',
        });
      } else if (route === '/chapters/15') {
        const step = page.locator('.forward-jump select');
        for (let i = 0; i < 23; i++) {
          await step.selectOption(String(i));
          await layoutAndLanguage(page, locale);
        }
        await page.locator('#scope').evaluate((el) => ((el as HTMLDetailsElement).open = true));
        await layoutAndLanguage(page, locale);
        await page.locator('.forward-current').screenshot({
          path: `test-results/${info.project.name}-chapter15-${locale}.png`,
          scale: 'css',
        });
      } else {
        for (const button of await page.locator('.mamba-flow button').all()) {
          await button.click();
          await layoutAndLanguage(page, locale);
        }
        await page
          .locator('.architecture-families details')
          .evaluateAll((elements) =>
            elements.forEach((el) => ((el as HTMLDetailsElement).open = true)),
          );
        await layoutAndLanguage(page, locale);
        await page.locator('.chapter-module-reading').screenshot({
          path: `test-results/${info.project.name}-mamba-${locale}.png`,
          scale: 'css',
        });
        await page.locator('.hybrid-layer-grid').screenshot({
          path: `test-results/${info.project.name}-hybrid-${locale}.png`,
          scale: 'css',
        });
      }
      expect(problems).toEqual([]);
    });
