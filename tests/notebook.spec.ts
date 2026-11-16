import { expect, test } from '@playwright/test';
import { openDirectory } from './helpers/navigation';
import { NOTE_DESTINATIONS } from '../src/content/experimentLinks';

const labs=[
  {id:'training',path:'/training-step',steps:6,choice:'容差内不变'},
  {id:'policy',path:'/policy-update',steps:7,choice:'容差内不变'},
  {id:'prompt-chain',path:'/prompt-engineering',steps:6,choice:'会改变'},
  // [5,5,3,5,3]: the first two candidates still elect 5.
  {id:'prompt-vote',path:'/prompt-voting',steps:6,choice:'容差内不变'},
  {id:'prompt-search',path:'/prompt-search',steps:6,choice:'会改变'},
];

for(const lab of labs) test(`${lab.id} 同一例子逐步核对、预测、回退与分享恢复`,async({page,context},info)=>{
  const errors:string[]=[];
  page.on('pageerror',e=>errors.push(e.message));
  await context.grantPermissions(['clipboard-read','clipboard-write']);
  await page.goto(lab.path);
  await expect(page.locator('.notebook-directory,.course-note-guide,.course-architecture-guide')).toHaveCount(0);
  await expect(page.locator('.lesson-progress')).toHaveText(`1 / ${lab.steps}`);
  await expect(page.locator('.diagram-panel figure')).toBeVisible();
  const directory=await openDirectory(page);
  await expect(directory.locator('ol a')).toHaveCount(23);
  await page.keyboard.press('Escape');
  for(let i=0;i<lab.steps;i++){
    await page.locator('.p1-steps button').nth(i).click();
    await expect(page.locator('.lesson-progress')).toHaveText(`${i+1} / ${lab.steps}`);
    await expect(page.locator('.lesson-reading table').first()).toBeVisible();
    // Scope to the lesson: collapsed "further reading" tools carry their own formulas.
    const lesson=page.locator('.p1-panel');
    await lesson.getByText('公式与符号说明',{exact:true}).click();
    await expect(lesson.getByLabel('本式符号说明')).toBeVisible();
    await expect(page.locator('.katex-error')).toHaveCount(0);
    await lesson.getByText('公式与符号说明',{exact:true}).click();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth)).toBeLessThanOrEqual(1);
  }
  await page.getByRole('button',{name:lab.choice,exact:true}).click();
  await page.getByRole('button',{name:'验证预测并执行 →',exact:true}).click();
  await expect(page.locator('.exp-experiment [role="status"]')).toContainText('与计算一致');
  await page.getByRole('button',{name:'复制实验链接',exact:false}).click();
  const url=await page.evaluate(()=>navigator.clipboard.readText());
  expect(url).toContain('#lab=');
  const state=JSON.parse(decodeURIComponent(new URL(url).hash.slice(5)));
  expect(state.topic).toBe(lab.id);
  await page.goto(url);
  await expect(page.locator('.lesson-progress')).toHaveText(`${lab.steps} / ${lab.steps}`);
  await expect(page.locator('.exp-experiment [role="status"]')).toHaveCount(0);
  await page.getByRole('button',{name:'← 上一步',exact:true}).click();
  await expect(page.locator('.lesson-progress')).toHaveText(`${lab.steps-1} / ${lab.steps}`);
  await page.getByRole('button',{name:'隐藏说明',exact:true}).click();
  await expect(page.locator('.lesson-explanation')).toHaveCount(0);
  await page.getByText('数值性质核对',{exact:false}).click();
  await expect(page.locator('.p1-checks')).not.toContainText('超出容差');
  await page.getByRole('button',{name:'恢复默认示例',exact:true}).click();
  await expect(page.locator('.lesson-progress')).toHaveText(`1 / ${lab.steps}`);
  await page.screenshot({path:`test-results/${info.project.name}-learning-${lab.id}.png`,scale:'css',fullPage:true});
  expect(errors).toEqual([]);
});

test('训练目标控制真实梯度，零输入与零学习率停止更新',async({page})=>{
  await page.goto('/training-step');
  await page.locator('.p1-steps button').nth(3).click();
  const gradient=page.getByRole('table',{name:'g_W = hᵀg_z',exact:true});
  const original=await gradient.innerText();
  await page.getByRole('combobox',{name:'目标 token',exact:true}).selectOption('0');
  await expect(gradient).not.toHaveText(original);
  await page.getByRole('button',{name:'零输入',exact:true}).click();
  await page.locator('.p1-steps button').nth(3).click();
  expect(await gradient.locator('tbody td').allTextContents()).toEqual(['0','0','0','0','0','0']);
  await page.getByRole('button',{name:'恢复默认示例',exact:true}).click();
  const rate=page.getByRole('slider',{name:'学习率 η',exact:true});
  await rate.focus(); await page.keyboard.press('Home');
  await page.locator('.p1-steps button').nth(5).click();
  const before=await page.getByRole('table',{name:'p',exact:true}).locator('tbody td').allTextContents();
  expect(await page.getByRole('table',{name:'p′',exact:true}).locator('tbody td').allTextContents()).toEqual(before);
});

test('提示链发现中间错误，多数票会答错，搜索需要扩大预算',async({page})=>{
  await page.goto('/prompt-engineering');
  // The step note; the collapsed property checks reuse the .exp-verdict class.
  const note=page.locator('.exp-widget .exp-verdict');
  await page.getByRole('button',{name:'中间结果多 2',exact:true}).click();
  await page.locator('.p1-steps button').nth(4).click();
  await expect(note).toContainText('核对失败');
  await page.goto('/prompt-voting');
  await page.getByRole('button',{name:'多数答错',exact:true}).click();
  await page.locator('.p1-steps button').nth(4).click();
  await expect(note).toContainText('多数答案未通过');
  await page.getByRole('button',{name:'平局',exact:true}).click();
  await page.locator('.p1-steps button').nth(3).click();
  await expect(note).toContainText('平局');
  await page.goto('/prompt-search');
  await page.locator('.p1-steps button').nth(5).click();
  await expect(note).toContainText('当前预算内没有找到解');
  await page.getByRole('button',{name:'检查全部路径',exact:true}).click();
  await page.locator('.p1-steps button').nth(5).click();
  await expect(note).toContainText('2 − 3 + 1 = 0');
  await expect(page.locator('.learning-tree .search-match')).toHaveCount(1);
});

test('旧笔记链接进入实验，训练补充工具仍计算并随分享恢复',async({page,context})=>{
  await context.grantPermissions(['clipboard-read','clipboard-write']);
  for(const [anchor,path] of [['rlhf','policy-update'],['prompt','prompt-engineering'],['self-consistency','prompt-voting'],['tot-react','prompt-search'],['learning-rate','training-step']]){
    await page.goto('/notes#'+anchor);
    await expect(page).toHaveURL(new RegExp('/'+path+'#lab='));
    await expect(page.locator('.notebook-directory')).toHaveCount(0);
  }
  await page.getByText('进一步理解',{exact:true}).click();
  await page.getByText('学习率轨迹与 Warmup',{exact:true}).click();
  await page.getByRole('button',{name:'η=2.5',exact:true}).click();
  await page.getByRole('button',{name:'不更小',exact:true}).click();
  await page.getByRole('button',{name:'执行 12 次更新',exact:true}).click();
  await expect(page.getByTestId('learning-result')).toContainText('预测与计算一致');
  await page.getByRole('button',{name:'复制实验链接',exact:false}).click();
  const url=await page.evaluate(()=>navigator.clipboard.readText());
  // A hash-only goto keeps the page (and its open panels); load the link fresh.
  await page.goto('about:blank');
  await page.goto(url);
  await page.getByText('进一步理解',{exact:true}).click();
  await page.getByText('学习率轨迹与 Warmup',{exact:true}).click();
  await expect(page.getByRole('slider',{name:'基础学习率 η',exact:true})).toHaveValue('2.5');
  // The compute ledger moved to the calculator; old #scaling links follow it.
  await page.goto('/notes#scaling');
  await expect(page).toHaveURL(/\/model-budget$/);
  await expect(page.getByTestId('budget-total')).toHaveText('124,439,808');
});

for(const locale of ['en','de']) test(`${locale} 五个新实验完整翻译且窄屏可读`,async({page},info)=>{
  const errors:string[]=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.text().includes('[i18n] missing')) errors.push(m.text());});
  for(const lab of labs){
    await page.goto('/'+locale+lab.path);
    for(let i=0;i<lab.steps;i++){
      await page.locator('.p1-steps button').nth(i).click();
      await expect(page.locator('.lesson-progress')).toHaveText(`${i+1} / ${lab.steps}`);
      expect(await page.locator('.lesson-reading').innerText()).not.toMatch(/[一-鿿]/);
      expect(await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth)).toBeLessThanOrEqual(1);
    }
    await page.screenshot({path:`test-results/${info.project.name}-${locale}-${lab.id}.png`,scale:'css'});
  }
  expect(errors).toEqual([]);
});

test('每条知识笔记都在对应页面的“进一步理解”中可读',async({page})=>{
  const byPath=new Map<string,string[]>();
  for(const [id,path] of Object.entries(NOTE_DESTINATIONS)) byPath.set(path,[...(byPath.get(path)??[]),id]);
  for(const [path,ids] of byPath){
    await page.goto(path);
    const panel=page.locator('.related-notes');
    await expect(panel,path).toHaveCount(1);
    await panel.locator(':scope > summary').click();
    for(const id of ids) await expect(panel.locator(`[data-note="${id}"] h3`),`${path} ${id}`).toBeVisible();
    if(path==='/policy-update') await expect(panel.locator('.rlhf-stages li')).toHaveCount(7);
  }
});
