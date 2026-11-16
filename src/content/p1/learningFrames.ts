import { T } from '../../i18n';
import { formatNumber as fmt } from '../../math/attention';
import { supervisedStep, policyStep, thoughtSearch } from '../../math/learning';
import { tallyAnswers } from '../../math/notebook';
import { check, frame, tensor, type Computation, type Frame, type Parameters } from '../../math/p1/shared';

const vector = (values: number[]) => '[' + values.map((value) => fmt(value, 3)).join(', ') + ']';
function withFlow(frames: Frame[], labels: string[], values: string[]): Frame[] {
  return frames.map((current, active) => ({ ...current, flow: {
    nodes: labels.map((label, i) => ({ label, value: i <= active ? values[i] : '…' })),
    active, caption: current.title,
  } }));
}

export function trainingComputation(input: number[], p: Parameters): Computation {
  const r = supervisedStep(input, p.target, p.eta, p.shift);
  const labels = ['h → z', 'Softmax', 'Loss', 'Backward', 'SGD', 'p′'];
  const metrics: Frame['metrics'] = [['p(target)', r.probabilities[p.target]], ['loss', r.loss], ['next loss', r.nextLoss]];
  const frames = [
    frame(T('隐藏状态乘输出权重'), [tensor('h', input), tensor('W', r.weights), tensor('z', r.logits)], metrics),
    frame(T('沿词表轴归一化'), [tensor('z', r.logits), tensor('p', r.probabilities)], metrics),
    frame(T('标签选中目标概率'), [tensor('onehot(y)', [0,1,2].map((i) => Number(i === p.target))), tensor('p', r.probabilities), tensor('L', [r.loss])], metrics),
    frame(T('链式法则得到每个权重的梯度'), [tensor('g_z = p − onehot(y)', r.logitGradient), tensor('g_W = hᵀg_z', r.weightGradient)], metrics),
    frame(T('执行一次 SGD'), [tensor('W', r.weights), tensor('−ηg_W', r.weightGradient.map((row) => row.map((v) => -p.eta * v))), tensor('W′', r.nextWeights)], metrics),
    frame(T('同一输入重新预测'), [tensor('p', r.probabilities), tensor('p′', r.nextProbabilities), tensor('[L, L′]', [r.loss, r.nextLoss])], metrics),
  ];
  frames[1].distribution = { names:['a','b','c'],before:r.probabilities,caption:T('三个候选的概率来自同一行 Softmax。') };
  frames[5].distribution = { names:['a','b','c'],before:r.probabilities,after:r.nextProbabilities,caption:T('对照更新前后的概率；每个候选保持同一位置。') };
  return { frames: withFlow(frames, labels, [vector(r.logits),vector(r.probabilities),fmt(r.loss),vector(r.logitGradient),vector(r.nextWeights.flat()),vector(r.nextProbabilities)]),
    signature:r.nextProbabilities,
    checks:[check(T('概率和为 1'),[r.probabilities.reduce((sum,v)=>sum+v,0)],[1]),check(T('logit 梯度之和为 0'),[r.logitGradient.reduce((sum,v)=>sum+v,0)],[0]),
      check(T('共同平移 logits 不改变更新结果'),r.nextProbabilities,supervisedStep(input,p.target,p.eta,p.shift+1).nextProbabilities)] };
}

export function policyComputation(input: number[], p: Parameters): Computation {
  const r = policyStep(input,p.oldLogit,p.baseline,p.eta,p.beta,p.epsilon);
  const metrics: Frame['metrics'] = [['E[R]',r.expectedReward],['next p(A)',r.after.probabilities[0]],['KL',r.after.kl],['DPO loss',r.dpoLoss]];
  const scoreDifference = input[0]-input[1];
  const preference = 1/(1+Math.exp(-scoreDifference));
  const frames = [
    frame(T('一个问题，两个回答'),[tensor('[r_A, r_B]',input),tensor('P(A ≻ B)',[preference])],metrics),
    frame(T('保留旧策略与参考策略'),[tensor('π_old',r.old),tensor('π_ref',r.before.reference)],metrics),
    frame(T('回报、价值与优势'),[tensor('R',input),tensor('V',[p.baseline,p.baseline]),tensor('A = R − V',r.before.advantages)], [...metrics,['Critic MSE',r.criticLoss]]),
    frame(T('目标对策略 logits 的梯度'),[tensor('g_θ',r.gradient),tensor('J_old',[r.before.objective])],metrics),
    frame(T('参数更新后的分布'),[tensor('θ',r.oldLogits),tensor('θ′',r.nextLogits),tensor('π′',r.after.probabilities)],metrics),
    frame(T('用旧策略核对这次更新'),[tensor('ρ = π′ / π_old',r.after.ratios),tensor('clip(ρ)',r.after.clipped),tensor('min(ρA, clip(ρ)A)',r.after.terms),tensor('KL contributions',r.after.klTerms)], [...metrics,['J_old',r.before.objective],['J_new',r.after.objective]]),
    frame(T('由奖励差连接到 DPO 概率比'),[tensor('log(π′ / π_ref)',r.after.probabilities.map((v,i)=>Math.log(v/r.after.reference[i]))),tensor('[DPO margin, loss]',[r.dpoMargin,r.dpoLoss])],metrics,
      T('DPO 在这里是另一条训练路径；固定 A 为偏好回答。')),
  ];
  frames[4].distribution={names:['A','B'],before:r.old,after:r.after.probabilities,caption:T('对照更新前后的概率；每个候选保持同一位置。')};
  return {frames:withFlow(frames,['r','π_old','R − V','∇J','θ′ → π′','ρ / KL','DPO'],[vector(input),vector(r.old),vector(r.before.advantages),vector(r.gradient),vector(r.after.probabilities),fmt(r.after.kl),fmt(r.dpoLoss)]),
    signature:r.after.probabilities,checks:[check(T('新策略概率和为 1'),[r.after.probabilities.reduce((sum,v)=>sum+v,0)],[1]),check(T('精确期望的更新不受价值基线平移影响'),r.after.probabilities,policyStep(input,p.oldLogit,p.baseline+1,p.eta,p.beta,p.epsilon).after.probabilities)]};
}

export function chainComputation(input: number[], p: Parameters): Computation {
  const [a,b,c] = input, distributed = b*c, used = distributed+p.delta, remainder = a-used, correct = a-b*c;
  const difference = Math.abs(remainder-correct);
  const metrics: Frame['metrics'] = [['distributed',distributed],['remainder',remainder],['error',difference]];
  const question = T('有 {#0} 个物品，给 {#1} 人每人 {#2} 个，还剩多少？',a,b,c);
  const stages = [
    frame(T('固定原始问题'),[tensor('[a, b, c]',input)],metrics,question),
    frame(T('提取命名字段'),[tensor('[a, b, c]',input)],metrics,T('提示模板：提取原有数量 a、人数 b、每人数量 c。模板字段：{#0}',JSON.stringify({a,b,c}))),
    frame(T('计算分出数量'),[tensor('[b, c, b×c]',[b,c,distributed])],metrics,T('提示模板：使用上一步字段，调用算术计算 b×c，返回分出数量 d。')),
    frame(T('传递中间结果并计算'),[tensor('[d, Δ, d_used]',[distributed,p.delta,used]),tensor('o = a − d_used',[remainder])],metrics,T('提示模板：使用原有数量与传入的分出数量，计算剩余量。当前传入 d_used={#0}。',used)),
    frame(T('用原始字段独立核对'),[tensor('[o, a−bc, error]',[remainder,correct,difference])],metrics,difference===0?T('核对通过。'):T('核对失败，返回中间步骤修正。')),
    frame(T('带检查状态的输出'),[tensor('[answer, verified]',[remainder,Number(difference===0)])],metrics,JSON.stringify({answer:remainder,verified:difference===0})),
  ];
  return {frames:withFlow(stages,[T('问题'),T('提取'),T('计算'),T('传递'),T('核对'),T('输出')],[question,JSON.stringify({a,b,c}),fmt(distributed),fmt(remainder),fmt(difference),JSON.stringify({answer:remainder,verified:difference===0})]),
    signature:[remainder],checks:[check(T('错误中间量对剩余数量的影响'),[remainder-correct],[-p.delta])]};
}

export function voteComputation(input: number[], p: Parameters): Computation {
  const r = tallyAnswers(input,p.count), winner = r.winner;
  const counts = r.ranked.map(([,count])=>count);
  const metrics: Frame['metrics'] = [['vote share',r.share],['count',p.count]];
  const result = winner===null ? T('平局，保留并列答案。') : T('投票选中 {#0}。',winner);
  const verdict = winner===null ? T('需要更多证据或明确的平局规则。') : winner===5 ? T('独立算术核对通过。') : T('多数答案未通过独立算术核对。');
  const table = {name:T('答案与票数'),values:r.ranked.map(([answer,count])=>[answer,count]),columns:[T('答案'),T('票数')]};
  const frames = [
    frame(T('固定问题与正确的算术结果'),[tensor('17−3×4',[5])],metrics,T('候选来自手工输入，实际采样发生在模型之外。')),
    frame(T('保留前 n 个候选'),[tensor('candidates',r.used as number[])],metrics),
    frame(T('按最终答案计票'),[table],metrics),
    frame(T('选择唯一最高票或报告平局'),[table],metrics,result),
    frame(T('核对多数答案'),[tensor('expected',[5]),table],metrics,verdict),
    frame(T('改变预算，重算同一组候选'),[tensor('candidates',r.used as number[]),table],metrics,result+' '+verdict),
  ];
  return {frames:withFlow(frames,[T('问题'),T('候选'),T('计票'),T('聚合'),T('验证'),T('对照')],['17−3×4',vector(r.used as number[]),vector(counts),result,verdict,String(p.count)]),
    // Compare the aggregate: [is tie, selected answer]. Vote counts always change with n.
    signature:[winner===null?1:0,winner??0],checks:[check(T('票数之和等于候选数'),[counts.reduce((sum,value)=>sum+value,0)],[p.count])]};
}

export function searchComputation(input: number[],p:Parameters): Computation {
  const r=thoughtSearch(input,p.target,p.budget);
  const expression=(i:number)=>`${input[0]} ${r.paths[i].first===1?'+':'−'} ${input[1]} ${r.paths[i].second===1?'+':'−'} ${input[2]}`;
  const table={name:T('路径、结果与误差'),values:r.paths.map((path,i)=>[i+1,path.intermediate,path.result,Math.abs(path.result-p.target)]),columns:['path','s1','s2','error']};
  const result=r.solution?T('找到解：{#0} = {#1}',expression(r.found),r.solution.result):T('当前预算内没有找到解。');
  const frames=[
    frame(T('固定输入、动作与目标'),[tensor('[a,b,c,τ]',[...input,p.target])]),
    frame(T('展开两个中间状态'),[tensor('s1',[r.paths[0].intermediate,r.paths[2].intermediate])]),
    frame(T('四条完整候选路径'),[table]),
    frame(T('检查预算内的路径'),[{...table,values:table.values.slice(0,p.budget)}]),
    frame(T('从失败路径回到下一个分支'),[{...table,values:table.values.slice(0,p.budget)}],[],result),
    frame(T('报告解或预算耗尽'),[{...table,values:table.values.slice(0,p.budget)}],[],result),
  ];
  return {frames:withFlow(frames,[T('目标'),T('展开'),T('候选'),T('检查'),T('回退'),T('结果')],[String(p.target),vector([r.paths[0].intermediate,r.paths[2].intermediate]),vector(r.paths.map(path=>path.result)),String(p.budget),result,result]).map((f,step)=>({...f,search:step<2?undefined:{root:input[0],middle:[r.paths[0].intermediate,r.paths[2].intermediate],operations:input.slice(1),leaves:r.paths.map((path,i)=>({label:expression(i),value:path.result,state:step<3||i>=p.budget?'unvisited':path.result===p.target?'match':'miss'})),caption:step<4?f.title:result}})),
    signature:[r.solution?1:0,r.solution?.result??0],checks:[check(T('每条路径保留原输入顺序'),r.paths.map(path=>path.result),[input[0]+input[1]+input[2],input[0]+input[1]-input[2],input[0]-input[1]+input[2],input[0]-input[1]-input[2]])]};
}
