import { useEffect } from 'react';
import { localePath, T } from '../../i18n';
import { restoreNotebookState } from '../../state/notebook';
import { NOTE_DESTINATIONS } from '../../content/experimentLinks';
import { initialLabState, serializeLabState } from '../../state/p1';
import { trainingTopic, policyTopic, chainTopic, voteTopic, searchTopic } from '../../content/p1/learning';

/** Old bookmarks open the corresponding experiment without a second navigation column. */
export default function KnowledgeNotebook() {
  useEffect(() => {
    const {state} = restoreNotebookState(window.location.hash);
    const path = NOTE_DESTINATIONS[state.view.selected];
    const topic = [trainingTopic,policyTopic,chainTopic,voteTopic,searchTopic].find(t => t.path===path);
    let hash = '';
    if(topic) {
      const next = initialLabState(topic);
      next.view.explanations = !state.view.recall;
      if(topic.id==='policy') next.step = Math.min(state.step,topic.steps.length-1);
      if(topic.id==='training') {
        next.config.eta = Math.min(2,Math.round(state.config.eta*20)/20);
        next.config.trajectoryEta = Math.round(state.config.eta*20)/20;
        next.config.schedule = Number(state.config.schedule);
        next.config.parametersB = Math.round(state.config.parametersB*10)/10;
        next.config.tokensB = Math.round(state.config.tokensB);
      }
      if(topic.id==='prompt-vote') {
        const answers=state.input.answers.filter((a):a is 3|5|7=>a!==null);
        next.input=Array.from({length:5},(_,i)=>answers[i]??5);
        next.config.count=Math.max(1,Math.min(state.config.samples,answers.length));
      }
      hash=serializeLabState(topic,next);
    }
    window.location.replace(localePath(path)+hash);
  },[]);
  return <div className="loading">{T('正在准备实验…')}</div>;
}
