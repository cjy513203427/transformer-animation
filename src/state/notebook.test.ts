import { expect, it } from 'vitest';
import { initialNotebookState, restoreNotebookState, serializeNotebookState } from './notebook';

it('笔记链接恢复公开输入、所选知识点与步骤，严格验证类型与边界', () => {
  const state = initialNotebookState();
  state.input.answers = [null, 3, 7, 5, 5];
  state.step = 6;
  state.view = { selected: 'prompt', recall: true };
  expect(restoreNotebookState(serializeNotebookState(state))).toEqual({ state, invalid: false });
  expect(restoreNotebookState('#learning-rate').state.view.selected).toBe('learning-rate');
  for (const value of [
    { ...state, version: 2 },
    { ...state, seed: 1 },
    { ...state, step: 7 },
    { ...state, input: { ...state.input, answers: [5] } },
    { ...state, config: { ...state.config, eta: '0.5' } },
    { ...state, config: { ...state.config, samples: 3.5 } },
    { ...state, view: { selected: 'unknown', recall: false } },
  ])
    expect(
      restoreNotebookState('#notes=' + encodeURIComponent(JSON.stringify(value))).invalid,
    ).toBe(true);
  for (const hash of ['#notes=%', '#notes=' + 'a'.repeat(2600), '#unknown'])
    expect(restoreNotebookState(hash).invalid).toBe(true);
});
