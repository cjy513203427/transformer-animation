import { expect, it } from 'vitest';
import { CHAPTERS } from './chapters';
import { KNOWLEDGE_NOTES, NOTE_GROUPS, NOTE_SOURCES, RLHF_STAGES } from './notebook';
import { NOTE_IDS } from '../state/notebook';
import { TOPICS } from './topics';

it('笔记覆盖全部章节的概念索引，链接到已有机制并保留范围边界', () => {
  expect(KNOWLEDGE_NOTES.map((n) => n.id)).toEqual(NOTE_IDS);
  const covered = new Set(KNOWLEDGE_NOTES.flatMap((n) => n.chapters));
  expect([...covered].sort()).toEqual(CHAPTERS.map((c) => c.id).sort());
  for (const note of KNOWLEDGE_NOTES) {
    expect(NOTE_GROUPS.some((g) => g.id === note.group)).toBe(true);
    expect(note.body.length).toBeGreaterThanOrEqual(2);
    expect(note.example.length).toBeGreaterThan(15);
    expect(note.boundary.length).toBeGreaterThan(15);
    expect(note.recall.length).toBeGreaterThan(10);
    note.topics.forEach((id) => expect(TOPICS.some((t) => t.id === id)).toBe(true));
    note.sources.forEach((id) => expect(NOTE_SOURCES[id]?.[1]).toMatch(/^https:\/\//));
  }
});

it('RLHF包含数据、奖励、rollout、价值、策略更新与独立评估，不把各角色混为一体', () => {
  expect(RLHF_STAGES).toHaveLength(7);
  expect(RLHF_STAGES[3].updates).toContain('π_old');
  expect(RLHF_STAGES[5].updates).toContain('π_ref');
  expect(RLHF_STAGES[6].updates).toContain('不直接反向更新');
  expect(KNOWLEDGE_NOTES.find((n) => n.id === 'rlhf')!.boundary).toContain('没有在浏览器训练');
});
