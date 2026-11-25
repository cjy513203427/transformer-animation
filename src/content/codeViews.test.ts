import { describe, expect, it } from 'vitest';
import {
  INFERENCE_PY_CODE,
  INFERENCE_PY_HIGHLIGHTS,
  MODEL_PY_CODE,
  MODEL_PY_HIGHLIGHTS,
  TRAIN_PY_CODE,
  TRAIN_PY_HIGHLIGHTS,
} from './codeViews';
import { FORWARD_CONTENT } from './forward';
import { inferenceSteps } from './inference';
import { trainingTopic } from './p1/learning';
import type { ForwardKind } from '../math/forward';

describe('第 18–20 章步骤联动代码视图', () => {
  it('model.py 覆盖第 15 章全部步骤类型，且高亮行号在代码范围内', () => {
    const kinds = Object.keys(FORWARD_CONTENT) as ForwardKind[];
    for (const kind of kinds) {
      const h = MODEL_PY_HIGHLIGHTS[kind];
      expect(h, kind).toBeDefined();
      expect(h.lines.length).toBeGreaterThan(0);
      expect(h.note.length).toBeGreaterThan(4);
      for (const line of h.lines) {
        expect(line).toBeGreaterThanOrEqual(1);
        expect(line).toBeLessThanOrEqual(MODEL_PY_CODE.lines.length);
      }
    }
  });

  it('train.py 与 inference.py 覆盖所有对应步骤且行号有效', () => {
    expect(TRAIN_PY_HIGHLIGHTS).toHaveLength(trainingTopic.steps.length);
    for (const h of TRAIN_PY_HIGHLIGHTS) {
      expect(h.lines.length).toBeGreaterThan(0);
      for (const line of h.lines) {
        expect(line).toBeGreaterThanOrEqual(1);
        expect(line).toBeLessThanOrEqual(TRAIN_PY_CODE.lines.length);
      }
    }

    expect(INFERENCE_PY_HIGHLIGHTS).toHaveLength(inferenceSteps.length);
    for (const h of INFERENCE_PY_HIGHLIGHTS) {
      expect(h.lines.length).toBeGreaterThan(0);
      for (const line of h.lines) {
        expect(line).toBeGreaterThanOrEqual(1);
        expect(line).toBeLessThanOrEqual(INFERENCE_PY_CODE.lines.length);
      }
    }
  });
});
