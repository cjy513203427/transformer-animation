import { describe, expect, it } from 'vitest';
import katex from 'katex';
import { steps } from '../attention';
import { positionSteps } from '../position';
import { inferenceSteps } from '../inference';
import { kvSteps } from '../kv';
import { moeSteps } from '../moe';
import { flashSteps } from '../flash';
import normalization from '../p1/normalization';
import ffn from '../p1/ffn';
import lora from '../p1/lora';
import quantization from '../p1/quantization';
import decoding from '../p1/decoding';
import preference from '../p1/preference';
import memory from '../p1/memory';
import ssm from '../p1/ssm';
import optimizer from '../p1/optimizer';
import context from '../p1/context';
import { multiheadSymbols } from './multihead';
import {
  projectionCellSymbols,
  scoreCellSymbols,
  outputCellSymbols,
  valueChangeSymbols,
} from './attention';

describe('公式的就地符号说明', () => {
  const lessons = [
    steps,
    positionSteps,
    inferenceSteps,
    kvSteps,
    moeSteps,
    flashSteps,
    ...[normalization, ffn, lora, quantization, decoding, preference, memory, ssm, optimizer, context].map(
      (topic) => topic.steps,
    ),
  ];

  it('每一步都有非空的符号说明，符号本身可严格排版且不重复', () => {
    const glossaries = [
      ...lessons.flatMap((group) => group.map((step) => step.symbols)),
      ...multiheadSymbols,
      projectionCellSymbols,
      scoreCellSymbols,
      outputCellSymbols,
      valueChangeSymbols,
    ];
    for (const symbols of glossaries) {
      expect(symbols?.length).toBeGreaterThan(0);
      expect(new Set(symbols.map(([tex]) => tex)).size).toBe(symbols.length);
      for (const [tex, meaning] of symbols) {
        expect(meaning.trim().length, tex).toBeGreaterThan(0);
        expect(
          () => katex.renderToString(tex, { throwOnError: true, strict: 'ignore' }),
          tex,
        ).not.toThrow();
      }
    }
  });

  it('同一个字母的释义跟随当前公式，不混用位置、概率与状态', () => {
    const meanings = (symbols: (typeof positionSteps)[number]['symbols']) =>
      symbols.map(([, meaning]) => meaning).join(' ');
    expect(meanings(positionSteps[3].symbols)).toContain('位置');
    expect(meanings(positionSteps[5].symbols)).toContain('ALiBi 的头斜率');
    expect(meanings(flashSteps[2].symbols)).toContain('最大分数');
    expect(meanings(decoding.steps[3].symbols)).toContain('Top-P 阈值');
    expect(meanings(memory.steps[3].symbols)).toContain('外积');
    expect(meanings(kvSteps[0].symbols)).toContain('批大小');
    expect(meanings(flashSteps[1].symbols)).toContain('块最多包含的行数');
  });
});
