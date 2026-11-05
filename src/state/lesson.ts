export const LESSON_MODES = ['overview', 'guide', 'numbers'] as const;
export type LessonMode = (typeof LESSON_MODES)[number];
export const ARCHITECTURE_NODES = [
  'text',
  'tokenization',
  'embedding',
  'position',
  'attention',
  'attention-add',
  'attention-norm',
  'ffn',
  'ffn-add',
  'ffn-norm',
  'linear',
  'softmax',
  'output',
] as const;
export type ArchitectureNode = (typeof ARCHITECTURE_NODES)[number];
