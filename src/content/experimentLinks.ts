import type { NoteId } from '../state/notebook';
/** Compatibility for previously shared notes; experiments own the learning paths. */
export const NOTE_DESTINATIONS: Record<NoteId,string> = {
  model:'/forward-pass', tokenization:'/tokenization', architecture:'/forward-pass',
  attention:'/attention', components:'/ffn-residual', position:'/position',
  training:'/training-step', 'learning-rate':'/training-step', generation:'/inference-cache',
  efficiency:'/flash-attention', memory:'/attention-variants', adaptation:'/lora',
  quantization:'/quantization', moe:'/moe', prompt:'/prompt-engineering',
  'few-shot':'/prompt-engineering', cot:'/prompt-search', 'self-consistency':'/prompt-voting',
  'tot-react':'/prompt-search', rlhf:'/policy-update', 'ppo-dpo':'/policy-update',
  'alignment-risks':'/policy-update', grpo:'/preference', reasoning:'/policy-update',
  scaling:'/model-budget', faq:'/inference-cache',
};
