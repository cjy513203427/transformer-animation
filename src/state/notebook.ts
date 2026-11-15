export const NOTE_IDS = [
  'model',
  'tokenization',
  'architecture',
  'attention',
  'components',
  'position',
  'training',
  'learning-rate',
  'generation',
  'efficiency',
  'memory',
  'adaptation',
  'quantization',
  'moe',
  'prompt',
  'few-shot',
  'cot',
  'self-consistency',
  'tot-react',
  'rlhf',
  'ppo-dpo',
  'alignment-risks',
  'grpo',
  'reasoning',
  'scaling',
  'faq',
] as const;
export type NoteId = (typeof NOTE_IDS)[number];
export interface NotebookState {
  version: 1;
  seed: 0;
  config: { samples: number; eta: number; schedule: boolean; parametersB: number; tokensB: number };
  input: {
    answers: (3 | 5 | 7 | null)[];
    examples: number;
    role: boolean;
    evidence: boolean;
    format: boolean;
  };
  step: number;
  view: { selected: NoteId; recall: boolean };
}
export const initialNotebookState = (): NotebookState => ({
  version: 1,
  seed: 0,
  config: { samples: 5, eta: 0.5, schedule: false, parametersB: 7, tokensB: 140 },
  input: { answers: [5, 5, 3, 5, 3], examples: 0, role: false, evidence: false, format: false },
  step: 0,
  view: { selected: 'rlhf', recall: false },
});
const record = (x: unknown): x is Record<string, unknown> =>
  !!x && typeof x === 'object' && !Array.isArray(x);
const finite = (x: unknown, min: number, max: number): x is number =>
  typeof x === 'number' && Number.isFinite(x) && x >= min && x <= max;
const integer = (x: unknown, min: number, max: number) =>
  finite(x, min, max) && Number.isInteger(x);
export const notebookAnchor = (hash: string): NoteId | undefined =>
  NOTE_IDS.find((id) => hash === '#' + id);
export const serializeNotebookState = (state: NotebookState): string =>
  '#notes=' + encodeURIComponent(JSON.stringify(state));
export function restoreNotebookState(hash: string): { state: NotebookState; invalid: boolean } {
  const anchor = notebookAnchor(hash);
  const fallback = { state: initialNotebookState(), invalid: !!hash && !anchor };
  if (anchor) fallback.state.view.selected = anchor;
  if (!hash.startsWith('#notes=') || hash.length > 2500) return fallback;
  try {
    const x: unknown = JSON.parse(decodeURIComponent(hash.slice(7)));
    if (
      !record(x) ||
      x.version !== 1 ||
      x.seed !== 0 ||
      !record(x.config) ||
      !record(x.input) ||
      !record(x.view)
    )
      return fallback;
    const c = x.config,
      i = x.input,
      v = x.view;
    if (
      !integer(c.samples, 1, 5) ||
      !finite(c.eta, 0.05, 2.5) ||
      typeof c.schedule !== 'boolean' ||
      !finite(c.parametersB, 0.1, 1000) ||
      !finite(c.tokensB, 1, 100000) ||
      !Array.isArray(i.answers) ||
      i.answers.length !== 5 ||
      i.answers.some((a) => a !== null && a !== 3 && a !== 5 && a !== 7) ||
      !integer(i.examples, 0, 2) ||
      typeof i.role !== 'boolean' ||
      typeof i.evidence !== 'boolean' ||
      typeof i.format !== 'boolean' ||
      !integer(x.step, 0, 6) ||
      !NOTE_IDS.includes(v.selected as NoteId) ||
      typeof v.recall !== 'boolean'
    )
      return fallback;
    // Copy only public, explicitly validated fields. Extra URL fields are discarded.
    return {
      invalid: false,
      state: {
        version: 1,
        seed: 0,
        config: {
          samples: c.samples as number,
          eta: c.eta,
          schedule: c.schedule,
          parametersB: c.parametersB,
          tokensB: c.tokensB,
        },
        input: {
          answers: [...i.answers] as NotebookState['input']['answers'],
          examples: i.examples as number,
          role: i.role,
          evidence: i.evidence,
          format: i.format,
        },
        step: x.step as number,
        view: { selected: v.selected as NoteId, recall: v.recall },
      },
    };
  } catch {
    return fallback;
  }
}
