import { isWellFormed, MAX_MERGES, MAX_TEXT_LENGTH, type CorpusId } from '../math/tokenization';

export const TOKENIZATION_STEPS = 5;
export interface TokenizationState {
  version: 1;
  config: { corpus: CorpusId; merges: number };
  input: { text: string };
  step: number;
  seed: number;
  view: { speed: number; explanations: boolean; round: number };
}

export const initialTokenizationState = (): TokenizationState => ({
  version: 1,
  config: { corpus: 'pairs', merges: 3 },
  input: { text: 'abab ab 猫猫' },
  step: 0,
  seed: 42,
  view: { speed: 1, explanations: true, round: 0 },
});

const PREFIX = '#bpe=';
const record = (x: unknown): x is Record<string, unknown> =>
  typeof x === 'object' && x !== null && !Array.isArray(x);
const integer = (x: unknown, min: number, max: number): x is number =>
  typeof x === 'number' && Number.isInteger(x) && x >= min && x <= max;
const viewSpeeds = [0.5, 1, 2];
export const validTokenizationText = (text: string) =>
  text.length <= MAX_TEXT_LENGTH && isWellFormed(text);

export function restoreTokenizationState(hash: string) {
  const fallback = { state: initialTokenizationState(), invalid: Boolean(hash) };
  if (!hash.startsWith(PREFIX) || hash.length > 1600) return fallback;
  try {
    const raw: unknown = JSON.parse(decodeURIComponent(hash.slice(PREFIX.length)));
    if (
      !record(raw) ||
      raw.version !== 1 ||
      !record(raw.config) ||
      !record(raw.input) ||
      !record(raw.view) ||
      !integer(raw.step, 0, TOKENIZATION_STEPS - 1) ||
      !integer(raw.seed, 1, 9999)
    )
      return fallback;
    const { corpus, merges } = raw.config;
    const { text } = raw.input;
    const { speed, explanations, round } = raw.view;
    if (
      (corpus !== 'pairs' && corpus !== 'repeats') ||
      !integer(merges, 0, MAX_MERGES) ||
      typeof text !== 'string' ||
      !validTokenizationText(text) ||
      typeof speed !== 'number' ||
      !viewSpeeds.includes(speed) ||
      typeof explanations !== 'boolean' ||
      !integer(round, 0, merges)
    )
      return fallback;
    const state: TokenizationState = {
      version: 1,
      config: { corpus, merges },
      input: { text },
      step: raw.step,
      seed: raw.seed,
      view: { speed, explanations, round },
    };
    return { state, invalid: false };
  } catch {
    return fallback;
  }
}

export function serializeTokenizationState({ config, input, step, seed, view }: TokenizationState) {
  const hash =
    PREFIX + encodeURIComponent(JSON.stringify({ version: 1, config, input, step, seed, view }));
  const clean = restoreTokenizationState(hash);
  if (clean.invalid) throw new Error('Invalid tokenization state');
  return PREFIX + encodeURIComponent(JSON.stringify(clean.state));
}
