import { T } from '../i18n';
import { Tx } from '../i18n/jsx';
/** Small controls shared by the experiments (styles in styles/experiment.css). */

export function Segment<T extends string | number>({
  label,
  options,
  value,
  onSelect,
  render = (option) => String(option),
}: {
  label: string;
  options: readonly T[];
  value: T;
  onSelect: (value: T) => void;
  render?: (option: T) => string;
}) {
  return (
    <div className="exp-segment" role="group" aria-label={label}>
      <span>{label}</span>
      {options.map((option) => (
        <button key={option} aria-pressed={option === value} onClick={() => onSelect(option)}>
          {render(option)}
        </button>
      ))}
    </div>
  );
}

/** Pick one answer before the page computes it. */
export function Prediction<T extends string>({
  question,
  options,
  choice,
  onChoose,
}: {
  question: string;
  options: { value: T; label: string }[];
  choice: T | null;
  onChoose: (choice: T) => void;
}) {
  return (
    <div className="exp-predict" role="group" aria-label={question}>
      <strong>{Tx('先预测：{#0}', question)}</strong>
      <div>
        {options.map((option) => (
          <button
            key={option.value}
            aria-pressed={choice === option.value}
            onClick={() => onChoose(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** One sentence comparing the learner's prediction with the computed answer. */
export function PredictionResult({ label, holds }: { label: string | null; holds: boolean }) {
  if (!label) return null;
  return (
    <>
      {Tx(
        ' 你的预测是“{#0}”，{#1}',
        label,
        holds ? T('与计算一致。') : T('与计算不同，对照左图再想一想。'),
      )}
    </>
  );
}
