/** Small deterministic checks for the notebook. None of these run a language model. */
export function tallyAnswers(answers: readonly (number | null)[], count: number) {
  const used = answers.slice(0, count);
  const counts = new Map<number, number>();
  for (const answer of used) if (answer !== null) counts.set(answer, (counts.get(answer) ?? 0) + 1);
  const ranked = [...counts].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const valid = ranked.reduce((sum, entry) => sum + entry[1], 0);
  const winners = ranked.filter((entry) => entry[1] === ranked[0]?.[1]).map(([answer]) => answer);
  return {
    used,
    ranked,
    valid,
    invalid: used.length - valid,
    winners,
    winner: winners.length === 1 ? winners[0] : null,
    share: valid ? (ranked[0]?.[1] ?? 0) / valid : 0,
  };
}

/** L(w)=w²/2, plain SGD, with optional linear warmup then cosine decay. */
export function gradientTrajectory(eta: number, steps: number, schedule: boolean) {
  let w = 2;
  return Array.from({ length: steps + 1 }, (_, step) => {
    const rate = !schedule
      ? eta
      : step < 3
        ? (eta * (step + 1)) / 3
        : (eta * (1 + Math.cos((Math.PI * (step - 3)) / Math.max(1, steps - 3)))) / 2;
    const row = { step, w, loss: (w * w) / 2, gradient: w, rate };
    w -= rate * w;
    return row;
  });
}
