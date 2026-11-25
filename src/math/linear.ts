export type Vec2 = [number, number];
export type Mat2 = [Vec2, Vec2];

export type LinearPresetId = 'shear' | 'rotate90' | 'stretch' | 'mirror' | 'collapse' | 'identity';

export const LINEAR_PRESETS: Record<LinearPresetId, { w: Mat2; x: Vec2; k: Vec2 }> = {
  shear: {
    w: [
      [1, 0],
      [0.75, 1],
    ],
    x: [1, 1],
    k: [1.5, 0.5],
  },
  rotate90: {
    w: [
      [0, 1],
      [-1, 0],
    ],
    x: [1.25, 0.5],
    k: [1, 1],
  },
  stretch: {
    w: [
      [1.5, 0],
      [0, 0.5],
    ],
    x: [1, 1.5],
    k: [1, 1],
  },
  mirror: {
    w: [
      [-1, 0],
      [0, 1],
    ],
    x: [1, 1.25],
    k: [1, 0.5],
  },
  collapse: {
    w: [
      [1, 0.5],
      [1, 0.5],
    ],
    x: [1, -0.5],
    k: [1, 0.5],
  },
  identity: {
    w: [
      [1, 0],
      [0, 1],
    ],
    x: [1, 0.75],
    k: [1.25, 0.5],
  },
};

/** Fixed second linear map W₂ used to verify composition x(W₁W₂) = (xW₁)W₂ and non-commutativity. */
export const SECOND_MATRIX: Mat2 = [
  [0.5, 1],
  [-0.5, 0.5],
];

/** Row-vector linear map: y = x W = x₀·W[0] + x₁·W[1]. */
export function transformVec(x: Vec2, w: Mat2): Vec2 {
  return [x[0] * w[0][0] + x[1] * w[1][0], x[0] * w[0][1] + x[1] * w[1][1]];
}

export function matmul2x2(a: Mat2, b: Mat2): Mat2 {
  return [
    [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
    [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
  ];
}

export function determinant2x2(w: Mat2): number {
  return w[0][0] * w[1][1] - w[0][1] * w[1][0];
}

export function vecNorm(v: Vec2): number {
  return Math.hypot(v[0], v[1]);
}

export function dot2(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}

export interface DotGeometry {
  dot: number;
  normQ: number;
  normK: number;
  /** Cosine similarity q·k / (||q|| ||k||); null when either vector is zero. */
  cosine: number | null;
  /** Angle between q and k in degrees; null when either vector is zero. */
  angleDeg: number | null;
  /** Signed scalar projection of q onto unit direction of k; 0 when k is zero. */
  scalarProj: number;
  /** Vector projection of q onto line spanned by k. */
  projOnK: Vec2;
  /** Geometric formula ||q|| ||k|| cos(theta), matching algebraic dot product. */
  geometricDot: number;
}

export function dotGeometry(q: Vec2, k: Vec2): DotGeometry {
  const dot = dot2(q, k);
  const normQ = vecNorm(q);
  const normK = vecNorm(k);
  if (normQ === 0 || normK === 0) {
    return {
      dot: 0,
      normQ,
      normK,
      cosine: null,
      angleDeg: null,
      scalarProj: 0,
      projOnK: [0, 0],
      geometricDot: 0,
    };
  }
  const cosine = Math.max(-1, Math.min(1, dot / (normQ * normK)));
  const angleDeg = (Math.acos(cosine) * 180) / Math.PI;
  const kSq = dot2(k, k);
  const factor = dot / kSq;
  return {
    dot,
    normQ,
    normK,
    cosine,
    angleDeg,
    scalarProj: dot / normK,
    projOnK: [k[0] * factor, k[1] * factor],
    geometricDot: normQ * normK * cosine,
  };
}

export function computeLinear(w: Mat2, x: Vec2, k: Vec2) {
  const e0: Vec2 = [w[0][0], w[0][1]];
  const e1: Vec2 = [w[1][0], w[1][1]];
  const y = transformVec(x, w);
  const det = determinant2x2(w);
  const orientation: 'preserved' | 'flipped' | 'collapsed' =
    Math.abs(det) < 1e-12 ? 'collapsed' : det > 0 ? 'preserved' : 'flipped';

  // Step 2: dot product geometry using transformed vector q = xW and reference key k.
  const dotInfo = dotGeometry(y, k);
  const doubledK: Vec2 = [k[0] * 2, k[1] * 2];
  const doubledDotInfo = dotGeometry(y, doubledK);

  // Step 3: composition (x W₁) W₂ vs x (W₁ W₂) and non-commutativity W₁W₂ vs W₂W₁.
  const step1 = y;
  const step2 = transformVec(step1, SECOND_MATRIX);
  const composedW = matmul2x2(w, SECOND_MATRIX);
  const composedY = transformVec(x, composedW);
  const reversedW = matmul2x2(SECOND_MATRIX, w);
  const reversedY = transformVec(x, reversedW);
  const composeError = Math.max(
    Math.abs(step2[0] - composedY[0]),
    Math.abs(step2[1] - composedY[1]),
  );
  const commuteGap = Math.max(
    Math.abs(composedW[0][0] - reversedW[0][0]),
    Math.abs(composedW[0][1] - reversedW[0][1]),
    Math.abs(composedW[1][0] - reversedW[1][0]),
    Math.abs(composedW[1][1] - reversedW[1][1]),
  );

  // Four corners of the transformed unit square [0,0], [1,0], [1,1], [0,1].
  const unitSquare: Vec2[] = [
    [0, 0],
    e0,
    [e0[0] + e1[0], e0[1] + e1[1]],
    e1,
  ];

  return {
    w,
    x,
    k,
    e0,
    e1,
    y,
    det,
    area: Math.abs(det),
    orientation,
    unitSquare,
    dotInfo,
    doubledK,
    doubledDotInfo,
    step1,
    step2,
    composedW,
    composedY,
    reversedW,
    reversedY,
    composeError,
    commuteGap,
  };
}

export type LinearResult = ReturnType<typeof computeLinear>;
