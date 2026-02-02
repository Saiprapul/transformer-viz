export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export const mulberry32 = (seed: number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
};

export const hashString = (value: string) => {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

export const seededVector = (seed: number, length: number, scale = 1) => {
  const rand = mulberry32(seed);
  return Array.from({ length }, () => (rand() * 2 - 1) * scale);
};

export const seededMatrix = (
  seed: number,
  rows: number,
  cols: number,
  scale = 1
) => {
  const rand = mulberry32(seed);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (rand() * 2 - 1) * scale)
  );
};

export const dot = (a: number[], b: number[]) =>
  a.reduce((sum, value, index) => sum + value * (b[index] ?? 0), 0);

export const addVectors = (a: number[], b: number[]) =>
  a.map((value, index) => value + (b[index] ?? 0));

export const sumVectors = (vectors: number[][]) => {
  const length = vectors[0]?.length ?? 0;
  return Array.from({ length }, (_, index) =>
    vectors.reduce((acc, vector) => acc + (vector?.[index] ?? 0), 0)
  );
};

export const scaleVector = (vector: number[], scale: number) =>
  vector.map((value) => value * scale);

export const vectorNorm = (vector: number[]) =>
  Math.sqrt(vector.reduce((acc, value) => acc + value * value, 0));

export const normalizeVector = (vector: number[]) => {
  const magnitude = vectorNorm(vector) || 1;
  return vector.map((value) => value / magnitude);
};

export const cosineSimilarity = (a: number[], b: number[]) => {
  const denom = (vectorNorm(a) || 1) * (vectorNorm(b) || 1);
  return denom === 0 ? 0 : dot(a, b) / denom;
};

export const softmax = (values: number[]) => {
  const max = Math.max(...values);
  const exps = values.map((value) =>
    value === -Infinity ? 0 : Math.exp(value - max)
  );
  const sum = exps.reduce((acc, value) => acc + value, 0) || 1;
  return exps.map((value) => value / sum);
};

export const formatNumber = (value: number, digits = 3) =>
  value.toFixed(digits);
