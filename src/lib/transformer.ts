import {
  addVectors,
  dot,
  hashString,
  normalizeVector,
  seededMatrix,
  seededVector,
  softmax,
  sumVectors,
  vectorNorm
} from "./math";

export const D_MODEL = 32;
export const NUM_HEADS = 4;
export const D_HEAD = D_MODEL / NUM_HEADS;
export const FF_DIM = 64;

const BASE_VOCAB = [
  "the",
  "a",
  "an",
  "of",
  "to",
  "and",
  "in",
  "on",
  "for",
  "with",
  "transformer",
  "attention",
  "model",
  "learns",
  "predicts",
  "think",
  "sentence",
  "word",
  "tokens",
  "sequence",
  ".",
  ",",
  "?",
  "!",
  ":",
  ";"
];

const EXTENDED_VOCAB = [
  ...BASE_VOCAB,
  "cat", "dog", "sat", "mat", "hat", "bat", "rat",
  "is", "was", "are", "were", "been", "be", "has", "had", "have",
  "it", "he", "she", "they", "we", "you", "that", "this", "which",
  "not", "but", "or", "if", "then", "so", "as", "by", "from", "at",
  "can", "will", "would", "could", "should", "may", "might",
  "do", "does", "did", "make", "take", "give", "get", "go", "come",
  "see", "know", "say", "tell", "find", "use", "try", "ask",
  "next", "first", "last", "new", "old", "good", "best", "same", "other",
  "time", "way", "day", "thing", "world", "life", "work", "part",
  "right", "left", "up", "down", "out", "over", "back", "still",
  "also", "just", "more", "very", "much", "well", "now", "here",
  "about", "into", "through", "between", "after", "before",
  "many", "some", "each", "every", "all", "both", "few",
  "because", "when", "where", "while", "how", "what", "who",
  "people", "one", "two", "three", "only", "most", "own",
  "think", "thought", "know", "knew", "said", "made",
  "long", "great", "little", "big", "small", "high", "low",
  "pattern", "data", "layer", "head", "key", "value", "query",
  "input", "output", "hidden", "weight", "score", "matrix",
  "network", "neural", "deep", "learning", "training",
  "language", "text", "meaning", "context", "embedding",
  "curious", "studied", "map", "guessed", "quick", "brown",
  "fox", "jumped", "lazy", "floor", "table", "chair",
  "robot", "garden", "economy", "jazz", "quantum"
];

export type TokenBundle = {
  tokens: string[];
  ids: number[];
  embeddings: number[][];
  positions: number[][];
  combined: number[][];
  vocab: Map<string, number>;
};

export type AttentionPack = {
  scores: number[][][];
  weights: number[][][];
  outputs: number[][];
  headOutputs: number[][][];
};

export type ResidualComponents = {
  embedding: number[][];
  position: number[][];
  attention: number[][];
  mlp: number[][];
  combined: number[][];
};

export const tokenize = (text: string) => {
  const matches = text.match(/[\w']+|[^\s\w]/g);
  return matches ? matches : [];
};

export const buildVocab = (tokens: string[]) => {
  const vocab = new Map<string, number>();
  let index = 0;
  BASE_VOCAB.forEach((token) => {
    vocab.set(token, index);
    index += 1;
  });
  tokens.forEach((token) => {
    const normalized = token.toLowerCase();
    if (!vocab.has(normalized)) {
      vocab.set(normalized, index);
      index += 1;
    }
  });
  return vocab;
};

export const tokensToIds = (tokens: string[], vocab: Map<string, number>) =>
  tokens.map((token) => vocab.get(token.toLowerCase()) ?? 0);

export const embeddingForId = (id: number) => seededVector(id + 11, D_MODEL, 0.9);

export const embeddingForToken = (token: string) =>
  seededVector(hashString(token) + 31, D_MODEL, 0.9);

export const positionalEncoding = (position: number) => {
  const values: number[] = [];
  for (let i = 0; i < D_MODEL; i += 2) {
    const divisor = Math.pow(10000, i / D_MODEL);
    values[i] = Math.sin(position / divisor);
    values[i + 1] = Math.cos(position / divisor);
  }
  return values;
};

export const bundleTokens = (tokens: string[]): TokenBundle => {
  const vocab = buildVocab(tokens);
  const ids = tokensToIds(tokens, vocab);
  const embeddings = ids.map((id) => embeddingForId(id));
  const positions = tokens.map((_, index) => positionalEncoding(index));
  const combined = embeddings.map((vector, index) =>
    addVectors(vector, positions[index])
  );
  return { tokens, ids, embeddings, positions, combined, vocab };
};

export const createAttentionWeights = (seed: number) => {
  const weights = Array.from({ length: NUM_HEADS }, (_, head) => {
    const baseSeed = seed + head * 97;
    return {
      Wq: seededMatrix(baseSeed + 1, D_MODEL, D_HEAD, 0.45),
      Wk: seededMatrix(baseSeed + 2, D_MODEL, D_HEAD, 0.45),
      Wv: seededMatrix(baseSeed + 3, D_MODEL, D_HEAD, 0.45)
    };
  });
  return weights;
};

const project = (vector: number[], matrix: number[][]) => {
  const cols = matrix[0]?.length ?? 0;
  const output = Array.from({ length: cols }, () => 0);
  for (let row = 0; row < matrix.length; row += 1) {
    const value = vector[row] ?? 0;
    for (let col = 0; col < cols; col += 1) {
      output[col] += value * matrix[row][col];
    }
  }
  return output;
};

const matrixVectorMultiply = (matrix: number[][], vector: number[]) => {
  return matrix.map((row) => dot(row, vector));
};

const gelu = (value: number) => {
  const x = value;
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
};

export const computeAttention = (
  inputs: number[][],
  weightSets: ReturnType<typeof createAttentionWeights>
): AttentionPack => {
  const scores: number[][][] = [];
  const weights: number[][][] = [];
  const headOutputs: number[][][] = [];
  const scale = 1 / Math.sqrt(D_HEAD);

  weightSets.forEach((headWeights, head) => {
    const qVectors = inputs.map((vector) => project(vector, headWeights.Wq));
    const kVectors = inputs.map((vector) => project(vector, headWeights.Wk));
    const vVectors = inputs.map((vector) => project(vector, headWeights.Wv));

    const headScores = qVectors.map((q, row) =>
      kVectors.map((k, col) => (col > row ? -Infinity : dot(q, k) * scale))
    );

    const headWeightsNormalized = headScores.map((row) => softmax(row));
    const headOutput = headWeightsNormalized.map((rowWeights) => {
      const output = Array.from({ length: D_HEAD }, () => 0);
      rowWeights.forEach((weight, col) => {
        const valueVector = vVectors[col];
        if (!valueVector) return;
        for (let i = 0; i < output.length; i += 1) {
          output[i] += weight * valueVector[i];
        }
      });
      return output;
    });

    scores[head] = headScores;
    weights[head] = headWeightsNormalized;
    headOutputs[head] = headOutput;
  });

  const outputs = inputs.map((_, tokenIndex) => {
    const combined: number[] = [];
    for (let head = 0; head < NUM_HEADS; head += 1) {
      combined.push(...(headOutputs[head]?.[tokenIndex] ?? []));
    }
    return combined;
  });

  return { scores, weights, outputs, headOutputs };
};

export type MLPResult = {
  outputs: number[][];
  hiddenActivations: number[][];
};

export const computeFeedForward = (inputs: number[][], seed = 2024): MLPResult => {
  const W1 = seededMatrix(seed + 1, D_MODEL, FF_DIM, 0.4);
  const W2 = seededMatrix(seed + 2, FF_DIM, D_MODEL, 0.4);

  const hiddenActivations: number[][] = [];
  const outputs = inputs.map((vector) => {
    const hidden = project(vector, W1).map(gelu);
    hiddenActivations.push(hidden);
    return project(hidden, W2);
  });

  return { outputs, hiddenActivations };
};

export const buildResiduals = (
  bundle: TokenBundle,
  attentionOutput: number[][],
  mlpOutput: number[][]
): ResidualComponents => {
  const combined = bundle.embeddings.map((embedding, index) =>
    sumVectors([
      embedding,
      bundle.positions[index],
      attentionOutput[index] ?? [],
      mlpOutput[index] ?? []
    ])
  );
  return {
    embedding: bundle.embeddings,
    position: bundle.positions,
    attention: attentionOutput,
    mlp: mlpOutput,
    combined
  };
};

export type Prediction = {
  token: string;
  logit: number;
  probability: number;
};

export const predictFromVector = (
  vector: number[],
  inputTokens: string[],
  topK = 5
): Prediction[] => {
  const allTokens = new Set<string>(EXTENDED_VOCAB);
  inputTokens.forEach((t) => allTokens.add(t.toLowerCase()));

  const scored = Array.from(allTokens).map((token) => ({
    token,
    logit: dot(vector, embeddingForToken(token))
  }));

  scored.sort((a, b) => b.logit - a.logit);
  const topScored = scored.slice(0, topK);
  const logits = topScored.map((s) => s.logit);
  const probs = softmax(logits);

  return topScored.map((s, i) => ({
    token: s.token,
    logit: s.logit,
    probability: probs[i]
  }));
};

export const logitsToProbabilities = (logits: number[]) => softmax(logits);

export const vectorSummary = (vector: number[]) => {
  if (!vector.length) {
    return { min: 0, max: 0, avg: 0 };
  }
  const max = Math.max(...vector);
  const min = Math.min(...vector);
  const avg = vector.reduce((acc, value) => acc + value, 0) / vector.length;
  return { min, max, avg };
};

const computeCovariance = (vectors: number[][]) => {
  const dims = vectors[0]?.length ?? 0;
  const mean = Array.from({ length: dims }, () => 0);
  vectors.forEach((vector) => {
    vector.forEach((value, index) => {
      mean[index] += value;
    });
  });
  const scale = vectors.length ? 1 / vectors.length : 1;
  for (let i = 0; i < mean.length; i += 1) {
    mean[i] *= scale;
  }

  const centered = vectors.map((vector) =>
    vector.map((value, index) => value - mean[index])
  );

  const covariance = Array.from({ length: dims }, () =>
    Array.from({ length: dims }, () => 0)
  );

  centered.forEach((vector) => {
    for (let i = 0; i < dims; i += 1) {
      for (let j = 0; j < dims; j += 1) {
        covariance[i][j] += vector[i] * vector[j];
      }
    }
  });

  const denom = centered.length > 1 ? centered.length - 1 : 1;
  for (let i = 0; i < dims; i += 1) {
    for (let j = 0; j < dims; j += 1) {
      covariance[i][j] /= denom;
    }
  }

  return { covariance, centered };
};

const powerIteration = (matrix: number[][], seed: number) => {
  const dims = matrix.length;
  let vector = normalizeVector(seededVector(seed, dims, 1));
  for (let i = 0; i < 32; i += 1) {
    const multiplied = matrixVectorMultiply(matrix, vector);
    vector = normalizeVector(multiplied);
  }
  return vector;
};

export const projectPCA2D = (vectors: number[][]) => {
  if (!vectors.length) return [] as { x: number; y: number }[];
  const { covariance, centered } = computeCovariance(vectors);
  const v1 = powerIteration(covariance, 77);
  const lambda1 = dot(v1, matrixVectorMultiply(covariance, v1));

  const deflated = covariance.map((row, i) =>
    row.map((value, j) => value - lambda1 * v1[i] * v1[j])
  );

  const v2 = powerIteration(deflated, 1337);

  const raw = centered.map((vector) => ({
    x: dot(vector, v1),
    y: dot(vector, v2)
  }));

  const maxAbs = raw.reduce(
    (acc, point) => Math.max(acc, Math.abs(point.x), Math.abs(point.y)),
    1
  );

  return raw.map((point) => ({
    x: point.x / maxAbs,
    y: point.y / maxAbs
  }));
};

export type LayerResult = {
  attention: AttentionPack;
  mlp: number[][];
  mlpHidden: number[][];
  combined: number[][];
};

export const layerMix = (
  inputs: number[][],
  attentionWeights: ReturnType<typeof createAttentionWeights>,
  seed = 2024
): LayerResult => {
  const attention = computeAttention(inputs, attentionWeights);
  const attentionAdded = inputs.map((vector, index) =>
    addVectors(vector, attention.outputs[index] ?? [])
  );
  const mlpResult = computeFeedForward(attentionAdded, seed);
  const combined = attentionAdded.map((vector, index) =>
    addVectors(vector, mlpResult.outputs[index] ?? [])
  );
  return { attention, mlp: mlpResult.outputs, mlpHidden: mlpResult.hiddenActivations, combined };
};

export const vectorNorms = (vectors: number[][]) =>
  vectors.map((vector) => vectorNorm(vector));

export type LogitLensEntry = {
  layer: number;
  predictions: Prediction[];
};

export const computeLogitLens = (
  layerDetails: LayerResult[],
  tokenIndex: number,
  inputTokens: string[],
  topK = 5
): LogitLensEntry[] => {
  return layerDetails.map((layer, index) => {
    const vector = layer.combined[tokenIndex] ?? [];
    return {
      layer: index + 1,
      predictions: predictFromVector(vector, inputTokens, topK)
    };
  });
};

export type MLPAnalysis = {
  sparsity: number;
  topNeurons: { index: number; activation: number }[];
  inputNorm: number;
  outputNorm: number;
  deltaNorm: number;
};

export const analyzeMLP = (
  inputVector: number[],
  outputDelta: number[],
  hiddenActivations: number[],
  topK = 8
): MLPAnalysis => {
  const activeCount = hiddenActivations.filter((v) => Math.abs(v) > 0.01).length;
  const sparsity = 1 - activeCount / Math.max(hiddenActivations.length, 1);

  const indexed = hiddenActivations.map((activation, index) => ({ index, activation }));
  indexed.sort((a, b) => Math.abs(b.activation) - Math.abs(a.activation));

  return {
    sparsity,
    topNeurons: indexed.slice(0, topK),
    inputNorm: vectorNorm(inputVector),
    outputNorm: vectorNorm(addVectors(inputVector, outputDelta)),
    deltaNorm: vectorNorm(outputDelta)
  };
};
