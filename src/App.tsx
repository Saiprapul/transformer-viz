import { useEffect, useMemo, useState } from "react";
import Section from "./components/Section";
import StickyInput from "./components/StickyInput";
import SineWaveCanvas from "./components/SineWaveCanvas";
import { MatrixHeatmap, VectorHeatmap } from "./components/Heatmap";
import Callout from "./components/Callout";
import ProjectionPlot from "./components/ProjectionPlot";
import ResidualStream, { ResidualStage } from "./components/ResidualStream";
import ModuleNav from "./components/ModuleNav";
import GuidedRun from "./components/GuidedRun";
import SocialLinks from "./components/SocialLinks";
import {
  analyzeMLP,
  buildResiduals,
  bundleTokens,
  computeLogitLens,
  createAttentionWeights,
  D_HEAD,
  D_MODEL,
  embeddingForToken,
  FF_DIM,
  layerMix,
  predictFromVector,
  projectPCA2D,
  vectorSummary
} from "./lib/transformer";
import { cosineSimilarity, dot, formatNumber, vectorNorm } from "./lib/math";

const MAX_TOKENS = 12;
const STACK_LAYERS = 6;

const MathBlock = ({ children }: { children: React.ReactNode }) => (
  <pre className="grid-bg overflow-x-auto rounded-2xl border border-white/10 bg-ink2/60 p-4 text-xs text-slate-200">
    {children}
  </pre>
);

const MathDetails = ({ summary, children }: { summary: string; children: React.ReactNode }) => (
  <details className="rounded-2xl border border-white/10 bg-ink2/60 p-4 text-xs text-slate-200">
    <summary className="cursor-pointer text-xs uppercase tracking-[0.3em] text-muted">
      {summary}
    </summary>
    <pre className="mt-3 whitespace-pre-wrap font-mono text-xs text-slate-200">{children}</pre>
  </details>
);

const App = () => {
  const [text, setText] = useState(
    "The curious cat studied the attention map and guessed the next word"
  );
  const [compareText, setCompareText] = useState(
    "The curious dog studied the attention map and guessed the next word"
  );
  const [compareMode, setCompareMode] = useState(false);
  const [showMath, setShowMath] = useState(true);
  const [showRawEmbedding, setShowRawEmbedding] = useState(false);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(
    null
  );
  const [activeStage, setActiveStage] = useState<ResidualStage>("final");
  const [activeModule, setActiveModule] = useState(0);
  const [guidedRun, setGuidedRun] = useState(false);
  const [layerIndex, setLayerIndex] = useState(0);

  const tokens = useMemo(() => {
    const raw = text.trim();
    if (!raw) return [];
    return raw.match(/[\w']+|[^\s\w]/g) ?? [];
  }, [text]);

  const trimmedTokens = tokens.slice(0, MAX_TOKENS);
  const isTruncated = tokens.length > MAX_TOKENS;
  const lastIndex = Math.max(trimmedTokens.length - 1, 0);

  useEffect(() => {
    if (selectedTokenIndex >= trimmedTokens.length) {
      setSelectedTokenIndex(Math.max(trimmedTokens.length - 1, 0));
    }
  }, [selectedTokenIndex, trimmedTokens.length]);

  const bundle = useMemo(() => bundleTokens(trimmedTokens), [trimmedTokens]);

  const layerWeightSets = useMemo(
    () =>
      Array.from({ length: STACK_LAYERS }, (_, index) =>
        createAttentionWeights(4242 + index * 17)
      ),
    []
  );

  const layerDetails = useMemo(() => {
    let vectors = bundle.combined;
    return layerWeightSets.map((weights, index) => {
      const mix = layerMix(vectors, weights, 3200 + index * 19);
      vectors = mix.combined;
      return mix;
    });
  }, [bundle.combined, layerWeightSets]);

  const primaryLayer = layerDetails[0];
  const lastLayerIndex = Math.max(layerDetails.length - 1, 0);
  const finalLayer = layerDetails[lastLayerIndex];

  const baseResiduals = useMemo(
    () => buildResiduals(bundle, primaryLayer?.attention.outputs ?? [], primaryLayer?.mlp ?? []),
    [bundle, primaryLayer]
  );
  const stackResiduals = useMemo(() => {
    const layer = layerDetails[layerIndex];
    return buildResiduals(bundle, layer?.attention.outputs ?? [], layer?.mlp ?? []);
  }, [bundle, layerDetails, layerIndex]);
  const finalResiduals = useMemo(() => {
    const layer = layerDetails[lastLayerIndex];
    return buildResiduals(bundle, layer?.attention.outputs ?? [], layer?.mlp ?? []);
  }, [bundle, layerDetails, lastLayerIndex]);

  const residualView =
    activeStage === "stack"
      ? stackResiduals
      : activeStage === "final"
      ? finalResiduals
      : baseResiduals;

  const finalVectors = finalLayer?.combined ?? baseResiduals.combined;

  const activeEmbedding = bundle.embeddings[selectedTokenIndex] ?? [];
  const activePosition = bundle.positions[selectedTokenIndex] ?? [];
  const activeCombined = bundle.combined[selectedTokenIndex] ?? [];

  const mlpAnalysis = useMemo(() => {
    const inputVec = bundle.combined[lastIndex] ?? [];
    const mlpDelta = primaryLayer?.mlp[lastIndex] ?? [];
    const hidden = primaryLayer?.mlpHidden[lastIndex] ?? [];
    return analyzeMLP(inputVec, mlpDelta, hidden);
  }, [bundle.combined, lastIndex, primaryLayer]);

  const predictions = useMemo(
    () => predictFromVector(finalVectors[lastIndex] ?? [], trimmedTokens),
    [finalVectors, lastIndex, trimmedTokens]
  );

  const attentionMatrix = primaryLayer?.attention.weights[selectedHead] ?? [];
  const selectedInfo = selectedCell
    ? {
        rowToken: trimmedTokens[selectedCell.row],
        colToken: trimmedTokens[selectedCell.col],
        value: attentionMatrix[selectedCell.row]?.[selectedCell.col] ?? 0,
        masked: selectedCell.col > selectedCell.row
      }
    : null;

  const similarityRows = useMemo(() => {
    return bundle.embeddings
      .map((vector, index) => ({
        token: trimmedTokens[index] ?? "",
        similarity: cosineSimilarity(activeEmbedding, vector),
        norm: vectorNorm(vector),
        index
      }))
      .sort((a, b) => b.similarity - a.similarity);
  }, [activeEmbedding, bundle.embeddings, trimmedTokens]);

  const projection = useMemo(() => projectPCA2D(bundle.embeddings), [bundle.embeddings]);

  const finalAttentionWeights = finalLayer?.attention.weights ?? [];
  const avgAttention = useMemo(() => {
    if (!trimmedTokens.length) return [];
    return trimmedTokens.map((_, col) => {
      const sum = finalAttentionWeights.reduce(
        (acc, head) => acc + (head[lastIndex]?.[col] ?? 0),
        0
      );
      return sum / Math.max(finalAttentionWeights.length, 1);
    });
  }, [finalAttentionWeights, lastIndex, trimmedTokens]);

  const topAttentionSources = useMemo(() => {
    return avgAttention
      .map((value, index) => ({ token: trimmedTokens[index], value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 3);
  }, [avgAttention, trimmedTokens]);

  const contributionTable = useMemo(() => {
    const components = {
      embedding: finalResiduals.embedding[lastIndex] ?? [],
      position: finalResiduals.position[lastIndex] ?? [],
      attention: finalResiduals.attention[lastIndex] ?? [],
      mlp: finalResiduals.mlp[lastIndex] ?? []
    };

    return predictions.slice(0, 4).map((item) => ({
      token: item.token,
      probability: item.probability,
      contributions: {
        embedding: dot(components.embedding, embeddingForToken(item.token)),
        position: dot(components.position, embeddingForToken(item.token)),
        attention: dot(components.attention, embeddingForToken(item.token)),
        mlp: dot(components.mlp, embeddingForToken(item.token))
      }
    }));
  }, [predictions, finalResiduals, lastIndex]);

  const logitLens = useMemo(
    () => computeLogitLens(layerDetails, lastIndex, trimmedTokens),
    [layerDetails, lastIndex, trimmedTokens]
  );

  const layerNorms = useMemo(
    () => layerDetails.map((layer) => vectorNorm(layer.combined[lastIndex] ?? [])),
    [layerDetails, lastIndex]
  );
  const maxLayerNorm = Math.max(...layerNorms, 1);

  const focusLayers = useMemo(() => {
    return layerDetails.map((layer, index) => {
      const weights = layer.attention.weights;
      const avg = trimmedTokens.map((_, col) => {
        const sum = weights.reduce(
          (acc, head) => acc + (head[selectedTokenIndex]?.[col] ?? 0),
          0
        );
        return sum / Math.max(weights.length, 1);
      });
      const topSources = avg
        .map((value, col) => ({ token: trimmedTokens[col], value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 3);

      return {
        layer: index + 1,
        topSources,
        mlpDelta: vectorNorm(layer.mlp[selectedTokenIndex] ?? [])
      };
    });
  }, [layerDetails, selectedTokenIndex, trimmedTokens]);

  const replaceLastToken = (word: string) => {
    if (!text.trim()) {
      setText(word);
      return;
    }
    const match = text.match(/(.*?)([\w']+)(\W*)$/);
    if (match) {
      setText(`${match[1]}${word}${match[3]}`);
    } else {
      setText(`${text.trim()} ${word}`);
    }
  };

  const compareTokens = useMemo(() => {
    const raw = compareText.trim();
    if (!raw) return [];
    return raw.match(/[\w']+|[^\s\w]/g) ?? [];
  }, [compareText]);
  const compareTrimmed = compareTokens.slice(0, MAX_TOKENS);
  const compareBundle = useMemo(() => bundleTokens(compareTrimmed), [compareTrimmed]);
  const compareLayerDetails = useMemo(() => {
    let vectors = compareBundle.combined;
    return layerWeightSets.map((weights, index) => {
      const mix = layerMix(vectors, weights, 5200 + index * 19);
      vectors = mix.combined;
      return mix;
    });
  }, [compareBundle.combined, layerWeightSets]);
  const compareFinalLayer = compareLayerDetails[lastLayerIndex];
  const compareFinalVectors = compareFinalLayer?.combined ?? compareBundle.combined;

  const comparePredictions = useMemo(
    () =>
      predictFromVector(
        compareFinalVectors[Math.max(compareTrimmed.length - 1, 0)] ?? [],
        compareTrimmed
      ),
    [compareFinalVectors, compareTrimmed]
  );

  const compareAttentionSources = useMemo(() => {
    const compareWeights = compareFinalLayer?.attention.weights ?? [];
    const compareLastIndex = Math.max(compareTrimmed.length - 1, 0);
    if (!compareTrimmed.length) return [];
    const avg = compareTrimmed.map((_, col) => {
      const sum = compareWeights.reduce(
        (acc, head) => acc + (head[compareLastIndex]?.[col] ?? 0),
        0
      );
      return sum / Math.max(compareWeights.length, 1);
    });
    return avg
      .map((value, index) => ({ token: compareTrimmed[index], value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 3);
  }, [compareFinalLayer, compareTrimmed]);

  const deltaRows = useMemo(() => {
    const primaryMap = new Map(predictions.map((item) => [item.token, item.probability]));
    const compareMap = new Map(comparePredictions.map((item) => [item.token, item.probability]));
    const tokensSet = new Set([...primaryMap.keys(), ...compareMap.keys()]);

    return Array.from(tokensSet)
      .map((token) => ({
        token,
        primary: primaryMap.get(token) ?? 0,
        compare: compareMap.get(token) ?? 0,
        delta: (primaryMap.get(token) ?? 0) - (compareMap.get(token) ?? 0)
      }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
      .slice(0, 4);
  }, [predictions, comparePredictions]);

  const topPrediction = predictions[0];

  const handleStageEnter = (stage: ResidualStage, module: number) => () => {
    setActiveStage(stage);
    setActiveModule(module);
  };

  return (
    <div className="min-h-screen">
      <div className="pointer-events-none fixed inset-0 opacity-50" aria-hidden="true">
        <div className="absolute left-[10%] top-[30%] h-72 w-72 rounded-full bg-accent/10 blur-[120px]" />
        <div className="absolute right-[15%] top-[10%] h-72 w-72 rounded-full bg-accent2/10 blur-[130px]" />
      </div>

      <ModuleNav activeModule={activeModule} />

      <StickyInput
        text={text}
        onChange={setText}
        tokens={trimmedTokens}
        tokenIds={bundle.ids}
        predictions={predictions}
        onReplaceLastToken={replaceLastToken}
        showMath={showMath}
        onToggleMath={() => setShowMath((prev) => !prev)}
        maxTokens={MAX_TOKENS}
        isTruncated={isTruncated}
        compareMode={compareMode}
        onToggleCompare={() => setCompareMode((prev) => !prev)}
      />

      <main className="relative z-10 mx-auto flex max-w-6xl flex-col gap-10 px-6 py-12">
        <ResidualStream
          tokens={trimmedTokens}
          components={{
            embedding: residualView.embedding,
            position: residualView.position,
            attention: residualView.attention,
            mlp: residualView.mlp
          }}
          stage={activeStage}
          highlightIndex={lastIndex}
        />

        <Section
          id="module-0"
          module="Module 0"
          title="The Prediction"
          kicker="Start at the end"
          onEnter={handleStageEnter("final", 0)}
        >
          <Callout label="Before" variant="question">
            What does "predict the next word" really mean in a transformer?
          </Callout>
          <p>
            The model takes your sentence and produces a probability distribution over the next
            token. Everything below will trace that distribution back to its source.
          </p>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">Top prediction right now</p>
              <p className="mt-3 text-2xl text-white">{topPrediction?.token ?? "—"}</p>
              <p className="mt-2 text-xs text-muted">
                Probability: {topPrediction ? formatNumber(topPrediction.probability * 100, 1) : "0"}%
              </p>
            </div>
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">We will trace the chain</p>
              <ul className="mt-3 space-y-2 text-sm text-slate-200">
                <li>Tokens → IDs</li>
                <li>IDs → vectors</li>
                <li>Position added</li>
                <li>Attention reads context</li>
                <li>MLP transforms meaning</li>
                <li>Stacked layers → logits</li>
              </ul>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setGuidedRun(true)}
            className="w-full rounded-2xl border border-accent/30 bg-accent/10 px-5 py-3 text-sm text-accent transition hover:bg-accent/20"
          >
            Walk me through this prediction
          </button>
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              The output isn't a single word — it's a ranked list of probabilities.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              The model "knows" the answer. It only scores possibilities and picks the highest.
            </Callout>
          </div>
        </Section>

        <Section
          id="module-1"
          module="Module 1"
          title="Text → Tokens → IDs"
          kicker="Tokenization"
          onEnter={handleStageEnter("raw", 1)}
        >
          <Callout label="Before" variant="question">
            What pieces of text does the model actually see?
          </Callout>
          <p>
            Tokenization is learned. It doesn’t respect words or syllables — it finds subword chunks
            that make prediction efficient.
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            {bundle.tokens.map((token, index) => (
              <div
                key={`${token}-${index}`}
                className="flex items-center justify-between rounded-2xl border border-white/10 bg-ink2/60 px-4 py-3"
              >
                <span className="text-white">{token}</span>
                <span className="font-mono text-xs text-muted">ID {bundle.ids[index]}</span>
              </div>
            ))}
          </div>
          {showMath ? (
            <>
              <MathBlock>
                {`tokens = split(text)
ids = vocab_lookup(tokens)`}
              </MathBlock>
              <MathDetails summary="Show derivation">
                {`Tokenizer learns a vocabulary V
text → byte pairs → merge rules → tokens
IDs = index of each token in V`}
              </MathDetails>
            </>
          ) : null}
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              Tokens can be parts of words. “unhappiness” might split into “un” + “happiness”.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              Tokens equal words. They don’t — punctuation and subword pieces are tokens too.
            </Callout>
          </div>
        </Section>

        <Section
          id="module-2"
          module="Module 2"
          title="Embeddings Are Geometry"
          kicker="Meaning as vectors"
          onEnter={handleStageEnter("embed", 2)}
        >
          <Callout label="Before" variant="question">
            How does a numeric vector capture meaning?
          </Callout>
          <p>
            The embedding isn’t meaningful per dimension. Meaning lives in *direction* and *distance*
            within the space.
          </p>
          <div className="flex flex-wrap gap-2">
            {bundle.tokens.map((token, index) => (
              <button
                key={`token-select-${index}`}
                type="button"
                onClick={() => setSelectedTokenIndex(index)}
                className={`chip transition ${
                  selectedTokenIndex === index ? "bg-accent/20" : "bg-white/5"
                }`}
              >
                {token}
              </button>
            ))}
          </div>
          <div className="grid gap-6 md:grid-cols-[1.1fr_0.9fr]">
            <ProjectionPlot
              points={projection.map((point, index) => ({
                x: point.x,
                y: point.y,
                label: trimmedTokens[index] ?? "",
                active: index === selectedTokenIndex
              }))}
            />
            <div className="space-y-4">
              <div className="glass rounded-2xl p-4">
                <p className="text-sm text-muted">Cosine similarity (to selected token)</p>
                <div className="mt-3 space-y-2">
                  {similarityRows.map((row) => (
                    <div key={`${row.token}-${row.index}`} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-white">{row.token}</span>
                        <span className="text-muted">{formatNumber(row.similarity, 2)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-white/5">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-accent2/80 to-accent/80"
                          style={{ width: `${Math.max(row.similarity, 0) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="glass rounded-2xl p-4">
                <p className="text-sm text-muted">Vector norms</p>
                <div className="mt-3 grid gap-2 text-xs">
                  {bundle.embeddings.map((vector, index) => (
                    <div key={`norm-${index}`} className="flex items-center justify-between">
                      <span className="text-white">{trimmedTokens[index]}</span>
                      <span className="text-muted">{formatNumber(vectorNorm(vector), 2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => setShowRawEmbedding((prev) => !prev)}
              className={`chip transition ${showRawEmbedding ? "bg-accent/20" : "bg-white/5"}`}
            >
              {showRawEmbedding ? "Hide raw vector" : "Show raw vector"}
            </button>
          </div>
          {showRawEmbedding ? (
            <div className="mt-4 grid gap-6 md:grid-cols-[1.1fr_0.9fr]">
              <VectorHeatmap values={activeEmbedding} columns={8} label="Embedding values" />
              <div className="space-y-4 rounded-2xl border border-white/10 bg-ink2/60 p-4">
                <p className="text-sm text-muted">Embedding summary</p>
                {(() => {
                  const summary = vectorSummary(activeEmbedding);
                  return (
                    <div className="space-y-2 text-sm">
                      <p>Min: {formatNumber(summary.min)}</p>
                      <p>Max: {formatNumber(summary.max)}</p>
                      <p>Avg: {formatNumber(summary.avg)}</p>
                    </div>
                  );
                })()}
                <p className="mt-3 narrative">
                  This view is intentionally hidden: the *pattern* matters, not any single cell.
                </p>
              </div>
            </div>
          ) : null}
          {showMath ? (
            <>
              <MathBlock>{`embedding = E[id]`}</MathBlock>
              <MathDetails summary="Show derivation">
                {`E ∈ R^{|V| × d}
embedding(token) = one_hot(id) · E`}
              </MathDetails>
            </>
          ) : null}
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              Similar meaning = nearby vectors, not matching individual dimensions.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              Each coordinate has semantic meaning. It doesn't — geometry is what matters.
            </Callout>
          </div>
          <Callout label="Later" variant="insight">
            In Module 8, you'll see how this starting vector gets edited by attention and MLPs —
            and why the final direction determines the prediction.
          </Callout>
        </Section>

        <Section
          id="module-3"
          module="Module 3"
          title="Add Position"
          kicker="Order matters"
          onEnter={handleStageEnter("position", 3)}
        >
          <Callout label="Before" variant="question">
            How does the model know which word came first?
          </Callout>
          <p>
            Positional encodings add wave patterns so the model can distinguish “cat sat” from “sat
            cat”.
          </p>
          <div className="grid gap-6 md:grid-cols-[1.2fr_0.8fr]">
            <SineWaveCanvas length={bundle.tokens.length} />
            <div className="space-y-4">
              <VectorHeatmap values={activePosition} columns={8} label="Position signal" />
              <VectorHeatmap values={activeCombined} columns={8} label="Embedding + position" />
            </div>
          </div>
          {showMath ? (
            <>
              <MathBlock>
                {`PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
input = embedding + PE`}
              </MathBlock>
              <MathDetails summary="Show derivation">
                {`PE injects periodic signals
This lets attention infer relative positions
input vectors keep original meaning + order`}
              </MathDetails>
            </>
          ) : null}
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              The same word in a new position gets a different vector.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              Position replaces the embedding. It is *added* to it in the residual stream.
            </Callout>
          </div>
          <Callout label="Later" variant="insight">
            This add-not-replace pattern is the residual stream. Every future operation — attention,
            MLP — also adds. The forensic breakdown traces exactly what each addition contributed.
          </Callout>
        </Section>

        <Section
          id="module-4"
          module="Module 4"
          title="Attention = Where to Read"
          kicker="Context routing"
          onEnter={handleStageEnter("attention", 4)}
        >
          <Callout label="Before" variant="question">
            Which earlier tokens should each token pay attention to?
          </Callout>
          <p>
            Attention decides *where to read from*. The feed-forward layers decide *what to do with
            what was read*.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            {Array.from({ length: 4 }, (_, index) => (
              <button
                key={`head-${index}`}
                type="button"
                className={`chip transition ${selectedHead === index ? "bg-accent/20" : "bg-white/5"}`}
                onClick={() => setSelectedHead(index)}
              >
                Head {index + 1}
              </button>
            ))}
            <span className="text-sm text-muted">Each head is a different perspective.</span>
          </div>
          <MatrixHeatmap
            matrix={attentionMatrix}
            tokens={trimmedTokens}
            selectedCell={selectedCell}
            onSelect={(row, col) => setSelectedCell({ row, col })}
          />
          <div className="grid gap-4 md:grid-cols-2">
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">Selected attention</p>
              {selectedInfo ? (
                <div className="mt-2 space-y-1 text-sm">
                  <p>
                    Query token: <span className="text-white">{selectedInfo.rowToken}</span>
                  </p>
                  <p>
                    Key token: <span className="text-white">{selectedInfo.colToken}</span>
                  </p>
                  <p>
                    Weight: <span className="text-white">{formatNumber(selectedInfo.value)}</span>
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-200 max-w-prose">
                    {selectedInfo.masked
                      ? "Masked (future token)."
                      : "Higher weight means stronger influence."}
                  </p>
                </div>
              ) : (
                <p className="mt-2 text-sm text-muted">Click a cell in the heatmap.</p>
              )}
            </div>
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">Dimensions</p>
              <div className="mt-2 space-y-1 text-sm">
                <p>Model width: {D_MODEL}</p>
                <p>Heads: 4</p>
                <p>Per-head width: {D_HEAD}</p>
                <p className="text-sm leading-6 text-slate-200">Causal mask prevents looking ahead.</p>
              </div>
            </div>
          </div>
          {showMath ? (
            <>
              <MathBlock>
                {`Q = XWq   K = XWk   V = XWv
scores = (Q · K^T) / sqrt(d_head)
weights = softmax(mask(scores))
output = weights · V`}
              </MathBlock>
              <MathDetails summary="Show derivation">
                {`Q, K, V ∈ R^{n × d_head}
mask sets future positions to -∞
softmax converts scores to probabilities`}
              </MathDetails>
            </>
          ) : null}
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              Attention redistributes information; it doesn’t decide the final answer.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              Attention weights are explanations. They're a routing mechanism, not a full story.
            </Callout>
          </div>
          <Callout label="Later" variant="insight">
            In the forensic breakdown, you'll see exactly which tokens attention pulled from at
            each layer — and how that routing shaped the final prediction.
          </Callout>
        </Section>

        <Section
          id="module-5"
          module="Module 5"
          title="Feed-Forward = The Thinker"
          kicker="MLP layers"
          onEnter={handleStageEnter("mlp", 5)}
        >
          <Callout label="Before" variant="question">
            Where does most of the computation actually happen?
          </Callout>
          <p>
            Attention gathers evidence from other tokens. The MLP decides how to reinterpret it —
            expanding each vector into a wider hidden space, applying nonlinearity, then compressing
            back. This is where 2/3 of all parameters live.
          </p>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">Before MLP</p>
              <p className="mt-2 text-2xl text-white">{formatNumber(mlpAnalysis.inputNorm, 2)}</p>
              <p className="mt-1 text-xs text-muted">Vector norm (last token)</p>
            </div>
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">MLP delta</p>
              <p className="mt-2 text-2xl text-accent3">{formatNumber(mlpAnalysis.deltaNorm, 2)}</p>
              <p className="mt-1 text-xs text-muted">How much the MLP changed the vector</p>
            </div>
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">After MLP</p>
              <p className="mt-2 text-2xl text-white">{formatNumber(mlpAnalysis.outputNorm, 2)}</p>
              <p className="mt-1 text-xs text-muted">New vector norm</p>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">Activation sparsity</p>
              <div className="mt-3 flex items-end gap-3">
                <p className="text-3xl text-accent3">{formatNumber(mlpAnalysis.sparsity * 100, 0)}%</p>
                <p className="pb-1 text-xs text-muted">of {FF_DIM} neurons inactive</p>
              </div>
              <div className="mt-3 h-3 overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full bg-accent3/70"
                  style={{ width: `${(1 - mlpAnalysis.sparsity) * 100}%` }}
                />
              </div>
              <p className="mt-2 narrative">
                Only a small fraction of neurons fire for any given token. The rest stay near zero
                after GELU — this is how the MLP specializes.
              </p>
            </div>
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">Most active neurons (by |activation|)</p>
              <div className="mt-3 space-y-2">
                {mlpAnalysis.topNeurons.map((neuron) => {
                  const maxAct = Math.abs(mlpAnalysis.topNeurons[0]?.activation ?? 1);
                  return (
                    <div key={`neuron-${neuron.index}`} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted">Neuron {neuron.index}</span>
                        <span className="text-white">{formatNumber(neuron.activation, 3)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-white/5">
                        <div
                          className="h-2 rounded-full"
                          style={{
                            width: `${(Math.abs(neuron.activation) / maxAct) * 100}%`,
                            background: neuron.activation >= 0 ? "#7bd389" : "#4fd1c5"
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="mt-3 narrative">
                Individual neurons don't have interpretable meanings. These are simply the
                highest-magnitude activations — "highly active" doesn't mean "detects X."
              </p>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">Parameter share (typical)</p>
              <div className="mt-3 space-y-3">
                <div>
                  <div className="flex items-center justify-between text-xs text-muted">
                    <span>Attention</span>
                    <span>~1/3</span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-white/5">
                    <div className="h-2 w-1/3 rounded-full bg-accent/70" />
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between text-xs text-muted">
                    <span>MLP</span>
                    <span>~2/3</span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-white/5">
                    <div className="h-2 w-2/3 rounded-full bg-accent3/70" />
                  </div>
                </div>
              </div>
            </div>
            <div className="glass rounded-2xl p-4">
              <p className="text-sm text-muted">MLP dimensions</p>
              <div className="mt-3 space-y-1 text-sm">
                <p>Model width: {D_MODEL}</p>
                <p>Hidden size: {FF_DIM}</p>
                <p>Expansion: {FF_DIM / D_MODEL}x</p>
                <p>Nonlinearity: GELU</p>
              </div>
            </div>
          </div>
          {showMath ? (
            <>
              <MathBlock>
                {`h = GELU(XW1)        // expand ${D_MODEL} → ${FF_DIM}
mlp_out = hW2          // compress ${FF_DIM} → ${D_MODEL}
residual = X + mlp_out // add back (not replace)`}
              </MathBlock>
              <MathDetails summary="Show derivation">
                {`W1 ∈ R^{${D_MODEL} × ${FF_DIM}}   (${D_MODEL * FF_DIM} params)
W2 ∈ R^{${FF_DIM} × ${D_MODEL}}   (${FF_DIM * D_MODEL} params)
Total MLP params per layer: ${2 * D_MODEL * FF_DIM}
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
Sparsity emerges from GELU zeroing negative pre-activations`}
              </MathDetails>
            </>
          ) : null}
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              Attention tells the model *where* to look. The MLP transforms *what it found* —
              most neurons stay silent, and a few fire strongly.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              Transformers are only attention. Most parameters and computation live in the MLPs.
            </Callout>
          </div>
        </Section>

        <Section
          id="module-6"
          module="Module 6"
          title="Stack Layers + Predict"
          kicker="Residual edits"
          onEnter={handleStageEnter("stack", 6)}
        >
          <Callout label="Before" variant="question">
            Why stack many layers instead of one?
          </Callout>
          <p>
            Each layer re-edits the same residual stream. Stacking creates depth: context becomes
            richer, and the final vector grows more specialized for prediction.
          </p>
          <div className="glass rounded-2xl p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-muted">Layer scrubber</p>
              <span className="text-xs text-muted">Layer {layerIndex + 1} of {STACK_LAYERS}</span>
            </div>
            <input
              type="range"
              min={1}
              max={STACK_LAYERS}
              step={1}
              value={layerIndex + 1}
              onChange={(event) => setLayerIndex(Number(event.target.value) - 1)}
              className="mt-4 w-full"
            />
            <p className="mt-3 narrative">
              Scrub to see the residual stream update with each layer’s attention and MLP edits.
            </p>
          </div>
          <div className="glass rounded-2xl p-4">
            <p className="text-sm text-muted">Residual norm across layers (last token)</p>
            <div className="mt-3 space-y-3">
              {layerNorms.map((normValue, index) => (
                <div key={`layer-${index}`} className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-muted">
                    <span>Layer {index + 1}</span>
                    <span>{formatNumber(normValue, 2)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/5">
                    <div
                      className="h-2 rounded-full bg-gradient-to-r from-accent/80 to-accent2/80"
                      style={{ width: `${Math.min(normValue / maxLayerNorm, 1) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          {showMath ? (
            <>
              <MathBlock>
                {`for each layer:
  X = X + attention(X)
  X = X + mlp(X)`}
              </MathBlock>
            </>
          ) : null}
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              Each layer is a small edit to the same vectors, not a brand new pipeline stage.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              Layers are independent modules. They are repeated edits to one residual stream.
            </Callout>
          </div>

          {/* From thought to choice — the hinge */}
          <div className="mt-4 rounded-2xl border border-accent/20 bg-accent/5 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-accent">From thought to choice</p>
            <div className="mt-4 space-y-4">
              <p className="narrative">
                After {STACK_LAYERS} layers of edits, the residual stream holds everything the model
                knows. But how does an internal vector become a word?
              </p>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-xl border border-white/10 bg-ink2/60 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted">1. State</p>
                  <p className="mt-2 narrative">
                    The last token's vector now encodes grammar, meaning, and context — all mixed
                    into {D_MODEL} dimensions.
                  </p>
                </div>
                <div className="rounded-xl border border-white/10 bg-ink2/60 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted">2. Question</p>
                  <p className="mt-2 narrative">
                    Each candidate token asks: "How compatible am I with this state?" The answer
                    is a dot product — the logit.
                  </p>
                </div>
                <div className="rounded-xl border border-white/10 bg-ink2/60 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted">3. Choice</p>
                  <p className="mt-2 narrative">
                    Softmax converts logits into probabilities. The token most aligned with the
                    residual state wins.
                  </p>
                </div>
              </div>
              <details className="rounded-2xl border border-white/10 bg-ink2/60 p-4">
                <summary className="cursor-pointer text-sm font-medium text-slate-200">
                  Show alignment details (advanced)
                  <span className="mt-1 block text-sm text-muted">
                    Optional: see how the final state compares against candidate tokens.
                  </span>
                </summary>
                <div className="mt-4 space-y-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-muted">
                      Final vector components
                    </p>
                    <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted">
                      <span className="inline-flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ background: "#4fd1c5" }} />
                        Embedding
                      </span>
                      <span className="inline-flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ background: "#8aa7ff" }} />
                        Position
                      </span>
                      <span className="inline-flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ background: "#ffb457" }} />
                        Attention
                      </span>
                      <span className="inline-flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ background: "#7bd389" }} />
                        MLP
                      </span>
                    </div>
                  </div>
                  <div className="glass rounded-2xl p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted">
                      Alignment: final vector vs. candidate embeddings
                    </p>
                    <div className="mt-3 space-y-2">
                      {predictions.map((item, rank) => (
                        <div key={`alignment-${item.token}`} className="flex items-center gap-3">
                          <span className="w-20 text-right text-xs text-white">{item.token}</span>
                          <div className="flex-1">
                            <div className="h-3 rounded-full bg-white/5">
                              <div
                                className="h-3 rounded-full"
                                style={{
                                  width: `${(item.probability / (predictions[0]?.probability || 1)) * 100}%`,
                                  background: rank === 0
                                    ? "linear-gradient(to right, #ffb457, #ff8a20)"
                                    : "linear-gradient(to right, rgba(79,209,197,0.5), rgba(79,209,197,0.2))"
                                }}
                              />
                            </div>
                          </div>
                          <span className="w-16 text-right text-xs text-muted">
                            {formatNumber(item.logit, 1)} → {formatNumber(item.probability * 100, 1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                    <p className="mt-3 text-sm leading-6 text-slate-200 max-w-prose">
                      Logit = dot(final_vector, token_embedding). Softmax turns these raw scores into
                      the probability distribution you see in the header.
                    </p>
                  </div>
                </div>
              </details>
            </div>
          </div>
          {showMath ? (
            <MathBlock>
              {`logit(token) = final_vector · embedding(token)
probabilities = softmax(logits)
prediction = argmax(probabilities)`}
            </MathBlock>
          ) : null}
        </Section>

        <Section
          id="module-7"
          module="Module 7"
          title="Token Focus Mode"
          kicker="Per-layer evidence"
          onEnter={handleStageEnter("stack", 7)}
        >
          <Callout label="Before" variant="question">
            How does a single token's evidence evolve layer by layer?
          </Callout>
          <p>
            Click a token to follow its attention sources and MLP deltas across the stack.
          </p>
          <div className="flex flex-wrap gap-2">
            {bundle.tokens.map((token, index) => (
              <button
                key={`focus-token-${index}`}
                type="button"
                onClick={() => setSelectedTokenIndex(index)}
                className={`chip transition ${
                  selectedTokenIndex === index ? "bg-accent/20" : "bg-white/5"
                }`}
              >
                {token}
              </button>
            ))}
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {focusLayers.map((layer) => (
              <div key={`focus-layer-${layer.layer}`} className="glass rounded-2xl p-4">
                <div className="flex items-center justify-between text-xs text-muted">
                  <span>Layer {layer.layer}</span>
                  <span>MLP Δ {formatNumber(layer.mlpDelta, 2)}</span>
                </div>
                <div className="mt-3 space-y-2 text-xs">
                  {layer.topSources.map((source) => (
                    <div key={`${layer.layer}-${source.token}`} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-white">{source.token}</span>
                        <span className="text-muted">{formatNumber(source.value, 2)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-white/5">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-accent/80 to-accent2/80"
                          style={{ width: `${source.value * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              The sources shift each layer as the token gathers different context.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              A token keeps the same meaning throughout. It evolves with each residual edit.
            </Callout>
          </div>
        </Section>

        <Section
          id="module-8"
          module="Module 8"
          title="Forensic Breakdown"
          kicker="Why this prediction?"
          onEnter={handleStageEnter("final", 8)}
        >
          <Callout label="Before" variant="question">
            Why did the model choose this next word over the others?
          </Callout>
          <p>
            Let's trace the evidence like a detective. The prediction didn't happen all at once — it
            emerged across layers, shaped by attention routing and MLP transformations.
          </p>

          {/* Step 1: The Competition */}
          <div className="glass rounded-2xl p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-accent">Step 1 — The competition</p>
            <p className="mt-2 narrative">
              The model considered many tokens. These were the serious contenders:
            </p>
            <div className="mt-4 space-y-2">
              {predictions.map((item, rank) => {
                const maxProb = predictions[0]?.probability ?? 1;
                return (
                  <div key={`forensic-${item.token}`} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-2">
                        <span className={rank === 0 ? "text-accent" : "text-white"}>{item.token}</span>
                        {rank === 0 && <span className="text-xs text-accent/60">winner</span>}
                      </span>
                      <span className="text-muted">{formatNumber(item.probability * 100, 1)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-white/5">
                      <div
                        className="h-2 rounded-full"
                        style={{
                          width: `${(item.probability / maxProb) * 100}%`,
                          background: rank === 0
                            ? "linear-gradient(to right, #ffb457, #ff8a20)"
                            : "linear-gradient(to right, rgba(79,209,197,0.6), rgba(79,209,197,0.3))"
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            {predictions.length >= 2 && (
              <p className="mt-3 text-sm leading-6 text-slate-200 max-w-prose">
                Margin: {formatNumber((predictions[0].probability - predictions[1].probability) * 100, 1)}
                pp between #{1} and #{2}
              </p>
            )}
          </div>

          {/* Step 2: The Logit Lens — when did the winner emerge? */}
          <div className="glass rounded-2xl p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-accent2">Step 2 — When did the winner emerge?</p>
            <p className="mt-2 narrative">
              At each layer, we ask: "If the model had to predict right now, what would it say?"
              Watch the top token's confidence build (or shift) across layers.
            </p>
            <div className="mt-4 space-y-3">
              {logitLens.map((entry) => {
                const top = entry.predictions[0];
                const runnerUp = entry.predictions[1];
                return (
                  <div key={`lens-${entry.layer}`} className="rounded-xl border border-white/10 bg-ink2/60 p-3">
                    <div className="flex items-center justify-between text-xs text-muted">
                      <span>Layer {entry.layer}</span>
                      <span>
                        Top: <span className="text-white">{top?.token ?? "—"}</span>{" "}
                        {top ? `${formatNumber(top.probability * 100, 1)}%` : ""}
                      </span>
                    </div>
                    <div className="mt-2 flex gap-1">
                      {entry.predictions.map((p, i) => (
                        <div
                          key={`lens-${entry.layer}-${p.token}`}
                          className="h-3 rounded-sm"
                          style={{
                            width: `${p.probability * 100}%`,
                            minWidth: "2px",
                            background: i === 0 ? "#ffb457" : i === 1 ? "#4fd1c5" : "rgba(255,255,255,0.15)"
                          }}
                          title={`${p.token}: ${formatNumber(p.probability * 100, 1)}%`}
                        />
                      ))}
                    </div>
                    {top && runnerUp && top.token !== predictions[0]?.token && (
                      <p className="mt-1 text-xs text-accent/60">
                        The final winner hasn't taken the lead yet
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Step 3: Attention routing — where did evidence come from? */}
          <div className="glass rounded-2xl p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-accent3">Step 3 — Where did the evidence come from?</p>
            <p className="mt-2 narrative">
              In the final layer, the last token drew most heavily from these earlier tokens:
            </p>
            <div className="mt-4 space-y-2">
              {topAttentionSources.map((source) => (
                <div key={`forensic-attn-${source.token}`} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-white">{source.token}</span>
                    <span className="text-muted">attention weight {formatNumber(source.value, 3)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/5">
                    <div
                      className="h-2 rounded-full bg-accent/70"
                      style={{ width: `${source.value * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-3 narrative">
              These tokens supplied the context that shaped the prediction vector.
            </p>
          </div>

          {/* Step 4: Component attribution */}
          <div className="glass rounded-2xl p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-rose-400">Step 4 — Who gets credit?</p>
            <p className="mt-2 narrative">
              The final logit for each candidate is a sum of contributions from embedding, position,
              attention, and MLP. Here's the breakdown for the top contenders:
            </p>
            <div className="mt-4 space-y-4">
              {contributionTable.map((row) => {
                const entries = Object.entries(row.contributions) as [string, number][];
                const maxContrib = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.01);
                return (
                  <div key={`forensic-contrib-${row.token}`} className="rounded-xl border border-white/10 bg-ink2/60 p-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-white">{row.token}</span>
                      <span className="text-muted">{formatNumber(row.probability * 100, 1)}%</span>
                    </div>
                    <div className="mt-2 space-y-1">
                      {entries.map(([key, value]) => (
                        <div key={`${row.token}-${key}`} className="flex items-center gap-2 text-xs">
                          <span className="w-16 capitalize text-muted">{key}</span>
                          <div className="flex-1">
                            <div className="flex h-2 items-center">
                              <div
                                className="h-2 rounded-sm"
                                style={{
                                  width: `${(Math.abs(value) / maxContrib) * 50}%`,
                                  marginLeft: value < 0 ? "auto" : "50%",
                                  marginRight: value >= 0 ? "auto" : "50%",
                                  background: value >= 0 ? "#7bd389" : "#f87171"
                                }}
                              />
                            </div>
                          </div>
                          <span className="w-12 text-right text-muted">{formatNumber(value, 2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Step 5: Verdict */}
          <div className="glass rounded-2xl border-accent/30 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-accent">Step 5 — The verdict</p>
            <p className="mt-3 narrative">
              After {STACK_LAYERS} layers of attention routing and MLP transformation, the last
              token's residual vector points most strongly toward{" "}
              <span className="text-accent">{predictions[0]?.token ?? "—"}</span>
              {predictions.length >= 2 && (
                <span>
                  , beating <span className="text-accent2">{predictions[1]?.token}</span> by{" "}
                  {formatNumber((predictions[0].probability - predictions[1].probability) * 100, 1)}
                  {" "}percentage points
                </span>
              )}
              . No single component decided this — it was the accumulation of many small edits to the
              residual stream.
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              The winner often isn't obvious in early layers. It emerges through accumulated edits.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              There is a single "reason" for the prediction. It's a sum of many small contributions.
            </Callout>
          </div>
        </Section>

        <Section
          id="module-9"
          module="Module 9"
          title="What-If Lab"
          kicker="Side-by-side comparison"
          onEnter={handleStageEnter("final", 9)}
        >
          <Callout label="Before" variant="question">
            How does a single word change attention and prediction?
          </Callout>
          {!compareMode ? (
            <div className="glass rounded-2xl p-4">
              <p className="text-base text-slate-200">Compare mode is off.</p>
              <p className="mt-2 narrative">
                Turn on Compare in the header to see two sentences side-by-side.
              </p>
              <button
                type="button"
                onClick={() => setCompareMode(true)}
                className="mt-4 chip transition hover:bg-accent2/20"
              >
                Enable Compare Mode
              </button>
            </div>
          ) : (
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-4">
                <div className="glass rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">Primary sentence</p>
                  <p className="mt-3 text-sm text-slate-200">{text}</p>
                </div>
                <div className="glass rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">Top predictions</p>
                  <div className="mt-3 space-y-2">
                    {predictions.map((item) => (
                      <div key={`primary-${item.token}`} className="flex items-center justify-between text-xs">
                        <span className="text-white">{item.token}</span>
                        <span className="text-muted">{formatNumber(item.probability * 100, 1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="glass rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">Attention sources</p>
                  <div className="mt-3 space-y-2">
                    {topAttentionSources.map((source) => (
                      <div key={`primary-attn-${source.token}`} className="flex items-center justify-between text-xs">
                        <span className="text-white">{source.token}</span>
                        <span className="text-muted">{formatNumber(source.value, 2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="glass rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">Alternate sentence</p>
                  <textarea
                    className="mt-3 min-h-[120px] w-full rounded-2xl border border-white/10 bg-ink2/80 p-4 text-sm text-slate-100 focus:border-accent2/60 focus:outline-none"
                    value={compareText}
                    onChange={(event) => setCompareText(event.target.value)}
                  />
                </div>
                <div className="glass rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">Top predictions</p>
                  <div className="mt-3 space-y-2">
                    {comparePredictions.map((item) => (
                      <div key={`compare-${item.token}`} className="flex items-center justify-between text-xs">
                        <span className="text-white">{item.token}</span>
                        <span className="text-muted">{formatNumber(item.probability * 100, 1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="glass rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">Attention sources</p>
                  <div className="mt-3 space-y-2">
                    {compareAttentionSources.map((source) => (
                      <div key={`compare-attn-${source.token}`} className="flex items-center justify-between text-xs">
                        <span className="text-white">{source.token}</span>
                        <span className="text-muted">{formatNumber(source.value, 2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="md:col-span-2">
                <div className="glass rounded-2xl p-4">
                  <p className="text-xs uppercase tracking-[0.3em] text-muted">Biggest shifts</p>
                  <div className="mt-4 space-y-3">
                    {deltaRows.map((row) => (
                      <div key={`delta-${row.token}`} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-white">{row.token}</span>
                          <span className="text-muted">
                            {formatNumber(row.primary * 100, 1)}% → {formatNumber(row.compare * 100, 1)}%
                          </span>
                        </div>
                        <div className="h-2 rounded-full bg-white/5">
                          <div
                            className="h-2 rounded-full"
                            style={{
                              width: `${Math.min(Math.abs(row.delta) * 160, 100)}%`,
                              background: row.delta >= 0 ? "#ffb457" : "#4fd1c5"
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
          <div className="grid gap-3 md:grid-cols-2">
            <Callout label="What to notice" variant="notice">
              Tiny input edits can reroute attention and reshuffle the probability distribution.
            </Callout>
            <Callout label="Common misconception" variant="misconception">
              The model is deterministic in meaning. It is sensitive to phrasing and context.
            </Callout>
          </div>
        </Section>
      </main>

      <GuidedRun active={guidedRun} onClose={() => setGuidedRun(false)} />

      <footer className="border-t border-white/10 px-6 py-12">
        <div className="mx-auto max-w-3xl space-y-6">
          <div className="rounded-2xl border border-white/10 bg-surface2/60 p-6">
            <p className="text-xs uppercase tracking-[0.3em] text-muted">What this does not show</p>
            <p className="mt-3 narrative">
              This visualization teaches inference — a single forward pass through a transformer.
              It is deliberately simplified and does not cover:
            </p>
            <ul className="mt-3 space-y-2 text-base leading-7 text-slate-200 max-w-prose">
              <li>
                <span className="text-slate-200">Training dynamics</span> — how weights are learned
                via backpropagation and gradient descent
              </li>
              <li>
                <span className="text-slate-200">RLHF / preference tuning</span> — how models are
                aligned to human preferences after pre-training
              </li>
              <li>
                <span className="text-slate-200">Long-context behavior</span> — attention patterns
                over thousands of tokens, KV caching, and context window effects
              </li>
              <li>
                <span className="text-slate-200">LayerNorm</span> — normalization steps omitted
                here for clarity, but critical in real models
              </li>
              <li>
                <span className="text-slate-200">Tool use and agents</span> — how models generate
                structured outputs that invoke external systems
              </li>
              <li>
                <span className="text-slate-200">Real model scale</span> — production models use
                768–12288 dimensions, 12–96 layers, and 50k+ token vocabularies
              </li>
            </ul>
            <p className="mt-4 narrative">
              The {D_MODEL}-dimensional, {STACK_LAYERS}-layer model here is a faithful miniature —
              same architecture, same math, smaller numbers. The intuitions transfer.
            </p>
          </div>
          <p className="text-center text-sm text-muted">
            Built with curiosity — thanks to the “Attention Is All You Need” paper!
          </p>
          <SocialLinks className="mt-3 justify-center" />
        </div>
      </footer>
    </div>
  );
};

export default App;
