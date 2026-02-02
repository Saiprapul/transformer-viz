import { sumVectors, vectorNorm } from "../lib/math";

export type ResidualStage =
  | "raw"
  | "embed"
  | "position"
  | "attention"
  | "mlp"
  | "stack"
  | "final";

type ComponentKey = "embedding" | "position" | "attention" | "mlp";

type ResidualComponents = Record<ComponentKey, number[][]>;

type ResidualStreamProps = {
  tokens: string[];
  components: ResidualComponents;
  stage: ResidualStage;
  highlightIndex?: number;
};

const STAGE_CONFIG: Record<ResidualStage, { title: string; note: string; keys: ComponentKey[] }>
  = {
    raw: {
      title: "Residual stream: waiting",
      note: "Tokens exist, vectors are not built yet.",
      keys: []
    },
    embed: {
      title: "Residual stream: embeddings",
      note: "Each token gets its base meaning vector.",
      keys: ["embedding"]
    },
    position: {
      title: "Residual stream: + position",
      note: "Order is injected by adding positional signals.",
      keys: ["embedding", "position"]
    },
    attention: {
      title: "Residual stream: + attention",
      note: "Tokens read from other tokens and add the result.",
      keys: ["embedding", "position", "attention"]
    },
    mlp: {
      title: "Residual stream: + MLP",
      note: "Each token runs a local transformation and adds it.",
      keys: ["embedding", "position", "attention", "mlp"]
    },
    stack: {
      title: "Residual stream: stacked layers",
      note: "The same edits repeat many times, deepening the context.",
      keys: ["embedding", "position", "attention", "mlp"]
    },
    final: {
      title: "Residual stream: from thought to choice",
      note: "Each candidate token asks: how compatible am I with this state?",
      keys: ["embedding", "position", "attention", "mlp"]
    }
  };

const COLORS: Record<ComponentKey, string> = {
  embedding: "#4fd1c5",
  position: "#8aa7ff",
  attention: "#ffb457",
  mlp: "#7bd389"
};

const LABELS: Record<ComponentKey, string> = {
  embedding: "Embedding",
  position: "Position",
  attention: "Attention",
  mlp: "MLP"
};

const ResidualStream = ({ tokens, components, stage, highlightIndex }: ResidualStreamProps) => {
  const config = STAGE_CONFIG[stage];

  return (
    <div className="section-card border border-white/10 bg-ink/80 backdrop-blur-xl">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-muted">{config.title}</p>
          <p className="mt-2 narrative">{config.note}</p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
          {config.keys.map((key) => (
            <span key={key} className="inline-flex items-center gap-2">
              <span className="h-2 w-2 rounded-full" style={{ background: COLORS[key] }} />
              {LABELS[key]}
            </span>
          ))}
        </div>
      </div>

      {tokens.length === 0 ? (
        <div className="mt-4 rounded-2xl border border-white/10 bg-ink2/60 p-4 text-sm text-muted">
          Type a sentence to initialize the residual stream.
        </div>
      ) : (
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {tokens.map((token, index) => {
          const activeVectors = config.keys.map((key) => components[key][index] ?? []);
          const totalVector = activeVectors.length
            ? sumVectors(activeVectors)
            : Array.from({ length: components.embedding[0]?.length ?? 0 }, () => 0);
          const totalNorm = vectorNorm(totalVector);
          const segmentTotals = activeVectors.map((vector) => vectorNorm(vector));
          const segmentSum = segmentTotals.reduce((acc, value) => acc + value, 0) || 1;

          return (
            <div
              key={`${token}-${index}`}
              className={`rounded-2xl border border-white/10 bg-ink2/70 p-3 transition ${
                highlightIndex === index ? "ring-2 ring-accent/60" : ""
              }`}
            >
              <div className="flex items-center justify-between text-sm">
                <span className="text-white">{token}</span>
                <span className="text-xs text-muted">||v|| {totalNorm.toFixed(2)}</span>
              </div>
              <div className="mt-3 flex h-2 overflow-hidden rounded-full bg-white/5">
                {config.keys.length === 0 ? (
                  <div className="h-full w-full bg-white/10" />
                ) : (
                  config.keys.map((key, keyIndex) => (
                    <div
                      key={`${token}-${key}`}
                      className="h-full"
                      style={{
                        width: `${(segmentTotals[keyIndex] / segmentSum) * 100}%`,
                        background: COLORS[key]
                      }}
                    />
                  ))
                )}
              </div>
            </div>
          );
        })}
        </div>
      )}
    </div>
  );
};

export default ResidualStream;
