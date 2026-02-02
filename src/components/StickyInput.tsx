import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { formatNumber } from "../lib/math";
import SocialLinks from "./SocialLinks";

type Prediction = {
  token: string;
  logit: number;
  probability: number;
};

type StickyInputProps = {
  text: string;
  onChange: (value: string) => void;
  tokens: string[];
  tokenIds: number[];
  predictions: Prediction[];
  onReplaceLastToken: (value: string) => void;
  showMath: boolean;
  onToggleMath: () => void;
  maxTokens: number;
  isTruncated: boolean;
  compareMode: boolean;
  onToggleCompare: () => void;
};

const replacements = ["robot", "garden", "economy", "jazz", "quantum"];

const lessonPrompts = [
  {
    label: "Coreference",
    text: "The cat sat on the mat because it was",
    hint: "Who is 'it' — the cat or the mat?"
  },
  {
    label: "Negation",
    text: "The dog did not chase the",
    hint: "Does 'not' change what follows?"
  },
  {
    label: "Garden path",
    text: "The old man the boat and the young",
    hint: "'Man' is a verb here — watch attention shift"
  }
];

const StickyInput = ({
  text,
  onChange,
  tokens,
  tokenIds,
  predictions,
  onReplaceLastToken,
  showMath,
  onToggleMath,
  maxTokens,
  isTruncated,
  compareMode,
  onToggleCompare
}: StickyInputProps) => {
  const [expanded, setExpanded] = useState(false);
  const topPrediction = predictions[0];

  return (
    <section className="sticky top-0 z-40 border-b border-white/10 bg-ink/80 backdrop-blur-2xl">
      <div className="mx-auto max-w-6xl px-6 py-4">
        {/* Compact bar — always visible */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-4">
            <h1 className="font-display text-xl text-white md:text-2xl">How Transformers Think</h1>
            <div className="hidden items-center gap-2 text-sm text-slate-300 md:flex">
              <span className="max-w-[280px] truncate">{text || "Type a sentence..."}</span>
              <span className="text-muted">→</span>
              <span className="text-base font-semibold text-accent">
                {topPrediction?.token ?? "—"}
              </span>
              <span className="text-sm text-slate-200">
                {topPrediction ? `${formatNumber(topPrediction.probability * 100, 1)}%` : ""}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onToggleMath}
              className={`chip transition ${showMath ? "bg-accent/20" : "bg-white/5"}`}
            >
              {showMath ? "Math On" : "Math Off"}
            </button>
            <button
              type="button"
              onClick={onToggleCompare}
              className={`chip transition ${compareMode ? "bg-accent2/20" : "bg-white/5"}`}
            >
              {compareMode ? "Compare On" : "Compare Off"}
            </button>
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="chip transition hover:bg-white/10"
            >
              {expanded ? "Collapse" : "Edit Input"}
            </button>
          </div>
        </div>
        <SocialLinks showDomains size="sm" className="mt-2 opacity-70" />

        {/* Expanded panel */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="overflow-hidden"
            >
              <div className="grid gap-6 pt-4 lg:grid-cols-[1.1fr_0.9fr]">
                <div className="space-y-3">
                  <label className="block text-sm text-muted">Type a sentence and watch it transform.</label>
                  <textarea
                    className="min-h-[80px] w-full rounded-2xl border border-white/10 bg-ink2/80 p-3 text-sm text-slate-100 shadow-glow focus:border-accent/60 focus:outline-none"
                    placeholder="The curious cat studied the attention map and..."
                    value={text}
                    onChange={(event) => onChange(event.target.value)}
                    onFocus={() => setExpanded(true)}
                  />
                  <div className="flex flex-wrap items-center gap-2">
                    {tokens.slice(0, maxTokens).map((token, index) => (
                      <span key={`${token}-${index}`} className="chip">
                        <span className="text-white">{token}</span>
                        <span className="text-muted">#{tokenIds[index]}</span>
                      </span>
                    ))}
                    {isTruncated ? (
                      <span className="text-xs text-muted">
                        Showing first {maxTokens} tokens.
                      </span>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
                    <span>Quick swap last word:</span>
                    {replacements.map((word) => (
                      <button
                        key={word}
                        type="button"
                        className="chip transition hover:bg-accent/20"
                        onClick={() => onReplaceLastToken(word)}
                      >
                        {word}
                      </button>
                    ))}
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
                    <span>Try a lesson:</span>
                    {lessonPrompts.map((prompt) => (
                      <button
                        key={prompt.label}
                        type="button"
                        className="chip transition hover:bg-accent2/20"
                        onClick={() => onChange(prompt.text)}
                        title={prompt.hint}
                      >
                        {prompt.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="glass rounded-2xl p-4">
                  <div className="flex items-center justify-between">
                    <p className="text-xs uppercase tracking-[0.3em] text-muted">Top 5 next tokens</p>
                    <span className="text-xs text-muted">live prediction</span>
                  </div>
                  <div className="mt-3 space-y-2">
                    {predictions.map((item) => (
                      <div key={item.token} className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-white">{item.token}</span>
                          <span className="text-muted">{formatNumber(item.probability * 100, 1)}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-white/5">
                          <div
                            className="h-2 rounded-full bg-gradient-to-r from-accent/80 to-accent2/80"
                            style={{ width: `${item.probability * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
};

export default StickyInput;
