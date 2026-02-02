import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Step = {
  moduleId: string;
  title: string;
  narration: string;
};

const STEPS: Step[] = [
  {
    moduleId: "module-0",
    title: "The Prediction",
    narration: "We start at the end — the model has already made a prediction. Now let's trace how it got here."
  },
  {
    moduleId: "module-1",
    title: "Tokenization",
    narration: "Your text is split into tokens — the atomic units the model works with. These aren't always words."
  },
  {
    moduleId: "module-2",
    title: "Embeddings",
    narration: "Each token becomes a vector. Similar meanings live nearby in this space."
  },
  {
    moduleId: "module-3",
    title: "Position",
    narration: "Word order is encoded as wave patterns and added to each vector. The residual stream is initialized."
  },
  {
    moduleId: "module-4",
    title: "Attention",
    narration: "Each token decides which earlier tokens to read from. This is routing, not reasoning."
  },
  {
    moduleId: "module-5",
    title: "Feed-Forward",
    narration: "The MLP transforms what attention gathered. Most of the model's parameters live here."
  },
  {
    moduleId: "module-6",
    title: "Stack & Predict",
    narration: "Layers repeat, editing the same residual stream. The final vector is compared against every possible next token."
  },
  {
    moduleId: "module-8",
    title: "Forensic Breakdown",
    narration: "Now we trace backwards: why did the model choose this word? Watch the evidence accumulate layer by layer."
  }
];

type GuidedRunProps = {
  active: boolean;
  onClose: () => void;
};

const GuidedRun = ({ active, onClose }: GuidedRunProps) => {
  const [stepIndex, setStepIndex] = useState(0);
  const step = STEPS[stepIndex];

  const scrollToStep = useCallback((index: number) => {
    const target = STEPS[index];
    if (!target) return;
    document.getElementById(target.moduleId)?.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });
  }, []);

  useEffect(() => {
    if (active) {
      setStepIndex(0);
      scrollToStep(0);
    }
  }, [active, scrollToStep]);

  const next = () => {
    if (stepIndex < STEPS.length - 1) {
      const nextIndex = stepIndex + 1;
      setStepIndex(nextIndex);
      scrollToStep(nextIndex);
    } else {
      onClose();
    }
  };

  const prev = () => {
    if (stepIndex > 0) {
      const prevIndex = stepIndex - 1;
      setStepIndex(prevIndex);
      scrollToStep(prevIndex);
    }
  };

  return (
    <AnimatePresence>
      {active && step && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="fixed bottom-6 left-1/2 z-50 w-full max-w-lg -translate-x-1/2 px-4"
        >
          <div className="rounded-2xl border border-white/10 bg-ink/90 p-5 shadow-2xl backdrop-blur-2xl">
            <div className="flex items-center justify-between text-xs text-muted">
              <span>
                Step {stepIndex + 1} of {STEPS.length}
              </span>
              <button
                type="button"
                onClick={onClose}
                className="text-muted transition hover:text-white"
              >
                Exit tour
              </button>
            </div>
            <p className="mt-2 text-sm font-medium text-accent">{step.title}</p>
            <p className="mt-1 narrative">{step.narration}</p>
            <div className="mt-4 flex items-center justify-between">
              <button
                type="button"
                onClick={prev}
                disabled={stepIndex === 0}
                className="chip transition hover:bg-white/10 disabled:opacity-30"
              >
                Back
              </button>
              <div className="flex gap-1">
                {STEPS.map((_, i) => (
                  <div
                    key={`dot-${i}`}
                    className={`h-1.5 w-1.5 rounded-full transition ${
                      i === stepIndex ? "bg-accent" : i < stepIndex ? "bg-accent/30" : "bg-white/10"
                    }`}
                  />
                ))}
              </div>
              <button
                type="button"
                onClick={next}
                className="chip transition hover:bg-accent/20"
              >
                {stepIndex === STEPS.length - 1 ? "Finish" : "Next"}
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default GuidedRun;
