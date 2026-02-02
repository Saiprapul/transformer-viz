import React from "react";

type CalloutVariant = "question" | "notice" | "misconception" | "insight";

type CalloutProps = {
  label: string;
  children: React.ReactNode;
  variant?: CalloutVariant;
};

const variantStyles: Record<CalloutVariant, string> = {
  question: "border-accent/[0.25] bg-accent/[0.08]",
  notice: "border-accent2/[0.25] bg-accent2/[0.08]",
  misconception: "border-rose-400/[0.25] bg-rose-400/[0.08]",
  insight: "border-accent3/[0.25] bg-accent3/[0.08]"
};

const Callout = ({ label, children, variant = "question" }: CalloutProps) => {
  return (
    <div className={`rounded-2xl border px-4 py-3 ${variantStyles[variant]}`}>
      <p className="text-xs uppercase tracking-[0.3em] text-muted">{label}</p>
      <div className="mt-2 text-sm leading-6 text-slate-200">{children}</div>
    </div>
  );
};

export default Callout;
