import { motion } from "framer-motion";
import React from "react";

type SectionProps = {
  id: string;
  module: string;
  title: string;
  kicker?: string;
  onEnter?: () => void;
  children: React.ReactNode;
};

const Section = ({ id, module, title, kicker, onEnter, children }: SectionProps) => {
  return (
    <motion.section
      id={id}
      className="section-card relative overflow-hidden"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: false, amount: 0.2 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      onViewportEnter={onEnter}
    >
      <div className="absolute -right-24 -top-24 h-48 w-48 rounded-full bg-accent/10 blur-3xl" />
      <div className="absolute -left-20 -bottom-20 h-48 w-48 rounded-full bg-accent2/10 blur-3xl" />
      <div className="relative z-10">
        <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.3em] text-muted">
          <span className="chip">{module}</span>
          {kicker ? <span>{kicker}</span> : null}
        </div>
        <h2 className="section-title mt-4">{title}</h2>
        <div className="section-content mt-6 space-y-6">{children}</div>
      </div>
    </motion.section>
  );
};

export default Section;
