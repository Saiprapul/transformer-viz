type Module = {
  id: string;
  number: number;
  title: string;
};

const MODULES: Module[] = [
  { id: "module-0", number: 0, title: "The Prediction" },
  { id: "module-1", number: 1, title: "Tokens & IDs" },
  { id: "module-2", number: 2, title: "Embeddings" },
  { id: "module-3", number: 3, title: "Position" },
  { id: "module-4", number: 4, title: "Attention" },
  { id: "module-5", number: 5, title: "Feed-Forward" },
  { id: "module-6", number: 6, title: "Stack & Predict" },
  { id: "module-7", number: 7, title: "Token Focus" },
  { id: "module-8", number: 8, title: "Forensic" },
  { id: "module-9", number: 9, title: "What-If Lab" }
];

type ModuleNavProps = {
  activeModule: number;
};

const ModuleNav = ({ activeModule }: ModuleNavProps) => {
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <nav className="fixed left-4 top-1/2 z-50 hidden -translate-y-1/2 xl:block">
      <div className="space-y-1">
        {MODULES.map((mod) => {
          const isActive = mod.number === activeModule;
          return (
            <button
              key={mod.id}
              type="button"
              onClick={() => scrollTo(mod.id)}
              className={`group flex items-center gap-2 rounded-lg px-2 py-1.5 text-left transition ${
                isActive ? "bg-white/10" : "hover:bg-white/5"
              }`}
              title={`Module ${mod.number}: ${mod.title}`}
            >
              <span
                className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-medium transition ${
                  isActive
                    ? "bg-accent/80 text-ink"
                    : "bg-white/10 text-muted group-hover:text-white"
                }`}
              >
                {mod.number}
              </span>
              <span
                className={`text-xs transition ${
                  isActive ? "text-white" : "text-muted/60 group-hover:text-muted"
                }`}
              >
                {mod.title}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
};

export default ModuleNav;
