export type ProjectionPoint = {
  x: number;
  y: number;
  label: string;
  active?: boolean;
};

type ProjectionPlotProps = {
  points: ProjectionPoint[];
};

const ProjectionPlot = ({ points }: ProjectionPlotProps) => {
  const width = 340;
  const height = 240;
  const padding = 28;

  const toX = (value: number) =>
    padding + ((value + 1) / 2) * (width - padding * 2);
  const toY = (value: number) =>
    padding + ((1 - (value + 1) / 2) * (height - padding * 2));

  return (
    <div className="rounded-2xl border border-white/10 bg-ink2/60 p-4">
      <p className="text-xs uppercase tracking-[0.3em] text-muted">2D projection</p>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="mt-3 h-56 w-full"
        role="img"
        aria-label="2D projection of token embeddings"
      >
        <rect x={0} y={0} width={width} height={height} rx={16} fill="rgba(8,12,16,0.6)" />
        <line
          x1={padding}
          y1={height / 2}
          x2={width - padding}
          y2={height / 2}
          stroke="rgba(255,255,255,0.08)"
        />
        <line
          x1={width / 2}
          y1={padding}
          x2={width / 2}
          y2={height - padding}
          stroke="rgba(255,255,255,0.08)"
        />
        {points.map((point, index) => (
          <g key={`${point.label}-${index}`}>
            <circle
              cx={toX(point.x)}
              cy={toY(point.y)}
              r={point.active ? 6 : 4}
              fill={point.active ? "#ffb457" : "#4fd1c5"}
              stroke="rgba(0,0,0,0.4)"
              strokeWidth={1}
            />
            <text
              x={toX(point.x) + 8}
              y={toY(point.y) - 6}
              fill="rgba(230,237,243,0.8)"
              fontSize={10}
            >
              {point.label}
            </text>
          </g>
        ))}
      </svg>
      <p className="mt-3 narrative">
        PCA projection preserves relative similarity, not exact geometry. Nearby points share
        direction in the full space — but distances here are approximate, not how the model
        reasons internally.
      </p>
    </div>
  );
};

export default ProjectionPlot;
