import React from "react";
import { clamp } from "../lib/math";

type VectorHeatmapProps = {
  values: number[];
  columns?: number;
  label?: string;
};

const valueToColor = (value: number) => {
  const intensity = clamp(Math.abs(value), 0, 1);
  if (value >= 0) {
    return `rgba(255, 180, 87, ${0.15 + intensity * 0.75})`;
  }
  return `rgba(79, 209, 197, ${0.15 + intensity * 0.75})`;
};

export const VectorHeatmap = ({ values, columns = 16, label }: VectorHeatmapProps) => {
  return (
    <div className="space-y-2">
      {label ? <div className="text-xs uppercase tracking-[0.25em] text-muted">{label}</div> : null}
      <div
        className="grid gap-1 rounded-2xl bg-ink2/60 p-3"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      >
        {values.map((value, index) => (
          <div
            key={`${index}-${value}`}
            className="h-5 rounded-md"
            style={{ backgroundColor: valueToColor(value) }}
            title={`Dim ${index + 1}: ${value.toFixed(3)}`}
          />
        ))}
      </div>
    </div>
  );
};

type MatrixHeatmapProps = {
  matrix: number[][];
  tokens: string[];
  selectedCell?: { row: number; col: number } | null;
  onSelect?: (row: number, col: number) => void;
};

export const MatrixHeatmap = ({ matrix, tokens, selectedCell, onSelect }: MatrixHeatmapProps) => {
  const size = matrix.length;
  const columns = size + 1;
  return (
    <div className="overflow-x-auto rounded-2xl bg-ink2/60 p-4">
      <div
        className="grid gap-1 text-xs"
        style={{ gridTemplateColumns: `160px repeat(${columns - 1}, minmax(30px, 1fr))` }}
      >
        <div className="text-muted">Query \ Key</div>
        {tokens.map((token, index) => (
          <div key={`col-${index}`} className="text-center text-muted">
            {token}
          </div>
        ))}
        {matrix.map((row, rowIndex) => (
          <React.Fragment key={`row-${rowIndex}`}>
            <div className="pr-2 text-right text-muted">{tokens[rowIndex]}</div>
            {row.map((value, colIndex) => {
              const masked = colIndex > rowIndex;
              const isSelected =
                selectedCell?.row === rowIndex && selectedCell?.col === colIndex;
              return (
                <button
                  key={`cell-${rowIndex}-${colIndex}`}
                  type="button"
                  className={`h-7 rounded-md border border-white/10 text-[10px] transition ${
                    isSelected ? "ring-2 ring-accent/80" : "hover:ring-1 hover:ring-white/20"
                  } ${masked ? "opacity-30" : ""}`}
                  style={{
                    backgroundColor: masked ? "rgba(255,255,255,0.04)" : valueToColor(value)
                  }}
                  onClick={() => onSelect?.(rowIndex, colIndex)}
                  title={masked ? "Masked" : `${value.toFixed(3)}`}
                />
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};
