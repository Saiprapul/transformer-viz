import { useEffect, useRef } from "react";

type SineWaveCanvasProps = {
  length: number;
  height?: number;
};

const colors = ["#ffb457", "#4fd1c5", "#7bd389", "#8aa7ff"];

const SineWaveCanvas = ({ length, height = 180 }: SineWaveCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const width = canvas.clientWidth;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    context.scale(window.devicePixelRatio, window.devicePixelRatio);

    context.clearRect(0, 0, width, height);
    context.lineWidth = 2;

    const steps = Math.max(length, 8);
    const xStep = width / (steps - 1);

    colors.forEach((color, index) => {
      context.beginPath();
      context.strokeStyle = color;
      for (let i = 0; i < steps; i += 1) {
        const x = i * xStep;
        const y =
          height / 2 +
          Math.sin(i / (2 + index * 0.6)) * (40 - index * 6) *
            Math.cos(i / (1.8 + index * 0.4));
        if (i === 0) {
          context.moveTo(x, y);
        } else {
          context.lineTo(x, y);
        }
      }
      context.stroke();
    });
  }, [length, height]);

  return (
    <div className="rounded-2xl border border-white/10 bg-ink2/60 p-3">
      <canvas ref={canvasRef} className="h-full w-full" style={{ height }} />
    </div>
  );
};

export default SineWaveCanvas;
