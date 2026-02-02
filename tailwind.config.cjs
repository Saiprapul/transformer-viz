/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"] ,
  theme: {
    extend: {
      colors: {
        ink: "rgb(var(--bg) / <alpha-value>)",
        ink2: "rgb(var(--bg-2) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        surface2: "rgb(var(--surface-2) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        accent2: "rgb(var(--accent-2) / <alpha-value>)",
        accent3: "rgb(var(--accent-3) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        grid: "rgb(var(--grid) / <alpha-value>)"
      },
      fontFamily: {
        sans: ["Space Grotesk", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "SFMono-Regular", "monospace"],
        display: ["Syne", "Space Grotesk", "system-ui", "sans-serif"]
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,180,87,0.2), 0 20px 40px rgba(0,0,0,0.3)",
        panel: "0 12px 30px rgba(0,0,0,0.35)"
      }
    }
  },
  plugins: []
};
