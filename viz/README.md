# WC 2026 Forecast — Web Viz

React + Vite + Tailwind. Consumes `public/predictions.json` produced by `notebooks/14_ensemble_bracket.ipynb`.

## Local dev (use the Node.js command prompt, not PowerShell)

```
cd viz
npm install
npm run dev
```

Then open the URL Vite prints (usually http://localhost:5173).

## Build for deploy

```
npm run build
```

Output goes to `dist/`. Vercel will auto-detect Vite and serve it.

## Regenerate predictions

After re-running the model in `notebooks/14_ensemble_bracket.ipynb`, the notebook writes a fresh `viz/public/predictions.json`. Commit and redeploy.
