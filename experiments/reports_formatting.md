# Report Formatting Style Guide

All experiment report scripts (`exp*_report.py`) must produce HTML using this exact CSS and structural conventions. The goal is a consistent academic look across all reports.

## CSS (copy verbatim into every report as `_CSS`)

```css
body {
  font-family: Georgia, serif;
  max-width: 1120px;
  margin: 0 auto;
  padding: 2em 2em 4em;
  color: #1e2a3a;
  background: #f8f9fb;
  line-height: 1.75;
}
h1 {
  color: #1a3a5c;
  border-bottom: 3px solid #1a3a5c;
  padding-bottom: 0.4em;
  font-size: 1.8em;
  margin-bottom: 0.3em;
}
h2 {
  color: #2c5282;
  margin-top: 2em;
  font-size: 1.25em;
  border-left: 4px solid #3182ce;
  padding-left: 0.6em;
}
h3 {
  color: #2d3748;
  margin-top: 1.4em;
  font-size: 1.05em;
}
.card {
  background: #fff;
  border: 1px solid #d0d9e8;
  border-radius: 8px;
  padding: 1.2em 1.6em;
  margin: 1em 0;
  box-shadow: 0 2px 6px rgba(0,0,0,.06);
}
.highlight {
  background: #ebf8ff;
  border-left: 4px solid #3182ce;
  border-radius: 0 6px 6px 0;
  padding: 0.7em 1.2em;
  margin: 1em 0;
}
.insight {
  background: #f0fff4;
  border-left: 4px solid #276749;
  border-radius: 0 6px 6px 0;
  padding: 0.7em 1.2em;
  margin: 1em 0;
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.88em;
  margin-top: 0.8em;
}
th {
  background: #2c5282;
  color: #fff;
  padding: 7px 12px;
  text-align: left;
  font-weight: 600;
}
td {
  padding: 6px 12px;
  border-bottom: 1px solid #e2e8f0;
}
tr:nth-child(even) td { background: #f7f9fc; }
tr:hover td { background: #ebf8ff; }
img.fig {
  max-width: 100%;
  border: 1px solid #d0d9e8;
  border-radius: 6px;
  margin: 0.8em 0;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
  display: block;
}
.formula {
  font-family: 'Courier New', monospace;
  background: #f0f4f8;
  border: 1px solid #d0d9e8;
  padding: 0.4em 0.8em;
  border-radius: 4px;
  display: inline-block;
  margin: 0.3em 0;
}
.caption {
  font-style: italic;
  color: #4a5568;
  margin: -0.4em 0 1.2em 0;
  font-size: 0.92em;
}
.pass { color: #276749; font-weight: bold; }
.fail { color: #c53030; font-weight: bold; }
.warn { color: #b7791f; font-weight: bold; }
code {
  background: #edf2f7;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.88em;
  font-family: 'Courier New', monospace;
}
.meta { color: #718096; font-size: 0.9em; }
```

## HTML Structure

Every report HTML must follow this skeleton:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Experiment N — Short Title</title>
  <style>{_CSS}</style>
</head>
<body>
<h1>Experiment N — Full Title</h1>
<p class="meta">Generated: {timestamp} · Results: {results_dir}</p>

<h2>1. About This Experiment</h2>
<!-- One or more .card divs with h3 subsections explaining the physics -->

<h2>2. Key Results</h2>
<!-- .card.highlight with summary table and main finding -->

<h2>3. ...</h2>
<!-- Figures with img.fig + p.caption pairs -->

</body>
</html>
```

## Conventions

- **Section headings**: always numbered (`1.`, `2.`, etc.)
- **Figures**: `<img class="fig" src="data:image/png;base64,...">` followed immediately by `<p class="caption">...</p>`
- **Equations**: `<p class="formula">...</p>` or `<span class="formula">...</span>`
- **Status indicators**: `<span class="pass">✓</span>`, `<span class="fail">✗</span>`, `<span class="warn">⚠</span>`
- **Cards**: use `.card` for explanation blocks; use `.card` + inner `.highlight` or `.insight` for key findings
- **No inline styles** except for `text-align:center` on summary paragraphs
- **Matplotlib figures**: use `plt.style.use('seaborn-v0_8-whitegrid')` or plain default; do NOT use dark themes. Preferred palette: `steelblue`, `tomato`, `seagreen`, `darkorange`, `mediumpurple`. Colormap: `RdBu_r` for diverging, `viridis` for sequential.
- **Figure function signature**: every figure function returns a base64 PNG string (call `_fig_to_b64(fig)` and `plt.close(fig)`)
- **Error handling**: wrap each figure generation in `try/except Exception` so one bad figure doesn't kill the whole report
