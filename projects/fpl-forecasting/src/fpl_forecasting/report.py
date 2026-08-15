"""Create a self-contained evaluation page with no external assets."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any

_MODEL_LABELS = {
    "last_gameweek": "Last gameweek",
    "rolling_3_mean": "Three-observation mean",
    "training_mean": "Training-window mean",
    "ridge_regression": "Ridge regression",
}


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_evaluation_html(report: Mapping[str, Any]) -> str:
    """Render a compact recruiter-facing view of one evaluation report."""

    data = report.get("data", {})
    protocol = report.get("protocol", {})
    models = report.get("models", {})
    top_k = int(protocol.get("ranking_top_k", 10))

    rows: list[str] = []
    for key, label in _MODEL_LABELS.items():
        metrics = models.get(key, {})
        rows.append(
            "<tr>"
            f"<th scope='row'>{escape(label)}</th>"
            f"<td>{_number(metrics.get('mae'))}</td>"
            f"<td>{_number(metrics.get('rmse'))}</td>"
            f"<td>{_number(metrics.get('spearman'))}</td>"
            f"<td>{_number(metrics.get(f'ndcg_at_{top_k}'))}</td>"
            f"<td>{_number(metrics.get(f'top_{top_k}_overlap'))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FPL Forecast Evaluation</title>
  <style>
    :root {{ color-scheme: light; --ink:#12212f; --muted:#526273; --line:#d9e1e8;
      --panel:#f5f8fb; --accent:#0d7c66; --accent-soft:#e4f4ef; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:16px/1.55 system-ui,-apple-system,sans-serif; color:var(--ink); background:#fff; }}
    main {{ width:min(1080px,calc(100% - 32px)); margin:0 auto; padding:56px 0 72px; }}
    .eyebrow {{ color:var(--accent); font-weight:750; letter-spacing:.08em; text-transform:uppercase; }}
    h1 {{ margin:.25rem 0 .75rem; font-size:clamp(2rem,5vw,4rem); line-height:1.05; max-width:850px; }}
    .lede {{ max-width:760px; color:var(--muted); font-size:1.15rem; }}
    .notice {{ margin:28px 0; padding:18px 20px; border-left:5px solid var(--accent); background:var(--accent-soft); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:30px 0; }}
    .card {{ border:1px solid var(--line); border-radius:14px; padding:18px; background:var(--panel); }}
    .card strong {{ display:block; font-size:1.65rem; }}
    .card span {{ color:var(--muted); }}
    h2 {{ margin-top:42px; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:14px; }}
    table {{ width:100%; border-collapse:collapse; min-width:720px; }}
    th,td {{ padding:13px 15px; text-align:right; border-bottom:1px solid var(--line); }}
    thead th {{ background:var(--panel); color:var(--muted); font-size:.88rem; }}
    th:first-child {{ text-align:left; }}
    tbody tr:last-child th,tbody tr:last-child td {{ border-bottom:0; }}
    ol {{ padding-left:1.25rem; }}
    code {{ background:var(--panel); padding:.1rem .35rem; border-radius:5px; }}
    footer {{ margin-top:48px; color:var(--muted); font-size:.92rem; }}
  </style>
</head>
<body>
<main>
  <p class="eyebrow">Forecast protocol</p>
  <h1>Next-gameweek forecasts evaluated in time order</h1>
  <p class="lede">Each test gameweek is later than every row used to fit its model. The comparison includes three simple baselines and standardized ridge regression.</p>
  <div class="notice"><strong>Evidence boundary:</strong> this page reflects only the supplied completed-gameweek files. It is not a claim about live-season performance.</div>

  <section class="grid" aria-label="Evaluation summary">
    <div class="card"><strong>{escape(str(data.get("players", "n/a")))}</strong><span>players</span></div>
    <div class="card"><strong>{escape(str(data.get("evaluated_rows", "n/a")))}</strong><span>future rows evaluated</span></div>
    <div class="card"><strong>{escape(str(data.get("folds", "n/a")))}</strong><span>rolling folds</span></div>
    <div class="card"><strong>{escape(str(data.get("evaluated_gameweek_start", "n/a")))} to {escape(str(data.get("evaluated_gameweek_end", "n/a")))}</strong><span>evaluated gameweeks</span></div>
  </section>

  <h2>Out-of-fold results</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th scope="col">Candidate</th><th scope="col">MAE ↓</th><th scope="col">RMSE ↓</th><th scope="col">Spearman ↑</th><th scope="col">NDCG@{top_k} ↑</th><th scope="col">Top-{top_k} overlap ↑</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>

  <h2>Leak controls</h2>
  <ol>
    <li>Empty, duplicate, gapped, and low-coverage gameweeks are rejected.</li>
    <li>Features for target gameweek <em>t+1</em> stop at completed gameweek <em>t</em>.</li>
    <li>The final supplied gameweek is a target only; unlabeled future rows are never invented.</li>
    <li>Model inputs come from a fixed numeric allow-list with boundary checks.</li>
    <li>Every reported prediction is out of fold under an expanding-window split.</li>
  </ol>

  <footer>Generated by the project CLI as a single local HTML file. No external scripts, fonts, trackers, or network requests are used.</footer>
</main>
</body>
</html>
"""


def write_evaluation_html(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_evaluation_html(report), encoding="utf-8")
    return destination
