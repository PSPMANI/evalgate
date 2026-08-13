"""Build the model-quality dashboard that CD publishes to GitHub Pages.

    python build_report.py

Reads metrics.json, merges it into the run history fetched from the currently
published dashboard (stateless history: the site itself is the store), and
writes a self-contained site/ folder.
"""
import json
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).parent
SITE = HERE / "site"
HISTORY_URL = "https://pspmani.github.io/evalgate/history.json"
MAX_HISTORY = 200

THRESHOLDS = {"roc_auc_test": 0.82, "accuracy_test": 0.75, "roc_auc_cv_mean": 0.82}


def fetch_history():
    try:
        with urllib.request.urlopen(HISTORY_URL, timeout=10) as r:
            return json.load(r)
    except Exception:
        return []


def spark_svg(values, width=560, height=90):
    if len(values) < 2:
        values = values * 2
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1e-9
    pts = []
    for i, v in enumerate(values):
        x = 10 + i * (width - 20) / (len(values) - 1)
        y = height - 12 - (v - lo) / span * (height - 24)
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">'
        f'<polyline fill="none" stroke="#4f46e5" stroke-width="2.5" points="{" ".join(pts)}"/>'
        "</svg>"
    )


def main():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    history = fetch_history()
    if not any(h.get("git_sha") == metrics["git_sha"]
               and h.get("trained_at_utc") == metrics["trained_at_utc"] for h in history):
        history.append(metrics)
    history = history[-MAX_HISTORY:]

    rows = []
    for m in reversed(history[-25:]):
        rows.append(
            "<tr><td>" + m.get("trained_at_utc", "-") + "</td>"
            "<td><code>" + m.get("git_sha", "-") + "</code></td>"
            f"<td>{m.get('roc_auc_test', 0):.4f}</td>"
            f"<td>{m.get('roc_auc_cv_mean', 0):.4f} +/- {m.get('roc_auc_cv_std', 0):.4f}</td>"
            f"<td>{m.get('accuracy_test', 0):.4f}</td></tr>"
        )
    gates = []
    for key, minimum in THRESHOLDS.items():
        val = metrics.get(key, 0)
        ok = val >= minimum
        gates.append(
            f"<tr><td>{key}</td><td>{val:.4f}</td><td>&gt;= {minimum}</td>"
            f"<td class='{'ok' if ok else 'bad'}'>{'PASS' if ok else 'FAIL'}</td></tr>"
        )

    auc_series = [m.get("roc_auc_test", 0) for m in history][-40:]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EvalGate - Model Quality Dashboard</title>
<style>
  :root {{ --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --brand:#4f46e5; --ok:#059669; --bad:#dc2626; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif; color:var(--ink);
         background:#f9fafb; line-height:1.55; padding:40px 20px; }}
  .wrap {{ max-width:900px; margin:0 auto; }}
  h1 {{ font-size:1.7rem; }} .sub {{ color:var(--muted); margin:6px 0 26px; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin-bottom:26px; }}
  .tile {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
  .tile b {{ font-size:1.5rem; color:var(--brand); display:block; }}
  .tile span {{ font-size:0.8rem; color:var(--muted); }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:20px 22px; margin-bottom:20px; }}
  h2 {{ font-size:1.05rem; margin-bottom:12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  .ok {{ color:var(--ok); font-weight:700; }} .bad {{ color:var(--bad); font-weight:700; }}
  code {{ background:#f3f4f6; padding:1px 6px; border-radius:5px; }}
  a {{ color:var(--brand); }}
  .foot {{ color:var(--muted); font-size:0.82rem; margin-top:24px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>EvalGate - Model Quality Dashboard</h1>
  <p class="sub">Published automatically by the CD pipeline after every gated deployment.
     Latest model: <code>{metrics['git_sha']}</code> trained {metrics['trained_at_utc']} UTC.</p>

  <div class="tiles">
    <div class="tile"><b>{metrics['roc_auc_test']:.4f}</b><span>Test ROC-AUC</span></div>
    <div class="tile"><b>{metrics['roc_auc_cv_mean']:.4f}</b><span>5-fold CV ROC-AUC</span></div>
    <div class="tile"><b>{metrics['accuracy_test']:.4f}</b><span>Test accuracy</span></div>
    <div class="tile"><b>{len(history)}</b><span>Gated deployments recorded</span></div>
  </div>

  <div class="card">
    <h2>Quality gate (enforced in CI - a failing row blocks deployment)</h2>
    <table><tr><th>Metric</th><th>Value</th><th>Required</th><th>Gate</th></tr>{''.join(gates)}</table>
  </div>

  <div class="card">
    <h2>Test ROC-AUC across deployments</h2>
    {spark_svg(auc_series)}
  </div>

  <div class="card">
    <h2>Deployment history</h2>
    <table><tr><th>Trained (UTC)</th><th>Commit</th><th>Test AUC</th><th>CV AUC</th><th>Accuracy</th></tr>{''.join(rows)}</table>
  </div>

  <p class="foot">Pipeline: push -> tests -> train -> quality gate -> artifact -> deploy.
     Source: <a href="https://github.com/PSPMANI/evalgate">github.com/PSPMANI/evalgate</a> |
     By <a href="https://pspmani.github.io">Pathi Manikanta</a></p>
</div>
</body>
</html>
"""
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(html, encoding="utf-8")
    (SITE / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"site/ built: {len(history)} history entries")


if __name__ == "__main__":
    main()
