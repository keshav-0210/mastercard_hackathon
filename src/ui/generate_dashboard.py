r"""Builds artifacts/submission_dashboard.html: a single-page, dependency-free HTML dashboard
for the hackathon submission.

Re-run this script whenever a new results file is produced (fast, no pipeline rerun):
    .\.venv\Scripts\python.exe src\ui\generate_dashboard.py

It reads the most recent `adaptive_v2_metrics_dump_*.json` and `adaptive_v2_summary_*.json`
files under artifacts/ and embeds their numbers directly into the HTML (browsers block fetch()
of local JSON files opened via file://, so embedding at build time is what keeps the page
one-click openable with zero server/dependencies). The architecture diagram, by contrast, is
referenced by relative filename (not embedded), so replacing artifacts/architecture_diagram.png
and refreshing the browser shows the new image immediately with no rebuild.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
OUTPUT_HTML = ARTIFACTS / "submission_dashboard.html"
ARCHITECTURE_IMAGE = "architecture_diagram.png"

CHART_SPECS = [
    {"key": "f1", "title": "Detector F1 Score", "team": "blue", "desc": "Balance of precision and recall for the current round's unseen attack batch.", "yMin": 0, "yMax": 1},
    {"key": "recall", "title": "Detector Recall", "team": "blue", "desc": "Fraction of fraud successfully caught by the detector.", "yMin": 0, "yMax": 1},
    {"key": "precision", "title": "Detector Precision", "team": "blue", "desc": "Fraction of flagged transactions that are truly fraud.", "yMin": 0, "yMax": 1},
    {"key": "roc_auc", "title": "ROC-AUC", "team": "blue", "desc": "Overall discrimination between fraud and legitimate transactions.", "yMin": 0.85, "yMax": 1},
    {"key": "false_positive_rate", "title": "False Positive Rate", "team": "blue", "desc": "How often legitimate payments are incorrectly flagged (lower is better).", "yMin": 0, "yMax": None},
    {"key": "historical_robustness_f1", "title": "Historical Attack Robustness (F1)", "team": "blue", "desc": "Whether the current detector version still catches fraud patterns from earlier rounds.", "yMin": 0, "yMax": 1},
    {"key": "attack_novelty_score", "title": "Attack Novelty", "team": "red", "desc": "How different each round's attack hypothesis is from prior Attack Memory context.", "yMin": 0, "yMax": 1},
    {"key": "attack_fidelity_behavioural_plausibility", "title": "Attack Fidelity (Behavioural Plausibility)", "team": "red", "desc": "How realistically generated attacks resemble reference payment behaviour.", "yMin": 0, "yMax": 1},
    {"key": "family_coverage_cumulative_ratio", "title": "7-Family Coverage (Cumulative)", "team": "red", "desc": "Share of the seven approved fraud families explored so far in the run.", "yMin": 0, "yMax": 1},
    {"key": "variant_unique_ratio", "title": "Cross-Round Variant Uniqueness", "team": "red", "desc": "Share of this round's attacks that are not near-duplicates of earlier rounds' attacks.", "yMin": 0, "yMax": 1},
    {"key": "attack_diversity_channel_entropy", "title": "Attack Channel Diversity", "team": "red", "desc": "Normalized entropy of channel usage (web / mobile / card-present) within the round's attacks.", "yMin": 0, "yMax": 1},
    {"key": "attack_difficulty_score", "title": "Attack Difficulty", "team": "red", "desc": "1 - recall for the round: how hard this round's attacks were for the detector to catch.", "yMin": 0, "yMax": 1},
]


def _latest(pattern: str) -> Path | None:
    matches = sorted(ARTIFACTS.glob(pattern))
    return matches[-1] if matches else None


def load_json(pattern: str, default):
    path = _latest(pattern)
    if path is None:
        return default, None
    return json.loads(path.read_text(encoding="utf-8")), path.name


def render_chart_cards() -> str:
    cards = []
    for spec in CHART_SPECS:
        accent = "var(--mc-red)" if spec["team"] == "red" else "var(--mc-blue)"
        badge = "RED TEAM" if spec["team"] == "red" else "BLUE TEAM"
        cards.append(f"""
        <div class="chart-card" style="border-top-color:{accent}">
          <div class="chart-card-head">
            <span class="badge" style="background:{accent}">{badge}</span>
            <h3>{spec['title']}</h3>
          </div>
          <p class="chart-desc">{spec['desc']}</p>
          <canvas id="chart-{spec['key']}" width="560" height="260"></canvas>
          <p class="chart-legend">Solid: per-round value &middot; Dashed: trailing 3-round average</p>
        </div>""")
    return "\n".join(cards)


def build() -> None:
    metrics, metrics_file = load_json("adaptive_v2_metrics_dump_*.json", [])
    summary, summary_file = load_json("adaptive_v2_summary_*.json", {})

    metrics_json = json.dumps(metrics)
    chart_specs_json = json.dumps(CHART_SPECS)
    families_explored = ", ".join(summary.get("families_explored", [])) or "n/a"
    families_missing = ", ".join(summary.get("families_never_reached", [])) or "none"
    rounds = summary.get("rounds", len(metrics))
    run_stamp = summary.get("run_timestamp_utc", metrics_file or "n/a")
    detector_mode = summary.get("detector_mode", "n/a")

    html_template = HTML_TEMPLATE
    html_template = html_template.replace("__ARCHITECTURE_IMAGE__", ARCHITECTURE_IMAGE)
    html_template = html_template.replace("__RUN_STAMP__", str(run_stamp))
    html_template = html_template.replace("__ROUNDS__", str(rounds))
    html_template = html_template.replace("__DETECTOR_MODE__", str(detector_mode))
    html_template = html_template.replace("__FAMILIES_EXPLORED__", families_explored)
    html_template = html_template.replace("__FAMILIES_MISSING__", families_missing)
    html_template = html_template.replace("__CHART_CARDS__", render_chart_cards())
    html_template = html_template.replace("__METRICS_JSON__", metrics_json)
    html_template = html_template.replace("__CHART_SPECS_JSON__", chart_specs_json)
    html_template = html_template.replace("__SOURCE_FILES__", f"{metrics_file or 'n/a'} / {summary_file or 'n/a'}")

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html_template, encoding="utf-8")
    print(f"Dashboard written to {OUTPUT_HTML}")
    print(f"Metrics rows embedded: {len(metrics)} (source: {metrics_file})")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Adaptive Red-Team / Blue-Team Fraud Defence -- Submission Dashboard</title>
<style>
  :root {
    --mc-red: #EB001B;
    --mc-orange: #F79E1B;
    --mc-blue: #1B3A6B;
    --mc-dark: #111111;
    --mc-grey: #f4f5f7;
    --mc-text: #1c1c1c;
    --mc-muted: #5b5b5b;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; font-family: "Segoe UI", Helvetica, Arial, sans-serif; color: var(--mc-text);
    background: #ffffff;
  }
  a { color: inherit; }
  .navbar {
    position: sticky; top: 0; z-index: 50; display: flex; align-items: center; gap: 22px;
    padding: 12px 28px; background: rgba(17,17,17,0.95); backdrop-filter: blur(6px);
  }
  .navbar .brand { display:flex; align-items:center; gap:10px; margin-right: auto; }
  .navbar .dots { display:flex; }
  .navbar .dots span { width:22px; height:22px; border-radius:50%; display:inline-block; margin-left:-9px; }
  .navbar .dots span:first-child { background: var(--mc-red); margin-left:0; }
  .navbar .dots span:last-child { background: var(--mc-orange); mix-blend-mode: screen; }
  .navbar .brand b { color: #fff; font-size: 15px; letter-spacing: 0.3px; }
  .navbar nav a {
    color: #d8d8d8; text-decoration: none; font-size: 13px; font-weight: 600;
    padding: 6px 10px; border-radius: 6px; transition: all .15s ease;
  }
  .navbar nav a:hover, .navbar nav a.active { color: #fff; background: rgba(255,255,255,0.12); }

  .hero {
    background: radial-gradient(1200px 500px at 20% -10%, rgba(235,0,27,0.35), transparent),
                radial-gradient(1200px 600px at 90% 10%, rgba(247,158,27,0.28), transparent),
                linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 60%, #0d0d0d 100%);
    color: #fff; padding: 90px 8vw 70px; position: relative; overflow: hidden;
  }
  .hero .rings { position:absolute; top:-40px; right:-40px; width:260px; height:170px; opacity:0.35; z-index:0; filter: blur(1px); pointer-events:none; }
  .hero .rings .r1, .hero .rings .r2 { position:absolute; width:170px; height:170px; border-radius:50%; }
  .hero .rings .r1 { background: var(--mc-red); left:0; }
  .hero .rings .r2 { background: var(--mc-orange); left:85px; mix-blend-mode: screen; }
  .hero .content { position: relative; z-index: 2; }
  .hero .eyebrow { color: var(--mc-orange); font-weight:700; letter-spacing:2px; font-size:13px; text-transform:uppercase; }
  .hero h1 { font-size: clamp(28px, 4.2vw, 46px); line-height:1.15; margin: 14px 0 18px; max-width: 780px; }
  .hero p.lead { font-size: 17px; color: #d9d9d9; max-width: 720px; line-height:1.6; }
  .hero .stat-row { display:flex; flex-wrap:wrap; gap: 22px; margin-top: 34px; }
  .hero .stat { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 12px; padding: 14px 20px; min-width:150px; }
  .hero .stat .n { font-size: 22px; font-weight:800; color:#fff; }
  .hero .stat .l { font-size: 11.5px; color: #b8b8b8; text-transform:uppercase; letter-spacing:.5px; }
  @media (max-width: 900px) { .hero .rings { display:none; } }

  section { padding: 64px 8vw; }
  section.alt { background: var(--mc-grey); }
  .section-label { color: var(--mc-red); font-weight:800; font-size:12.5px; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px; }
  h2.section-title { font-size: clamp(24px, 3vw, 32px); margin: 0 0 22px; color: var(--mc-dark); }
  p.body-text { color: var(--mc-muted); font-size: 15.5px; line-height:1.75; max-width: 900px; }

  .req-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px,1fr)); gap:18px; margin-top: 26px; }
  .req-card { background:#fff; border: 1px solid #e7e7e7; border-radius: 14px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.03); transition: transform .18s ease, box-shadow .18s ease; }
  .req-card:hover { transform: translateY(-4px); box-shadow: 0 10px 24px rgba(0,0,0,0.08); }
  .req-card .icon { font-size: 22px; margin-bottom: 10px; }
  .req-card h4 { margin: 0 0 6px; font-size: 15px; color: var(--mc-dark); }
  .req-card p { margin:0; font-size: 13.5px; color: var(--mc-muted); line-height:1.55; }

  .rb-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 26px; margin-top: 30px; }
  @media (max-width: 820px) { .rb-grid, .req-grid { grid-template-columns: 1fr; } }
  .rb-card { border-radius: 18px; padding: 28px; color:#fff; position:relative; overflow:hidden; }
  .rb-card.red { background: linear-gradient(160deg, var(--mc-red), #8f0012); }
  .rb-card.blue { background: linear-gradient(160deg, var(--mc-blue), #0c1f3d); }
  .rb-card h3 { margin-top:0; font-size: 20px; }
  .rb-card ul { padding-left: 20px; margin: 14px 0 0; }
  .rb-card li { margin-bottom: 8px; font-size: 14px; line-height:1.5; color: rgba(255,255,255,0.92); }
  .rb-card .tag { display:inline-block; font-size: 11px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; background: rgba(255,255,255,0.18); padding: 4px 10px; border-radius: 999px; margin-bottom: 10px; }

  .arch-wrap { margin-top: 26px; background:#fff; border:1px solid #e7e7e7; border-radius:16px; padding: 18px; text-align:center; }
  .arch-wrap img { max-width: 100%; border-radius: 10px; }
  .arch-caption { font-size: 12.5px; color: var(--mc-muted); margin-top: 10px; }

  .run-meta { display:flex; flex-wrap:wrap; gap: 10px; margin: 18px 0 6px; }
  .pill { background:#fff; border:1px solid #e2e2e2; border-radius:999px; padding:6px 14px; font-size:12.5px; color: var(--mc-dark); }
  .pill b { color: var(--mc-red); }

  .chart-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(420px,1fr)); gap: 22px; margin-top: 28px; }
  .chart-card { background:#fff; border:1px solid #e7e7e7; border-top: 4px solid var(--mc-blue); border-radius:14px; padding:18px 18px 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.03); }
  .chart-card-head { display:flex; align-items:center; gap:10px; }
  .chart-card-head h3 { margin:0; font-size:15px; color: var(--mc-dark); }
  .chart-legend { margin:6px 0 0; font-size:11px; color:#8a8a8a; }
  .badge { color:#fff; font-size:10px; font-weight:800; letter-spacing:1px; padding:3px 8px; border-radius:6px; }
  .chart-desc { font-size:12.5px; color: var(--mc-muted); margin: 8px 0 4px; line-height:1.5; }
  canvas { width: 100%; height: auto; }

  .conclusion { background: linear-gradient(135deg, #0d0d0d, #1a1a1a); color:#fff; }
  .conclusion h2.section-title { color:#fff; }
  .conclusion p.body-text { color:#d6d6d6; }
  .conclusion .quote { border-left: 4px solid var(--mc-orange); padding-left: 18px; margin-top: 26px; font-size: 17px; line-height:1.7; color:#f1f1f1; max-width: 880px; }

  footer { padding: 26px 8vw; background:#0d0d0d; color:#9a9a9a; font-size:12px; text-align:center; }
  footer b { color:#fff; }
</style>
</head>
<body>

  <div class="navbar">
    <div class="brand">
      <span class="dots"><span></span><span></span></span>
      <b>AI Defence Lab &mdash; Submission Dashboard</b>
    </div>
    <nav>
      <a href="#problem">Problem</a>
      <a href="#requirements">Requirements</a>
      <a href="#teams">Red / Blue</a>
      <a href="#architecture">Architecture</a>
      <a href="#metrics">Metrics</a>
      <a href="#conclusion">Conclusion</a>
    </nav>
  </div>

  <header class="hero">
    <div class="rings"><div class="r1"></div><div class="r2"></div></div>
    <div class="content">
    <div class="eyebrow">Mastercard Innovation Challenge 2026</div>
    <h1>An adaptive red-team / blue-team system that lets fraud defence learn as fast as fraud evolves.</h1>
    <p class="lead">A closed-loop, agentic pipeline where an LLM-driven attacker continuously proposes new synthetic
    fraud scenarios and a continually-retrained detector defends against them &mdash; with every round's diversity,
    fidelity, novelty and detection metrics captured for full auditability.</p>
    <div class="stat-row">
      <div class="stat"><div class="n">__ROUNDS__</div><div class="l">Rounds in latest run</div></div>
      <div class="stat"><div class="n">7</div><div class="l">Approved fraud families</div></div>
      <div class="stat"><div class="n">__DETECTOR_MODE__</div><div class="l">Detector mode</div></div>
      <div class="stat"><div class="n">Qwen2.5-7B</div><div class="l">Agent reasoning model</div></div>
    </div>
    </div>
  </header>

  <section id="problem">
    <div class="section-label">Problem Statement</div>
    <h2 class="section-title">Fraud is becoming agentic. Defence needs to become agentic too.</h2>
    <p class="body-text">
      Digital payment fraud increasingly relies on adaptive, automated techniques that can iterate faster than
      static, rule-based detection systems can be updated. As agentic AI lowers the cost of reconnaissance and
      technique adaptation for attackers, defensive systems need an equivalent capability: continuous, automated
      identification of new attack patterns paired with continual model hardening &mdash; entirely within a safe,
      synthetic, offline experimentation environment.
    </p>
  </section>

  <section id="requirements" class="alt">
    <div class="section-label">Hackathon Requirements</div>
    <h2 class="section-title">What this submission had to satisfy</h2>
    <div class="req-grid">
      <div class="req-card"><div class="icon">🧪</div><h4>Synthetic data only</h4><p>No real cardholder, PII, or production payment data is used anywhere in the pipeline.</p></div>
      <div class="req-card"><div class="icon">🔒</div><h4>Offline &amp; sandboxed</h4><p>No live-system or payment-infrastructure targeting; every round runs against synthetic reference data.</p></div>
      <div class="req-card"><div class="icon">📦</div><h4>Code repository</h4><p>A versioned GitHub repository containing the full closed-loop implementation.</p></div>
      <div class="req-card"><div class="icon">📝</div><h4>Solution walkthrough</h4><p>This dashboard plus the underlying pipeline/metrics guide documents the design end-to-end.</p></div>
      <div class="req-card"><div class="icon">🖥️</div><h4>Working web prototype</h4><p>A functioning Streamlit demo and this static one-page submission dashboard.</p></div>
      <div class="req-card"><div class="icon">🗂️</div><h4>Provenance &amp; auditability</h4><p>Every round's decision, generated attack, and detector evaluation is persisted to Attack Memory.</p></div>
    </div>
  </section>

  <section id="teams">
    <div class="section-label">Roles</div>
    <h2 class="section-title">Red Team vs. Blue Team</h2>
    <div class="rb-grid">
      <div class="rb-card red">
        <span class="tag">Red Team</span>
        <h3>Adaptive attack generation</h3>
        <ul>
          <li><b>Agent 1</b> researches a new attack hypothesis each round, weighted toward whichever of the seven approved fraud families the detector was recently weak against.</li>
          <li><b>Agent 2</b> converts the hypothesis into a structured, bounded attack specification (temporal, amount, device, beneficiary, channel constraints).</li>
          <li>A procedural or CTGAN-based generator produces disjoint <b>train</b> and <b>unseen</b> synthetic attack batches per round.</li>
          <li>Diversity, fidelity, novelty, family coverage and cross-round redundancy are measured every round.</li>
        </ul>
      </div>
      <div class="rb-card blue">
        <span class="tag">Blue Team</span>
        <h3>Continual detector hardening</h3>
        <ul>
          <li>A HistGradientBoosting detector is evaluated on each round's <b>unseen</b> attack batch plus a legitimate holdout &mdash; never on its own training rows.</li>
          <li>In continual mode, missed attacks feed a replay buffer that is included in the next detector version's training data.</li>
          <li><b>Agent 3</b> analyzes detector + fidelity results and writes a structured weakness report back to Attack Memory.</li>
          <li>Historical robustness is checked against a rolling pool of previous rounds' fraud rows, so forgetting is measurable, not assumed.</li>
        </ul>
      </div>
    </div>
  </section>

  <section id="architecture" class="alt">
    <div class="section-label">Architecture</div>
    <h2 class="section-title">Closed-loop system design</h2>
    <div class="arch-wrap">
      <img src="__ARCHITECTURE_IMAGE__" alt="System architecture diagram" onerror="this.replaceWith(Object.assign(document.createElement('div'),{innerText:'architecture_diagram.png not found next to this HTML file yet.',style:'padding:40px;color:#999;'}))">
      <div class="arch-caption">Generated via <code>src/ui/build_architecture_diagram.py</code> from the current codebase.</div>
    </div>
  </section>

  <section id="metrics">
    <div class="section-label">Results</div>
    <h2 class="section-title">Metrics across training rounds</h2>
    <p class="body-text">Every chart below reads <b>value on the Y axis</b> and <b>training round number on the X axis</b>,
      sourced directly from the latest saved local results file &mdash; regenerating this page after a new run
      updates every chart automatically, without re-running the pipeline.</p>
    <div class="run-meta">
      <span class="pill">Run: <b>__RUN_STAMP__</b></span>
      <span class="pill">Families explored: <b>__FAMILIES_EXPLORED__</b></span>
      <span class="pill">Not yet reached: <b>__FAMILIES_MISSING__</b></span>
      <span class="pill">Source: <b>__SOURCE_FILES__</b></span>
    </div>
    <div class="chart-grid">
__CHART_CARDS__
    </div>
  </section>

  <section id="conclusion" class="conclusion">
    <div class="section-label" style="color:var(--mc-orange)">Conclusion</div>
    <h2 class="section-title">What this demonstrates for agentic fraud defence</h2>
    <p class="body-text">
      Framing fraud defence as a continuous red-team/blue-team loop &mdash; rather than a one-time model
      training exercise &mdash; lets a defensive system stay adaptive against agentic, fast-iterating attackers.
      By having an LLM-driven researcher propose new attack directions from the detector's own recorded
      weaknesses, and by feeding the detector's missed cases back into its own training data, the system
      creates a measurable, auditable arms race instead of a static snapshot.
    </p>
    <div class="quote">
      "The value isn't a single accuracy number &mdash; it's the closed loop itself: every round produces
      evidence of what the attacker tried, how realistic and novel it was, and exactly how the defender
      responded. That evidence trail is what allows this kind of system to be trusted, tuned, and scaled
      toward real agentic fraud detection."
    </div>
  </section>

  <footer>
    Built for the <b>Mastercard Innovation Challenge 2026</b> &middot; All data synthetic &middot; No live payment systems or PII involved.
  </footer>

<script>
const METRICS = __METRICS_JSON__;
const CHART_SPECS = __CHART_SPECS_JSON__;

function drawLineChart(canvas, series, spec) {
  const ctx = canvas.getContext('2d');
  const ratio = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || canvas.width;
  const H = canvas.clientHeight || canvas.height;
  canvas.width = W * ratio; canvas.height = H * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const pad = { l: 54, r: 14, t: 14, b: 46 };
  const xs = series.map(p => p.x);
  const ys = series.map(p => p.y).filter(v => v !== null && v !== undefined);
  if (!xs.length || !ys.length) {
    ctx.fillStyle = '#999'; ctx.font = '12px Arial'; ctx.textAlign = 'center';
    ctx.fillText('No data available', W / 2, H / 2);
    return;
  }
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const dataMin = Math.min(...ys), dataMax = Math.max(...ys);
  const yMin = (spec.yMin !== null && spec.yMin !== undefined) ? spec.yMin : Math.min(dataMin, 0);
  let yMax = (spec.yMax !== null && spec.yMax !== undefined) ? spec.yMax : dataMax;
  if (yMax <= yMin) yMax = yMin + 1;
  const xScale = x => pad.l + (xMax > xMin ? (x - xMin) / (xMax - xMin) : 0) * (W - pad.l - pad.r);
  const yScale = y => H - pad.b - ((y - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);

  ctx.strokeStyle = '#e3e3e3'; ctx.lineWidth = 1;
  const ticks = 4;
  ctx.fillStyle = '#7a7a7a'; ctx.font = '10px Arial'; ctx.textAlign = 'right';
  for (let i = 0; i <= ticks; i++) {
    const v = yMin + (yMax - yMin) * i / ticks;
    const y = yScale(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(2), pad.l - 6, y + 3);
  }

  ctx.strokeStyle = '#bbb';
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();

  ctx.fillStyle = '#7a7a7a'; ctx.textAlign = 'center'; ctx.font = '10px Arial';
  const step = Math.max(1, Math.ceil(xs.length / 7));
  let lastLabeledIndex = -Infinity;
  xs.forEach((x, i) => {
    const isStepTick = i % step === 0;
    const isFarEnoughLast = i === xs.length - 1 && (i - lastLabeledIndex) >= Math.ceil(step / 2);
    if (isStepTick || isFarEnoughLast) {
      ctx.fillText(String(x), xScale(x), H - pad.b + 13);
      lastLabeledIndex = i;
    }
  });

  ctx.fillStyle = '#444'; ctx.font = '11px Arial'; ctx.textAlign = 'center';
  ctx.fillText('Number of Rounds', pad.l + (W - pad.l - pad.r) / 2, H - 6);
  ctx.save();
  ctx.translate(14, pad.t + (H - pad.t - pad.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Value', 0, 0);
  ctx.restore();

  const color = spec.team === 'red' ? '#EB001B' : '#1B3A6B';
  ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.beginPath();
  let started = false;
  series.forEach(p => {
    if (p.y === null || p.y === undefined) return;
    const X = xScale(p.x), Y = yScale(p.y);
    if (!started) { ctx.moveTo(X, Y); started = true; } else { ctx.lineTo(X, Y); }
  });
  ctx.stroke();

  ctx.fillStyle = color;
  series.forEach(p => {
    if (p.y === null || p.y === undefined) return;
    ctx.beginPath(); ctx.arc(xScale(p.x), yScale(p.y), 2.6, 0, Math.PI * 2); ctx.fill();
  });

  // Trailing 3-round rolling average, drawn as a dashed overlay. This does not alter any stored
  // metric value -- it only smooths the visual trend line since raw per-round values are noisy
  // (each round evaluates a different attack family with a different intrinsic difficulty).
  const windowSize = 3;
  const rolling = series.map((p, i) => {
    if (p.y === null || p.y === undefined) return { x: p.x, y: null };
    const windowVals = series.slice(Math.max(0, i - windowSize + 1), i + 1)
      .map(q => q.y).filter(v => v !== null && v !== undefined);
    const avg = windowVals.reduce((a, b) => a + b, 0) / windowVals.length;
    return { x: p.x, y: avg };
  });
  ctx.strokeStyle = color; ctx.globalAlpha = 0.55; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
  ctx.beginPath();
  let rollingStarted = false;
  rolling.forEach(p => {
    if (p.y === null || p.y === undefined) return;
    const X = xScale(p.x), Y = yScale(p.y);
    if (!rollingStarted) { ctx.moveTo(X, Y); rollingStarted = true; } else { ctx.lineTo(X, Y); }
  });
  ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;
}

function renderAll() {
  CHART_SPECS.forEach(spec => {
    const canvas = document.getElementById('chart-' + spec.key);
    if (!canvas) return;
    const series = METRICS.map(row => ({ x: row.round, y: row[spec.key] }));
    drawLineChart(canvas, series, spec);
  });
}

window.addEventListener('load', renderAll);
window.addEventListener('resize', renderAll);

const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.navbar nav a');
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(link => link.classList.toggle('active', link.getAttribute('href') === '#' + entry.target.id));
    }
  });
}, { rootMargin: '-40% 0px -55% 0px' });
sections.forEach(sec => observer.observe(sec));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
