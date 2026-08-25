from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mastercard_defence.loop import ClosedLoop, load_config


st.set_page_config(page_title="AI Defence Lab", page_icon="M", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background: #f4f1ea; color: #162a2a; }
    [data-testid="stSidebar"] { background: #102f2f; }
    [data-testid="stSidebar"] * { color: #f4f1ea !important; }
    .hero { padding: 2.2rem 2.4rem; background: linear-gradient(120deg, #123c3b 0%, #1f625b 100%); color: #f8f4e9; border-radius: 4px; margin-bottom: 1.2rem; }
    .hero h1 { font-size: 2.7rem; line-height: 1; margin: 0 0 .65rem; letter-spacing: 0; }
    .hero p { max-width: 760px; margin: 0; color: #d9e8df; font-size: 1.05rem; }
    .flow { display: flex; gap: .55rem; align-items: stretch; margin: 1rem 0 1.5rem; }
    .node { flex: 1; min-height: 104px; padding: 1rem; background: #fffdf7; border: 1px solid #d5cdbd; border-top: 4px solid #d56b3d; }
    .node strong { display: block; color: #123c3b; margin-bottom: .35rem; }
    .node small { color: #5d6a65; line-height: 1.35; }
    .arrow { align-self: center; color: #d56b3d; font-size: 1.5rem; }
    .section-label { color: #d56b3d; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .76rem; }
    div[data-testid="stMetric"] { background: #fffdf7; border: 1px solid #d5cdbd; padding: .8rem; }
    @media (max-width: 800px) { .flow { flex-direction: column; } .arrow { transform: rotate(90deg); align-self: center; } .hero h1 { font-size: 2rem; } }
</style>
""", unsafe_allow_html=True)


config = load_config()
artifact_paths = sorted(
    list((ROOT / config["paths"]["artifacts"]).glob("robustness_results_*.json"))
    + list((ROOT / config["paths"]["artifacts"]).glob("*ctgan*_results_*.json")),
    reverse=True,
)

st.markdown("""
<div class="hero">
  <div class="section-label">Synthetic payment security laboratory</div>
  <h1>AI Defence Lab</h1>
  <p>Identify emerging fraud patterns, generate controlled attack scenarios, and defend against unseen synthetic transactions in one auditable loop.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Closed-loop architecture</div>', unsafe_allow_html=True)
st.markdown("""
<div class="flow">
  <div class="node"><strong>01 · Identify</strong><small>RAG evidence, attack memory, and the research agent select a new family and hypothesis.</small></div>
  <div class="arrow">→</div>
  <div class="node"><strong>02 · Generate</strong><small>The specification agent defines a constrained scenario for synthetic transaction generation.</small></div>
  <div class="arrow">→</div>
  <div class="node"><strong>03 · Defend</strong><small>The blue-team detector scores unseen attacks and separates fidelity from detectability.</small></div>
  <div class="arrow">→</div>
  <div class="node"><strong>04 · Learn</strong><small>Weakness findings return to memory and steer the next research round.</small></div>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_artifact(path_string: str) -> dict:
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


with st.sidebar:
    st.markdown("### Control room")
    st.metric("Execution mode", config["run_mode"])
    st.metric("Data policy", "Synthetic only")
    st.caption("Select a saved run to explore its evidence trail.")

if artifact_paths:
    selected_path = st.sidebar.selectbox("Saved experiment", artifact_paths, format_func=lambda path: path.name)
    artifact = load_artifact(str(selected_path))
else:
    artifact = None
    st.warning("No saved robustness artifact found in the artifacts directory.")

if artifact:
    summary = artifact.get("summary", {})
    overview, architecture, rounds_tab, evidence = st.tabs(["Overview", "Architecture", "Round explorer", "Evidence trail"])
    with overview:
        st.markdown('<div class="section-label">Validated baseline</div>', unsafe_allow_html=True)
        st.subheader(f"{artifact.get('seed_count', 0)} seeds · {artifact.get('rounds', 0)} rounds")
        metrics = st.columns(4)
        for column, name in zip(metrics, ("f1", "recall", "precision", "roc_auc")):
            column.metric(name.upper(), f"{summary.get(name, {}).get('mean', 0.0):.4f}", f"std {summary.get(name, {}).get('std', 0.0):.4f}")
        family_rows = pd.DataFrame(artifact.get("family_analysis", []))
        if family_rows.empty:
            family_rows = pd.DataFrame([
                {
                    "seed": run["seed"],
                    "round": result["round"],
                    "attack_family": result.get("attack_family", "unknown"),
                    "f1": result.get("detection", {}).get("f1", 0.0),
                    "recall": result.get("detection", {}).get("recall", 0.0),
                    "precision": result.get("detection", {}).get("precision", 0.0),
                    "roc_auc": result.get("detection", {}).get("roc_auc", 0.0),
                }
                for run in artifact.get("by_seed", [])
                for result in run.get("results", [])
            ])
        if not family_rows.empty:
            st.subheader("Detector response by attack family")
            selected_families = st.multiselect("Families", sorted(family_rows["attack_family"].unique()), default=sorted(family_rows["attack_family"].unique()))
            visible = family_rows[family_rows["attack_family"].isin(selected_families)]
            st.dataframe(visible, width="stretch", hide_index=True)
    with architecture:
        st.subheader("How one round moves through the system")
        stages = ["Research hypothesis", "Attack specification", "Synthetic generation", "Unseen evaluation", "Weakness memory"]
        chosen_stage = st.radio("Inspect stage", stages, horizontal=True)
        descriptions = {
            stages[0]: "Agent 1 combines reviewed evidence with recent weakness memory and selects an unused attack family.",
            stages[1]: "Agent 2 turns the hypothesis into bounded temporal, amount, device, beneficiary, and evasion constraints.",
            stages[2]: "The generator creates labelled synthetic rows while preserving family, round, and generation metadata.",
            stages[3]: "The detector is trained on one batch and evaluated on disjoint unseen attack rows plus legitimate holdout data.",
            stages[4]: "Agent 3 records detector and fidelity findings so the next research round can change direction.",
        }
        st.info(descriptions[chosen_stage])
        st.code("Agent 1 → Hypothesis → Agent 2 → Specification → Generator → Detector → Agent 3 → Attack Memory → Agent 1", language="text")
    with rounds_tab:
        runs = artifact.get("by_seed", [])
        seed_values = [run["seed"] for run in runs]
        if seed_values:
            selected_seed = st.selectbox("Seed run", seed_values)
            selected_run = next(run for run in runs if run["seed"] == selected_seed)
            round_values = [result["round"] for result in selected_run["results"]]
            selected_round = st.slider("Round", min(round_values), max(round_values), min(round_values))
            result = next(item for item in selected_run["results"] if item["round"] == selected_round)
            left, right = st.columns(2)
            with left:
                st.subheader("Research and generation")
                st.write({"attack_family": result.get("attack_family", "unknown"), "generator": artifact.get("generator_backend", "procedural")})
                if "hypothesis" in result:
                    st.json(result["hypothesis"])
                if "specification" in result:
                    st.json(result["specification"])
            with right:
                st.subheader("Evaluation and learning")
                st.json({key: result[key] for key in ("fidelity", "diversity", "detection", "weakness") if key in result})
    with evidence:
        st.subheader("Experiment provenance")
        st.write({"artifact": selected_path.name, "timestamp_utc": artifact.get("run_timestamp_utc"), "families": artifact.get("families_per_run"), "protocol": "unseen attack rows and legitimate holdout"})
        st.caption("All metrics are internal synthetic-experiment evidence, not official Mastercard scores or live-payment performance claims.")

st.divider()
st.markdown('<div class="section-label">Live local demonstration</div>', unsafe_allow_html=True)
demo_rounds = st.slider("Demo rounds", 1, 3, 2)
if st.button("Run closed loop", type="primary", width="stretch"):
    demo_config = dict(config)
    demo_config["paths"] = dict(config["paths"])
    demo_config["paths"]["memory_db"] = tempfile.mktemp(suffix=".sqlite")
    loop = ClosedLoop(demo_config)
    try:
        results = loop.run(rounds=demo_rounds)
    finally:
        loop.close()
    for result in results:
        with st.expander(f"Round {result['round']} · {result['specification'].attack_family}", expanded=True):
            st.json({"hypothesis": result["hypothesis"].model_dump(), "specification": result["specification"].model_dump(), "fidelity": result["fidelity"], "detection": result["detection"], "weakness": result["weakness"].model_dump()})
