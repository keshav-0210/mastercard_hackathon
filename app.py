from __future__ import annotations

import json

import streamlit as st

from mastercard_defence.loop import ClosedLoop, load_config

st.set_page_config(page_title="AI Defence Lab", page_icon="M", layout="wide")
st.title("AI Defence Lab")
st.caption("Synthetic red-team / blue-team closed-loop demonstration")

config = load_config()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Run mode", config["run_mode"])
with col2:
    st.metric("Configured rounds", config["pipeline"]["rounds"])
with col3:
    st.metric("Data policy", "Synthetic only")

if st.button("Run two-round closed loop", type="primary"):
    loop = ClosedLoop(config)
    try:
        results = loop.run(rounds=2)
    finally:
        loop.close()
    for result in results:
        st.subheader(f"Round {result['round']}")
        left, right = st.columns(2)
        with left:
            st.write("**Research hypothesis**")
            st.json(result["hypothesis"].model_dump())
            st.write("**Attack specification**")
            st.json(result["specification"].model_dump())
        with right:
            st.write("**Fidelity**")
            st.json(result["fidelity"])
            st.write("**Detection**")
            st.json(result["detection"])
            st.write("**Agent 3 -> Attack Memory**")
            st.json(result["weakness"].model_dump())
        st.divider()

st.info("GPU workloads are intentionally excluded from this local demonstration. Kaggle GPU execution will be added behind RUN_MODE=KAGGLE_GPU.")
