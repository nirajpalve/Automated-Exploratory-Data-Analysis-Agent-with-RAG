import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from agent import run
from report import build_html

st.set_page_config(page_title="EDA Agent", page_icon="📊", layout="wide")
st.title("📊 Automated EDA Agent")
st.caption("Upload a CSV. The agent plans its own analysis, follows EDA best practices "
           "retrieved from its knowledge base, and writes a report.")

uploaded = st.file_uploader("Upload a CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write(f"**{uploaded.name}**: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))

    if st.button("Run analysis", type="primary"):
        save_dir = Path("data/uploads")
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / uploaded.name
        df.to_csv(path, index=False)
        with st.spinner("Planning, analysing and writing the report... (about a minute)"):
            try:
                st.session_state["result"] = run(str(path))
            except Exception as e:
                st.error(f"The analysis failed: {e}")

result = st.session_state.get("result")
if result:
    tab_report, tab_plots, tab_plan = st.tabs(["Report", "Plots", "Agent plan"])

    with tab_report:
        st.markdown(result["report"])

    with tab_plots:
        plots = sorted(Path("outputs/plots").glob("*.png"))
        cols = st.columns(2)
        for i, p in enumerate(plots):
            cols[i % 2].image(str(p), caption=p.stem)

    with tab_plan:
        st.subheader("Steps the agent chose")
        for i, s in enumerate(result["plan"], 1):
            st.markdown(f"**{i}. {s['tool']}** {s['args'] or ''}  \n{s['reason']}")
        st.subheader("Best-practice sections retrieved")
        st.json(result["guidance_used"])

    st.divider()
    c1, c2 = st.columns(2)
    c1.download_button("Download report (HTML)", build_html(result["report"]),
                       file_name="eda_report.html", mime="text/html")
    c2.download_button("Download report (Markdown)", result["report"],
                       file_name="eda_report.md")