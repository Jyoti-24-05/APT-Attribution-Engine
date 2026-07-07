
import json
import pathlib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from inference import predict_top3

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="APT Attribution Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── load supporting artifacts ─────────────────────────────────
ROOT      = pathlib.Path(__file__).parent.resolve()
ARTIFACTS = ROOT / "artifacts"

@st.cache_resource(show_spinner=False)
def load_eval_summary():
    p = ARTIFACTS / "eval_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_resource(show_spinner=False)
def load_shap_table():
    p = ARTIFACTS / "shap_importance.json"
    return json.loads(p.read_text()) if p.exists() else []

eval_summary = load_eval_summary()
shap_table   = load_shap_table()

# ── preset known APT signatures ───────────────────────────────
PRESETS = {
    "APT28 — Fancy Bear (Russia)": [
        "T1059","T1078","T1566","T1053","T1027",
        "T1105","T1036","T1016","T1057","T1083",
    ],
    "Lazarus Group (North Korea)": [
        "T1059","T1003","T1071","T1041","T1105",
        "T1547","T1055","T1070","T1140","T1486",
    ],
    "APT29 — Cozy Bear (Russia)": [
        "T1078","T1566","T1027","T1036","T1071",
        "T1090","T1102","T1560","T1048","T1070",
    ],
    "APT41 — Winnti (China)": [
        "T1059","T1078","T1021","T1047","T1053",
        "T1027","T1036","T1055","T1070","T1112",
    ],
    "Minimal incident — 2 techniques": [
        "T1059","T1003",
    ],
}

RANK_COLORS  = ["#1D9E75", "#378ADD", "#888780"]
RANK_BG      = ["rgba(29,158,117,0.08)", "rgba(55,138,221,0.07)", "rgba(136,135,128,0.07)"]
RANK_MEDALS  = ["", "", ""]

# ── global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── top banner ────────────── */
.banner {
    background: linear-gradient(135deg, #0A0F1E 0%, #0D2137 60%, #0A2B1A 100%);
    border-radius: 12px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
    border: 1px solid rgba(29,158,117,0.2);
    position: relative;
    overflow: hidden;
}
.banner::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 24px,
        rgba(29,158,117,0.04) 24px, rgba(29,158,117,0.04) 25px
    );
    pointer-events: none;
}
.banner-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 0.15em;
    color: #1D9E75; text-transform: uppercase;
    margin-bottom: 8px;
}
.banner-title {
    font-size: 26px; font-weight: 600; color: #F0F4F8;
    line-height: 1.2; margin-bottom: 6px;
}
.banner-sub {
    font-size: 13px; color: rgba(240,244,248,0.5);
    font-family: 'IBM Plex Mono', monospace;
}

/* ── metric pill ───────────── */
.metric-row { display: flex; gap: 10px; margin-top: 16px; }
.metric-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 10px 16px; flex: 1;
}
.metric-pill .val {
    font-size: 22px; font-weight: 600; color: #F0F4F8;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-pill .lbl {
    font-size: 10px; color: rgba(240,244,248,0.45);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px;
}

/* ── result cards ──────────── */
.result-card {
    border-radius: 10px; padding: 16px 20px;
    margin-bottom: 10px; border-left: 4px solid;
    position: relative;
}
.result-card .rank-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.55; margin-bottom: 4px;
}
.result-card .group-name {
    font-size: 17px; font-weight: 500; margin-bottom: 2px;
}
.result-card .conf-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; opacity: 0.55; margin-top: 4px;
}
.result-card .conf-bar-bg {
    height: 3px; background: rgba(128,128,128,0.15);
    border-radius: 2px; margin-top: 10px; overflow: hidden;
}
.result-card .conf-bar-fill {
    height: 100%; border-radius: 2px; transition: width 0.4s ease;
}

/* ── input area ────────────── */
.section-label {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: rgba(128,128,128,0.7); margin-bottom: 8px;
}

/* ── sidebar ───────────────── */
.sidebar-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #1D9E75;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── banner ────────────────────────────────────────────────────
f1_val  = eval_summary.get("macro_f1",     None)
t3_val  = eval_summary.get("top3_accuracy", None)
n_test  = eval_summary.get("n_test_samples", "—")
n_cls   = eval_summary.get("n_classes_in_test", "—")

f1_str = f"{f1_val:.3f}" if isinstance(f1_val, float) else "—"
t3_str = f"{t3_val:.3f}" if isinstance(t3_val, float) else "—"

st.markdown(f"""
<div class="banner">
  <div class="banner-eyebrow">MITRE ATT&amp;CK · ML Attribution Engine · v1.0</div>
  <div class="banner-title">APT Attribution Engine</div>
  <div class="banner-sub">XGBoost + Random Forest + SVM Ensemble &nbsp;·&nbsp; Optuna-tuned &nbsp;·&nbsp; SHAP Explainability</div>
  <div class="metric-row">
    <div class="metric-pill"><div class="val">{f1_str}</div><div class="lbl">Macro F1</div></div>
    <div class="metric-pill"><div class="val">{t3_str}</div><div class="lbl">Top-3 Accuracy</div></div>
    <div class="metric-pill"><div class="val">{n_test}</div><div class="lbl">Test Samples</div></div>
    <div class="metric-pill"><div class="val">{n_cls}</div><div class="lbl">Attributable Groups</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-head">Preset Incidents</div>', unsafe_allow_html=True)
    st.caption("Load a known APT technique signature for demo.")
    preset_choice = st.selectbox(
        "Select preset", ["— enter manually —"] + list(PRESETS.keys()),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="sidebar-head">About</div>', unsafe_allow_html=True)
    st.markdown("""
**Data source**  
MITRE ATT&CK STIX 2.1 enterprise bundle

**Feature encoding**  
Binary technique presence (root IDs only).  
30 synthetic samples per group via random sub-sampling.

**Model**  
Soft-voting ensemble:  
XGBoost + Random Forest + Calibrated SVM

**Tuning**  
Optuna · 60 trials · macro F1 objective

**Explainability**  
SHAP TreeExplainer on XGBoost base
""")

    st.divider()
    st.caption("⚠ Groups with < 3 techniques excluded from training. Known limitation documented in engineering report.")

# ── main layout ───────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: input ───────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">Incident Technique IDs</div>', unsafe_allow_html=True)

    default_text = ""
    if preset_choice and preset_choice != "— enter manually —":
        default_text = "\n".join(PRESETS[preset_choice])

    raw_input = st.text_area(
        "technique_ids",
        value=default_text,
        height=220,
        placeholder="T1059\nT1003\nT1566\nT1071\nT1027",
        help="One ATT&CK technique ID per line, or comma-separated. Sub-technique IDs (T1059.001) are auto-collapsed to root.",
        label_visibility="collapsed",
    )

    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        run = st.button("Attribute →", type="primary", use_container_width=True)
    with col_info:
        if raw_input.strip():
            import re
            parsed = [t.strip().upper() for t in re.split(r"[\n,]+", raw_input) if t.strip()]
            st.caption(f"{len(parsed)} technique ID(s) entered")
        else:
            parsed = []
            st.caption("Enter technique IDs above")

    # How to use expander
    with st.expander("How to use"):
        st.markdown("""
**Step 1** — Enter ATT&CK technique IDs from the incident  
_(e.g. from an EDR alert, SIEM rule, or threat intel report)_

**Step 2** — Click **Attribute →**

**Step 3** — Review the top-3 candidate threat groups with confidence scores

**Tips:**
- More techniques → more confident attribution
- Sub-technique IDs (T1059.001) are automatically collapsed to root
- Use the preset panel on the left to explore known APT signatures
""")

# ── RIGHT: results ────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">Attribution Results</div>', unsafe_allow_html=True)

    if run:
        if not parsed:
            st.warning("Enter at least one technique ID to attribute.")
        else:
            with st.spinner("Running inference…"):
                try:
                    results = predict_top3(parsed)

                    for r in results:
                        pct   = r["confidence"] * 100
                        color = RANK_COLORS[r["rank"] - 1]
                        bg    = RANK_BG[r["rank"] - 1]
                        medal = RANK_MEDALS[r["rank"] - 1]
                        bar_w = int(r["confidence"] * 100)

                        st.markdown(f"""
<div class="result-card" style="border-color:{color}; background:{bg};">
  <div class="rank-badge">{medal} Rank #{r["rank"]}</div>
  <div class="group-name">{r["group"]}</div>
  <div class="conf-label">Confidence: {r["confidence"]:.4f} &nbsp;({pct:.1f}%)</div>
  <div class="conf-bar-bg">
    <div class="conf-bar-fill" style="width:{bar_w}%; background:{color};"></div>
  </div>
</div>
""", unsafe_allow_html=True)

                    # Plotly bar chart
                    fig = go.Figure(go.Bar(
                        x=[r["confidence"] for r in results],
                        y=[r["group"]      for r in results],
                        orientation="h",
                        marker_color=RANK_COLORS[:len(results)],
                        text=[f"{r['confidence']:.4f}" for r in results],
                        textposition="outside",
                        cliponaxis=False,
                    ))
                    max_conf = results[0]["confidence"]
                    fig.update_layout(
                        xaxis=dict(
                            title="Confidence (probability)",
                            range=[0, min(1.0, max_conf * 1.35)],
                        ),
                        yaxis=dict(autorange="reversed"),
                        height=210,
                        margin=dict(l=0, r=60, t=8, b=36),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=12),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Techniques that matched
                    with st.expander(f"Matched techniques ({len(parsed)} submitted)"):
                        import joblib as _jl
                        fv = _jl.load(ARTIFACTS / "feature_vocab.joblib")
                        matched = [t for t in [
                            t.split(".")[0] if "." in t else t for t in parsed
                        ] if t in fv]
                        unknown = [t for t in parsed if (t.split(".")[0] if "." in t else t) not in fv]
                        if matched:
                            st.success(f"**{len(matched)} known:** {', '.join(sorted(set(matched)))}")
                        if unknown:
                            st.warning(f"**{len(unknown)} unknown (skipped):** {', '.join(unknown)}")

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    else:
        st.markdown("""
<div style="border:1.5px dashed rgba(128,128,128,0.2); border-radius:10px;
            padding:40px 20px; text-align:center; color:rgba(128,128,128,0.5);">
  <div style="font-size:32px; margin-bottom:8px;"></div>
  <div style="font-size:14px;">Enter technique IDs and click <strong>Attribute →</strong></div>
</div>
""", unsafe_allow_html=True)

# ── SHAP panel ────────────────────────────────────────────────
st.divider()

with st.expander(" SHAP Feature Importance — which techniques drive predictions?"):
    if shap_table:
        top_n  = st.slider("Show top N techniques", 5, min(30, len(shap_table)), 15)
        rows   = shap_table[:top_n]
        tids   = [r["technique_id"]  for r in rows]
        vals   = [r["mean_abs_shap"] for r in rows]

        fig2 = go.Figure(go.Bar(
            x=vals[::-1], y=tids[::-1],
            orientation="h",
            marker=dict(
                color=vals[::-1],
                colorscale=[[0,"#E1F5EE"],[0.5,"#1D9E75"],[1,"#085041"]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in vals[::-1]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig2.update_layout(
            xaxis_title="Mean |SHAP value|",
            height=max(300, top_n * 22),
            margin=dict(l=0, r=60, t=8, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "SHAP values computed on XGBoost base estimator across all test samples. "
            "Higher = more discriminative for attributing groups."
        )
    else:
        st.info("Run notebook 03_evaluation.ipynb first to generate SHAP data.")

# ── engineering report panel ──────────────────────────────────
with st.expander(" Evaluation Summary"):
    if eval_summary:
        c1, c2, c3, c4 = st.columns(4)
        def pill(col, val, lbl, ok=None):
            icon = ("✓ " if ok else "✗ " if ok is False else "") if ok is not None else ""
            col.metric(lbl, f"{icon}{val:.4f}" if isinstance(val, float) else str(val))
        pill(c1, eval_summary.get("macro_f1",        0), "Macro F1",
             ok=eval_summary.get("macro_f1", 0) >= eval_summary.get("target_macro_f1", 0.62))
        pill(c2, eval_summary.get("top3_accuracy",   0), "Top-3 Accuracy",
             ok=eval_summary.get("top3_accuracy", 0) >= eval_summary.get("target_top3_acc", 0.80))
        pill(c3, eval_summary.get("macro_precision",0), "Macro Precision")
        pill(c4, eval_summary.get("macro_recall",   0), "Macro Recall")

        st.caption(f"Eval method: `{eval_summary.get('eval_method', '—')}`  ·  "
                   f"{eval_summary.get('n_test_samples','—')} test samples  ·  "
                   f"{eval_summary.get('n_failing_groups','—')} groups with F1=0")

        note = eval_summary.get("note", "")
        if note:
            st.warning(f"⚠ {note}")
    else:
        st.info("Run notebook 03_evaluation.ipynb to generate eval_summary.json.")