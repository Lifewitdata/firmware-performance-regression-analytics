"""Firmware v2.0 rollout dashboard.  Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_design import CALENDAR_START, INTERVAL_MIN, ROLLOUT_AT  # noqa: E402
from src.performance_metrics import METRICS, fleet_timeseries  # noqa: E402
from src.reliability_analysis import bearing_health, survival_by  # noqa: E402

P = ROOT / "data" / "processed"
FW = ROOT / "data" / "firmware_metadata"
ROLLOUT_TS = CALENDAR_START + pd.Timedelta(minutes=ROLLOUT_AT * INTERVAL_MIN)
COLORS = {"CONTINUE": "#1a7f37", "INVESTIGATE": "#bf8700", "ROLLBACK": "#cf222e"}
VERDICT_COLORS = {"improved": "#1a7f37", "regressed": "#cf222e", "no significant change": "#6e7781"}

st.set_page_config(page_title="Firmware Regression Analytics", layout="wide")


REQUIRED_OUTPUTS = ("summary.json", "telemetry.parquet", "device_period_kpis.parquet", "ims_features.parquet",
                    "comparison_overall.csv", "comparison_by_segment.csv", "device_regression_scores.csv",
                    "event_rate_tests.csv", "naive_comparison.csv", "mtbf.csv", "time_to_first_reboot.csv",
                    "nab_detector_evaluation.csv", "bearing_degradation.csv")


@st.cache_data
def load():
    missing = [f for f in REQUIRED_OUTPUTS if not (P / f).exists()]
    missing += [f"firmware_metadata/{f}" for f in ("firmware_effects_ground_truth.json", "devices.csv",
                                                   "firmware_releases.csv") if not (FW / f).exists()]
    if missing:
        return {"missing": missing}
    return {
        "summary": json.loads((P / "summary.json").read_text()),
        "tel": pd.read_parquet(P / "telemetry.parquet"),
        "kpis": pd.read_parquet(P / "device_period_kpis.parquet"),
        "overall": pd.read_csv(P / "comparison_overall.csv"),
        "seg": pd.read_csv(P / "comparison_by_segment.csv"),
        "scores": pd.read_csv(P / "device_regression_scores.csv"),
        "rates": pd.read_csv(P / "event_rate_tests.csv"),
        "naive": pd.read_csv(P / "naive_comparison.csv"),
        "mtbf": pd.read_csv(P / "mtbf.csv"),
        "ttf": pd.read_csv(P / "time_to_first_reboot.csv"),
        "nab": pd.read_csv(P / "nab_detector_evaluation.csv"),
        "degr": pd.read_csv(P / "bearing_degradation.csv"),
        "features": pd.read_parquet(P / "ims_features.parquet"),
        "truth": json.loads((FW / "firmware_effects_ground_truth.json").read_text()),
        "devices": pd.read_csv(FW / "devices.csv"),
    }


D = load()
if "missing" in D:
    st.error("Missing analysis outputs: " + ", ".join(D["missing"]))
    st.caption(f"Looked in `{P.resolve()}`.")
    st.markdown("These files are committed in the repo, so an outdated checkout is the usual cause:")
    st.code("git pull origin main\n\n# or regenerate them from the raw inputs:\npython -m src.pipeline", language="bash")
    st.stop()

st.title("Firmware v2.0 Performance & Hardware Regression Analytics")
st.caption("Real hardware signals: NASA IMS bearing vibration + NAB request latency. "
           "Firmware versions, fleet assignment and firmware effects are an **experimental, clearly labelled layer** "
           "(see *Data provenance*).")

tabs = st.tabs(["Decision", "KPI comparison", "Trends", "Device regressions", "Reliability",
                "Real sensor data", "Data provenance"])

# ------------------------------------------------------------------ Decision
with tabs[0]:
    s = D["summary"]
    st.subheader("Recommendation")
    st.info(s["headline"])
    cols = st.columns(len(s["decisions"]))
    for col, (seg, d) in zip(cols, s["decisions"].items()):
        with col:
            st.markdown(f"<div style='border-left:6px solid {COLORS[d['decision']]};padding:8px 12px'>"
                        f"<div style='font-size:0.85rem;color:#57606a'>{seg}</div>"
                        f"<div style='font-size:1.6rem;font-weight:700;color:{COLORS[d['decision']]}'>{d['decision']}</div>"
                        "</div>", unsafe_allow_html=True)
            for r in d["reasons"]:
                st.markdown(f"- {r}")

    st.subheader("Answers to the engineering questions")
    seg = D["seg"].set_index(["segment", "metric"])
    ov = D["overall"].set_index("metric")
    rows = []
    for m in METRICS:
        r = ov.loc[m.name]
        rows.append({"Question": m.question, "Metric": m.label, "Fleet": r.verdict,
                     "Rev A": seg.loc[("hw_revision=A", m.name), "verdict"],
                     "Rev B": seg.loc[("hw_revision=B", m.name), "verdict"],
                     "Fleet DiD %": round(r.relative_change_pct, 1)})
    q = pd.DataFrame(rows)
    st.dataframe(q.style.map(lambda v: f"color:{VERDICT_COLORS.get(v, 'inherit')}", subset=["Fleet", "Rev A", "Rev B"]),
                 hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Updated devices", s["treated_devices"])
    c2.metric("Devices individually flagged", s["flagged_devices"])
    c3.metric("Control devices (stayed on v1.0)", int((D["devices"].group == "control").sum()))

# ------------------------------------------------------------------ KPI comparison
with tabs[1]:
    st.subheader("Difference-in-differences: v2.0 effect relative to the v1.0 control arm")
    seg_opt = ["fleet"] + sorted(D["seg"].segment.unique())
    pick = st.selectbox("Segment", seg_opt)
    res = D["overall"].assign(segment="fleet") if pick == "fleet" else D["seg"][D["seg"].segment == pick]
    fig = go.Figure()
    for _, r in res.iterrows():
        rel = lambda v: 100 * v / r.treatment_pre
        fig.add_trace(go.Scatter(x=[rel(r.did)], y=[r.label], mode="markers",
                                 marker=dict(size=12, color=VERDICT_COLORS[r.verdict]),
                                 error_x=dict(type="data", symmetric=False, array=[rel(r.ci_high) - rel(r.did)],
                                              arrayminus=[rel(r.did) - rel(r.ci_low)]),
                                 name=r.verdict, showlegend=False,
                                 hovertemplate=f"{r.label}<br>DiD {r.relative_change_pct:+.1f}%<br>adj. p={r.p_value_adj:.3g}"))
    fig.add_vline(x=0, line_dash="dash", line_color="#6e7781")
    fig.update_layout(xaxis_title="Change vs control, % of pre-rollout level (95% bootstrap CI)", height=420,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    show = res[["label", "treatment_pre", "treatment_post", "control_pre", "control_post", "did", "ci_low", "ci_high",
                "relative_change_pct", "p_value_adj", "p_value_ols_cluster", "cohens_d", "verdict"]]
    st.dataframe(show.round(4), hide_index=True, use_container_width=True)

    st.subheader("Why a control arm matters")
    st.caption("Naive before/after on updated devices only. Bearings wear during the observation window, so "
               "hardware-driven metrics drift even without a firmware change.")
    nv = D["naive"].merge(D["overall"][["metric", "did", "label"]], on="metric")
    st.dataframe(nv[["label", "pre", "post", "naive_diff", "p_value", "did"]].rename(
        columns={"naive_diff": "naive before/after diff", "did": "DiD (controls for drift)"}).round(4),
        hide_index=True, use_container_width=True)

    st.subheader("Exact event-rate tests (post-rollout, v2.0 vs v1.0)")
    st.dataframe(D["rates"].round(4), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ Trends
with tabs[2]:
    c1, c2, c3 = st.columns(3)
    metric = c1.selectbox("Metric", ["fw_errors_per_100", "p50_response_ms", "uptime_pct", "mean_rms",
                                     "alarm_rate_pct", "reboots"])
    rev = c2.radio("Hardware revision", ["A", "B", "both"], horizontal=True)
    bucket = c3.select_slider("Bucket (hours)", [1, 2, 4, 8], value=2)
    ts = fleet_timeseries(D["tel"] if rev == "both" else D["tel"][D["tel"].hw_revision == rev], bucket * 6)
    ts = ts.groupby(["bucket", "group"]).agg(timestamp=("timestamp", "min"), value=(metric, "mean")).reset_index()
    fig = px.line(ts, x="timestamp", y="value", color="group", markers=True,
                  color_discrete_map={"treatment": "#0969da", "control": "#6e7781"}, labels={"value": metric})
    fig.add_vline(x=ROLLOUT_TS, line_dash="dash", line_color="#cf222e")
    fig.add_annotation(x=ROLLOUT_TS, y=1, yref="paper", text="v2.0 rollout", showarrow=False, xanchor="left")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ Device regressions
with tabs[3]:
    sc = D["scores"]
    st.subheader("Updated devices ranked by how unusual their change is vs control devices")
    st.caption("Robust z-score of each device's (post - pre) change against the control distribution. Positive = worse. "
               "Flag = any guardrail metric with z > 3.")
    only = st.checkbox("Show flagged devices only", value=True)
    view = sc[sc.regression_flag] if only else sc
    st.dataframe(view.round(2), hide_index=True, use_container_width=True, height=320)
    fig = px.scatter(sc, x="z_p95_response_ms", y="z_fw_errors_per_100", color="hw_revision",
                     symbol="regression_flag", hover_data=["device_id", "region", "worst_metric"],
                     labels={"z_p95_response_ms": "P95 response z (worse ->)", "z_fw_errors_per_100": "FW error z (worse ->)"})
    fig.add_hline(y=3, line_dash="dot")
    fig.add_vline(x=3, line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)

    dev = st.selectbox("Drill into device", sc.device_id)
    t = D["tel"][D["tel"].device_id == dev]
    c1, c2 = st.columns(2)
    f1 = px.line(t, x="timestamp", y=["real_rms", "vib_rms"], title="Vibration RMS: real vs reported by firmware")
    f1.add_vline(x=ROLLOUT_TS, line_dash="dash", line_color="#cf222e")
    c1.plotly_chart(f1, use_container_width=True)
    f2 = px.line(t, x="timestamp", y="response_time_ms", title="Response time (ms)")
    f2.add_vline(x=ROLLOUT_TS, line_dash="dash", line_color="#cf222e")
    c2.plotly_chart(f2, use_container_width=True)
    st.json(D["devices"].set_index("device_id").loc[dev].astype(str).to_dict())

# ------------------------------------------------------------------ Reliability
with tabs[4]:
    st.subheader("MTBF (firmware reboots) and availability")
    st.dataframe(D["mtbf"].round(2), hide_index=True, use_container_width=True)
    st.subheader("Time to first reboot after rollout (Kaplan-Meier)")
    ttf = D["ttf"].assign(cohort=lambda d: d.firmware_version + " / rev " + d.hw_revision)
    fig = go.Figure()
    for name, km in survival_by(ttf, "cohort").items():
        fig.add_trace(go.Scatter(x=km.time, y=km.survival, mode="lines", line_shape="hv", name=name))
    fig.update_layout(xaxis_title="Hours since rollout", yaxis_title="Share of devices without a reboot",
                      height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.write({f"rev {k}": f"log-rank p = {v['p_value']:.3g}" for k, v in D["summary"]["logrank_time_to_first_reboot"].items()})

    st.subheader("Real hardware degradation (NASA IMS run-to-failure)")
    st.caption("Hardware faults come from the real bearing signal and are independent of firmware; "
               "they are reported separately so bearing wear is not blamed on the update.")
    st.dataframe(D["degr"].round(2), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ Real sensor data
with tabs[5]:
    f = D["features"]
    c1, c2 = st.columns([1, 3])
    test = c1.selectbox("IMS test", sorted(f.test_id.unique()))
    feat = c1.selectbox("Feature", ["rms", "kurtosis", "peak", "crest_factor", "hf_energy_ratio"])
    h = bearing_health(f[f.test_id == test])
    fig = px.line(h, x="timestamp", y=feat, color=h.bearing.astype(str) + " (" + h.failure_mode + ")",
                  labels={"color": "bearing"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    c2.plotly_chart(fig, use_container_width=True)
    st.subheader("Anomaly detector validated on NAB labelled anomaly windows")
    st.dataframe(D["nab"].round(3), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ Provenance
with tabs[6]:
    truth = D["truth"]
    st.warning(truth["WARNING"])
    st.subheader("Column origin")
    st.json(truth["column_origin"])
    st.subheader("Validation: injected effect vs estimated effect")
    eff = truth["effects_v2_vs_v1"]
    seg = D["seg"].set_index(["segment", "metric"])
    val = []
    for rev in ("A", "B"):
        r = seg.loc[(f"hw_revision={rev}", "fw_errors_per_100")]
        val.append({"hw_revision": rev, "quantity": "FW error rate multiplier",
                    "injected": eff["fw_error_rate"][rev], "estimated": round(1 + r.did / r.treatment_pre, 3)})
        r = seg.loc[(f"hw_revision={rev}", "p50_response_ms")]
        val.append({"hw_revision": rev, "quantity": "Response time multiplier",
                    "injected": eff["response_time"][rev], "estimated": round(1 + r.did / r.treatment_pre, 3)})
        r = seg.loc[(f"hw_revision={rev}", "mean_rms")]
        val.append({"hw_revision": rev, "quantity": "RMS calibration gain",
                    "injected": eff["rms_gain"][rev], "estimated": round(1 + r.did / r.treatment_pre, 3)})
    st.dataframe(pd.DataFrame(val), hide_index=True, use_container_width=True)
    st.subheader("Firmware releases")
    st.dataframe(pd.read_csv(FW / "firmware_releases.csv"), hide_index=True, use_container_width=True)
