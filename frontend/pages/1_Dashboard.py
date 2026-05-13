import streamlit as st
import os
import time
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go

from api_client import RAGAPIClient

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Live Agent Dashboard")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

if "api_client" not in st.session_state:
    st.session_state.api_client = RAGAPIClient(API_BASE_URL)

client: RAGAPIClient = st.session_state.api_client

# ── Refresh controls ───────────────────────────────────────────────────────────
_, c2, c3, c4 = st.columns([5, 1, 1, 1])
with c2:
    auto_refresh = st.toggle("Auto-refresh", value=True)
with c3:
    paused = st.toggle("Pause", value=False)
with c4:
    refresh_secs = st.selectbox("Every (s)", [3, 5, 10], index=1, label_visibility="collapsed")

# ── Fetch stats ────────────────────────────────────────────────────────────────
try:
    s = client.get_stats()
except Exception as e:
    st.error(f"Cannot reach backend: {e}")
    st.stop()

st.caption(
    f"Tracking since **{s.get('started_at', 'unknown')}**  ·  "
    f"Last updated: {time.strftime('%H:%M:%S')}"
)

# ── KPI row ────────────────────────────────────────────────────────────────────
st.subheader("Overview")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Queries",     s["total_queries"])
k2.metric("Retrieval Route",   f"{s['retrieval_route']}  ({s['retrieval_pct']}%)")
k3.metric("General Route",     f"{s['general_route']}  ({s['general_pct']}%)")
k4.metric("Success Rate",      f"{s.get('success_rate', 100.0)}%")
k5.metric("Avg Latency",       f"{s.get('avg_latency_s', 0)}s")
k6.metric("Errors",            s["errors"])

st.divider()

# ── Charts row 1: Route Pie + Query Volume ─────────────────────────────────────
queries = s.get("recent_queries", [])

col_pie, col_vol = st.columns(2)

with col_pie:
    st.subheader("Route Distribution")
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Retrieval", "General"],
        values=[max(s["retrieval_route"], 0), max(s["general_route"], 0)],
        hole=0.45,
        marker_colors=["#a5b4fc", "#34d399"],
        textinfo="label+percent",
    )])
    fig_pie.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_vol:
    st.subheader("Query Volume Over Time")
    if queries:
        timeline: dict = defaultdict(int)
        for q in queries:
            ts = q["ts"]
            minute = ts[11:16] if len(ts) >= 16 else ts[:5]
            timeline[minute] += 1
        df_tl = pd.DataFrame(sorted(timeline.items()), columns=["Time", "Queries"])
        fig_vol = go.Figure(data=[go.Bar(
            x=df_tl["Time"], y=df_tl["Queries"], marker_color="#a5b4fc",
        )])
        fig_vol.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e2130"),
            yaxis=dict(gridcolor="#1e2130"),
            margin=dict(t=10, b=30, l=30, r=10), height=260,
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("No query data yet.")

# ── Charts row 2: Model Usage + Latency Trend ─────────────────────────────────
col_model, col_lat = st.columns(2)

with col_model:
    st.subheader("Model Usage")
    if queries:
        model_counts: dict = defaultdict(int)
        for q in queries:
            model_counts[q.get("model", "unknown")] += 1
        df_m = pd.DataFrame(model_counts.items(), columns=["Model", "Queries"])
        fig_m = go.Figure(data=[go.Bar(
            x=df_m["Model"], y=df_m["Queries"], marker_color="#34d399",
        )])
        fig_m.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e2130"),
            yaxis=dict(gridcolor="#1e2130"),
            margin=dict(t=10, b=30, l=30, r=10), height=260,
        )
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("No query data yet.")

with col_lat:
    st.subheader("Latency Trend (last 30 queries)")
    if queries:
        recent_30 = list(reversed(queries[:30]))
        latencies = [round(q["latency_ms"] / 1000, 2) for q in recent_30]
        fig_lat = go.Figure(data=[go.Scatter(
            y=latencies, mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=5),
        )])
        fig_lat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e2130", title="Query #"),
            yaxis=dict(gridcolor="#1e2130", title="Seconds"),
            margin=dict(t=10, b=40, l=40, r=10), height=260,
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.info("No query data yet.")

st.divider()

# ── Recent Queries Table ───────────────────────────────────────────────────────
st.subheader("Recent Queries")
if queries:
    df_q = pd.DataFrame(queries[:20])
    df_q["ts"] = df_q["ts"].str[:19].str.replace("T", " ")
    df_q["latency_s"] = (df_q["latency_ms"] / 1000).round(1)
    df_q["success"] = df_q["success"].apply(lambda x: "✓" if x else "✗")
    df_q = df_q[["ts", "query", "route", "model", "retries", "latency_s", "success"]]
    df_q.columns = ["Time", "Query", "Route", "Model", "Retries", "Latency (s)", "OK"]
    st.dataframe(df_q, use_container_width=True, hide_index=True)
else:
    st.info("No queries yet. Send a message in the Chat page.")

st.divider()

# ── Agent Pipeline ─────────────────────────────────────────────────────────────
st.subheader("Multi-Agent Pipeline")
agents = [
    ("Router",               "Classifies query:\nretrieval vs general",  s["total_queries"]),
    ("Retrieval",            "Fetches top-4 chunks\nfrom FAISS",          s["retrieval_route"]),
    ("Relevance Grader",     "Filters irrelevant docs\n(retries once)",   s["retrieval_route"]),
    ("Generator",            "Produces answer\nfrom filtered context",    s["total_queries"]),
    ("Hallucination Grader", "Checks answer is\ngrounded in context",     s["retrieval_route"]),
]
cols = st.columns(len(agents))
for col, (name, role, calls) in zip(cols, agents):
    col.markdown(
        f"""<div style="background:#1e2130;border:1px solid #3b3f6b;border-radius:10px;
            padding:14px 10px;text-align:center;">
          <div style="color:#a5b4fc;font-weight:700;font-size:.9rem">{name}</div>
          <div style="color:#6b7280;font-size:.72rem;margin:6px 0;white-space:pre-line">{role}</div>
          <div style="color:#22c55e;font-size:.8rem;font-weight:600">{calls} calls</div>
        </div>""",
        unsafe_allow_html=True,
    )
st.caption(
    "Router → Retrieval → Grader ⟳(retry) → Generator → Hallucination Grader ⟳(regenerate) → END"
)

st.divider()

# ── Document Store + Log ───────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Document Store")
    st.metric("Chunks in FAISS",         s["document_count"])
    st.metric("PDFs uploaded (session)", s["docs_uploaded"])

    st.subheader("Retry Rates")
    total = max(s["total_queries"], 1)
    st.progress(
        s["grader_retries"] / total,
        text=f"Grader retry  {s['grader_retry_pct']}%",
    )
    st.progress(
        s["hallucination_retries"] / total,
        text=f"Halluc. retry  {s['halluc_retry_pct']}%",
    )

with right:
    st.subheader("Agent Execution Log")

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        log_search = st.text_input(
            "Search logs", placeholder="Filter by keyword...",
            label_visibility="collapsed",
        )
    with sc2:
        log_level = st.selectbox(
            "Level", ["All", "Error", "Retry", "Success", "Step"],
            label_visibility="collapsed",
        )

    logs = s.get("recent_logs", [])
    filtered = []
    for entry in reversed(logs):
        msg = entry["msg"]
        if log_level == "Error" and "error" not in msg.lower():
            continue
        if log_level == "Retry" and "retry" not in msg.lower() and "not_grounded" not in msg:
            continue
        if log_level == "Success" and not (
            "passed" in msg
            or ("grounded" in msg and "not_grounded" not in msg)
        ):
            continue
        if log_level == "Step" and not msg.startswith("["):
            continue
        if log_search and log_search.lower() not in msg.lower():
            continue
        filtered.append(entry)

    if not filtered:
        st.info("No log entries match the current filter." if (log_search or log_level != "All")
                else "No queries yet. Send a message in the Chat page to see live logs.")
    else:
        lines = []
        for entry in filtered:
            msg = entry["msg"]
            if "error" in msg.lower():
                colour = "#ef4444"
            elif "retry" in msg.lower() or "not_grounded" in msg:
                colour = "#f59e0b"
            elif "→" in msg or "passed" in msg or (
                "grounded" in msg and "not_grounded" not in msg
            ):
                colour = "#22c55e"
            elif msg.startswith("["):
                colour = "#60a5fa"
            else:
                colour = "#6b7280"
            lines.append(
                f'<span style="color:#4b5563">{entry["ts"]}</span> '
                f'<span style="color:{colour}">{msg}</span>'
            )
        st.markdown(
            f"""<div style="background:#0d0f1a;border-radius:8px;padding:14px 16px;
                font-family:monospace;font-size:.75rem;line-height:1.8;
                max-height:380px;overflow-y:auto;">
              {"<br>".join(lines)}
            </div>""",
            unsafe_allow_html=True,
        )

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh and not paused:
    time.sleep(refresh_secs)
    st.rerun()
