import streamlit as st
import os
import time
from api_client import RAGAPIClient

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Live Agent Dashboard")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

if "api_client" not in st.session_state:
    st.session_state.api_client = RAGAPIClient(API_BASE_URL)

client: RAGAPIClient = st.session_state.api_client

# ── Refresh control ────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([6, 1])
with col_refresh:
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_secs = st.selectbox("Interval", [3, 5, 10], index=1, label_visibility="collapsed")

# ── Fetch stats ────────────────────────────────────────────────────────────────
try:
    s = client.get_stats()
    api_ok = True
except Exception as e:
    st.error(f"Cannot reach backend: {e}")
    st.stop()

# ── KPI row ────────────────────────────────────────────────────────────────────
st.subheader("Overview")
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Queries",      s["total_queries"])
k2.metric("Retrieval Route",    f"{s['retrieval_route']}  ({s['retrieval_pct']}%)")
k3.metric("General Route",      f"{s['general_route']}  ({s['general_pct']}%)")
k4.metric("Grader Retries",     f"{s['grader_retries']}  ({s['grader_retry_pct']}%)")
k5.metric("Halluc. Retries",    f"{s['hallucination_retries']}  ({s['halluc_retry_pct']}%)")

st.divider()

# ── Agent pipeline ─────────────────────────────────────────────────────────────
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
        f"""
        <div style="
            background:#1e2130; border:1px solid #3b3f6b; border-radius:10px;
            padding:14px 10px; text-align:center;
        ">
          <div style="color:#a5b4fc;font-weight:700;font-size:.9rem">{name}</div>
          <div style="color:#6b7280;font-size:.72rem;margin:6px 0;white-space:pre-line">{role}</div>
          <div style="color:#22c55e;font-size:.8rem;font-weight:600">{calls} calls</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Router → Retrieval → Grader ⟳(retry) → Generator → Hallucination Grader ⟳(regenerate) → END"
)

st.divider()

# ── Stats + logs side by side ──────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Document Store")
    st.metric("Chunks in FAISS", s["document_count"])
    st.metric("PDFs uploaded (session)", s["docs_uploaded"])
    st.metric("Errors", s["errors"])

    st.subheader("Retry Rates")
    total = max(s["total_queries"], 1)
    st.progress(s["grader_retries"] / total,
                text=f"Grader retry  {s['grader_retry_pct']}%")
    st.progress(s["hallucination_retries"] / total,
                text=f"Halluc. retry  {s['halluc_retry_pct']}%")

with right:
    st.subheader("Agent Execution Log")
    logs = s.get("recent_logs", [])
    if not logs:
        st.info("No queries yet. Send a message in the Chat page to see live logs.")
    else:
        log_lines = []
        for entry in reversed(logs):
            msg = entry["msg"]
            if "Error" in msg or "error" in msg:
                colour = "#ef4444"
            elif "retry" in msg.lower() or "not_grounded" in msg:
                colour = "#f59e0b"
            elif "→" in msg or "passed" in msg or "grounded" in msg:
                colour = "#22c55e"
            elif msg.startswith("["):
                colour = "#60a5fa"
            else:
                colour = "#6b7280"

            log_lines.append(
                f'<span style="color:#4b5563">{entry["ts"]}</span> '
                f'<span style="color:{colour}">{msg}</span>'
            )

        st.markdown(
            f"""
            <div style="
                background:#0d0f1a; border-radius:8px; padding:14px 16px;
                font-family:monospace; font-size:.75rem; line-height:1.8;
                max-height:380px; overflow-y:auto;
            ">
              {"<br>".join(log_lines)}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Auto-refresh ───────────────────────────────────────────────────────────────
st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()
