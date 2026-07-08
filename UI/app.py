"""
UI/app.py — Streamlit front-end for the Incoming Request Processing Workflow POC

Run from project root with:
    streamlit run UI/app.py
"""

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Path setup so imports from parent package work ──────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from graph.graph import app, examples       # compiled LangGraph + test scenarios
from graph.nodes import apply_override      # direct-branch override (no full graph re-run)
from graph.state import TicketState

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Request Processing Workflow",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design constants ──────────────────────────────────────────────────────────
DB_PATH = os.path.join(ROOT, "cases.db")

TYPE_CONFIG = {
    "Complaint":       {"color": "#EA580C", "bg": "#FFF7ED", "border": "#FED7AA", "icon": "🟠"},
    "Enquiry":         {"color": "#2563EB", "bg": "#EFF6FF", "border": "#BFDBFE", "icon": "🔵"},
    "Service Request": {"color": "#059669", "bg": "#ECFDF5", "border": "#A7F3D0", "icon": "🟢"},
    "Escalation":      {"color": "#DC2626", "bg": "#FEF2F2", "border": "#FECACA", "icon": "🔴"},
}

URGENCY_CONFIG = {
    "Low":      {"color": "#059669", "bg": "#ECFDF5", "border": "#A7F3D0"},
    "Medium":   {"color": "#D97706", "bg": "#FFFBEB", "border": "#FDE68A"},
    "High":     {"color": "#EA580C", "bg": "#FFF7ED", "border": "#FED7AA"},
    "Critical": {"color": "#DC2626", "bg": "#FEF2F2", "border": "#FECACA"},
}

STATUS_CONFIG = {
    "resolved":      {"color": "#059669", "label": "✓ Resolved"},
    "open":          {"color": "#2563EB", "label": "↺ Open"},
    "escalated":     {"color": "#EA580C", "label": "↑ Escalated"},
    "pending_human": {"color": "#DC2626", "label": "⚠ Pending Human"},
}

# Display labels for the simulated inbox dropdown (matches examples[] order)
INBOX_LABELS = [
    "🟠  Portal outage complaint — user1@company.com",
    "🔵  CI/CD documentation enquiry — dev@company.com",
    "🟢  Jira access service request — newhire@company.com",
    "🔴  Repeated outage + legal threat (escalation) — angry_client@enterprise.com",
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    color: white;
}
.main-header h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }
.main-header p  { margin: 6px 0 0 0; font-size: 14px; opacity: 0.85; }

.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid;
    margin-right: 6px;
}

.result-header {
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
    border-left: 4px solid;
}

.case-id {
    font-family: 'Courier New', monospace;
    font-weight: 700;
    font-size: 15px;
    color: #374151;
    background: #F3F4F6;
    padding: 4px 10px;
    border-radius: 6px;
}

.info-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F1F5F9;
    color: #475569;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    margin-right: 6px;
    margin-top: 4px;
}

.action-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid #F3F4F6;
    font-size: 14px;
    color: #374151;
}
.action-step:last-child { border-bottom: none; }
.step-check { color: #059669; font-weight: 700; font-size: 15px; flex-shrink: 0; }

.reasoning-box {
    background: #F8FAFC;
    border-left: 3px solid #94A3B8;
    padding: 14px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #374151;
    font-style: italic;
    margin-top: 8px;
    min-height: 80px;
}

.response-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 20px;
    font-size: 14px;
    line-height: 1.8;
    color: #1E293B;
    white-space: pre-wrap;
    margin-top: 8px;
}

.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Helper: render HTML badges ────────────────────────────────────────────────

def type_badge(cls: str) -> str:
    cfg = TYPE_CONFIG.get(cls, {"color": "#6B7280", "bg": "#F3F4F6", "border": "#D1D5DB", "icon": "⚪"})
    return (f'<span class="badge" style="color:{cfg["color"]};background:{cfg["bg"]};'
            f'border-color:{cfg["border"]}">{cfg["icon"]} {cls}</span>')

def urgency_badge(urgency: str) -> str:
    cfg = URGENCY_CONFIG.get(urgency, {"color": "#6B7280", "bg": "#F3F4F6", "border": "#D1D5DB"})
    return (f'<span class="badge" style="color:{cfg["color"]};background:{cfg["bg"]};'
            f'border-color:{cfg["border"]}">⚡ {urgency} Urgency</span>')

def status_badge(status: str) -> str:
    cfg = STATUS_CONFIG.get(status, {"color": "#6B7280", "label": status or "—"})
    return (f'<span class="badge" style="color:{cfg["color"]};background:white;'
            f'border-color:{cfg["color"]}">{cfg["label"]}</span>')


# ── Helper: render the full result card ───────────────────────────────────────

def render_result(result: dict):
    cls     = result.get("classification", "Unknown")
    urgency = result.get("urgency", "")
    status  = result.get("status", "")
    cfg     = TYPE_CONFIG.get(cls, {"color": "#6B7280", "bg": "#F9FAFB", "border": "#D1D5DB", "icon": "⚪"})

    # --- Coloured header strip ---
    extra_tags = ""
    if result.get("sla_timer_hours"):
        extra_tags += f'<span class="info-tag">⏱ SLA: {result["sla_timer_hours"]}h</span>'
    if result.get("followup_hours"):
        extra_tags += f'<span class="info-tag">🔔 Follow-up: {result["followup_hours"]}h</span>'
    if result.get("human_review_flag"):
        extra_tags += '<span class="info-tag" style="color:#DC2626;background:#FEF2F2">🚨 Human Review Required</span>'
    if result.get("auto_resolution_paused"):
        extra_tags += '<span class="info-tag" style="color:#DC2626;background:#FEF2F2">⏸ Auto-Resolution Paused</span>'

    st.markdown(
        f'<div class="result-header" style="background:{cfg["bg"]};border-left-color:{cfg["color"]}">'
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:10px">'
        f'{type_badge(cls)}{urgency_badge(urgency)}{status_badge(status)}'
        f'</div>'
        f'<div>'
        f'<span class="info-tag">📁 {result.get("case_id","—")}</span>'
        f'<span class="info-tag">👤 Routed → {result.get("route_to","—")}</span>'
        f'{extra_tags}'
        f'</div></div>',
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**🔍 AI Reasoning**")
        st.markdown(
            f'<div class="reasoning-box">{result.get("reasoning", "No reasoning provided.")}</div>',
            unsafe_allow_html=True
        )

    with col_right:
        st.markdown("**✅ Remediation Steps Taken**")
        actions = result.get("actions_taken") or []
        rows = "".join(
            f'<div class="action-step"><span class="step-check">✓</span><span>{a}</span></div>'
            for a in actions
        )
        st.markdown(f'<div style="margin-top:8px">{rows}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📝 Generated Response / Draft**")
    response = result.get("response_draft") or "No response generated."
    st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
    st.download_button(
        label="📋 Download Response as .txt",
        data=response,
        file_name=f"{result.get('case_id', 'response')}.txt",
        mime="text/plain",
        key=f"dl_{result.get('case_id', 'none')}_{uuid.uuid4().hex[:8]}"
    )


# ── Helpers: load cases from SQLite ──────────────────────────────────────────

def load_cases() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM cases ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_case_detail(case_id: str) -> dict | None:
    """Load one case by ID and reconstruct the result dict used by render_result()."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "case_id":                row["case_id"],
            "classification":         row["classification"],
            "urgency":                row["urgency"],
            "reasoning":              row["reasoning"],
            "status":                 row["status"],
            "route_to":               row["route_to"],
            "response_draft":         row["response_draft"],
            "sla_timer_hours":        row["sla_timer_hours"],
            "followup_hours":         row["followup_hours"],
            "human_review_flag":      bool(row["human_review_flag"]),
            "auto_resolution_paused": bool(row["auto_resolution_paused"]),
            "is_human_override":      bool(row.get("is_human_override", 0)),
            # actions_taken was stored as a JSON string — parse it back to a list
            "actions_taken":          json.loads(row["actions_taken"] or "[]"),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🔄 Incoming Request Processing Workflow</h1>
    <p>AI-powered classification and automated remediation for operations teams &nbsp;·&nbsp;
    Powered by LangGraph + Qwen3-235B via Nebius AI</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["📥  Process Request", "📋  Case Dashboard"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Process Request
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    input_method = st.radio(
        "**Select Input Method**",
        options=["📝 Form Input", "📂 File Upload", "📬 Simulated Inbox"],
        horizontal=True,
    )
    st.divider()

    sender   = ""
    subject  = ""
    raw_text = ""

    # ── Form Input ──────────────────────────────────────────────────────────
    if input_method == "📝 Form Input":
        c1, c2 = st.columns(2)
        with c1:
            sender  = st.text_input("Sender Email", placeholder="customer@example.com")
        with c2:
            subject = st.text_input("Subject", placeholder="Brief description of issue")
        raw_text = st.text_area(
            "Request / Message Body",
            height=220,
            placeholder="Paste or type the customer's message here…",
        )

    # ── File Upload ─────────────────────────────────────────────────────────
    elif input_method == "📂 File Upload":
        c1, c2 = st.columns(2)
        with c1:
            sender  = st.text_input("Sender Email", placeholder="customer@example.com")
        with c2:
            subject = st.text_input("Subject", placeholder="Brief description of issue")

        uploaded = st.file_uploader(
            "Upload request file (.txt or .eml)",
            type=["txt", "eml"],
            help="File contents will populate the request body below for review."
        )
        if uploaded:
            file_content = uploaded.read().decode("utf-8", errors="ignore")
            raw_text = st.text_area(
                "File Preview (editable before processing)",
                value=file_content,
                height=200,
                key="file_preview"
            )

    # ── Simulated Inbox ─────────────────────────────────────────────────────
    elif input_method == "📬 Simulated Inbox":
        st.info(
            "Select one of the four pre-loaded test scenarios. "
            "This demonstrates all four classification branches in the workflow.",
            icon="💡"
        )
        selected = st.selectbox("Choose a scenario", options=INBOX_LABELS)
        idx      = INBOX_LABELS.index(selected)
        ex       = examples[idx]
        raw_text = ex["raw_input"]
        sender   = ex["input_metadata"].get("sender", "")
        subject  = ex["input_metadata"].get("subject", "")
        with st.expander("📄 Preview selected message"):
            st.text(raw_text)

    st.divider()

    process_btn = st.button("⚡ Process Request", type="primary", use_container_width=False)

    if process_btn:
        if not raw_text.strip():
            st.error("Please provide a request message before processing.", icon="⚠️")
        else:
            source_map = {
                "📝 Form Input":      "form",
                "📂 File Upload":     "file_upload",
                "📬 Simulated Inbox": "simulated_inbox",
            }
            input_source = source_map[input_method]
            task = TicketState(
                input_source=input_source,
                raw_input=raw_text.strip(),
                input_metadata={
                    "sender":    sender,
                    "subject":   subject,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )

            with st.status("⚙️ Processing request…", expanded=True) as status:
                st.write("📨 Request received and validated")
                st.write("🤖 Classifying request type and urgency via LLM…")

                result = app.invoke(task)

                cls     = result.get("classification", "Unknown")
                urgency = result.get("urgency", "")
                case_id = result.get("case_id", "—")

                st.write(f"🔍 Classified as **{cls}** · **{urgency}** urgency")
                st.write(f"🔀 Executing **{cls}** branch remediation steps…")
                st.write(f"💾 Case logged to audit trail → **{case_id}**")
                status.update(label=f"✅ Complete — {case_id}", state="complete")

            # Persist result + inputs so the override panel can reuse them
            st.session_state["current_result"]    = result
            st.session_state["last_raw_input"]    = raw_text.strip()
            st.session_state["last_sender"]       = sender
            st.session_state["last_subject"]      = subject
            st.session_state["last_input_source"] = input_source

    # ── Result card (persists across reruns via session_state) ────────────────
    if st.session_state.get("current_result"):
        result_to_show = st.session_state["current_result"]
        is_override    = result_to_show.get("_is_override", False)

        st.markdown("### Result")
        if is_override:
            st.info(
                "✏️ Showing **overridden** result — the original AI classification "
                "was replaced by the operator.",
                icon="👤"
            )
        render_result(result_to_show)

        # ── Human-in-the-loop override panel ─────────────────────────────────
        st.markdown("")
        with st.expander("⚙️ Override Classification  *(Human-in-the-Loop)*", expanded=False):
            st.warning(
                "Use this only if the AI classification above appears incorrect. "
                "The system will **skip the LLM** and re-run the workflow through "
                "the branch you select.",
                icon="⚠️"
            )
            ov_c1, ov_c2 = st.columns(2)
            
            type_options = ["Complaint", "Enquiry", "Service Request", "Escalation"]
            current_type = result_to_show.get("classification")
            type_idx = type_options.index(current_type) if current_type in type_options else 0
            
            urgency_options = ["Low", "Medium", "High", "Critical"]
            current_urgency = result_to_show.get("urgency")
            urgency_idx = urgency_options.index(current_urgency) if current_urgency in urgency_options else 0

            with ov_c1:
                new_type = st.selectbox(
                    "Reclassify as",
                    type_options,
                    index=type_idx,
                    key="override_type_select",
                )
            with ov_c2:
                new_urgency = st.selectbox(
                    "Urgency",
                    urgency_options,
                    index=urgency_idx,
                    key="override_urgency_select",
                )

            if st.button("🔄 Re-process with Override", type="secondary"):
                original_case_id = st.session_state["current_result"].get("case_id")
                with st.status("⚙️ Re-processing with human override…", expanded=True) as ov_status:
                    st.write(f"👤 Override applied: **{new_type}** / **{new_urgency}**")
                    st.write("⏭️ Skipping LLM classification → running corrected branch only…")
                    override_result = apply_override(
                        case_id=original_case_id,
                        raw_input=st.session_state["last_raw_input"],
                        input_metadata={
                            "sender":  st.session_state.get("last_sender", ""),
                            "subject": st.session_state.get("last_subject", ""),
                        },
                        new_classification=new_type,
                        new_urgency=new_urgency,
                    )
                    st.write(f"💾 Case **{original_case_id}** updated in audit trail (no new row)")
                    ov_status.update(label=f"✅ Override complete — {original_case_id}", state="complete")

                override_result["_is_override"] = True
                st.session_state["current_result"] = override_result
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Case Dashboard
# ──────────────────────────────────────────────────────────────────────────────

with tab2:
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

    df = load_cases()

    if df.empty:
        st.info(
            "No cases logged yet. Process a request in the **Process Request** tab first.",
            icon="📭"
        )
    else:
        # ── Metric cards ────────────────────────────────────────────────────
        total    = len(df)
        resolved = len(df[df["status"] == "resolved"])
        open_c   = len(df[df["status"] == "open"])
        escalated = len(df[df["status"] == "escalated"])
        pending  = len(df[df["status"] == "pending_human"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📁 Total Cases",     total)
        m2.metric("✓ Resolved",         resolved)
        m3.metric("↺ Open",             open_c)
        m4.metric("↑ Escalated",        escalated)
        m5.metric("⚠ Pending Human",    pending)

        st.divider()

        # ── Filters ─────────────────────────────────────────────────────────
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            filter_type = st.selectbox(
                "Filter by Type",
                ["All"] + sorted(df["classification"].dropna().unique().tolist())
            )
        with c2:
            filter_status = st.selectbox(
                "Filter by Status",
                ["All"] + sorted(df["status"].dropna().unique().tolist())
            )

        filtered = df.copy()
        if filter_type   != "All": filtered = filtered[filtered["classification"] == filter_type]
        if filter_status != "All": filtered = filtered[filtered["status"]         == filter_status]

        # ── Case log table (click a row to view full case detail) ────────────
        st.markdown("### Case Log")
        st.caption("💡 Click any row to expand the full case detail below the table.")
        display_cols = ["case_id", "timestamp", "classification", "urgency",
                        "status", "route_to", "sla_timer_hours", "followup_hours",
                        "is_human_override"]
        available = [c for c in display_cols if c in filtered.columns]

        # Build display dataframe with a human-readable Decision column
        display_df = filtered[available].copy()
        if "is_human_override" in display_df.columns:
            display_df["decision"] = display_df["is_human_override"].apply(
                lambda v: "👤 Human Override" if int(v or 0) == 1 else "🤖 AI"
            )
            display_df = display_df.drop(columns=["is_human_override"])
        else:
            display_df["decision"] = "🤖 AI"

        event = st.dataframe(
            display_df.rename(columns={
                "case_id":         "Case ID",
                "timestamp":       "Timestamp",
                "classification":  "Type",
                "urgency":         "Urgency",
                "status":          "Status",
                "route_to":        "Routed To",
                "sla_timer_hours": "SLA (hrs)",
                "followup_hours":  "Follow-up (hrs)",
                "decision":        "Decision",
            }),
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True,
            hide_index=True,
        )

        # ── Case detail panel ────────────────────────────────────────────────
        selected_rows = event.selection.rows
        if selected_rows:
            # Map the selected display-row index back to the filtered df
            selected_case_id = filtered.iloc[selected_rows[0]]["case_id"]
            case_detail = load_case_detail(selected_case_id)
            if case_detail:
                st.markdown("---")
                is_overridden = bool(case_detail.get("is_human_override", 0))
                detail_title = (
                    f"👤 Human-Overridden Case — `{selected_case_id}`"
                    if is_overridden else
                    f"🔍 Case Detail — `{selected_case_id}`"
                )
                st.markdown(f"### {detail_title}")
                if is_overridden:
                    st.info(
                        "👤 This case was **manually corrected by a human operator**. "
                        "The AI's original classification was overridden.",
                        icon="✏️"
                    )
                render_result(case_detail)
            else:
                st.warning("Could not load case details. Try refreshing.", icon="⚠️")

        st.divider()

        # ── Bar chart ────────────────────────────────────────────────────────
        if "classification" in df.columns and not df["classification"].dropna().empty:
            st.markdown("### Requests by Classification Type")
            chart_data = (
                df["classification"]
                .value_counts()
                .reset_index()
                .rename(columns={"classification": "Type", "count": "Count"})
            )
            st.bar_chart(chart_data.set_index("Type"))
