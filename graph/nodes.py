"""
nodes.py — LangGraph node functions for the Incoming Request Processing Workflow

Each node receives the full TicketState and returns a dict of ONLY the fields it updates.
LangGraph merges the returned dict back into the state automatically.

Node map:
  classify_node          → 1 LLM call  → classification, urgency, reasoning, timestamp
  complaint_node         → 1 LLM call  → response_draft, escalated_to, followup_hours, route_to, status, actions_taken
  enquiry_node           → 1 LLM call  → sub_topic, response_draft, status, actions_taken
  service_request_node   → 1 LLM call  → extracted_details, route_to, response_draft, sla_timer_hours, status, actions_taken
  escalation_node        → 1 LLM call  → response_draft, human_review_flag, supervisor_notified, auto_resolution_paused, route_to, status, actions_taken
  log_node               → 0 LLM calls → case_id  (writes to SQLite)
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import TicketState

load_dotenv()

# ── LangChain ChatOpenAI pointed at Nebius ───────────────────────────────────
# Nebius is OpenAI-compatible — only base_url and api_key need to change.
# Temperature is NOT set here; it is passed per-call via llm.bind() so each
# node can tune it independently (0.1 for classify, 0.3-0.4 for generation).
llm = ChatOpenAI(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cases.db")


# ── SQLite setup (called once at import time) ──────────────────────────────────
def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id                TEXT PRIMARY KEY,
            timestamp              TEXT,
            input_source           TEXT,
            classification         TEXT,
            urgency                TEXT,
            reasoning              TEXT,
            status                 TEXT,
            route_to               TEXT,
            response_draft         TEXT,
            sla_timer_hours        INTEGER,
            followup_hours         INTEGER,
            human_review_flag      INTEGER,
            auto_resolution_paused INTEGER,
            actions_taken          TEXT,
            raw_input              TEXT,
            is_human_override      INTEGER DEFAULT 0
        )
    """)
    # Safe migration — adds the column on existing databases that pre-date this field
    try:
        conn.execute("ALTER TABLE cases ADD COLUMN is_human_override INTEGER DEFAULT 0")
    except Exception:
        pass  # column already exists — no action needed
    conn.commit()
    conn.close()


_init_db()


# ── LLM helper functions ──────────────────────────────────────────────────────
def _call_llm_json(system_prompt: str, user_content: str, temperature: float = 0.2) -> dict:
    """
    LLM call expecting structured JSON back.
    Uses llm.bind() to pass response_format + temperature without creating
    a new ChatOpenAI instance each time.
    On any failure → returns a fallback dict that routes to Escalation.
    """
    try:
        chain = llm.bind(
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        response = chain.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return json.loads(response.content)   # AIMessage.content is a plain string
    except Exception as e:
        print(f"[LLM ERROR] {e} — routing to Escalation as fallback")
        return {"__fallback__": True, "error": str(e)}


def _call_llm_text(system_prompt: str, user_content: str, temperature: float = 0.3) -> str:
    """
    LLM call expecting a plain-text response (acknowledgements, drafts).
    Returns AIMessage.content directly — no JSON parsing needed.
    """
    try:
        chain = llm.bind(temperature=temperature)
        response = chain.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return response.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "We have received your message and a team member will be in touch shortly."


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — classify_node
# ══════════════════════════════════════════════════════════════════════════════

_CLASSIFY_SYSTEM = """
You are a request classification engine for an operations team.

Classify the incoming customer request into EXACTLY ONE of these four types:

- "Complaint": Customer expressing dissatisfaction, reporting a failure, or demanding
  resolution for something that went wrong. The customer is unhappy with a past or
  ongoing experience.

- "Enquiry": Customer seeking information, asking a question, or requesting guidance.
  No action needs to be performed — only information needs to be provided.

- "Service Request": Customer requesting a specific action to be carried out — such as
  a setup, configuration change, access grant, delivery, or any task that requires
  someone to DO something.

- "Escalation": Request involving threats of legal action, regulatory complaints,
  repeated failures after prior attempts to resolve, demands for senior management
  involvement, or any situation that must bypass normal workflow and go directly
  to a supervisor.

Also assess urgency based on BUSINESS IMPACT — not the customer's emotional tone:
- "Critical": Operations halted, data loss, security breach, regulatory deadline, or
  legal threat. Immediate action required.
- "High": User(s) blocked from working, service degraded for a team, or SLA at risk.
- "Medium": Service impaired but a workaround exists. No immediate business stoppage.
- "Low": General question, future need, no time pressure whatsoever.

Return ONLY valid JSON in this exact format — no extra text, no markdown:
{
  "classification": "<Complaint|Enquiry|Service Request|Escalation>",
  "urgency": "<Low|Medium|High|Critical>",
  "reasoning": "<one sentence explaining the classification and urgency choice>"
}
""".strip()


def classify_node(state: TicketState) -> dict:
    """
    Reads raw_input → classifies type and urgency via one LLM call.
    On any LLM failure, falls back to Escalation / High so a human handles it.
    If override_classification is set by the UI, skips the LLM entirely.
    """
    # ── Human-in-the-loop override ───────────────────────────────────────────
    # If an operator has manually selected a classification, bypass the LLM.
    if state.get("override_classification"):
        return {
            "classification": state["override_classification"],
            "urgency":        state.get("override_urgency", "Medium"),
            "reasoning":      (
                f"⚠️ Manually overridden by operator: "
                f"{state['override_classification']} / {state.get('override_urgency', 'Medium')}. "
                "Original LLM classification was bypassed."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── LLM classification ───────────────────────────────────────────────────
    result = _call_llm_json(_CLASSIFY_SYSTEM, state["raw_input"], temperature=0.1)

    if result.get("__fallback__"):
        return {
            "classification": "Escalation",
            "urgency": "High",
            "reasoning": f"Auto-classification failed — routed to human review. Error: {result.get('error')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "classification": result.get("classification", "Escalation"),
        "urgency":        result.get("urgency", "High"),
        "reasoning":      result.get("reasoning", ""),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — complaint_node
# ══════════════════════════════════════════════════════════════════════════════

# Maps urgency → who the complaint is escalated to
_ESCALATION_TARGETS = {
    "Critical": "Senior Manager",
    "High":     "Team Lead",
    "Medium":   "Customer Relations Specialist",
    "Low":      "Support Agent",
}

_COMPLAINT_SYSTEM = """
You are a customer relations specialist drafting a professional acknowledgement
for a complaint. You have been given the urgency level and who the case is
escalated to.

Your message must:
1. Acknowledge receipt of the complaint warmly and directly
2. Express genuine understanding of the customer's frustration
3. State that the case has been escalated to the appropriate handler
4. Confirm that a follow-up will occur within 2 hours
5. NOT promise specific outcomes or timelines for resolution

Tone by urgency:
- Critical / High : Empathetic, urgent, direct. Acknowledge severity without dismissing.
- Medium          : Professional and reassuring.
- Low             : Friendly, calm, and helpful.

Respond with ONLY the message body — no subject line, no "Dear [Name]" salutation,
no metadata.
""".strip()


def complaint_node(state: TicketState) -> dict:
    """
    Steps:
      1. Acknowledge receipt      → LLM generates response_draft
      2. Escalate to handler      → logic: map urgency → escalated_to
      3. Log with priority flag   → appended to actions_taken
      4. Set 2-hour follow-up     → followup_hours = 2
    """
    urgency      = state.get("urgency", "Medium")
    escalated_to = _ESCALATION_TARGETS.get(urgency, "Support Agent")

    user_msg = (
        f"Urgency: {urgency}\n"
        f"Escalated to: {escalated_to}\n\n"
        f"Original complaint:\n{state['raw_input']}"
    )
    response_draft = _call_llm_text(_COMPLAINT_SYSTEM, user_msg, temperature=0.3)

    actions = [
        f"Step 1: Acknowledgement drafted (urgency={urgency})",
        f"Step 2: Escalated to → {escalated_to}",
        f"Step 3: Case logged with {urgency} priority flag",
        "Step 4: 2-hour follow-up reminder set",
    ]

    return {
        "response_draft": response_draft,
        "escalated_to":   escalated_to,
        "followup_hours": 2,
        "route_to":       escalated_to,
        "status":         "escalated" if urgency in ("High", "Critical") else "open",
        "actions_taken":  actions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — enquiry_node
# ══════════════════════════════════════════════════════════════════════════════

_ENQUIRY_SYSTEM = """
You are a knowledgeable customer support agent answering an enquiry.

For the incoming enquiry, provide:
1. A sub_topic label — 2 to 4 words that precisely describe the specific topic
   of this enquiry (e.g. "Password Reset Process", "Billing Invoice Query",
   "Product Feature Availability").
2. A clear, accurate, helpful response that directly addresses the question.

Tone by urgency:
- High    : Concise and direct. Offer an immediate callback or escalation if needed.
- Medium  : Balanced — helpful explanation with next steps.
- Low     : Comprehensive and friendly. Can include background context.

Return ONLY valid JSON — no extra text, no markdown:
{
  "sub_topic": "<2-4 word label>",
  "response":  "<full response message to send to the customer>"
}
""".strip()


def enquiry_node(state: TicketState) -> dict:
    """
    Steps:
      1. Classify sub-topic       → combined with step 2 in one LLM call
      2. Generate AI response     → combined with step 1
      3. Send response            → simulated (response_draft written to state)
      4. Log as resolved          → status = "resolved"
    """
    urgency  = state.get("urgency", "Low")
    user_msg = f"Urgency: {urgency}\n\nCustomer enquiry:\n{state['raw_input']}"

    result = _call_llm_json(_ENQUIRY_SYSTEM, user_msg, temperature=0.4)

    if result.get("__fallback__"):
        result = {"sub_topic": "General Enquiry", "response": _call_llm_text(_ENQUIRY_SYSTEM, user_msg)}

    sub_topic = result.get("sub_topic", "General Enquiry")
    response  = result.get("response", "")

    actions = [
        f"Step 1: Sub-topic classified as '{sub_topic}'",
        "Step 2: AI response generated from knowledge base",
        "Step 3: Response dispatched to requester",
        "Step 4: Case logged as resolved",
    ]

    return {
        "sub_topic":      sub_topic,
        "response_draft": response,
        "route_to":       "Customer Support",
        "status":         "resolved",
        "actions_taken":  actions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — service_request_node
# ══════════════════════════════════════════════════════════════════════════════

# SLA timer (hours) mapped directly from urgency — pure logic, no LLM needed
_SLA_HOURS = {"Critical": 2, "High": 4, "Medium": 8, "Low": 24}

_SERVICE_REQUEST_SYSTEM = """
You are a service desk agent processing an incoming service request.

From the request text, extract and produce:
1. extracted_details — a structured summary of what is being requested, including:
   who is making the request, what exactly they need, any relevant specifics
   (dates, systems, quantities, access levels, etc.)
2. department — the single most appropriate team to fulfil this request.
   Choose from: IT Support | Customer Relations | Finance | Operations | HR |
   Technical Support | Sales | Legal
3. confirmation_message — a professional confirmation to send to the requester,
   stating that their request has been received, routed to the correct department,
   and will be handled within the SLA window provided.

Return ONLY valid JSON — no extra text, no markdown:
{
  "extracted_details":    "<structured summary of the request>",
  "department":           "<department name>",
  "confirmation_message": "<confirmation message for the requester>"
}
""".strip()


def service_request_node(state: TicketState) -> dict:
    """
    Steps:
      1. Extract required details   → combined LLM call
      2. Route to department        → from LLM output (department field)
      3. Generate confirmation      → combined LLM call (confirmation_message field)
      4. Set SLA timer              → logic: urgency → sla_timer_hours
    """
    urgency   = state.get("urgency", "Medium")
    sla_hours = _SLA_HOURS.get(urgency, 8)

    user_msg = (
        f"Urgency: {urgency}\n"
        f"SLA window: {sla_hours} hours\n\n"
        f"Service request:\n{state['raw_input']}"
    )
    result = _call_llm_json(_SERVICE_REQUEST_SYSTEM, user_msg, temperature=0.3)

    if result.get("__fallback__"):
        result = {
            "extracted_details":    "Details could not be extracted — manual review required.",
            "department":           "Operations",
            "confirmation_message": "Your service request has been received and is being reviewed.",
        }

    actions = [
        "Step 1: Required details extracted from the request",
        f"Step 2: Routed to → {result.get('department', 'Operations')}",
        "Step 3: Confirmation message generated for requester",
        f"Step 4: SLA timer set to {sla_hours} hours (urgency={urgency})",
    ]

    return {
        "extracted_details": result.get("extracted_details", ""),
        "route_to":          result.get("department", "Operations"),
        "response_draft":    result.get("confirmation_message", ""),
        "sla_timer_hours":   sla_hours,
        "status":            "open",
        "actions_taken":     actions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — escalation_node
# ══════════════════════════════════════════════════════════════════════════════

_ESCALATION_SYSTEM = """
You are a senior customer service manager handling an urgent escalation that has
been flagged for immediate human review.

Write a professional, urgent acknowledgement message. It must:
1. Immediately acknowledge the seriousness and urgency of their situation
2. Confirm that a supervisor has been personally notified
3. State that this case has been removed from the automated queue and a human
   will handle it directly
4. Convey empathy and authority — the customer must feel heard and that their
   situation is being taken seriously at the highest level
5. NOT make specific promises about outcomes or resolution timeframes

Respond with ONLY the message body — no subject line, no salutation, no metadata.
""".strip()


def escalation_node(state: TicketState) -> dict:
    """
    Steps:
      1. Flag for human review      → logic: human_review_flag = True
      2. Draft urgent acknowledgement → LLM generates response_draft
      3. Notify supervisor          → logic: supervisor_notified = True
      4. Pause auto-resolution      → logic: auto_resolution_paused = True
    """
    response_draft = _call_llm_text(_ESCALATION_SYSTEM, state["raw_input"], temperature=0.2)

    actions = [
        "Step 1: Ticket flagged for immediate human review",
        "Step 2: Urgent acknowledgement drafted",
        "Step 3: Supervisor notified",
        "Step 4: Auto-resolution paused — awaiting human decision",
    ]

    return {
        "human_review_flag":      True,
        "response_draft":         response_draft,
        "supervisor_notified":    True,
        "auto_resolution_paused": True,
        "route_to":               "Supervisor / Human Review Queue",
        "status":                 "pending_human",
        "actions_taken":          actions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — log_node
# ══════════════════════════════════════════════════════════════════════════════

def log_node(state: TicketState) -> dict:
    """
    Pure logic — no LLM call.
    Generates a case_id (or reuses one for overrides) and persists the full case record to SQLite.
    Every branch converges here before END.
    """
    case_id = state.get("case_id") or f"CASE-{uuid.uuid4().hex[:8].upper()}"

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO cases (
            case_id, timestamp, input_source, classification, urgency,
            reasoning, status, route_to, response_draft, sla_timer_hours,
            followup_hours, human_review_flag, auto_resolution_paused,
            actions_taken, raw_input, is_human_override
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        state.get("timestamp"),
        state.get("input_source"),
        state.get("classification"),
        state.get("urgency"),
        state.get("reasoning"),
        state.get("status"),
        state.get("route_to"),
        state.get("response_draft"),
        state.get("sla_timer_hours"),
        state.get("followup_hours"),
        int(state.get("human_review_flag") or 0),
        int(state.get("auto_resolution_paused") or 0),
        json.dumps(state.get("actions_taken") or []),
        state.get("raw_input"),
        0,  # AI decision — not a human override
    ))
    conn.commit()
    conn.close()

    print(f"[LOG] Case {case_id} written to SQLite — status: {state.get('status')}")
    return {"case_id": case_id}


# ══════════════════════════════════════════════════════════════════════════════
# OVERRIDE HELPER — bypasses the full LangGraph graph
# ══════════════════════════════════════════════════════════════════════════════

def apply_override(
    case_id: str,
    raw_input: str,
    input_metadata: dict,
    new_classification: str,
    new_urgency: str,
) -> dict:
    """
    Human-in-the-loop override handler.

    Bypasses the LangGraph graph entirely to avoid state-threading issues.
    Steps:
      1. Builds a minimal state with the operator-selected classification + urgency.
      2. Calls ONLY the correct branch node (no LLM classify call).
      3. Does a direct SQL UPDATE on the existing case row (same case_id, no new row).
      4. Returns the full updated case dict for the UI to render.
    """
    # ── 1. Build mock state ───────────────────────────────────────────────────
    mock_state: dict = {
        "input_source":   input_metadata.get("input_source", "form"),
        "raw_input":      raw_input,
        "input_metadata": input_metadata,
        "classification": new_classification,
        "urgency":        new_urgency,
        "reasoning": (
            f"⚠️ Manually overridden by operator: "
            f"{new_classification} / {new_urgency}. "
            "Original AI classification was bypassed."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ── 2. Call only the relevant branch node ─────────────────────────────────
    _branch_map = {
        "Complaint":       complaint_node,
        "Enquiry":         enquiry_node,
        "Service Request": service_request_node,
        "Escalation":      escalation_node,
    }
    branch_fn = _branch_map.get(new_classification, escalation_node)
    branch_result = branch_fn(mock_state)

    # Merge branch outputs into mock state
    merged = {**mock_state, **branch_result, "case_id": case_id}

    # ── 3. UPDATE the existing SQLite row (no INSERT, so count never changes) ─
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        UPDATE cases SET
            classification      = ?,
            urgency             = ?,
            reasoning           = ?,
            status              = ?,
            route_to            = ?,
            response_draft      = ?,
            sla_timer_hours     = ?,
            followup_hours      = ?,
            human_review_flag   = ?,
            auto_resolution_paused = ?,
            actions_taken       = ?,
            is_human_override   = 1
        WHERE case_id = ?
        """,
        (
            merged.get("classification"),
            merged.get("urgency"),
            merged.get("reasoning"),
            merged.get("status"),
            merged.get("route_to"),
            merged.get("response_draft"),
            merged.get("sla_timer_hours"),
            merged.get("followup_hours"),
            int(merged.get("human_review_flag") or 0),
            int(merged.get("auto_resolution_paused") or 0),
            json.dumps(merged.get("actions_taken") or []),
            case_id,
        ),
    )
    conn.commit()
    conn.close()

    print(f"[OVERRIDE] Case {case_id} updated → {new_classification} / {new_urgency}")
    return merged
