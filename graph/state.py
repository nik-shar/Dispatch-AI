from typing import TypedDict, Optional


class TicketState(TypedDict):

    # ── Input layer ────────────────────────────────────────────────────────
    input_source: str               # "form" | "file_upload" | "simulated_inbox"
    raw_input: str                  # the full text of the incoming request
    input_metadata: Optional[dict]  # {"sender": "...", "subject": "...", "filename": "..."}
    timestamp: Optional[str]        # ISO format — when the request was received

    # ── Classification layer ───────────────────────────────────────────────
    classification: Optional[str]           # Complaint | Enquiry | Service Request | Escalation
    urgency: Optional[str]                  # Low | Medium | High | Critical
    reasoning: Optional[str]                # LLM explanation for the classification decision
    # Human-in-the-loop override — if set, classify_node skips the LLM entirely
    override_classification: Optional[str]  # operator-selected type to force
    override_urgency: Optional[str]         # operator-selected urgency to force

    # ── Enquiry branch ─────────────────────────────────────────────────────
    sub_topic: Optional[str]        # Step 1: sub-topic detected within the enquiry

    # ── Service Request branch ─────────────────────────────────────────────
    extracted_details: Optional[str]    # Step 1: key details pulled from the request
    sla_timer_hours: Optional[int]      # Step 4: SLA window in hours (e.g. 4, 8, 24)

    # ── Complaint branch ───────────────────────────────────────────────────
    escalated_to: Optional[str]     # Step 2: name/role of senior handler assigned
    followup_hours: Optional[int]   # Step 4: follow-up reminder window (2hrs for Complaint)

    # ── Escalation branch ──────────────────────────────────────────────────
    human_review_flag: Optional[bool]       # Step 1: ticket flagged for immediate human review
    supervisor_notified: Optional[bool]     # Step 3: confirmation that supervisor was alerted
    auto_resolution_paused: Optional[bool]  # Step 4: prevents any automated resolution

    # ── Shared outputs (used across multiple branches) ─────────────────────
    response_draft: Optional[str]   # draft message back to the requester (all branches)
    route_to: Optional[str]         # team or department the ticket is routed to

    # ── Audit & logging ────────────────────────────────────────────────────
    case_id: Optional[str]          # unique ID assigned at log_node
    status: Optional[str]           # open | resolved | escalated | pending_human
    actions_taken: Optional[list]   # ordered record of every step taken (the audit trail)
