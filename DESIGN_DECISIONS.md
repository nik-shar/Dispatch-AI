# POC Design Decisions & Discussion Log
> **Assignment:** Incoming Request Processing Workflow — AI Engineer Role Assessment
> **Timeline:** 5 days | **Stack:** LangGraph + Nebius LLM + Streamlit + SQLite

---

## 1. Are Request Type and Urgency Linked?

**Q: Are type and urgency linked, or are there different permutations?**

**They are partially linked by default, but independent by design.**

- **Type** = *what the request is about* (category/nature)
- **Urgency** = *how quickly it must be resolved* (time criticality)

Any combination is valid — visualised as a matrix:

```
                    URGENCY
                Low   Medium   High   Critical
            ┌─────────────────────────────────
 Complaint  │  ✓      ✓        ✓       ✓
 Enquiry    │  ✓      ✓        ✓       ✓ (rare)
 Serv. Req  │  ✓      ✓        ✓       ✓
 Escalation │  ✗      ✗        ✓       ✓
```

> `✓` = valid combination | `✗` = near-impossible by definition | `✓ (rare)` = theoretically valid but uncommon in practice

Some combinations have natural tendencies:
- Escalation almost always comes with High or Critical urgency
- Enquiry trends toward Low or Medium
- Complaint can genuinely land anywhere

### Graph design: 4 type branches, urgency modifies behaviour WITHIN each branch

Rejected: 16 permutation branches (4 types × 4 urgencies) — redundant logic, unmaintainable.

**How urgency modifies each branch:**
- **Complaint:** tone of response, who it escalates to (Team Lead vs Senior Manager)
- **Enquiry:** response depth, offered follow-up (email vs phone callback)
- **Service Request:** `sla_timer_hours` directly maps from urgency:
  ```
  Critical → 2 hours | High → 4 hours | Medium → 8 hours | Low → 24 hours
  ```
- **Escalation:** always High or Critical; Critical may additionally CC senior leadership

### Edge case: Service Request + Critical
Example: "Our payment gateway is completely down, I need this restored immediately."

Two approaches:
1. Trust classification → Service Request branch runs with Critical urgency and 2-hour SLA
2. Override rule → if `Service Request + Critical` → re-route to Escalation branch

**Decision for POC:** Option 1 (trust classification). Keeps the graph simple and reliable.
**Mention in presentation:** Option 2 as a planned enhancement — this directly addresses the rubric's "edge case handling" criterion.

---

## Presentation Slide Mapping

| Slide | Title | Key Points from This Document |
|---|---|---|
| 1 | Problem Understanding | Section 1 (dataset), the manual process problem, what AI automates |
| 2 | Solution Architecture | Section 5 (LangGraph), the 6-node graph diagram, state flow |
| 3 | Implementation Highlights | Section 2 (LLM vs ML decision), Section 7 (node design), prompt engineering |
| 4 | Challenges & Learnings | Section 3 (why Qwen 7B was rejected), Section 8 (edge cases), type/urgency independence |
| 5 | Demo Summary & Next Steps | Live demo, hybrid ML/LLM as v2, Option 2 edge case override as enhancement |


## 2. Dataset — What Public Data is Available?

**Q: What public datasets exist that are similar and relevant to this problem?**

| Dataset | Source | Why Relevant | Gap |
|---|---|---|---|
| **Bitext Customer Support LLM Dataset** | HuggingFace | 26,872 rows · 27 intents · 10 categories. Intents like complaint, refund, billing map directly to our 4 classes. Apache 2.0. | Synthetic data — clean but not messy real-world emails |
| **CFPB Consumer Complaints** | consumerfinance.gov / Kaggle | Real complaint narratives, 2M+ rows | Heavy finance skew, needs label remapping |
| **Customer IT Support Ticket Dataset** | Kaggle (gauravtopre) | Has priority/urgency field out of the box | IT domain only, small dataset |
| **Customer Support Intent Dataset** | Kaggle (danofer) | Intent labels for billing, feedback, requests | Needs consolidation into 4 macro-classes |

**Decision:** Use the Bitext dataset as base reference. The problem statement explicitly allows AI-generated data, so synthetic augmentation is acceptable. Dataset is not the bottleneck for this POC.

---

## 3. Classification Approach — Classic ML or LLM?

**Q: Should we use a classic ML model (TF-IDF + Logistic Regression / BERT) or directly use an LLM for classification?**

### Classic ML
- Fast inference, near-zero cost, explainable feature weights
- Needs thousands of well-labelled examples matching our exact taxonomy
- Cannot understand tone, implied urgency, or sarcasm
- Training + evaluation takes 2–3 days minimum

### LLM for Classification
- Zero-shot — no training data needed
- Understands nuanced language, tone, business impact
- Returns structured JSON with `class`, `urgency`, `confidence`, `reasoning`
- 1–3 second latency, small per-token cost

### Why NOT a hybrid (ML for type + LLM for urgency)?
Considered and rejected because:
- The available fine-tuned DistilBERT model outputs ITSM labels (`Incident/Request/Problem/Change`) — these do NOT match our required taxonomy (`Complaint/Enquiry/Service Request/Escalation`)
- Adding a mapping/translation layer between two mismatched models introduces compounding errors
- An evaluator will immediately ask: "why not just let the LLM classify directly?" — no good answer

**Decision: Use LLM directly for classification.** The 4 classes are semantically nuanced (especially Complaint vs Escalation, and Request vs Enquiry) — this requires language understanding, not keyword counting.

> For a production system (beyond POC), the right approach would be: fine-tune a small transformer (DistilBERT) on a labelled dataset mapped to our 4 classes, use it as a primary fast classifier, fall back to LLM for low-confidence cases. That is the hybrid pattern worth mentioning on the slide.

---

## 4. Fine-tuned Model Found: W-L/SFT-Customer-Ticket-Qwen2.5-7b-ins

**Q: Is this model useful? Can it replace the frontier LLM?**

**What this model is:**
- Base: `Qwen/Qwen2.5-7B-Instruct` — Alibaba's 7B parameter instruction-tuned LLM
- Fine-tuned via: SFT (Supervised Fine-Tuning) on `Tobi-Bueck/customer-support-tickets` dataset
- Pipeline tag: `question-answering` — it is a **generative model**, not a classifier

**Hardware requirements:**
| Format | RAM/VRAM Needed | Speed on CPU |
|---|---|---|
| float16 | ~14 GB | 3–10 min/response |
| 4-bit quantized | ~4–5 GB | 1–3 min/response |
| Google Colab (T4 free) | 15 GB available | Usable with 4-bit quant |

**Can it run on Google Colab?** Yes — with `BitsAndBytesConfig(load_in_4bit=True)` it fits on the free T4 GPU. Colab free sessions disconnect after ~90 minutes of inactivity — fine for testing, not for a live demo.

**Quality vs frontier models:**
- The SFT fine-tuning improves domain understanding (customer support language) but causes **catastrophic forgetting** — it loses some general reasoning and structured output reliability of the base model
- Frontier models (GPT-4o, Gemini, Claude) are 10–100x more capable at following complex JSON format instructions reliably
- For a POC requiring structured, parseable outputs that drive branching logic — an unreliable 7B model is a risk not worth taking

**Decision: Do not use this model as the primary AI layer.** Interesting to explore in Colab for learning. For the POC submission, use a frontier LLM API.

---

## 5. LLM Provider — Nebius AI

**Q: Which model to use given available Nebius credits?**

Nebius provides an **OpenAI-compatible API** — integrates with the `openai` Python library by changing only the `base_url`. No new SDK required.

**Available models evaluated:**

| Model | Cost (In/Out per 1M) | Assessment |
|---|---|---|
| Qwen3-235B-A22B-Instruct-2507 | $0.20 / $0.60 | ✅ **Top pick** — frontier quality, best value |
| Llama-3.3-70B-Instruct | $0.13 / $0.40 | ✅ Proven, reliable, widely benchmarked |
| gpt-oss-120b (OpenAI) | $0.15 / $0.60 | ✅ Strong JSON instruction following |
| Qwen3-32B | $0.10 / $0.30 | ✅ Budget option, good enough |
| DeepSeek-V4-Pro | $1.75 / $3.50 | ⚠️ Overkill — premium reasoning not needed |
| Nemotron-Nano-Omni | $0.06 / $0.24 | ❌ Too small for reliable structured output |

**Decision: Qwen3-235B-A22B-Instruct-2507**
- 235B parameters — frontier-class quality comparable to GPT-4o
- POC will use ~50,000–100,000 tokens total = less than $0.10 total cost
- Strong instruction following and structured JSON output

---

## 6. Framework — LangChain or LangGraph?

**Q: Should we use LangChain, LangGraph, or neither?**

| | LangChain | LangGraph |
|---|---|---|
| Best for | Sequential chains: A → B → C | Conditional branching: A → classify → route to B or C or D |
| Your POC fit | ⚠️ Partial | ✅ **Perfect match** |

**Why LangGraph is the right choice:**
- The problem statement explicitly requires "multi-step branching logic" — this is exactly what LangGraph models
- One input → classify → 4 separate branch paths with their own steps — that IS a graph
- The architecture diagram on Slide 2 of the presentation directly mirrors the LangGraph node structure
- Demonstrates production-grade AI engineering patterns used at companies like LinkedIn, Replit, Uber

**Decision: LangGraph** (which includes LangChain functionality)

---

## 7. State Design (`state.py`)

**Q: What fields does the shared state need?**

The state is the data object that flows through every node in the graph. Every field must either:
- Be READ by a downstream node, OR
- Appear in the final output/audit log

### Final State Structure

```
Input Layer          → input_source, raw_input, input_metadata, timestamp
Classification       → classification, urgency, reasoning
Enquiry branch       → sub_topic
Service Req branch   → extracted_details, sla_timer_hours
Complaint branch     → escalated_to, followup_hours
Escalation branch    → human_review_flag, supervisor_notified, auto_resolution_paused
Shared outputs       → response_draft, route_to
Audit                → case_id, status, actions_taken
```

### Key decisions made:
- `sla_flag: bool` replaced with `sla_timer_hours: int` — a boolean is not a timer; the workflow needs an actual hours value
- `escalation_alert: str` replaced with `escalated_to: str` — names who received the escalation, not the content (content goes into `response_draft`)
- `actions_taken: list` is the single most important audit field — every step a branch takes appends to this list, forming the complete audit trail shown in the UI

### `status` field values:
```
"open"          → in progress (Complaint, Service Request)
"resolved"      → closed automatically (Enquiry)
"escalated"     → senior handler assigned (Complaint with high urgency)
"pending_human" → waiting for human review (Escalation branch)
```

---

## 8. Node Design (`nodes.py`)

### How many nodes?
6 nodes total:
`classify_node` → `complaint_node` | `enquiry_node` | `service_request_node` | `escalation_node` → `log_node`

### Key design decisions:

**classify_node scope:** Classify only (type + urgency + reasoning). Does NOT generate the response draft.
*Reason:* If the LLM misclassifies, a response draft would also be wrong and wasted. Separation of concerns — classify node has one job.

**LLM calls per branch:**
| Branch | LLM Calls | What the LLM Does |
|---|---|---|
| Complaint | 1 | Generate acknowledgement response draft |
| Enquiry | 1 combined | Return sub_topic + response in one JSON |
| Service Request | 1 combined | Return extracted_details + confirmation message in one JSON |
| Escalation | 1 | Draft urgent acknowledgement |
| log_node | 0 | Pure logic — write to SQLite, generate case_id, set final status |

**Error / fallback strategy:** If the LLM returns malformed JSON or the API call fails → classify as `"Escalation"` with urgency `"High"`. Rationale: an unclassifiable request should always go to a human. This is the correct semantic behaviour AND a defensible engineering decision to present.

**Log storage:** SQLite (not CSV, not in-memory)
- Takes ~30 minutes to implement
- Satisfies the "audit trail" optional enhancement (15% creativity score)
- Can be displayed as a live table in the Streamlit UI

---

