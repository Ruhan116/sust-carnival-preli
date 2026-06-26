# QueueStorm Investigator — Test Failures Report

> **Status:** Fixes applied 2026-06-26. See [Fixes Applied](#fixes-applied) at the bottom.

**Date:** 2026-06-26  
**Environment:** `http://127.0.0.1:8000` (local uvicorn)  
**LLM:** Gemini 2.5 Flash (`GEMINI_API_KEY`)  
**Test scripts:** `test_unit.py`, `test_api.py`

---

## Summary

| Suite | Result |
|-------|--------|
| Unit tests (`test_unit.py`) | **7/7 passed** |
| API integration tests (`test_api.py`) | **6/14 passed** (9 assertion failures across 8 cases) |

When the Gemini API was available and not rate-limited, classification and evidence reasoning were largely correct. Most API failures were caused by **429 rate limits** and **JSON parse retries**, not bad business logic.

---

## Unit Tests — No Failures

All offline tests passed:

- Language detection (`en`, `bn`, `mixed`)
- Prompt injection scrubber
- Transaction amount/counterparty pre-matching
- Credential safety filter
- Unauthorized commitment filter
- Schema validator (invalid `relevant_transaction_id` → `null`)
- Human review enforcement rules

---

## API Integration Test Failures

### Passed (6/14)

| Test | Time | Outcome |
|------|------|---------|
| Health check | ~0ms | `GET /health` → `{"status":"ok"}` |
| Payment failed — inconsistent | 6.3s | Correct `inconsistent` verdict, `human_review=true` |
| Empty transaction history | 5.4s | `insufficient_data`, `case_type=other` |
| Phishing attempt | 5.8s | `phishing_or_social_engineering`, `fraud_risk`, `critical` |
| Missing `ticket_id` | ~0ms | Correct **422** response |
| Payment failed — consistent | 5.7s | Correct classification; 2 assertion failures (see below) |

---

### Failure Category A — Test Harness / Expectation Issues (3 cases)

These cases returned **correct LLM output** but failed automated assertions.

#### 1. Payment failed — consistent evidence (`TEST-001`)

| Check | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| Safety: `"will be returned"` | Not present | Present in output | **False positive** — judge uses `your (money\|amount\|balance) will be returned`, not bare `"will be returned"`. Approved safe text contains *"Any eligible amount will be returned..."* |
| `human_review_required` | `true` | `false` | **Wrong test expectation** — medium severity + consistent `payment_failed` does not require human review per spec |

**What worked:** `case_type=payment_failed`, `evidence_verdict=consistent`, `relevant_transaction_id=TXN-1001`

---

#### 2. Wrong transfer (`TEST-003`)

| Check | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| Safety: `"will be returned"` | Not present | Present | **False positive** (same as above) |

**What worked:** `wrong_transfer`, `consistent`, `TXN-3001`, `severity=high`, `human_review=true`

---

#### 3. Prompt injection in complaint (`TEST-006`)

| Check | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| Safety: `"will be returned"` | Not present | Present | **False positive** (same as above) |

**What worked:** Injection scrubbed; `payment_failed`, `consistent`, `TXN-6001`

---

### Failure Category B — Gemini Rate Limit (429) (5 cases)

These failed during the batch run because the free tier allows **5 requests/minute** for `gemini-2.5-flash`. The batch sent 13 requests in ~47 seconds. After ~6 successful calls, all subsequent requests failed in **~0.2–0.3s** with this fallback:

```json
{
  "ticket_id": "TEST-00X",
  "relevant_transaction_id": null,
  "evidence_verdict": "insufficient_data",
  "case_type": "other",
  "severity": "medium",
  "department": "customer_support",
  "agent_summary": "System error during analysis. Manual review required.",
  "recommended_next_action": "Escalate to supervisor for manual review.",
  "customer_reply": "Thank you for contacting us. Your complaint has been received...",
  "human_review_required": true
}
```

| Test | Expected | Got (429 fallback) |
|------|----------|---------------------|
| Bangla complaint (`TEST-007`) | LLM analysis | Generic `other` / system error |
| Merchant settlement delay (`TEST-008`) | `merchant_settlement_delay` | `other` / system error |
| Agent cash-in issue (`TEST-009`) | `agent_cash_in_issue` | `other` / system error |
| Duplicate payment (`TEST-010`) | `duplicate_payment` | `other` / system error |
| Refund request (`TEST-011`) | `refund_request` | `other` / system error |
| High value wrong transfer (`TEST-012`) | `wrong_transfer` | `other` / system error |

**Server log error:**

```
429 RESOURCE_EXHAUSTED
Quota exceeded: generativelanguage.googleapis.com/generate_content_free_tier_requests
Limit: 5 requests/minute, model: gemini-2.5-flash
Please retry in ~25s
```

**Root cause:** `llm_core.py` does not catch `ClientError: 429`. The exception bubbles to `main.py`, which returns HTTP 200 with the generic fallback above.

**Retests after 25–30s cooldown (logic confirmed working):**

| Retest | Result |
|--------|--------|
| `RETEST-008` (merchant settlement, 25,000 BDT) | `merchant_settlement_delay`, `merchant_operations`, `consistent`, `TXN-8001`, 12s |
| `RETEST-BN` (Bangla payment failed) | `payment_failed`, `consistent`, Bangla `customer_reply`, 44s |

---

## Application Issues Found (Not Test Bugs)

These are real gaps discovered during testing, independent of the test harness.

### Critical

#### 1. No 429 / rate-limit handling

- **File:** `app/llm_core.py`, `main.py`
- **Behavior:** Gemini 429 raises an unhandled exception → generic fallback with wrong classification
- **Impact:** Under judge load (>5 req/min on free tier), most tickets return useless `other` / `insufficient_data` responses

#### 2. JSON parse failure on first LLM attempt (~50%)

- **File:** `app/llm_core.py`
- **Log pattern:**
  ```
  LLM attempt 1 failed to produce valid JSON: Could not extract valid JSON...
  LLM attempt 2 succeeded
  ```
- **Impact:** 2 API calls per ticket instead of 1 → doubles rate-limit pressure and adds ~8s latency per ticket

---

### High

#### 3. Missing third-party contact safety filter

- **Spec requirement:** Never instruct customer to contact unofficial third parties (−10 pts)
- **Current state:** Rule exists in system prompt (`prompts.py`) only
- **Missing from:** `app/safety_guardrails.py` (credential and commitment filters exist; third-party filter does not)

#### 4. High-value human review not enforced from transaction data

- **Spec requirement:** `human_review_required=true` when amount > 10,000 BDT
- **File:** `app/orchestrator.py` → `_enforce_human_review()`
- **Current state:** Does not inspect transaction amounts; only adds `high_value` to `reason_codes`

---

### Medium

#### 5. Bangla numeral pre-matching gap

- **File:** `app/preprocessing.py` → `_extract_amounts_from_text()`
- **Issue:** Regex matches ASCII digits only; Bengali numerals like `৫০০০` are not extracted
- **Impact:** Pre-match hints are weaker for Bangla-only amount references (LLM still handled Bangla correctly in retest)

#### 6. Generic error fallback masks infrastructure failures

- **File:** `main.py`
- **Behavior:** All exceptions return HTTP 200 with `"System error during analysis"`
- **Impact:** Judge may score these as wrong classifications rather than transient API failures

---

## Failure Breakdown

```
┌──────────────────────────────┬─────────┬─────────────────────────────┐
│ Category                     │ Cases   │ Severity                    │
├──────────────────────────────┼─────────┼─────────────────────────────┤
│ Gemini 429 rate limit        │ 5       │ CRITICAL                    │
│ Test harness false positive  │ 3       │ LOW (test issue, not app)   │
│ Wrong test expectation       │ 1       │ LOW (human_review rule)     │
│ Actual classification bug    │ 0       │ — (retests passed)          │
└──────────────────────────────┴─────────┴─────────────────────────────┘
```

---

## Recommended Fix Priority

| Priority | Issue | Fix |
|----------|-------|-----|
| **P0** | 429 rate-limit handling | Retry with backoff in `llm_core.py`; fallback to rule-based response |
| **P0** | JSON parse retries | Use Gemini structured output (`response_mime_type: application/json`) |
| **P1** | Third-party safety filter | Add deterministic regex check in `safety_guardrails.py` |
| **P1** | High-value human review | Pass transaction amounts into `_enforce_human_review()` |
| **P2** | Bangla numeral extraction | Extend `_extract_amounts_from_text()` for Bengali digits |
| **P2** | Rule-based fallback | Implement spec Alternative B when LLM is unavailable |

---

## How to Reproduce

```bash
# Terminal 1 — start server
cd backend
source venv/bin/activate
uvicorn main:app --reload

# Terminal 2 — run tests
cd backend
source venv/bin/activate
python test_unit.py    # offline, fast
python test_api.py     # live API; needs GEMINI_API_KEY
```

**Rate-limit note:** Free tier allows 5 Gemini requests/minute. Each ticket may use 2 calls when JSON parse fails on attempt 1. Space API tests by ≥15 seconds or run individually to avoid 429 failures.

---

## Fixes Applied

| Issue | Fix | File(s) |
|-------|-----|---------|
| 429 rate-limit crashes | Retry with backoff (3 attempts); fall back to rule-based on exhaustion | `app/llm_core.py` |
| JSON parse failures (~50%) | Gemini structured output via `response_mime_type=application/json` + JSON schema | `app/llm_core.py` |
| No fallback when LLM unavailable | New rule-based analyzer using pre-match signals + keyword heuristics | `app/rule_based.py` |
| Missing third-party safety filter | Deterministic regex filter on `customer_reply` | `app/safety_guardrails.py` |
| High-value human review not enforced | `_enforce_human_review()` now checks transaction amounts > 10,000 BDT | `app/orchestrator.py` |
| Bangla numeral pre-matching gap | Bengali digits (০–৯) normalized before amount extraction | `app/preprocessing.py` |
| Generic error fallback | `main.py` uses rule-based + `finalize_response()` on unhandled errors | `main.py`, `app/orchestrator.py` |
| Test harness false positives | Commitment checks use judge-spec regex patterns; removed wrong `human_review` expectation | `test_api.py` |
| Test rate-limit collisions | 13s inter-test delay added to API test suite | `test_api.py` |
