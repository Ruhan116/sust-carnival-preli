"""
QueueStorm Investigator — Test Runner
======================================
Fires all 10 public sample cases at the deployed POST /analyze-ticket endpoint,
records response times, validates schema + enum correctness, and writes:
  • report/llm_responses.json   — raw LLM output for every test case
  • report/test_results.json    — machine-readable pass/fail per field
  • report/summary_report.html  — human-readable HTML report with timing charts
"""

import json
import os
import sys
import time
import datetime
import requests
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("API_BASE_URL", "https://sust-carnival-preli-deku.vercel.app").rstrip("/")
CASES_FILE = Path(__file__).parent / "SUST_Preli_Sample_Cases.json"
REPORT_DIR = Path(__file__).parent / "report"
TIMEOUT_SECONDS = 35          # 30-second enforced limit + 5-second buffer
HEALTH_TIMEOUT  = 60          # max wait for /health

# ── Allowed enum values (from problem statement §7) ────────────────────────────
ALLOWED_ENUMS = {
    "evidence_verdict": {"consistent", "inconsistent", "insufficient_data"},
    "case_type": {
        "wrong_transfer", "payment_failed", "refund_request",
        "duplicate_payment", "merchant_settlement_delay",
        "agent_cash_in_issue", "phishing_or_social_engineering", "other",
    },
    "severity": {"low", "medium", "high", "critical"},
    "department": {
        "customer_support", "dispute_resolution", "payments_ops",
        "merchant_operations", "agent_operations", "fraud_risk",
    },
}

REQUIRED_RESPONSE_FIELDS = [
    "ticket_id", "relevant_transaction_id", "evidence_verdict",
    "case_type", "severity", "department", "agent_summary",
    "recommended_next_action", "customer_reply", "human_review_required",
]

# Safety patterns that must NOT appear in customer_reply
SAFETY_VIOLATIONS = [
    "your pin", "your otp", "your password", "share your pin",
    "share your otp", "share your password", "give us your pin",
    "provide your pin", "provide your otp", "provide your password",
    "enter your pin", "enter your otp",
    "we will refund", "we will reverse", "we will unblock",
    "we will recover", "your money will be refunded",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_health() -> dict:
    """Hit GET /health and return status info."""
    url = f"{BASE_URL}/health"
    try:
        t0 = time.perf_counter()
        resp = requests.get(url, timeout=HEALTH_TIMEOUT)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "url": url,
            "status_code": resp.status_code,
            "body": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
            "ok": resp.status_code == 200,
            "elapsed_ms": elapsed,
        }
    except Exception as exc:
        return {"url": url, "ok": False, "error": str(exc), "elapsed_ms": None}


def send_ticket(input_payload: dict) -> dict:
    """POST /analyze-ticket and return timing + response info."""
    url = f"{BASE_URL}/analyze-ticket"
    try:
        t0 = time.perf_counter()
        resp = requests.post(url, json=input_payload, timeout=TIMEOUT_SECONDS)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return {
            "status_code": resp.status_code,
            "body": body,
            "elapsed_ms": elapsed,
            "timed_out": False,
            "error": None,
        }
    except requests.exceptions.Timeout:
        return {
            "status_code": None,
            "body": None,
            "elapsed_ms": TIMEOUT_SECONDS * 1000,
            "timed_out": True,
            "error": "Request timed out",
        }
    except Exception as exc:
        return {
            "status_code": None,
            "body": None,
            "elapsed_ms": None,
            "timed_out": False,
            "error": str(exc),
        }


def validate_response(body: dict | None, expected: dict, input_payload: dict) -> dict:
    """Validate the actual response body against expected output + schema rules."""
    if body is None or not isinstance(body, dict):
        return {
            "schema_valid": False,
            "missing_fields": REQUIRED_RESPONSE_FIELDS,
            "enum_errors": [],
            "field_matches": {},
            "safety_violations": [],
            "overall": "FAIL",
        }

    missing = [f for f in REQUIRED_RESPONSE_FIELDS if f not in body]
    enum_errors = []
    field_matches = {}

    # --- enum validation ---
    for field, allowed in ALLOWED_ENUMS.items():
        val = body.get(field)
        if val is not None and val not in allowed:
            enum_errors.append({"field": field, "value": val, "allowed": sorted(allowed)})

    # --- ticket_id echo ---
    field_matches["ticket_id"] = (
        body.get("ticket_id") == input_payload.get("ticket_id")
    )

    # --- key functional fields vs expected ---
    for field in ("relevant_transaction_id", "evidence_verdict", "case_type", "department"):
        field_matches[field] = (body.get(field) == expected.get(field))

    # --- human_review_required type ---
    hrr = body.get("human_review_required")
    field_matches["human_review_required_type"] = isinstance(hrr, bool)

    # --- confidence range ---
    conf = body.get("confidence")
    if conf is not None:
        field_matches["confidence_in_range"] = isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0
    else:
        field_matches["confidence_in_range"] = None  # optional field absent

    # --- safety check ---
    reply_lower = (body.get("customer_reply") or "").lower()
    next_action_lower = (body.get("recommended_next_action") or "").lower()
    combined = reply_lower + " " + next_action_lower
    found_violations = [v for v in SAFETY_VIOLATIONS if v in combined]

    schema_valid = len(missing) == 0 and len(enum_errors) == 0
    key_fields_ok = all(v for k, v in field_matches.items() if isinstance(v, bool))
    overall = "PASS" if (schema_valid and key_fields_ok and not found_violations) else "FAIL"

    return {
        "schema_valid": schema_valid,
        "missing_fields": missing,
        "enum_errors": enum_errors,
        "field_matches": field_matches,
        "safety_violations": found_violations,
        "overall": overall,
    }


# ── HTML report builder ────────────────────────────────────────────────────────

def build_html_report(health: dict, results: list, run_at: str) -> str:
    total = len(results)
    passed = sum(1 for r in results if r["validation"]["overall"] == "PASS")
    failed = total - passed
    avg_ms = round(sum(r["elapsed_ms"] for r in results if r["elapsed_ms"]) / max(1, total), 1)
    max_ms = max((r["elapsed_ms"] or 0) for r in results)
    min_ms = min((r["elapsed_ms"] or float("inf")) for r in results)

    health_badge = (
        '<span class="badge pass">✓ OK</span>'
        if health.get("ok")
        else '<span class="badge fail">✗ DOWN</span>'
    )

    rows = ""
    for r in results:
        status_cls = "pass" if r["validation"]["overall"] == "PASS" else "fail"
        status_txt = r["validation"]["overall"]
        fm = r["validation"]["field_matches"]

        def cell(val):
            if val is True:
                return '<td class="match">✓</td>'
            elif val is False:
                return '<td class="mismatch">✗</td>'
            else:
                return '<td class="na">—</td>'

        enum_err_txt = ""
        if r["validation"]["enum_errors"]:
            enum_err_txt = "<br>".join(
                f'<span class="enum-err">{e["field"]}: got <b>{e["value"]}</b></span>'
                for e in r["validation"]["enum_errors"]
            )

        safety_txt = ""
        if r["validation"]["safety_violations"]:
            safety_txt = "<br>".join(
                f'<span class="safety-err">⚠ "{v}"</span>'
                for v in r["validation"]["safety_violations"]
            )

        missing_txt = ", ".join(r["validation"]["missing_fields"]) if r["validation"]["missing_fields"] else ""

        elapsed = f'{r["elapsed_ms"]} ms' if r["elapsed_ms"] is not None else "timeout"
        timeout_warn = " ⚠" if r.get("timed_out") else ""

        rows += f"""
        <tr>
          <td><b>{r["case_id"]}</b><br><small>{r["label"]}</small></td>
          <td class="{status_cls}">{status_txt}</td>
          <td>{elapsed}{timeout_warn}</td>
          <td>{r["http_status"] or "—"}</td>
          {cell(fm.get("ticket_id"))}
          {cell(fm.get("relevant_transaction_id"))}
          {cell(fm.get("evidence_verdict"))}
          {cell(fm.get("case_type"))}
          {cell(fm.get("department"))}
          {cell(fm.get("human_review_required_type"))}
          <td>{'<span class="ok">✓</span>' if not r["validation"]["safety_violations"] else safety_txt}</td>
          <td>{enum_err_txt or '<span class="ok">✓</span>'}</td>
          <td class="missing">{missing_txt or "—"}</td>
        </tr>"""

    bar_data = json.dumps([
        {"id": r["case_id"], "ms": r["elapsed_ms"] or 0, "pass": r["validation"]["overall"] == "PASS"}
        for r in results
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>QueueStorm Investigator — Test Report</title>
  <style>
    :root{{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3e;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#3b82f6;--text:#e2e8f0;--muted:#8892a4;}}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);padding:32px 24px;min-height:100vh}}
    h1{{font-size:1.8rem;font-weight:700;margin-bottom:4px;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    .subtitle{{color:var(--muted);font-size:.9rem;margin-bottom:28px}}
    .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:32px}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center}}
    .card .num{{font-size:2.2rem;font-weight:700;line-height:1.1}}
    .card .lbl{{font-size:.78rem;color:var(--muted);margin-top:6px;text-transform:uppercase;letter-spacing:.05em}}
    .green{{color:var(--green)}} .red{{color:var(--red)}} .yellow{{color:var(--yellow)}} .blue{{color:var(--blue)}}
    .badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.8rem;font-weight:600}}
    .badge.pass{{background:#14532d;color:var(--green)}} .badge.fail{{background:#450a0a;color:var(--red)}}
    .health-bar{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin-bottom:28px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
    .health-bar span{{color:var(--muted);font-size:.88rem}}
    .chart-wrap{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:32px}}
    .chart-wrap h2{{font-size:1rem;font-weight:600;margin-bottom:18px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
    canvas{{width:100%!important;max-width:100%}}
    table{{width:100%;border-collapse:collapse;background:var(--card);border-radius:12px;overflow:hidden;border:1px solid var(--border)}}
    th{{background:#12151f;color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;padding:10px 12px;text-align:left;white-space:nowrap}}
    td{{padding:10px 12px;border-top:1px solid var(--border);font-size:.83rem;vertical-align:top}}
    td.pass{{color:var(--green);font-weight:600}} td.fail{{color:var(--red);font-weight:600}}
    td.match{{color:var(--green);text-align:center}} td.mismatch{{color:var(--red);text-align:center}} td.na{{color:var(--muted);text-align:center}}
    .ok{{color:var(--green)}} .enum-err{{color:var(--red);font-size:.78rem}} .safety-err{{color:var(--yellow);font-size:.78rem}}
    .missing{{color:var(--red);font-size:.78rem}}
    .footer{{margin-top:32px;text-align:center;color:var(--muted);font-size:.8rem}}
    tr:hover td{{background:#1f2335}}
    .tbl-wrap{{overflow-x:auto;border-radius:12px}}
  </style>
</head>
<body>
  <h1>QueueStorm Investigator — Test Report</h1>
  <p class="subtitle">Run at {run_at} &nbsp;·&nbsp; Endpoint: <code>{BASE_URL}</code></p>

  <div class="health-bar">
    <strong>Health Check</strong>
    {health_badge}
    <span>GET /health → {health.get("status_code","—")} &nbsp;·&nbsp; {health.get("elapsed_ms","—")} ms</span>
  </div>

  <div class="cards">
    <div class="card"><div class="num blue">{total}</div><div class="lbl">Total Cases</div></div>
    <div class="card"><div class="num green">{passed}</div><div class="lbl">Passed</div></div>
    <div class="card"><div class="num red">{failed}</div><div class="lbl">Failed</div></div>
    <div class="card"><div class="num yellow">{avg_ms} ms</div><div class="lbl">Avg Response</div></div>
    <div class="card"><div class="num {'green' if max_ms < 30000 else 'red'}">{max_ms} ms</div><div class="lbl">Max Response</div></div>
    <div class="card"><div class="num blue">{min_ms} ms</div><div class="lbl">Min Response</div></div>
  </div>

  <div class="chart-wrap">
    <h2>Response Time per Case (ms)</h2>
    <canvas id="timingChart" height="80"></canvas>
  </div>

  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>Case</th><th>Result</th><th>Time</th><th>HTTP</th>
          <th>ticket_id</th><th>txn_id</th><th>evidence</th><th>case_type</th>
          <th>dept</th><th>hrr</th><th>Safety</th><th>Enums</th><th>Missing</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <p class="footer">Generated by QueueStorm Test Runner · {run_at}</p>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script>
    const data = {bar_data};
    const ctx = document.getElementById('timingChart').getContext('2d');
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: data.map(d => d.id),
        datasets: [{{
          label: 'Response Time (ms)',
          data: data.map(d => d.ms),
          backgroundColor: data.map(d => d.pass ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'),
          borderColor: data.map(d => d.pass ? '#22c55e' : '#ef4444'),
          borderWidth: 1,
          borderRadius: 6,
        }}]
      }},
      options: {{
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: ctx => ctx.parsed.y + ' ms' }} }}
        }},
        scales: {{
          x: {{ grid: {{ color: '#1e2233' }}, ticks: {{ color: '#8892a4' }} }},
          y: {{
            grid: {{ color: '#1e2233' }},
            ticks: {{ color: '#8892a4', callback: v => v + ' ms' }},
            beginAtZero: true,
            suggestedMax: 30000,
          }}
        }}
      }}
    }});
  </script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"  QueueStorm Investigator — Test Runner")
    print(f"  Target : {BASE_URL}")
    print(f"  Run at : {run_at}")
    print(f"{'='*60}\n")

    # ── Health check ────────────────────────────────────────────────────────
    print("[ 0 ] Checking GET /health …", end=" ", flush=True)
    health = check_health()
    if health.get("ok"):
        print(f"✓  ({health['elapsed_ms']} ms)")
    else:
        print(f"✗  — {health.get('error', health.get('status_code'))}")
        print("     Warning: health check failed. Tests will still run.\n")

    # ── Load test cases ──────────────────────────────────────────────────────
    with open(CASES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]
    print(f"Loaded {len(cases)} test cases from {CASES_FILE.name}\n")

    results = []
    llm_responses = []

    # ── Run each case ────────────────────────────────────────────────────────
    for idx, case in enumerate(cases, start=1):
        case_id = case["id"]
        label   = case["label"]
        inp     = case["input"]
        exp     = case["expected_output"]

        print(f"[{idx:2d}] {case_id} — {label}")
        print(f"      POST /analyze-ticket  ticket_id={inp['ticket_id']} …", end=" ", flush=True)

        result = send_ticket(inp)
        elapsed = result["elapsed_ms"]
        http_status = result["status_code"]

        if result["error"] and not result["timed_out"]:
            print(f"✗  ERROR: {result['error']}")
        elif result["timed_out"]:
            print(f"✗  TIMEOUT (>{TIMEOUT_SECONDS}s)")
        else:
            print(f"→ HTTP {http_status}  ({elapsed} ms)")

        validation = validate_response(result["body"], exp, inp)
        overall    = validation["overall"]
        print(f"      Validation: {overall}", end="")
        if validation["missing_fields"]:
            print(f"  |  missing: {validation['missing_fields']}", end="")
        if validation["enum_errors"]:
            print(f"  |  enum errors: {[e['field'] for e in validation['enum_errors']]}", end="")
        if validation["safety_violations"]:
            print(f"  |  ⚠ SAFETY VIOLATION", end="")
        print()
        print()

        results.append({
            "case_id":    case_id,
            "label":      label,
            "ticket_id":  inp["ticket_id"],
            "elapsed_ms": elapsed,
            "http_status": http_status,
            "timed_out":  result["timed_out"],
            "error":      result["error"],
            "validation": validation,
        })

        llm_responses.append({
            "case_id":         case_id,
            "label":           label,
            "ticket_id":       inp["ticket_id"],
            "elapsed_ms":      elapsed,
            "http_status":     http_status,
            "request":         inp,
            "expected_output": exp,
            "actual_response": result["body"],
            "validation":      validation,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for r in results if r["validation"]["overall"] == "PASS")
    failed = total - passed
    timings = [r["elapsed_ms"] for r in results if r["elapsed_ms"] is not None]
    avg_ms  = round(sum(timings) / len(timings), 1) if timings else 0

    print(f"{'='*60}")
    print(f"  RESULTS : {passed}/{total} passed  |  {failed} failed")
    print(f"  Avg time: {avg_ms} ms  |  Min: {min(timings or [0])} ms  |  Max: {max(timings or [0])} ms")
    print(f"{'='*60}\n")

    # ── Write outputs ─────────────────────────────────────────────────────────
    llm_path   = REPORT_DIR / "llm_responses.json"
    result_path = REPORT_DIR / "test_results.json"
    html_path   = REPORT_DIR / "summary_report.html"

    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_at":    run_at,
            "base_url":  BASE_URL,
            "health":    health,
            "responses": llm_responses,
        }, f, indent=2, ensure_ascii=False)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_at":   run_at,
            "base_url": BASE_URL,
            "summary": {
                "total":   total,
                "passed":  passed,
                "failed":  failed,
                "avg_ms":  avg_ms,
                "min_ms":  min(timings or [0]),
                "max_ms":  max(timings or [0]),
            },
            "health":  health,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    html = build_html_report(health, results, run_at)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Reports written to  {REPORT_DIR}/")
    print(f"  • llm_responses.json   — raw LLM responses for every case")
    print(f"  • test_results.json    — machine-readable pass/fail details")
    print(f"  • summary_report.html  — interactive HTML report with timing chart\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
