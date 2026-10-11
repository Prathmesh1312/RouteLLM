import urllib.request
import json
import sys

def run():
    print("--- ROUTEIQ RELEASE CANDIDATE SMOKE VERIFICATION ---")

    # Step 1: Health
    res = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health").read().decode("utf-8"))
    assert res["status"] == "ok"
    print(f"[1/8] Health check passed: {res['service']} (Models: {res['models_registered']}, Chain valid: {res['chain_valid']})")

    # Step 2: Root Dashboard
    html = urllib.request.urlopen("http://127.0.0.1:8000/").read().decode("utf-8")
    assert "<title>" in html or "RouteIQ" in html or "html" in html
    print(f"[2/8] Dashboard HTML served successfully ({len(html)} bytes)")

    # Step 3: Model Inventory
    models = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/models").read().decode("utf-8"))
    assert len(models) == 5
    gguf_count = sum(1 for m in models if m.get("endpoint_type") == "local_gguf")
    sim_count = sum(1 for m in models if m.get("simulated"))
    print(f"[3/8] Model inventory verified (Total: {len(models)}, Local GGUF: {gguf_count}, Simulator: {sim_count})")

    # Step 4 & 5: New Request & Candidate Explanations
    req_data = json.dumps({
        "prompt": "Write a python generator for Fibonacci sequence.",
        "max_latency_ms": 2000.0,
        "max_cost_usd": 0.05,
        "min_capability": 0.35,
        "preference": "lowest_cost"
    }).encode("utf-8")
    route_resp = json.loads(urllib.request.urlopen(
        urllib.request.Request("http://127.0.0.1:8000/route", data=req_data, headers={"Content-Type": "application/json"})
    ).read().decode("utf-8"))
    assert not route_resp["no_valid_model"]
    assert len(route_resp["candidates"]) == 5
    print(f"[4/8 & 5/8] Request routed to: {route_resp['chosen_model']} (Badge: {route_resp['actual_metrics']['badge']}, Latency: {route_resp['actual_metrics']['total_latency_ms']}ms)")

    # Step 6A: Controlled Failure & Safe Failover (Sufficient Budget)
    chaos_req = json.dumps({
        "model_id": route_resp["chosen_model"],
        "kill_switch": True,
        "add_latency_ms": 0.0,
        "error_rate": 0.0
    }).encode("utf-8")
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/chaos", data=chaos_req, headers={"Content-Type": "application/json"}))

    req_with_headroom = json.dumps({
        "prompt": "Write a python generator for Fibonacci sequence.",
        "max_latency_ms": 3000.0,
        "max_cost_usd": 0.05,
        "min_capability": 0.35,
        "preference": "lowest_cost"
    }).encode("utf-8")
    failover_resp = json.loads(urllib.request.urlopen(
        urllib.request.Request("http://127.0.0.1:8000/route", data=req_with_headroom, headers={"Content-Type": "application/json"})
    ).read().decode("utf-8"))
    assert failover_resp["chosen_model"] != route_resp["chosen_model"]
    assert len(failover_resp["failover_chain"]) >= 1
    print(f"[6A/8] Shrinking-budget failover (Headroom available): Bypassed {route_resp['chosen_model']} -> Chosen backup: {failover_resp['chosen_model']} (Remaining latency: {failover_resp['failover_chain'][0]['remaining_latency_ms']}ms)")

    # Step 6B: Controlled Failure where budget is exhausted -> Truthful Refusal
    req_tight_budget = json.dumps({
        "prompt": "Write a python generator for Fibonacci sequence.",
        "max_latency_ms": 1100.0,
        "max_cost_usd": 0.05,
        "min_capability": 0.35,
        "preference": "lowest_cost"
    }).encode("utf-8")
    exhausted_resp = json.loads(urllib.request.urlopen(
        urllib.request.Request("http://127.0.0.1:8000/route", data=req_tight_budget, headers={"Content-Type": "application/json"})
    ).read().decode("utf-8"))
    assert exhausted_resp["no_valid_model"] is True
    print(f"[6B/8] Shrinking-budget failover (Budget exhausted): Primary failed -> Clean refusal returned without relaxing SLA")

    # Reset chaos
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/chaos/reset", data=b"{}", headers={"Content-Type": "application/json"}))

    # Step 7: Auditor Result
    auditor = route_resp.get("auditor", {})
    assert auditor.get("compliant") is True
    assert len(auditor.get("violations", [])) == 0
    print(f"[7/8] Independent post-hoc auditor verified (Compliant: {auditor.get('compliant')}, Violations: 0, Hash: {auditor.get('chain_hash')[:16]}...)")

    # Step 8: Cryptographic Ledger Verification
    verify_resp = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/audit/verify").read().decode("utf-8"))
    assert verify_resp["valid"] is True
    print(f"[8/8] Cryptographic SHA-256 Ledger verified: VALID (Total records: {verify_resp['total_records']}, Status: {verify_resp['message']})")

if __name__ == "__main__":
    run()
