import json
from backend.app.main import ESTIMATOR_CONSENSUS, EXECUTOR, RouteRequest

test_cases = [
    ("What is 2 + 2?", "lowest_cost", 1000.0, 0.01, 0.0),
    ("What is the capital of Japan?", "lowest_cost", 1000.0, 0.01, 0.0),
    ("Summarize the meeting notes into three bullet points: team aligned on roadmap, delayed sprint 4, launched beta.", "lowest_cost", 1000.0, 0.01, 0.0),
    ("Write a Python function to parse JSON with error handling.", "lowest_cost", 2000.0, 0.01, 0.0),
    ("Write a Python function to parse JSON with error handling.", "lowest_latency", 2000.0, 0.01, 0.0),
    ("Write a Python function to parse JSON with error handling.", "highest_capability", 2000.0, 0.01, 0.0),
    ("Write a Python function to parse JSON with error handling.", "balanced_pareto", 2000.0, 0.01, 0.0),
    ("Derive the Nash equilibrium in a Cournot oligopoly.", "lowest_cost", 2000.0, 0.05, 0.0),
    ("Derive the Nash equilibrium in a Cournot oligopoly.", "highest_capability", 2000.0, 0.05, 0.0),
    # Tight latency constraint (150ms -> only Llama 3.2 1b at 180ms or less can fit, or refusal)
    ("What is the capital of France?", "lowest_cost", 100.0, 0.01, 0.0),
    # Tight cost constraint ($0.0001 -> only cheapest can fit)
    ("Explain binary search.", "lowest_cost", 2000.0, 0.0001, 0.0),
]

for p, pref, max_lat, max_cost, min_cap in test_cases:
    est = ESTIMATOR_CONSENSUS.evaluate(p, min_cap)
    req = RouteRequest(prompt=p, max_latency_ms=max_lat, max_cost_usd=max_cost, min_capability=min_cap, preference=pref)
    resp = EXECUTOR.execute_request(req, simulate_delay=False)
    passed_models = [c.model_id for c in resp.candidates if c.passed]
    print(f"Prompt: {p[:35]}... | Pref: {pref} | LatMax: {max_lat}ms | CostMax: ${max_cost}")
    print(f"  Estimator: {est['tier']} (raw: {est['raw_tier']}, conf: {est['confidence']:.2f}, elevated: {est['elevated']}) -> ReqCap: {est['required_capability']}")
    print(f"  Passed Candidates: {passed_models}")
    print(f"  Chosen: {resp.chosen_model} | Refusal: {resp.no_valid_model}")
    print("-" * 60)
