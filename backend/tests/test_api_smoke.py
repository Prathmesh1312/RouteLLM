import unittest
import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://127.0.0.1:8000"

class TestAPISmoke(unittest.TestCase):
    def _get(self, path: str):
        req = urllib.request.Request(f"{BASE_URL}{path}", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, resp.read()

    def _post(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}{path}",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_01_health_endpoint(self):
        status, raw = self._get("/health")
        self.assertEqual(status, 200)
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "RouteIQ AI Gateway")
        self.assertGreaterEqual(data["models_registered"], 5)
        self.assertTrue(data["chain_valid"])

    def test_02_dashboard_html_served(self):
        status, raw = self._get("/")
        self.assertEqual(status, 200)
        html = raw.decode("utf-8")
        self.assertIn("RouteIQ", html)
        self.assertIn("Rules decide. AI only advises.", html)

    def test_03_models_inventory(self):
        status, raw = self._get("/models")
        self.assertEqual(status, 200)
        models = json.loads(raw.decode("utf-8"))
        self.assertGreaterEqual(len(models), 5)
        model_ids = [m["id"] for m in models]
        self.assertIn("deepseek-r1-distill-14b", model_ids)
        self.assertIn("qwen-2.5-coder-7b", model_ids)
        self.assertIn("mistral-7b-instruct", model_ids)
        self.assertIn("llama-3.2-1b", model_ids)

        # Check GGUF metadata attached
        qwen = next(m for m in models if m["id"] == "qwen-2.5-coder-7b")
        self.assertTrue(qwen.get("gguf_meta", {}).get("valid", False))

    def test_04_successful_route_request(self):
        payload = {
            "prompt": "Implement binary search in Python with O(log n) complexity proof.",
            "max_latency_ms": 2500.0,
            "max_cost_usd": 0.05,
            "min_capability": 0.70,
            "preference": "lowest_cost"
        }
        status, res = self._post("/route", payload)
        self.assertEqual(status, 200)
        self.assertFalse(res["no_valid_model"])
        self.assertIsNotNone(res["chosen_model"])
        self.assertIn(res["chosen_model"], ["qwen-2.5-coder-7b", "mistral-7b-instruct", "deepseek-r1-distill-14b"])
        self.assertTrue(res["auditor"]["compliant"])
        self.assertEqual(len(res["auditor"]["violations"]), 0)
        self.assertIn("chain_hash", res["auditor"])

    def test_05_impossible_constraint_structured_refusal(self):
        payload = {
            "prompt": "Solve complex quantum simulation",
            "max_latency_ms": 10.0,  # Impossible latency
            "max_cost_usd": 0.000001,  # Impossible budget
            "min_capability": 0.99   # Exceeds models
        }
        status, res = self._post("/route", payload)
        self.assertEqual(status, 200)
        self.assertTrue(res["no_valid_model"])
        self.assertIsNone(res["chosen_model"])
        self.assertIn("[RouteIQ Refusal]", res["response_text"])
        self.assertIsNotNone(res["closest_miss_suggestion"])
        self.assertTrue(res["auditor"]["compliant"])  # Refusal is compliant

    def test_06_chaos_and_shrinking_budget_failover(self):
        # 1. Reset all chaos first
        self._post("/chaos/reset", {})

        # 2. Kill the primary cheap model (Llama)
        self._post("/chaos", {
            "model_id": "llama-3.2-1b",
            "kill_switch": True,
            "add_latency_ms": 0.0,
            "error_rate": 0.0
        })

        # 3. Route request with lowest_cost
        payload = {
            "prompt": "What is the capital of France?",
            "max_latency_ms": 2500.0,
            "max_cost_usd": 0.01,
            "min_capability": 0.0,
            "preference": "lowest_cost"
        }
        status, res = self._post("/route", payload)
        self.assertEqual(status, 200)
        self.assertFalse(res["no_valid_model"])
        
        # Verify failover chain recorded the killed model
        self.assertGreater(len(res["failover_chain"]), 0)
        first_fail = res["failover_chain"][0]
        self.assertEqual(first_fail["model_id"], "llama-3.2-1b")
        self.assertIn("Chaos Kill-Switch", first_fail["error"])
        self.assertIn("remaining_latency_ms", first_fail)
        self.assertIn("remaining_cost_usd", first_fail)

        # 4. Reset chaos
        self._post("/chaos/reset", {})

    def test_07_audit_ledger_and_chain_verification(self):
        status, raw = self._get("/audit/verify")
        self.assertEqual(status, 200)
        verif = json.loads(raw.decode("utf-8"))
        self.assertTrue(verif["valid"], f"Audit chain must be valid: {verif.get('message')}")
        self.assertIsNone(verif["broken_at_id"])

if __name__ == "__main__":
    unittest.main()
