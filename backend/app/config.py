import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_REGISTRY_PATH = APP_DIR / "models" / "registry.json"
DB_PATH = DATA_DIR / "routeiq_audit.db"
ESTIMATOR_MODEL_PATH = DATA_DIR / "estimator_tfidf.joblib"
PROMPTS_PATH = DATA_DIR / "prompts_300.json"
BENCHMARK_PATH = DATA_DIR / "benchmark_30.json"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(BASE_DIR.parent / "model")))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Hard Rule & Circuit Breaker Thresholds
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 15.0
ESTIMATOR_CONFIDENCE_THRESHOLD = 0.70

# Independent Auditor Tolerances
AUDITOR_LATENCY_TOLERANCE_MS = 5.0
AUDITOR_COST_TOLERANCE_RATIO = 1.01

# System 1 / Laya Decision Engine Mapping
TIER_CAPABILITY_MAP = {
    "easy": 0.35,
    "medium": 0.65,
    "hard": 0.85
}
