import json
import uuid
from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

if HAS_PYDANTIC:
    class RouteRequest(BaseModel):
        prompt: str = Field(..., min_length=1, description="Prompt text")
        max_latency_ms: float = Field(..., gt=0, description="Hard maximum latency in milliseconds")
        max_cost_usd: float = Field(..., gt=0, description="Hard maximum budget in USD")
        min_capability: float = Field(0.0, ge=0.0, le=1.0, description="Minimum capability score")
        required_context: Optional[int] = Field(None, ge=1, description="Required context window in tokens")
        preference: str = Field("lowest_cost", pattern="^(lowest_cost|lowest_latency|highest_capability|balanced_pareto|pareto)$")
        expected_output_tokens: Optional[int] = Field(150, ge=1)

    class CandidateAudit(BaseModel):
        model_id: str
        model_name: str
        passed: bool
        rejection_reasons: List[str]
        context_fit: bool
        predicted_cost_usd: float
        cost_fit: bool
        predicted_latency_ms: float
        latency_fit: bool
        capability_score: float
        capability_fit: bool
        circuit_state: str
        score: float = 0.0

    class RouteResponse(BaseModel):
        request_id: str
        chosen_model: Optional[str]
        response_text: str
        no_valid_model: bool
        closest_miss_suggestion: Optional[str]
        estimator: Dict[str, Any]
        candidates: List[CandidateAudit]
        failover_chain: List[Dict[str, Any]]
        actual_metrics: Dict[str, Any]
        auditor: Dict[str, Any]
        timestamp: str

    class ChaosSetting(BaseModel):
        model_id: str
        add_latency_ms: float = 0.0
        kill_switch: bool = False
        error_rate: float = 0.0

else:
    # Pure Python robust fallback
    class BaseSchema:
        def model_dump(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        def dict(self) -> Dict[str, Any]:
            return self.model_dump()

    class RouteRequest(BaseSchema):
        def __init__(
            self,
            prompt: str,
            max_latency_ms: float,
            max_cost_usd: float,
            min_capability: float = 0.0,
            required_context: Optional[int] = None,
            preference: str = "lowest_cost",
            expected_output_tokens: int = 150
        ):
            if not prompt or len(prompt.strip()) == 0:
                raise ValueError("prompt must not be empty")
            if max_latency_ms <= 0:
                raise ValueError("max_latency_ms must be > 0")
            if max_cost_usd <= 0:
                raise ValueError("max_cost_usd must be > 0")
            if not (0.0 <= min_capability <= 1.0):
                raise ValueError("min_capability must be between 0.0 and 1.0")
            valid_prefs = ["lowest_cost", "lowest_latency", "highest_capability", "balanced_pareto", "pareto"]
            if preference not in valid_prefs:
                raise ValueError(f"preference must be one of {valid_prefs}")

            self.prompt = prompt
            self.max_latency_ms = float(max_latency_ms)
            self.max_cost_usd = float(max_cost_usd)
            self.min_capability = float(min_capability)
            self.required_context = int(required_context) if required_context else None
            self.preference = preference
            self.expected_output_tokens = int(expected_output_tokens) if expected_output_tokens else 150

    class CandidateAudit(BaseSchema):
        def __init__(
            self,
            model_id: str,
            model_name: str,
            passed: bool,
            rejection_reasons: List[str],
            context_fit: bool,
            predicted_cost_usd: float,
            cost_fit: bool,
            predicted_latency_ms: float,
            latency_fit: bool,
            capability_score: float,
            capability_fit: bool,
            circuit_state: str,
            score: float = 0.0
        ):
            self.model_id = model_id
            self.model_name = model_name
            self.passed = passed
            self.rejection_reasons = rejection_reasons
            self.context_fit = context_fit
            self.predicted_cost_usd = predicted_cost_usd
            self.cost_fit = cost_fit
            self.predicted_latency_ms = predicted_latency_ms
            self.latency_fit = latency_fit
            self.capability_score = capability_score
            self.capability_fit = capability_fit
            self.circuit_state = circuit_state
            self.score = score

    class RouteResponse(BaseSchema):
        def __init__(
            self,
            request_id: str,
            chosen_model: Optional[str],
            response_text: str,
            no_valid_model: bool,
            closest_miss_suggestion: Optional[str],
            estimator: Dict[str, Any],
            candidates: List[Any],
            failover_chain: List[Dict[str, Any]],
            actual_metrics: Dict[str, Any],
            auditor: Dict[str, Any],
            timestamp: str
        ):
            self.request_id = request_id
            self.chosen_model = chosen_model
            self.response_text = response_text
            self.no_valid_model = no_valid_model
            self.closest_miss_suggestion = closest_miss_suggestion
            self.estimator = estimator
            self.candidates = candidates
            self.failover_chain = failover_chain
            self.actual_metrics = actual_metrics
            self.auditor = auditor
            self.timestamp = timestamp

        def model_dump(self) -> Dict[str, Any]:
            d = dict(self.__dict__)
            d["candidates"] = [c.model_dump() if hasattr(c, "model_dump") else c for c in self.candidates]
            return d

    class ChaosSetting(BaseSchema):
        def __init__(
            self,
            model_id: str,
            add_latency_ms: float = 0.0,
            kill_switch: bool = False,
            error_rate: float = 0.0
        ):
            self.model_id = model_id
            self.add_latency_ms = float(add_latency_ms)
            self.kill_switch = bool(kill_switch)
            self.error_rate = float(error_rate)
