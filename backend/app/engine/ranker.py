from typing import List
from ..models.schemas import CandidateAudit

def rank_candidates(candidates: List[CandidateAudit], preference: str = "lowest_cost") -> List[CandidateAudit]:
    """
    Ranks compliant candidates based on user stated preference.
    """
    if not candidates:
        return []

    if preference == "lowest_cost":
        return sorted(candidates, key=lambda c: (c.predicted_cost_usd, c.predicted_latency_ms))

    elif preference == "lowest_latency":
        return sorted(candidates, key=lambda c: (c.predicted_latency_ms, c.predicted_cost_usd))

    elif preference == "highest_capability":
        return sorted(candidates, key=lambda c: (-c.capability_score, c.predicted_cost_usd))

    elif preference in ("balanced_pareto", "pareto"):
        costs = [c.predicted_cost_usd for c in candidates]
        lats = [c.predicted_latency_ms for c in candidates]
        caps = [c.capability_score for c in candidates]

        min_cost, max_cost = min(costs), max(costs)
        min_lat, max_lat = min(lats), max(lats)
        min_cap, max_cap = min(caps), max(caps)

        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
        lat_range = max_lat - min_lat if max_lat > min_lat else 1.0
        cap_range = max_cap - min_cap if max_cap > min_cap else 1.0

        for c in candidates:
            norm_cost = (c.predicted_cost_usd - min_cost) / cost_range
            norm_lat = (c.predicted_latency_ms - min_lat) / lat_range
            norm_cap = (c.capability_score - min_cap) / cap_range

            # Higher is better
            c.score = round((0.4 * (1.0 - norm_cost)) + (0.3 * (1.0 - norm_lat)) + (0.3 * norm_cap), 4)

        return sorted(candidates, key=lambda c: -c.score)

    return sorted(candidates, key=lambda c: c.predicted_cost_usd)
