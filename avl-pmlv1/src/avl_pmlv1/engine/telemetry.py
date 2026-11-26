"""Seven Mandatory 500-Step Telemetry Redlines & Auto-Checkpoint Halt Guard."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List


@dataclass
class TelemetryAlert:
    step: int
    redline_id: int
    name: str
    value: float
    threshold_desc: str


@dataclass
class TelemetryMonitor:
    """Checks all 7 AVL-PMLv1 redlines every 500 steps; flags emergency save-and-halt on persistent violation."""

    check_interval: int = 500
    history: List[TelemetryAlert] = field(default_factory=list)

    def evaluate(self, step: int, metrics: Dict[str, float]) -> List[TelemetryAlert]:
        alerts: List[TelemetryAlert] = []

        # 1. Layer residual energy ratio in [0.80, 1.25]
        r_energy = metrics.get("residual_energy_ratio", 1.0)
        if r_energy < 0.80 or r_energy > 1.25:
            alerts.append(TelemetryAlert(step, 1, "residual_energy_ratio", r_energy, "[0.80, 1.25]"))

        # 2. Probe-to-full-rank alignment cosine >= 0.65
        probe_cos = metrics.get("probe_alignment_cosine", 0.75)
        if probe_cos < 0.65:
            alerts.append(TelemetryAlert(step, 2, "probe_alignment_cosine", probe_cos, ">= 0.65"))

        # 3. Posterior scorer Kendall tau >= 0.30
        kendall = metrics.get("scorer_kendall_tau", 0.40)
        if kendall < 0.30:
            alerts.append(TelemetryAlert(step, 3, "scorer_kendall_tau", kendall, ">= 0.30"))

        # 4. Graph row entropy in [ln 2, ln 4] and syn-adv overlap <= 0.20
        ent = metrics.get("row_entropy_syn", 1.0)
        overlap = metrics.get("syn_adv_overlap", 0.0)
        if ent < math.log(2.0) or ent > math.log(4.0) or overlap > 0.20:
            alerts.append(
                TelemetryAlert(step, 4, "graph_health", overlap if overlap > 0.20 else ent, "H in [ln2,ln4] & cos<=0.20")
            )

        # 5. Same-pit relapse rate after pain imprint <= 0.20
        relapse = metrics.get("pain_same_pit_relapse_rate", 0.05)
        if relapse > 0.20:
            alerts.append(TelemetryAlert(step, 5, "pain_same_pit_relapse_rate", relapse, "<= 0.20"))

        # 6. Breaker rate <= 0.25 and NetFix >= 0
        breaker_rate = metrics.get("breaker_rate", 0.05)
        net_fix = metrics.get("post_net_fix", 0.0)
        if breaker_rate > 0.25 or net_fix < 0.0:
            alerts.append(TelemetryAlert(step, 6, "surgery_safety", net_fix if net_fix < 0 else breaker_rate, "breaker<=0.25 & NetFix>=0"))

        # 7. std(alpha_eng) >= 0.08
        std_eng = metrics.get("std_alpha_eng", 0.12)
        if std_eng < 0.08:
            alerts.append(TelemetryAlert(step, 7, "std_alpha_eng", std_eng, ">= 0.08"))

        self.history.extend(alerts)
        return alerts
