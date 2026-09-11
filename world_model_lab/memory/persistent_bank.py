"""
Persistent Cross-Run Memory Bank (v1 - vN Continuous Learning)
Enables knowledge persistence across distinct execution runs and competition versions:
- Exports verified causal rules, contingency statistics, and trajectory solutions.
- Imports and fuses knowledge from past versions (v1, v2, v3 -> v4).
- Warm-starts world models with empirical Bayesian priors rather than uniform ignorance.
- Packages accumulated intelligence for submission to competitions.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from world_model_lab.core.types import Hypothesis, ConditionType, EffectType, Action


@dataclass
class CausalRuleRecord:
    rule_id: str
    condition: str
    cause_color: str
    trigger_color: Optional[str]
    effect: str
    target_color: str
    new_state: str
    confidence: float
    success_count: int
    falsification_count: int
    versions_observed: List[str] = field(default_factory=list)


@dataclass
class RunMetadata:
    version: str
    timestamp: str
    games_evaluated: List[str]
    total_episodes: int
    solved_episodes: int
    success_rate: float


class PersistentMemoryBank:
    """
    Persistent Memory Bank that survives process termination.
    Allows Version N to import, build upon, and update memory from Versions 1...N-1.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "experiments", "memory_banks"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self.verified_rules: Dict[str, CausalRuleRecord] = {}
        self.semantic_contingencies: Dict[str, Dict[str, int]] = {}
        self.successful_trajectories: Dict[str, List[str]] = {}
        self.version_history: List[RunMetadata] = []

    def save_run_memory(
        self,
        version: str,
        hypotheses: List[Hypothesis],
        contingencies: Optional[Dict[str, Dict[str, int]]] = None,
        trajectories: Optional[Dict[str, List[str]]] = None,
        games: Optional[List[str]] = None,
        total_episodes: int = 0,
        solved_episodes: int = 0,
        filename: Optional[str] = None
    ) -> str:
        """Serializes current agent world model knowledge into a versioned bank."""
        # 1. Store high-confidence and verified hypotheses
        for h in hypotheses:
            if h.success_count > 0 or h.confidence > 0.05:
                sig = f"{h.condition.value}:{h.cause_color}:{h.trigger_color or 'none'}->{h.effect_target_color}:{h.effect_new_state}"
                if sig in self.verified_rules:
                    rec = self.verified_rules[sig]
                    rec.success_count += h.success_count
                    rec.falsification_count += h.falsification_count
                    rec.confidence = max(rec.confidence, h.confidence)
                    if version not in rec.versions_observed:
                        rec.versions_observed.append(version)
                else:
                    self.verified_rules[sig] = CausalRuleRecord(
                        rule_id=h.id,
                        condition=h.condition.value,
                        cause_color=h.cause_color,
                        trigger_color=h.trigger_color,
                        effect=h.effect.value,
                        target_color=h.effect_target_color,
                        new_state=h.effect_new_state,
                        confidence=h.confidence,
                        success_count=h.success_count,
                        falsification_count=h.falsification_count,
                        versions_observed=[version]
                    )

        # 2. Store contingencies
        if contingencies:
            for k, effect_map in contingencies.items():
                self.semantic_contingencies.setdefault(k, {})
                for eff, cnt in effect_map.items():
                    self.semantic_contingencies[k][eff] = self.semantic_contingencies[k].get(eff, 0) + cnt

        # 3. Store trajectories
        if trajectories:
            self.successful_trajectories.update(trajectories)

        # 4. Record metadata
        meta = RunMetadata(
            version=version,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            games_evaluated=games or [],
            total_episodes=total_episodes,
            solved_episodes=solved_episodes,
            success_rate=(solved_episodes / total_episodes) if total_episodes > 0 else 0.0
        )
        self.version_history.append(meta)

        # Write to JSON
        fname = filename or f"memory_bank_{version}.json"
        filepath = os.path.join(self.storage_dir, fname)
        data = {
            "version": version,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "verified_rules": {k: asdict(v) for k, v in self.verified_rules.items()},
            "semantic_contingencies": self.semantic_contingencies,
            "successful_trajectories": self.successful_trajectories,
            "version_history": [asdict(m) for m in self.version_history]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Also update cumulative master memory bank
        master_path = os.path.join(self.storage_dir, "master_memory_bank.json")
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_memory(self, filepath: str) -> bool:
        """Loads and imports knowledge from a specific memory bank file."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for k, v in data.get("verified_rules", {}).items():
            self.verified_rules[k] = CausalRuleRecord(**v)

        for k, v in data.get("semantic_contingencies", {}).items():
            self.semantic_contingencies.setdefault(k, {})
            for eff, cnt in v.items():
                self.semantic_contingencies[k][eff] = self.semantic_contingencies[k].get(eff, 0) + cnt

        for k, v in data.get("successful_trajectories", {}).items():
            self.successful_trajectories[k] = v

        for m in data.get("version_history", []):
            self.version_history.append(RunMetadata(**m))

        return True

    def import_all_versions(self) -> int:
        """Scans storage directory and imports all version banks (v1, v2, v3...)."""
        if not os.path.exists(self.storage_dir):
            return 0
        loaded_count = 0
        for fname in sorted(os.listdir(self.storage_dir)):
            if fname.startswith("memory_bank_") and fname.endswith(".json"):
                path = os.path.join(self.storage_dir, fname)
                if self.load_memory(path):
                    loaded_count += 1
        return loaded_count

    def apply_prior_to_hypotheses(self, hypotheses: List[Hypothesis]) -> None:
        """
        Warm-starts an agent's hypothesis space by boosting prior weights
        for causal rules that were verified in prior versions.
        """
        if not self.verified_rules:
            return

        for h in hypotheses:
            sig = f"{h.condition.value}:{h.cause_color}:{h.trigger_color or 'none'}->{h.effect_target_color}:{h.effect_new_state}"
            if sig in self.verified_rules:
                rec = self.verified_rules[sig]
                # Boost confidence proportionally to cross-version empirical verification
                evidence_ratio = (rec.success_count + 1) / (rec.falsification_count + 1)
                boost = min(5.0, 1.0 + (evidence_ratio * 0.5))
                h.confidence *= boost
                h.success_count = rec.success_count
                h.falsification_count = rec.falsification_count

        # Re-normalize hypothesis confidences
        total_conf = sum(h.confidence for h in hypotheses)
        if total_conf > 0:
            for h in hypotheses:
                h.confidence /= total_conf
