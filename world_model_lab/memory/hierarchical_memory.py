"""
Hierarchical 4-Tier Memory Architecture

Explicit 4-Layer Taxonomy:
┌────────────────────┬─────────────────────────────────────────────────┬──────────────────────────────┐
│ Layer              │ Purpose                                         │ Lifetime                     │
├────────────────────┼─────────────────────────────────────────────────┼──────────────────────────────┤
│ run_memory         │ Temporary observations from the current run     │ One run                      │
│ game_memory        │ Rules and discoveries for one game              │ Across versions              │
│ global_memory      │ General strategies useful across games          │ Across the entire project    │
│ submission_memory  │ Read-only snapshot used during competition      │ Frozen at submission         │
└────────────────────┴─────────────────────────────────────────────────┴──────────────────────────────┘

Guarantees:
- Never dumps raw transcripts into next context (prevents context overflow & noise).
- Distills structured causal rules, empirical probabilities, and validated invariants.
- Supports immutable frozen snapshots for competition evaluation.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class DistilledRule:
    """Distilled causal rule stored across versions."""
    rule_signature: str
    condition: str
    cause_color: str
    trigger_color: Optional[str]
    effect: str
    target_color: str
    new_state: str
    confidence: float
    verification_count: int
    falsification_count: int
    first_seen_version: str
    last_verified_version: str


@dataclass
class DistilledRunSummary:
    """Structured knowledge distilled from a single run, avoiding raw transcript dumping."""
    run_id: str
    steps_taken: int
    task_solved: bool
    verified_rules: List[Dict[str, Any]]
    falsified_rule_signatures: List[str]
    contingency_deltas: Dict[str, Dict[str, int]]
    net_process_reward: float


class RunMemory:
    """
    Layer 1: Temporary observations from current execution.
    Lifetime: One run (cleared at run end after distillation).
    """

    def __init__(self):
        self.step_transitions: List[Dict[str, Any]] = []
        self.active_hypotheses_history: List[Dict[str, float]] = []
        self.step_prediction_errors: List[Dict[str, Any]] = []
        self.temp_contingencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.cumulative_reward: float = 0.0

    def record_step(
        self,
        step: int,
        action_name: str,
        diff_summary: Dict[str, Any],
        prediction_error: bool,
        reward: float
    ) -> None:
        self.step_transitions.append({
            "step": step,
            "action": action_name,
            "diff": diff_summary,
            "prediction_error": prediction_error,
            "reward": reward
        })
        self.cumulative_reward += reward

    def record_contingency(self, antecedent: str, effect: str) -> None:
        self.temp_contingencies[antecedent][effect] += 1

    def distill(self, run_id: str, task_solved: bool, verified_hypotheses: List[Any], falsified_signatures: List[str]) -> DistilledRunSummary:
        """
        Distills raw step logs into compact structured knowledge.
        Prevents transcript dumping and context window overflow.
        """
        verified_data = []
        for h in verified_hypotheses:
            cond_val = h.condition.value.lower() if hasattr(h.condition, 'value') else str(h.condition).lower()
            eff_val = h.effect.value.lower() if hasattr(h.effect, 'value') else str(h.effect).lower()
            verified_data.append({
                "signature": f"{cond_val}:{h.cause_color}:{h.trigger_color or 'none'}->{h.effect_target_color}:{h.effect_new_state}",
                "condition": cond_val,
                "cause_color": h.cause_color,
                "trigger_color": h.trigger_color,
                "effect": eff_val,
                "target_color": h.effect_target_color,
                "new_state": h.effect_new_state,
                "confidence": h.confidence,
                "success_count": getattr(h, "success_count", 1),
                "falsification_count": getattr(h, "falsification_count", 0)
            })

        summary = DistilledRunSummary(
            run_id=run_id,
            steps_taken=len(self.step_transitions),
            task_solved=task_solved,
            verified_rules=verified_data,
            falsified_rule_signatures=list(falsified_signatures),
            contingency_deltas={k: dict(v) for k, v in self.temp_contingencies.items()},
            net_process_reward=self.cumulative_reward
        )
        return summary

    def clear(self) -> None:
        """Purges raw per-step trajectory data."""
        self.step_transitions.clear()
        self.active_hypotheses_history.clear()
        self.step_prediction_errors.clear()
        self.temp_contingencies.clear()
        self.cumulative_reward = 0.0


class GameMemory:
    """
    Layer 2: Rules and discoveries for specific games.
    Lifetime: Across versions (v1 -> v2 -> v3 -> v4).
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir
        # game_id -> {rule_signature: DistilledRule}
        self.game_rules: Dict[str, Dict[str, DistilledRule]] = defaultdict(dict)
        # game_id -> {antecedent: {effect: count}}
        self.game_contingencies: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # game_id -> solved status across versions
        self.game_solution_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def absorb_run_distillation(self, game_id: str, version: str, summary: DistilledRunSummary) -> None:
        """Integrates distilled knowledge from a single run into cross-version game memory."""
        # 1. Update verified rules
        for r_data in summary.verified_rules:
            sig = r_data["signature"]
            if sig in self.game_rules[game_id]:
                existing = self.game_rules[game_id][sig]
                existing.confidence = max(existing.confidence, r_data["confidence"])
                existing.verification_count += r_data["success_count"]
                existing.last_verified_version = version
            else:
                self.game_rules[game_id][sig] = DistilledRule(
                    rule_signature=sig,
                    condition=r_data["condition"],
                    cause_color=r_data["cause_color"],
                    trigger_color=r_data["trigger_color"],
                    effect=r_data["effect"],
                    target_color=r_data["target_color"],
                    new_state=r_data["new_state"],
                    confidence=r_data["confidence"],
                    verification_count=r_data["success_count"],
                    falsification_count=r_data["falsification_count"],
                    first_seen_version=version,
                    last_verified_version=version
                )

        # 2. Penalize falsified rules
        for sig in summary.falsified_rule_signatures:
            if sig in self.game_rules[game_id]:
                self.game_rules[game_id][sig].falsification_count += 1
                self.game_rules[game_id][sig].confidence *= 0.5

        # 3. Accumulate contingencies
        for k, v in summary.contingency_deltas.items():
            for eff, cnt in v.items():
                self.game_contingencies[game_id][k][eff] += cnt

        # 4. Record solution event
        self.game_solution_records[game_id].append({
            "version": version,
            "steps": summary.steps_taken,
            "solved": summary.task_solved,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    def get_game_priors(self, game_id: str) -> Dict[str, float]:
        """Returns prior boost factors for hypotheses in this game."""
        priors = {}
        if game_id in self.game_rules:
            for sig, rule in self.game_rules[game_id].items():
                evidence_ratio = (rule.verification_count + 1) / (rule.falsification_count + 1)
                priors[sig] = float(min(5.0, 1.0 + (evidence_ratio * 0.5)))
        return priors

    def export_json(self, filepath: str) -> None:
        data = {
            "game_rules": {
                g: {sig: asdict(r) for sig, r in rules.items()}
                for g, rules in self.game_rules.items()
            },
            "game_contingencies": {
                g: {k: dict(v) for k, v in cont.items()}
                for g, cont in self.game_contingencies.items()
            },
            "game_solution_records": dict(self.game_solution_records)
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_json(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for g, rules in data.get("game_rules", {}).items():
            for sig, r_dict in rules.items():
                self.game_rules[g][sig] = DistilledRule(**r_dict)
        for g, cont in data.get("game_contingencies", {}).items():
            for k, v in cont.items():
                for eff, cnt in v.items():
                    self.game_contingencies[g][k][eff] += cnt
        for g, recs in data.get("game_solution_records", {}).items():
            self.game_solution_records[g].extend(recs)
        return True


class GlobalMemory:
    """
    Layer 3: General strategies and invariants useful across games.
    Lifetime: Across the entire project.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir
        # Meta-rules e.g. "touch" condition frequency vs "adjacency" condition frequency
        self.condition_type_success: Dict[str, int] = defaultdict(int)
        self.condition_type_falsifications: Dict[str, int] = defaultdict(int)
        # Spatial invariant observations: e.g. solid barrier blocking
        self.spatial_invariants: Dict[str, float] = {
            "wall_blocks_motion": 1.0,
            "state_change_requires_contact": 0.95,
            "dormant_objects_immobile": 0.90
        }
        # Strategy effectiveness registry: strategy_name -> mean_information_gain
        self.strategy_efficiency: Dict[str, float] = {
            "predictive_disagreement": 0.85,
            "uncertainty_only": 0.60,
            "random_exploration": 0.20,
            "reactive_greedy": 0.10
        }

    def update_condition_prior(self, condition: str, success: bool) -> None:
        if success:
            self.condition_type_success[condition] += 1
        else:
            self.condition_type_falsifications[condition] += 1

    def get_condition_weight(self, condition: str) -> float:
        succ = self.condition_type_success.get(condition, 1)
        fals = self.condition_type_falsifications.get(condition, 0)
        return succ / (succ + fals + 1e-6)

    def export_json(self, filepath: str) -> None:
        data = {
            "condition_type_success": dict(self.condition_type_success),
            "condition_type_falsifications": dict(self.condition_type_falsifications),
            "spatial_invariants": self.spatial_invariants,
            "strategy_efficiency": self.strategy_efficiency
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_json(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.condition_type_success.update(data.get("condition_type_success", {}))
        self.condition_type_falsifications.update(data.get("condition_type_falsifications", {}))
        self.spatial_invariants.update(data.get("spatial_invariants", {}))
        self.strategy_efficiency.update(data.get("strategy_efficiency", {}))
        return True


class SubmissionMemory:
    """
    Layer 4: Read-only snapshot used during competition.
    Lifetime: Frozen at submission.
    
    Guarantees:
    - Zero write mutation at test-time.
    - Eliminates model drift, catastrophic forgetting, and non-deterministic behavior.
    - Compact distilled tables: O(1) query latency, minimal memory footprint.
    """

    def __init__(self, frozen_data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = frozen_data or {}
        self.is_frozen: bool = True
        self.frozen_timestamp: str = self._data.get("frozen_timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.submission_version: str = self._data.get("submission_version", "v_final")

    @classmethod
    def compile_from_active_memories(
        cls,
        game_memory: GameMemory,
        global_memory: GlobalMemory,
        version_tag: str = "v_submission"
    ) -> "SubmissionMemory":
        """Compiles distilled game & global memory into an immutable competition snapshot."""
        frozen = {
            "submission_version": version_tag,
            "frozen_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game_priors": {
                g: game_memory.get_game_priors(g)
                for g in game_memory.game_rules.keys()
            },
            "verified_rules_summary": {
                g: [asdict(r) for r in rules.values() if r.confidence >= 0.70]
                for g, rules in game_memory.game_rules.items()
            },
            "spatial_invariants": dict(global_memory.spatial_invariants),
            "strategy_efficiency": dict(global_memory.strategy_efficiency)
        }
        return cls(frozen_data=frozen)

    def query_prior(self, game_id: str, rule_signature: str) -> float:
        """Read-only query for rule prior boost factor."""
        game_priors = self._data.get("game_priors", {}).get(game_id, {})
        return game_priors.get(rule_signature, 1.0)

    def get_verified_rules_for_game(self, game_id: str) -> List[Dict[str, Any]]:
        """Read-only retrieval of pre-verified causal dynamics."""
        return self._data.get("verified_rules_summary", {}).get(game_id, [])

    def export_frozen_snapshot(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    @classmethod
    def load_frozen_snapshot(cls, filepath: str) -> "SubmissionMemory":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(frozen_data=data)


class HierarchicalMemoryManager:
    """
    Orchestrates the 4 memory layers:
    1. run_memory (One run)
    2. game_memory (Across versions)
    3. global_memory (Across entire project)
    4. submission_memory (Frozen at submission)
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        version_tag: str = "v1",
        mode: str = "active"  # "active" or "submission"
    ):
        self.version_tag = version_tag
        self.mode = mode
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "experiments", "hierarchical_memory"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

        self.run_memory = RunMemory()
        self.game_memory = GameMemory(storage_dir=self.storage_dir)
        self.global_memory = GlobalMemory(storage_dir=self.storage_dir)
        self.submission_memory: Optional[SubmissionMemory] = None

        if self.mode == "active":
            # Load existing cross-version and project-wide memory if available
            self.game_memory.import_json(os.path.join(self.storage_dir, "game_memory.json"))
            self.global_memory.import_json(os.path.join(self.storage_dir, "global_memory.json"))
        elif self.mode == "submission":
            sub_path = os.path.join(self.storage_dir, "submission_memory.json")
            if os.path.exists(sub_path):
                self.submission_memory = SubmissionMemory.load_frozen_snapshot(sub_path)

    def distill_and_persist_run(
        self,
        game_id: str,
        task_solved: bool,
        verified_hypotheses: List[Any],
        falsified_signatures: List[str]
    ) -> DistilledRunSummary:
        """
        Distills current run memory into game and global layers,
        then purges run memory so raw transcripts never spill into future contexts.
        """
        if self.mode == "submission":
            # In submission mode, memories are strictly immutable
            self.run_memory.clear()
            return DistilledRunSummary(
                run_id="frozen_run",
                steps_taken=0,
                task_solved=task_solved,
                verified_rules=[],
                falsified_rule_signatures=[],
                contingency_deltas={},
                net_process_reward=0.0
            )

        # 1. Distill run memory
        summary = self.run_memory.distill(
            run_id=f"run_{int(time.time())}",
            task_solved=task_solved,
            verified_hypotheses=verified_hypotheses,
            falsified_signatures=falsified_signatures
        )

        # 2. Update Layer 2: GameMemory (Across versions)
        self.game_memory.absorb_run_distillation(
            game_id=game_id,
            version=self.version_tag,
            summary=summary
        )

        # 3. Update Layer 3: GlobalMemory (Across project)
        for r in summary.verified_rules:
            self.global_memory.update_condition_prior(r["condition"], success=True)
        for sig in summary.falsified_rule_signatures:
            cond = sig.split(":")[0] if ":" in sig else "unknown"
            self.global_memory.update_condition_prior(cond, success=False)

        # 4. Persist to disk
        self.game_memory.export_json(os.path.join(self.storage_dir, "game_memory.json"))
        self.global_memory.export_json(os.path.join(self.storage_dir, "global_memory.json"))

        # 5. Clear Layer 1: RunMemory (Prevent context overflow & transcript dumping)
        self.run_memory.clear()

        return summary

    def compile_submission_artifact(self, output_filepath: Optional[str] = None) -> SubmissionMemory:
        """Compiles and freezes memory for competition submission."""
        filepath = output_filepath or os.path.join(self.storage_dir, "submission_memory.json")
        sub_mem = SubmissionMemory.compile_from_active_memories(
            game_memory=self.game_memory,
            global_memory=self.global_memory,
            version_tag=self.version_tag
        )
        sub_mem.export_frozen_snapshot(filepath)
        self.submission_memory = sub_mem
        return sub_mem
