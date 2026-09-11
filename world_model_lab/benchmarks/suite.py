"""
Leak-Free Benchmark Suite (v2)
- No target_color or target_new_state in agent constructors
- No oracle updates after rule mutation
- Agent discovers everything through prediction error alone
- Supports all 5 selection strategies for fair comparison
- Raw per-seed trajectory logging to .jsonl
"""
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional

from world_model_lab.core.types import ConditionType, EffectType, Action
from world_model_lab.environment.grid_lab import GridLabEnv, HiddenRule
from world_model_lab.agents.baseline_agent import BaselineAgent
from world_model_lab.agents.active_agent import ActiveWorldModelAgent
from world_model_lab.experimentation.selector import SelectionStrategy


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "experiments", "data")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _log_trajectory(filename: str, records: List[Dict[str, Any]]):
    _ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _create_standard_env(seed: int, max_steps: int = 50, randomize_layout: bool = True) -> GridLabEnv:
    rule = HiddenRule(
        rule_id="r_adj_red_green",
        condition=ConditionType.ADJACENCY,
        cause_color="red",
        trigger_color="green",
        effect=EffectType.STATE_CHANGE,
        target_color="blue",
        target_new_state="active"
    )
    return GridLabEnv(size=7, max_steps=max_steps, rules=[rule], randomize_layout=randomize_layout, seed=seed)


def _check_any_entity_activated(obs) -> Optional[str]:
    """Check if any non-agent entity reached 'active' state."""
    for e in obs.entities:
        if not e.is_agent and e.state == "active":
            return e.color
    return None


class BenchmarkSuite:
    """
    Leak-Free Automated Benchmark Suite:
    - Experiment 1: Rule Discovery (all 5 strategies)
    - Experiment 2: Scaling hypothesis space (2/4/8 entities)
    - Experiment 3: Misleading evidence recovery
    - Experiment 4: Rule Mutation (zero notification to agent)
    - Experiment 5: Structural Transfer (colour permutation)
    """

    def __init__(self, seeds: int = 100, max_steps: int = 50):
        self.seeds = seeds
        self.max_steps = max_steps

    # ── EXPERIMENT 1: Core Hypothesis — Strategy Comparison ────────────────

    def run_strategy_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare all 5 selection strategies + baseline on rule discovery."""
        results = {}
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        strategies = [
            ("Baseline_Random", None),
            ("A_Random", SelectionStrategy.RANDOM),
            ("B_Greedy", SelectionStrategy.REACTIVE_GREEDY),
            ("C_Uncertainty", SelectionStrategy.UNCERTAINTY_ONLY),
            ("D_Disagreement", SelectionStrategy.PREDICTIVE_DISAGREEMENT),
            ("E_InfoGain", SelectionStrategy.EXPECTED_INFO_GAIN),
        ]

        for name, strategy in strategies:
            print(f"  [{name}] Running {self.seeds} seeds...")
            steps_list = []
            solved_count = 0
            confidences = []
            trajectory_records = []

            for s in range(self.seeds):
                env = _create_standard_env(seed=s, max_steps=self.max_steps)
                obs = env.reset(seed=s)

                if strategy is None:
                    # Baseline agent
                    agent = BaselineAgent(seed=s)
                else:
                    agent = ActiveWorldModelAgent(
                        grid_size=7, strategy=strategy, seed=s
                    )

                solved = False
                step_taken = self.max_steps

                for step in range(1, self.max_steps + 1):
                    if strategy is None:
                        act = agent.select_action(obs)
                    else:
                        reward = env.step(Action.WAIT).reward if step == 1 else last_reward
                        if step == 1:
                            act = agent.select_action(obs)
                        else:
                            act = agent.select_action(obs, last_reward=last_reward)

                    res = env.step(act)
                    obs = res.obs
                    last_reward = res.reward

                    activated = _check_any_entity_activated(obs)
                    if activated:
                        solved = True
                        step_taken = step
                        break

                if solved:
                    solved_count += 1

                steps_list.append(step_taken)

                conf = 0.0
                if strategy is not None and hasattr(agent, 'hypothesis_manager'):
                    best_h = agent.hypothesis_manager.get_best_hypothesis()
                    conf = best_h.confidence if best_h else 0.0
                confidences.append(conf)

                trajectory_records.append({
                    "experiment": "strategy_comparison",
                    "strategy": name,
                    "seed": s,
                    "steps": step_taken,
                    "solved": solved,
                    "confidence": round(conf, 6),
                    "timestamp": timestamp
                })

            _log_trajectory(f"exp1_strategy_comparison_{timestamp}.jsonl",
                            trajectory_records)

            arr = np.array(steps_list)
            ci_95 = 1.96 * np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

            results[name] = {
                "success_rate": solved_count / self.seeds,
                "mean_steps": float(np.mean(arr)),
                "median_steps": float(np.median(arr)),
                "std_steps": float(np.std(arr)),
                "ci_95": float(ci_95),
                "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "n_seeds": self.seeds
            }

        return results

    # ── EXPERIMENT 4: Rule Mutation — Zero Notification ───────────────────

    def run_rule_mutation(self) -> Dict[str, Any]:
        """
        Agent learns Rule 1 for 20 steps.
        Environment silently mutates to Rule 2.
        Agent receives ZERO notification — must detect via prediction error.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        recovery_steps = []
        recovered_count = 0
        prediction_errors_before_recovery = []
        trajectory_records = []

        learning_phase = 20
        recovery_budget = 30

        for s in range(self.seeds):
            env = _create_standard_env(seed=s, max_steps=learning_phase + recovery_budget)
            agent = ActiveWorldModelAgent(grid_size=7, seed=s)
            obs = env.reset(seed=s)
            last_reward = 0.0

            # Phase 1: Learn Rule 1 for learning_phase steps
            for step in range(1, learning_phase + 1):
                act = agent.select_action(obs, last_reward=last_reward)
                res = env.step(act)
                obs = res.obs
                last_reward = res.reward

            # Reset any activated entities before mutation
            for e in env.entities:
                if e.state == "active":
                    e.state = "dormant"

            # Silently mutate the underlying hidden physics:
            # Rule shifts from ADJACENCY(red, green) -> blue:active
            # to TOUCH(green) -> blue:active.
            # NOTE: Agent receives ZERO notification, ZERO parameter updates.
            rule = env.rules[0]
            rule.condition = ConditionType.TOUCH
            rule.cause_color = "green"
            rule.trigger_color = None
            rule.target_color = "blue"
            rule.target_new_state = "active"

            # Phase 2: Measure recovery
            recovered = False
            steps_to_recover = recovery_budget
            errors_detected = 0

            for step in range(1, recovery_budget + 1):
                act = agent.select_action(obs, last_reward=last_reward)
                res = env.step(act)
                obs = res.obs
                last_reward = res.reward

                # Count prediction errors
                if hasattr(agent, 'diagnostic_engine'):
                    recent = agent.diagnostic_engine.error_history
                    errors_detected = len([e for e in recent if e.discrepancy])

                activated = _check_any_entity_activated(obs)
                if activated == "blue":
                    recovered = True
                    steps_to_recover = step
                    break

            if recovered:
                recovered_count += 1
            recovery_steps.append(steps_to_recover)
            prediction_errors_before_recovery.append(errors_detected)

            trajectory_records.append({
                "experiment": "rule_mutation",
                "seed": s,
                "learning_phase": learning_phase,
                "recovery_steps": steps_to_recover,
                "recovered": recovered,
                "prediction_errors": errors_detected,
                "timestamp": timestamp
            })

        _log_trajectory(f"exp4_rule_mutation_{timestamp}.jsonl", trajectory_records)

        arr = np.array(recovery_steps)
        ci_95 = 1.96 * np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

        return {
            "adaptation_rate": recovered_count / self.seeds,
            "mean_recovery_steps": float(np.mean(arr)),
            "median_recovery_steps": float(np.median(arr)),
            "std_recovery_steps": float(np.std(arr)),
            "ci_95": float(ci_95),
            "mean_prediction_errors": float(np.mean(prediction_errors_before_recovery)),
            "n_seeds": self.seeds
        }

    # ── EXPERIMENT 5: Structural Transfer ─────────────────────────────────

    def run_structural_transfer(self) -> Dict[str, Any]:
        """
        Learn rule with original colours, then transfer to reskinned env.
        Agent must re-discover the same causal structure with new surface features.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        transfer_steps = []
        transfer_solved = 0
        trajectory_records = []

        for s in range(self.seeds):
            env = _create_standard_env(seed=s, max_steps=self.max_steps)
            obs = env.reset(seed=s)

            # Reskin AFTER reset
            color_map = {"red": "purple", "green": "orange", "blue": "cyan"}
            env.reskin(color_map)
            obs = env._get_obs()

            agent = ActiveWorldModelAgent(grid_size=7, seed=s)

            solved = False
            step_taken = self.max_steps
            last_reward = 0.0

            for step in range(1, self.max_steps + 1):
                act = agent.select_action(obs, last_reward=last_reward)
                res = env.step(act)
                obs = res.obs
                last_reward = res.reward

                activated = _check_any_entity_activated(obs)
                if activated:
                    solved = True
                    step_taken = step
                    break

            if solved:
                transfer_solved += 1
            transfer_steps.append(step_taken)

            trajectory_records.append({
                "experiment": "structural_transfer",
                "seed": s,
                "steps": step_taken,
                "solved": solved,
                "timestamp": timestamp
            })

        _log_trajectory(f"exp5_transfer_{timestamp}.jsonl", trajectory_records)

        arr = np.array(transfer_steps)
        ci_95 = 1.96 * np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

        return {
            "transfer_success_rate": transfer_solved / self.seeds,
            "mean_transfer_steps": float(np.mean(arr)),
            "median_transfer_steps": float(np.median(arr)),
            "std_transfer_steps": float(np.std(arr)),
            "ci_95": float(ci_95),
            "n_seeds": self.seeds
        }
