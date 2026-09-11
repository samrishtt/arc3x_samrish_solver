"""
Leak-Free Ablation Study (v2)
- No target_color in agent constructors
- Systematic component removal with proper monkey-patching
- Raw per-seed data logged to .jsonl
"""
import json
import os
import time
import numpy as np
from typing import List, Dict, Any

from world_model_lab.core.types import ConditionType, EffectType, Action
from world_model_lab.environment.grid_lab import GridLabEnv, HiddenRule
from world_model_lab.agents.active_agent import ActiveWorldModelAgent
from world_model_lab.experimentation.selector import SelectionStrategy

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "experiments", "data")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _create_env(seed: int, max_steps: int = 50, randomize_layout: bool = True) -> GridLabEnv:
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


class AblationStudy:
    """
    Leak-Free Ablation Study:
    Systematically removes one component at a time from the full agent
    to measure each component's causal contribution.

    Configurations:
    1. Full System (all components active)
    2. −Active Selection (random actions, keep everything else)
    3. −Uncertainty (collapse to point estimate)
    4. −Mental Simulator (zero disagreement signal)
    5. −Memory (no episodic/semantic storage)
    6. −Self-Correction (no prediction-error diagnosis)
    """

    def __init__(self, seeds: int = 100, max_steps: int = 50):
        self.seeds = seeds
        self.max_steps = max_steps

    def run_all_ablations(self) -> Dict[str, Dict[str, Any]]:
        configurations = [
            "Full System",
            "No Active Selection",
            "No Uncertainty (Point Est)",
            "No Mental Simulator",
            "No Memory",
            "No Self-Correction",
        ]
        results = {}
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        for config in configurations:
            print(f"  [Ablation: {config}] Running {self.seeds} seeds...")
            steps_list = []
            solved_count = 0
            trajectory_records = []

            for s in range(self.seeds):
                env = _create_env(seed=s, max_steps=self.max_steps)
                agent = ActiveWorldModelAgent(grid_size=7, seed=s)
                obs = env.reset(seed=s)
                last_reward = 0.0

                # Apply ablation
                if config == "No Active Selection":
                    rng = np.random.RandomState(s)
                    original_select = agent.experiment_selector.select_action
                    agent.experiment_selector.select_action = \
                        lambda *a, _rng=rng, **kw: (_rng.choice(Action.all_actions()), "ABLATED_RANDOM", 0.0)

                elif config == "No Uncertainty (Point Est)":
                    def point_est_update(hypothesis_id, verified=False, lr=0.5,
                                         _agent=agent):
                        for h in _agent.hypothesis_manager.hypotheses:
                            h.confidence = 1.0 if h.id == hypothesis_id else 0.0
                    agent.hypothesis_manager.update_hypothesis = point_est_update

                elif config == "No Mental Simulator":
                    agent.simulator.compute_disagreement = lambda *a, **kw: 0.0

                elif config == "No Memory":
                    agent.memory.record_transition = lambda *a, **kw: None

                elif config == "No Self-Correction":
                    agent.diagnostic_engine.diagnose_and_revise = \
                        lambda *a, **kw: None

                solved = False
                steps_taken = self.max_steps

                for step in range(1, self.max_steps + 1):
                    act = agent.select_action(obs, last_reward=last_reward)
                    res = env.step(act)
                    obs = res.obs
                    last_reward = res.reward

                    # Check if ANY entity activated (leak-free)
                    activated = False
                    for e in obs.entities:
                        if not e.is_agent and e.state == "active":
                            activated = True
                            break

                    if activated:
                        solved = True
                        steps_taken = step
                        break

                if solved:
                    solved_count += 1
                steps_list.append(steps_taken)

                trajectory_records.append({
                    "experiment": "ablation",
                    "config": config,
                    "seed": s,
                    "steps": steps_taken,
                    "solved": solved,
                    "timestamp": timestamp
                })

            # Log raw data
            _ensure_data_dir()
            path = os.path.join(DATA_DIR, f"exp6_ablation_{timestamp}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                for r in trajectory_records:
                    f.write(json.dumps(r) + "\n")

            arr = np.array(steps_list)
            ci_95 = 1.96 * np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

            results[config] = {
                "success_rate": solved_count / self.seeds,
                "mean_steps": float(np.mean(arr)),
                "median_steps": float(np.median(arr)),
                "std_steps": float(np.std(arr)),
                "ci_95": float(ci_95),
                "n_seeds": self.seeds
            }

        return results
