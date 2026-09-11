"""
Superintelligent Process-Reward RL Agent
Implements verifiable process-oriented reinforcement learning:
- Rewarding the entire reasoning process rather than sparse outcome rewards.
- R = +10 (task solved) + 2 (verified causal rule) - 1 (prediction error) - 0.01 (step penalty).
- Integrates with PersistentMemoryBank for continuous cross-version knowledge transfer (v1, v2, v3 -> v4).
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from world_model_lab.core.types import Observation, Action, Hypothesis
from world_model_lab.agents.active_agent import ActiveWorldModelAgent
from world_model_lab.experimentation.selector import SelectionStrategy
from world_model_lab.memory.persistent_bank import PersistentMemoryBank


class VerifiableProcessRewardEngine:
    """
    Computes grounded, verifiable process rewards:
    Encourages active causal discovery and penalizes hallucinations/redundancy.
    """

    def __init__(
        self,
        reward_task_solved: float = 10.0,
        reward_verified_rule: float = 2.0,
        penalty_prediction_error: float = 1.0,
        penalty_step: float = 0.01
    ):
        self.reward_task_solved = reward_task_solved
        self.reward_verified_rule = reward_verified_rule
        self.penalty_prediction_error = penalty_prediction_error
        self.penalty_step = penalty_step

    def compute_reward(
        self,
        task_solved: bool,
        verified_hypotheses_count: int,
        prediction_discrepancies_count: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates process reward decomposition:
        R = R_outcome + 2 * N_verified - 1 * N_falsified - 0.01
        """
        r_outcome = self.reward_task_solved if task_solved else 0.0
        r_verified = self.reward_verified_rule * verified_hypotheses_count
        r_falsified = -self.penalty_prediction_error * prediction_discrepancies_count
        r_step = -self.penalty_step

        total_r = r_outcome + r_verified + r_falsified + r_step
        breakdown = {
            "outcome_reward": r_outcome,
            "verification_reward": r_verified,
            "prediction_error_penalty": r_falsified,
            "step_penalty": r_step,
            "total_process_reward": total_r
        }
        return total_r, breakdown


class ProcessRewardRLAgent(ActiveWorldModelAgent):
    """
    Continuous Learning Process-Reward Agent.
    - Inherits all-entity active world-modeling and counterfactual simulation.
    - Guides policy updates with verifiable process rewards.
    - Connects to PersistentMemoryBank to accumulate knowledge across runs (v1, v2, v3 -> v4).
    """

    def __init__(
        self,
        grid_size: int = 7,
        version_tag: str = "v1",
        import_prior_versions: bool = True,
        strategy: SelectionStrategy = SelectionStrategy.PREDICTIVE_DISAGREEMENT,
        confidence_threshold: float = 0.70,
        memory_bank: Optional[PersistentMemoryBank] = None,
        seed: int = 42
    ):
        super().__init__(
            grid_size=grid_size,
            strategy=strategy,
            confidence_threshold=confidence_threshold,
            seed=seed
        )
        self.version_tag = version_tag
        self.reward_engine = VerifiableProcessRewardEngine()
        self.memory_bank = memory_bank or PersistentMemoryBank()
        self.import_prior_versions = import_prior_versions
        self.cumulative_process_reward = 0.0
        self.process_reward_history: List[float] = []

        # Cross-run memory import
        if import_prior_versions:
            self.memory_bank.import_all_versions()

    def select_action(self, obs: Observation, last_reward: float = 0.0) -> Action:
        if not self.initialized:
            self.hypothesis_manager.generate_candidate_hypotheses(obs)
            if self.import_prior_versions:
                self.memory_bank.apply_prior_to_hypotheses(self.hypothesis_manager.hypotheses)
            self.initialized = True
        return super().select_action(obs, last_reward=last_reward)

    def process_step_with_verifiable_reward(
        self,
        obs: Observation,
        task_solved: bool = False
    ) -> Tuple[Action, float, Dict[str, float]]:
        """
        Selects an action and updates world model using verifiable process rewards.
        """
        # 1. Choose action via predictive disagreement
        act = self.select_action(obs, last_reward=self.cumulative_process_reward)

        # 2. Extract recent verification vs error events from diagnostic engine
        verified_count = 0
        error_count = 0
        if self.diagnostic_engine.error_history:
            recent_err = self.diagnostic_engine.error_history[-1]
            if recent_err.discrepancy:
                error_count += 1
            if recent_err.attribution_notes and "Verified" in recent_err.attribution_notes:
                verified_count += 1

        # 3. Calculate process reward
        step_reward, breakdown = self.reward_engine.compute_reward(
            task_solved=task_solved,
            verified_hypotheses_count=verified_count,
            prediction_discrepancies_count=error_count
        )

        self.cumulative_process_reward += step_reward
        self.process_reward_history.append(step_reward)

        return act, step_reward, breakdown

    def finalize_and_save_run(
        self,
        games_evaluated: Optional[List[str]] = None,
        total_episodes: int = 1,
        solved_episodes: int = 1
    ) -> str:
        """
        Saves current learned causal knowledge to persistent memory for next version.
        """
        saved_path = self.memory_bank.save_run_memory(
            version=self.version_tag,
            hypotheses=self.hypothesis_manager.hypotheses,
            contingencies=self.memory.contingency_table,
            games=games_evaluated or ["procedural_lab"],
            total_episodes=total_episodes,
            solved_episodes=solved_episodes
        )
        return saved_path
