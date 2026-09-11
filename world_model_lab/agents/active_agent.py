from typing import List, Dict, Tuple, Optional, Any
from world_model_lab.core.types import (
    Action, Observation, Hypothesis, MentalRollout, PredictionError
)
from world_model_lab.perception.extractor import PerceptionModule
from world_model_lab.memory.store import EpisodicSemanticMemory
from world_model_lab.world_model.hypothesis_manager import HypothesisManager
from world_model_lab.world_model.simulator import CounterfactualSimulator
from world_model_lab.experimentation.selector import ActiveExperimentSelector, SelectionStrategy
from world_model_lab.diagnostics.revision import PredictionErrorDiagnosticEngine

class ActiveWorldModelAgent:
    """
    Target-Agnostic Active World-Model Agent (Option 1: All-Entity Dynamics):
    - Begins without target identity, target color, or pre-defined knowledge of important entities.
    - Represents all observable entities and generic state/spatial relations.
    - Maintains an internal world model with competing hypotheses across all entities.
    - Mentally simulates counterfactual rollouts across candidate actions.
    - Actively selects experiments based on predictive disagreement or configured strategy.
    - Diagnoses prediction errors against reality and revises world-model hypotheses autonomously.
    """

    def __init__(
        self,
        grid_size: int = 7,
        strategy: SelectionStrategy = SelectionStrategy.PREDICTIVE_DISAGREEMENT,
        confidence_threshold: float = 0.75,
        seed: int = 42
    ):
        self.grid_size = grid_size
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.seed = seed

        # Modules
        self.perception = PerceptionModule()
        self.memory = EpisodicSemanticMemory()
        self.hypothesis_manager = HypothesisManager()
        self.simulator = CounterfactualSimulator(grid_size=grid_size)
        self.experiment_selector = ActiveExperimentSelector(
            simulator=self.simulator,
            strategy=strategy,
            confidence_threshold=confidence_threshold,
            seed=seed
        )
        self.diagnostic_engine = PredictionErrorDiagnosticEngine(
            hypothesis_manager=self.hypothesis_manager
        )

        # Internal state
        self.initialized = False
        self.last_action: Optional[Action] = None
        self.last_rollouts: Dict[str, MentalRollout] = {}
        self.last_obs: Optional[Observation] = None
        self.reward_history: List[float] = []
        self.step_count = 0

    def reset(self) -> None:
        self.perception.reset()
        self.memory.reset()
        self.hypothesis_manager.reset()
        self.diagnostic_engine.error_history.clear()
        self.initialized = False
        self.last_action = None
        self.last_rollouts.clear()
        self.last_obs = None
        self.reward_history.clear()
        self.step_count = 0

    def select_action(self, obs: Observation, last_reward: float = 0.0) -> Action:
        self.step_count += 1
        if self.step_count > 1:
            self.reward_history.append(last_reward)

        # 1. Perception & Temporal diff extraction from previous step
        diff = self.perception.extract_diff(obs)
        relations = self.perception.extract_spatial_relations(obs)

        if not self.initialized:
            # Generate all-entity hypothesis space on first observation
            self.hypothesis_manager.generate_candidate_hypotheses(obs)
            self.initialized = True
        else:
            # 2. Prediction-Error Diagnosis & Model Revision across all entities
            if self.last_action is not None and self.last_obs is not None:
                self.memory.record_transition(
                    step=self.step_count - 1,
                    action=self.last_action,
                    prev_obs=self.last_obs,
                    next_obs=obs,
                    diff=diff,
                    spatial_relations=relations,
                    reward=last_reward
                )
                self.diagnostic_engine.diagnose_and_revise(
                    step=self.step_count - 1,
                    action=self.last_action,
                    hypotheses=self.hypothesis_manager.hypotheses,
                    rollouts=self.last_rollouts,
                    diff=diff,
                    obs=obs
                )

        # 3. Competing Hypotheses state
        hypotheses = self.hypothesis_manager.hypotheses
        best_h = self.hypothesis_manager.get_best_hypothesis()

        # 4. Active Experiment Selection
        action, rationale, score = self.experiment_selector.select_action(
            obs=obs,
            hypotheses=hypotheses,
            best_hypothesis=best_h,
            reward_history=self.reward_history
        )

        # 5. Pre-compute mental rollouts for chosen action to evaluate prediction next step
        self.last_rollouts = {
            h.id: self.simulator.simulate_forward(obs, action, h)
            for h in hypotheses
        }
        self.last_action = action
        self.last_obs = obs

        return action

    def get_world_model_state(self) -> Dict[str, Any]:
        best_h = self.hypothesis_manager.get_best_hypothesis()
        return {
            "total_hypotheses": len(self.hypothesis_manager.hypotheses),
            "best_hypothesis": best_h.description() if best_h else "None",
            "best_confidence": best_h.confidence if best_h else 0.0,
            "hypotheses_detail": [
                {"id": h.id, "desc": h.description(), "confidence": round(h.confidence, 4)}
                for h in sorted(self.hypothesis_manager.hypotheses, key=lambda h: h.confidence, reverse=True)[:5]
            ],
            "total_prediction_errors": len([e for e in self.diagnostic_engine.error_history if e.discrepancy])
        }
