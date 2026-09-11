from typing import List, Dict, Tuple, Optional
from world_model_lab.core.types import (
    Action, Observation, PredictionError, Hypothesis, MentalRollout
)
from world_model_lab.perception.extractor import StateTransitionDiff
from world_model_lab.world_model.hypothesis_manager import HypothesisManager

class PredictionErrorDiagnosticEngine:
    """
    All-Entity Prediction-Error Diagnostic Engine:
    - Compares mental simulation predictions against actual environmental outcomes.
    - Diagnoses which hypothesis assumptions were falsified vs. verified across all entities.
    - Triggers autonomous self-correction and model revision upon prediction error.
    """

    def __init__(self, hypothesis_manager: HypothesisManager):
        self.hypothesis_manager = hypothesis_manager
        self.error_history: List[PredictionError] = []
        self.peak_confidence: float = 0.0

    def reset(self) -> None:
        self.error_history.clear()
        self.peak_confidence = 0.0

    def diagnose_and_revise(
        self,
        step: int,
        action: Action,
        hypotheses: List[Hypothesis],
        rollouts: Dict[str, MentalRollout],
        diff: StateTransitionDiff,
        obs: Observation
    ) -> PredictionError:
        actual_events = diff.events
        actual_state_changes = set(e for e in actual_events if e.startswith("STATE_CHANGE"))

        discrepancy_detected = False
        faulty_hypotheses: List[str] = []
        verified_hypotheses: List[str] = []
        notes = []

        for h in hypotheses:
            rollout = rollouts.get(h.id)
            if not rollout:
                continue

            pred_state_changes = set(e for e in rollout.predicted_events if e.startswith("STATE_CHANGE"))

            if pred_state_changes and not actual_state_changes:
                # False Positive: predicted state change, but reality was unchanged
                discrepancy_detected = True
                faulty_hypotheses.append(h.id)
                self.hypothesis_manager.update_hypothesis(h.id, verified=False, lr=0.8)
                notes.append(f"Falsified {h.id}: predicted {pred_state_changes} but none occurred.")

            elif pred_state_changes and actual_state_changes:
                if pred_state_changes.intersection(actual_state_changes):
                    # True Positive: accurate prediction
                    verified_hypotheses.append(h.id)
                    self.hypothesis_manager.update_hypothesis(h.id, verified=True, lr=0.9)
                    notes.append(f"Verified {h.id}: predicted {pred_state_changes} matched reality.")
                else:
                    # Predicted wrong effect
                    discrepancy_detected = True
                    faulty_hypotheses.append(h.id)
                    self.hypothesis_manager.update_hypothesis(h.id, verified=False, lr=0.8)
                    notes.append(f"Falsified {h.id}: predicted {pred_state_changes} but actual was {actual_state_changes}.")

            elif actual_state_changes and not pred_state_changes:
                # Target entity changed state, but this hypothesis did not anticipate it
                # If this hypothesis specifically claims to govern that target entity, penalize it
                for sc in actual_state_changes:
                    parts = sc.split(":")
                    if len(parts) >= 2 and h.effect_target_color == parts[1]:
                        self.hypothesis_manager.update_hypothesis(h.id, verified=False, lr=0.5)

        # Autonomous Self-Correction / Model Recovery:
        # If the agent previously had high confidence (e.g. peak_confidence >= 0.25)
        # and best hypothesis collapsed below uniform (1/k), OR all hypotheses are repeatedly falsified:
        best_h = self.hypothesis_manager.get_best_hypothesis()
        if best_h and best_h.confidence > self.peak_confidence:
            self.peak_confidence = best_h.confidence

        k = len(hypotheses)
        all_falsified = (k > 0 and all(h.falsification_count >= 2 for h in hypotheses))
        collapse_after_convergence = (self.peak_confidence >= 0.25 and best_h is not None and best_h.confidence < (1.0 / k))

        if all_falsified or collapse_after_convergence:
            notes.append("AUTONOMOUS MODEL RECOVERY: Model invalidated; re-generating candidate hypotheses.")
            self.hypothesis_manager.generate_candidate_hypotheses(obs)
            self.peak_confidence = 0.0

        error_record = PredictionError(
            step=step,
            action=action,
            predicted_events=[e for r in rollouts.values() for e in r.predicted_events],
            actual_events=actual_events,
            discrepancy=discrepancy_detected,
            faulty_hypothesis_id=",".join(faulty_hypotheses) if faulty_hypotheses else None,
            attribution_notes="; ".join(notes)
        )
        self.error_history.append(error_record)
        return error_record
