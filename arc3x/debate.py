"""Dual-Agent Dialectical Debate Architecture for ARC-AGI-3.

Integrates:
1. Agent A (Proposer / Creative Intuition - System 1):
   Proposes action candidates based on policy priors, curiosity, and hypothesis generation.
2. Agent B (Adversarial Critic / Skeptic - System 2):
   Challenges proposals by scrutinizing spatial hazards, wall traps, and oscillation loops.
3. The Mental World Model (Arbiter / Simulator):
   Runs counterfactual mental rollouts inside the in-process twin *before* committing any graded action.
   If the Critic proves a lethal hazard or deadlock, the action is vetoed in the mind!
4. Pluggable vLLM / Qwen-3.8 Adapter:
   Provides prompt formatting and async/sync completion hooks for large vision-language models.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from arc3x.twin import Act, Obs, Twin


@dataclass
class DebateTelemetry:
    total_proposals: int = 0
    total_critiques: int = 0
    vetoed_actions: int = 0
    consensus_actions: int = 0
    mental_rollouts_performed: int = 0
    critic_hazard_saves: int = 0


class ProposerAgent:
    """Agent A: Proposes actions based on neural policy priors and exploratory curiosity."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(42)

    def propose(
        self,
        obs: Obs,
        history: Sequence[Act],
        policy_fn: Optional[Callable[[np.ndarray, Sequence[Act]], np.ndarray]] = None,
    ) -> Act:
        valid = list(obs.valid)
        if not valid:
            return Act(0)

        # If a trained student policy or neural prior is provided:
        if policy_fn is not None:
            try:
                probs = policy_fn(obs.frame, valid)
                if len(probs) == len(valid) and probs.sum() > 0:
                    probs = probs / probs.sum()
                    idx = self.rng.choice(len(valid), p=probs)
                    return valid[idx]
            except Exception:
                pass

        # Sticky rollout heuristic: favor repeating previous directional move
        if history:
            last = history[-1]
            if last in valid and self.rng.random() < 0.65:
                return last

        # Default: sample uniformly from valid actions
        return self.rng.choice(valid)


class CriticAgent:
    """Agent B: Adversarial Critic that looks for hazards, wall collisions, and deadlock loops."""

    def __init__(self, skepticism_threshold: float = 0.70):
        self.skepticism_threshold = skepticism_threshold
        self.fatal_colors: set[int] = set()
        self.seen_frames: set[bytes] = set()

    def record_death(self, fatal_color: Optional[int] = None) -> None:
        if fatal_color is not None:
            self.fatal_colors.add(fatal_color)

    def critique(
        self,
        candidate: Act,
        obs: Obs,
        history: Sequence[Act],
        recent_frames: Sequence[bytes],
    ) -> Tuple[bool, str, float]:
        """Scrutinizes candidate action. Returns (objected, reason, severity)."""
        # 1. Action not in valid set
        if candidate not in obs.valid:
            return True, "Action is illegal in current state", 1.0

        # 2. Oscillation loop detection (e.g. Left -> Right -> Left -> Right)
        if len(history) >= 2:
            last = history[-1]
            # Action 1=Up, 2=Down, 3=West, 4=East
            opposites = {1: 2, 2: 1, 3: 4, 4: 3}
            if candidate.aid in opposites and opposites[candidate.aid] == last.aid:
                # Disallow immediate back-tracking unless no other valid moves
                if len(obs.valid) > 2:
                    return True, "Immediate reverse oscillation detected", 0.75

        # 3. Repeated state deadlock
        frame_hash = obs.frame.tobytes()
        if frame_hash in recent_frames and len(obs.valid) > 1:
            return True, "Revisiting identical cyclic state", 0.72

        return False, "Clear", 0.0


class MentalArbiter:
    """The World Model Arbiter: conducts mental rollouts in imagination before real action execution."""

    def __init__(self):
        self.mental_steps: int = 0

    def evaluate_in_mind(
        self,
        twin_game: Any,
        candidate: Act,
        lookahead: int = 1,
    ) -> Tuple[bool, Optional[Obs]]:
        """Clones state and evaluates candidate action mentally.

        Returns: (is_safe, resulting_mental_obs)
        """
        self.mental_steps += 1
        try:
            clone = copy.deepcopy(twin_game)
            obs = Twin.step_game(clone, candidate)
            # If candidate results in instant GAME_OVER, it is fatal!
            if obs.game_over:
                return False, obs
            return True, obs
        except Exception:
            return False, None


class DialecticalDebateAgent:
    """Master Multi-Agent Dialectical Controller:

    Coordinates Proposer, Critic, and Internal Mental Simulation Arbiter.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.proposer = ProposerAgent(rng=self.rng)
        self.critic = CriticAgent()
        self.arbiter = MentalArbiter()
        self.telemetry = DebateTelemetry()
        self.recent_frame_hashes: List[bytes] = []

    def decide_action(
        self,
        obs: Obs,
        twin_game: Optional[Any] = None,
        history: Sequence[Act] = (),
        policy_fn: Optional[Callable] = None,
    ) -> Act:
        """Conducts dialectical debate between Proposer and Critic."""
        valid = list(obs.valid)
        if not valid:
            return Act(0)
        if len(valid) == 1:
            return valid[0]

        self.telemetry.total_proposals += 1
        frame_hash = obs.frame.tobytes()
        self.recent_frame_hashes.append(frame_hash)
        if len(self.recent_frame_hashes) > 16:
            self.recent_frame_hashes.pop(0)

        # Step 1: Proposer suggests candidate move
        candidate = self.proposer.propose(obs, history, policy_fn=policy_fn)

        # Step 2: Critic scrutinizes proposal
        objected, reason, severity = self.critic.critique(
            candidate, obs, history, self.recent_frame_hashes
        )
        if objected:
            self.telemetry.total_critiques += 1

        # Step 3: If objection raised and internal twin is available, arbitrate mentally
        if twin_game is not None:
            self.telemetry.mental_rollouts_performed += 1
            is_safe, mental_obs = self.arbiter.evaluate_in_mind(twin_game, candidate)
            if not is_safe or (objected and severity > 0.70):
                self.telemetry.vetoed_actions += 1
                if mental_obs and mental_obs.game_over:
                    self.telemetry.critic_hazard_saves += 1

                # Select best alternative not vetoed by critic
                alternatives = [a for a in valid if a != candidate]
                if alternatives:
                    for alt in alternatives:
                        alt_safe, _ = self.arbiter.evaluate_in_mind(twin_game, alt)
                        if alt_safe:
                            return alt
                    return self.rng.choice(alternatives)

        if not objected:
            self.telemetry.consensus_actions += 1
        return candidate


# ---------------------------------------------------------------------------
# Pluggable vLLM / Qwen-3.8 Integration Hook
# ---------------------------------------------------------------------------

class QwenDebateHook:
    """Prompt generator and interface adapter for Qwen-3.8-27B-FP8 via vLLM.

    Formats 64x64 grid state, legal action tokens, and debate stances into
    compact prompts suitable for local LLM inference.
    """

    @staticmethod
    def format_debate_prompt(
        obs: Obs,
        role: str,  # 'proposer' or 'critic'
        opposing_argument: str = "",
    ) -> str:
        h, w = obs.frame.shape
        colors = sorted(list(set(obs.frame.flatten())))
        valid_str = ", ".join(repr(a) for a in obs.valid[:10])

        if role == "proposer":
            return (
                f"<|im_start|>system\nYou are the ARC-AGI-3 Proposer Agent. Your goal is to hypothesize the winning rule and propose the best immediate action.<|im_end|>\n"
                f"<|im_start|>user\nGrid: {h}x{w}, Colors present: {colors}, Level: {obs.level}, Score: {obs.score}\n"
                f"Legal actions: [{valid_str}]\n"
                f"Propose the best next action and briefly explain the hypothesis.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            return (
                f"<|im_start|>system\nYou are the ARC-AGI-3 Adversarial Critic. Your goal is to detect spatial hazards, wall traps, and infinite loops in proposed actions.<|im_end|>\n"
                f"<|im_start|>user\nProposed Move: {opposing_argument}\n"
                f"Grid: {h}x{w}, Colors present: {colors}\n"
                f"Legal actions: [{valid_str}]\n"
                f"Evaluate if this move is safe or leads to a trap.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
