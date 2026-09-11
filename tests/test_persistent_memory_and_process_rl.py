"""
Unit Tests for Persistent Cross-Version Memory Bank & Verifiable Process-Reward RL
Verifies:
1. Version 1 writes learned causal hypotheses to persistent memory.
2. Version 2 loads v1 memory and warm-starts with boosted priors.
3. Version 3 and v4 continuously update cumulative knowledge.
4. Verifiable process rewards reward verified rules and penalize prediction errors.
"""

import os
import shutil
import tempfile
import unittest

from world_model_lab.core.types import Action, ConditionType, EffectType
from world_model_lab.environment.grid_lab import GridLabEnv, HiddenRule
from world_model_lab.memory.persistent_bank import PersistentMemoryBank
from world_model_lab.agents.process_rl_agent import ProcessRewardRLAgent, VerifiableProcessRewardEngine


class TestPersistentMemoryAndProcessRL(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.rule = HiddenRule(
            rule_id="r_test",
            condition=ConditionType.ADJACENCY,
            cause_color="red",
            trigger_color="green",
            effect=EffectType.STATE_CHANGE,
            target_color="blue",
            target_new_state="active"
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_verifiable_process_reward_engine(self):
        engine = VerifiableProcessRewardEngine()
        # Case 1: Normal step without error or verification
        r, b = engine.compute_reward(task_solved=False, verified_hypotheses_count=0, prediction_discrepancies_count=0)
        self.assertAlmostEqual(r, -0.01)

        # Case 2: Step with a verified causal rule
        r, b = engine.compute_reward(task_solved=False, verified_hypotheses_count=1, prediction_discrepancies_count=0)
        self.assertAlmostEqual(r, 1.99) # 2.0 - 0.01

        # Case 3: Step with prediction error
        r, b = engine.compute_reward(task_solved=False, verified_hypotheses_count=0, prediction_discrepancies_count=1)
        self.assertAlmostEqual(r, -1.01) # -1.0 - 0.01

        # Case 4: Task solved
        r, b = engine.compute_reward(task_solved=True, verified_hypotheses_count=1, prediction_discrepancies_count=0)
        self.assertAlmostEqual(r, 11.99) # 10.0 + 2.0 - 0.01

    def test_cross_version_memory_transfer(self):
        bank = PersistentMemoryBank(storage_dir=self.test_dir)

        # ── RUN VERSION 1 ──
        agent_v1 = ProcessRewardRLAgent(
            grid_size=7,
            version_tag="v1",
            import_prior_versions=False,
            memory_bank=bank,
            seed=1
        )
        env_v1 = GridLabEnv(size=7, max_steps=10, rules=[self.rule], seed=1)
        obs = env_v1.reset()

        # Run 5 steps to verify rules
        for _ in range(5):
            act, r, b = agent_v1.process_step_with_verifiable_reward(obs)
            res = env_v1.step(act)
            obs = res.obs

        # Save v1 memory
        v1_path = agent_v1.finalize_and_save_run(games_evaluated=["game_1"], total_episodes=1, solved_episodes=1)
        self.assertTrue(os.path.exists(v1_path))
        self.assertTrue(len(bank.verified_rules) > 0)

        # ── RUN VERSION 2 (IMPORTS V1) ──
        bank_v2 = PersistentMemoryBank(storage_dir=self.test_dir)
        agent_v2 = ProcessRewardRLAgent(
            grid_size=7,
            version_tag="v2",
            import_prior_versions=True,
            memory_bank=bank_v2,
            seed=2
        )
        env_v2 = GridLabEnv(size=7, max_steps=10, rules=[self.rule], seed=2)
        obs_v2 = env_v2.reset()
        agent_v2.select_action(obs_v2)

        # Check that agent_v2 has non-uniform boosted hypotheses from v1
        confidences = [h.confidence for h in agent_v2.hypothesis_manager.hypotheses]
        self.assertGreater(len(confidences), 0)
        self.assertFalse(all(abs(c - confidences[0]) < 1e-6 for c in confidences),
                         "Version 2 should have non-uniform prior weights learned from Version 1")

        # Save v2 memory
        v2_path = agent_v2.finalize_and_save_run(games_evaluated=["game_2"], total_episodes=1, solved_episodes=1)
        self.assertTrue(os.path.exists(v2_path))

        # ── RUN VERSION 3 & 4 CUMULATIVE MERGE ──
        bank_v4 = PersistentMemoryBank(storage_dir=self.test_dir)
        loaded = bank_v4.import_all_versions()
        self.assertGreaterEqual(loaded, 2, "Version 4 should import both v1 and v2 memory banks")


if __name__ == "__main__":
    unittest.main()
