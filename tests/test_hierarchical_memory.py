"""
Unit tests for the 4-Tier Hierarchical Memory Architecture:
1. run_memory (One run)
2. game_memory (Across versions)
3. global_memory (Across entire project)
4. submission_memory (Frozen at submission)
"""

import os
import shutil
import tempfile
import unittest

from world_model_lab.core.types import Hypothesis, ConditionType, EffectType
from world_model_lab.memory.hierarchical_memory import (
    HierarchicalMemoryManager,
    RunMemory,
    GameMemory,
    GlobalMemory,
    SubmissionMemory
)


class TestHierarchicalMemory(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_run_memory_distillation_and_cleanup(self):
        """Test Layer 1: records single run steps, distills structured summary, purges raw data."""
        run_mem = RunMemory()
        run_mem.record_step(1, "MOVE_RIGHT", {"red": "moved"}, False, -0.01)
        run_mem.record_step(2, "INTERACT", {"blue": "activated"}, False, 10.0)
        run_mem.record_contingency("touch:red:blue", "activate")

        self.assertEqual(len(run_mem.step_transitions), 2)
        self.assertAlmostEqual(run_mem.cumulative_reward, 9.99)

        h = Hypothesis(
            id="H_test",
            condition=ConditionType.TOUCH,
            cause_color="red",
            trigger_color=None,
            effect=EffectType.STATE_CHANGE,
            effect_target_color="blue",
            effect_new_state="active",
            confidence=0.92,
            success_count=4,
            falsification_count=0
        )

        summary = run_mem.distill(
            run_id="run_001",
            task_solved=True,
            verified_hypotheses=[h],
            falsified_signatures=["adjacency:green:blue->blue:active"]
        )

        self.assertEqual(summary.steps_taken, 2)
        self.assertTrue(summary.task_solved)
        self.assertEqual(len(summary.verified_rules), 1)
        self.assertEqual(len(summary.falsified_rule_signatures), 1)

        # Purge raw memory
        run_mem.clear()
        self.assertEqual(len(run_mem.step_transitions), 0)
        self.assertEqual(run_mem.cumulative_reward, 0.0)

    def test_cross_version_game_memory_and_global_memory(self):
        """Test Layer 2 (Across versions) & Layer 3 (Across project)."""
        mgr_v1 = HierarchicalMemoryManager(storage_dir=self.test_dir, version_tag="v1")

        h = Hypothesis(
            id="H_test",
            condition=ConditionType.TOUCH,
            cause_color="red",
            trigger_color=None,
            effect=EffectType.STATE_CHANGE,
            effect_target_color="blue",
            effect_new_state="active",
            confidence=0.95,
            success_count=5,
            falsification_count=0
        )

        h2 = Hypothesis(
            id="H_test2",
            condition=ConditionType.TOUCH,
            cause_color="green",
            trigger_color=None,
            effect=EffectType.STATE_CHANGE,
            effect_target_color="blue",
            effect_new_state="active",
            confidence=0.88,
            success_count=3,
            falsification_count=0
        )

        # Distill v1 run
        summary = mgr_v1.distill_and_persist_run(
            game_id="puzzle_alpha",
            task_solved=True,
            verified_hypotheses=[h, h2],
            falsified_signatures=["touch:yellow:blue->blue:active"]
        )

        # Raw run memory was purged immediately
        self.assertEqual(len(mgr_v1.run_memory.step_transitions), 0)

        # Load v2 manager pointing to the same storage
        mgr_v2 = HierarchicalMemoryManager(storage_dir=self.test_dir, version_tag="v2")
        priors = mgr_v2.game_memory.get_game_priors("puzzle_alpha")

        # The verified rule signature should have boosted prior in v2
        rule_sig = "touch:red:none->blue:active"
        self.assertIn(rule_sig, priors)
        self.assertGreater(priors[rule_sig], 1.0)

        # Global memory should record touch success
        weight = mgr_v2.global_memory.get_condition_weight("touch")
        self.assertGreater(weight, 0.5)

    def test_submission_memory_frozen_snapshot(self):
        """Test Layer 4: Compiles frozen immutable snapshot for competition."""
        mgr = HierarchicalMemoryManager(storage_dir=self.test_dir, version_tag="v3")
        h = Hypothesis(
            id="H_verified",
            condition=ConditionType.TOUCH,
            cause_color="red",
            trigger_color=None,
            effect=EffectType.STATE_CHANGE,
            effect_target_color="blue",
            effect_new_state="active",
            confidence=0.99,
            success_count=10,
            falsification_count=0
        )
        mgr.distill_and_persist_run(
            game_id="competition_level_1",
            task_solved=True,
            verified_hypotheses=[h],
            falsified_signatures=[]
        )

        sub_mem = mgr.compile_submission_artifact()
        self.assertTrue(sub_mem.is_frozen)

        # Query read-only prior
        rule_sig = "touch:red:none->blue:active"
        boost = sub_mem.query_prior("competition_level_1", rule_sig)
        self.assertGreater(boost, 1.0)

        rules = sub_mem.get_verified_rules_for_game("competition_level_1")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["rule_signature"], rule_sig)

        # In submission mode, run memory does not mutate stored knowledge
        mgr_sub = HierarchicalMemoryManager(storage_dir=self.test_dir, mode="submission")
        res = mgr_sub.distill_and_persist_run("competition_level_1", True, [], [])
        self.assertEqual(res.run_id, "frozen_run")


if __name__ == "__main__":
    unittest.main()
