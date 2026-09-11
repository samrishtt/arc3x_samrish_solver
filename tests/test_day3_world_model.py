from world_model_lab.core.types import Action, Position, ConditionType, EffectType
from world_model_lab.environment.grid_lab import GridLabEnv
from world_model_lab.world_model.hypothesis_manager import HypothesisManager
from world_model_lab.world_model.simulator import CounterfactualSimulator

def test_hypothesis_generation_and_simulation():
    env = GridLabEnv(size=7)
    obs = env.reset()

    manager = HypothesisManager()
    hypotheses = manager.generate_candidate_hypotheses(obs)

    assert len(hypotheses) > 0
    # Should have hypotheses covering multiple targets (blue, yellow, etc.)
    targets = set(h.effect_target_color for h in hypotheses)
    assert "blue" in targets
    assert "yellow" in targets

    simulator = CounterfactualSimulator(grid_size=7)

    # Test rollout of an action
    rollout = simulator.simulate_forward(obs, Action.INTERACT, hypotheses[0])
    assert rollout.action == Action.INTERACT

    # Compute multi-entity disagreement across all actions
    disagreements = {
        act: simulator.compute_disagreement(obs, act, hypotheses)
        for act in Action.all_actions()
    }
    assert all(d >= 0.0 for d in disagreements.values())
    print("Day 3 world model and simulator test passed successfully!")

if __name__ == "__main__":
    test_hypothesis_generation_and_simulation()
