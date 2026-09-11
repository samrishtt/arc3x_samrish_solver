from world_model_lab.core.types import Action, Position, ConditionType, EffectType
from world_model_lab.environment.grid_lab import GridLabEnv, HiddenRule
from world_model_lab.perception.extractor import PerceptionModule
from world_model_lab.memory.store import EpisodicSemanticMemory

def test_perception_and_memory():
    env = GridLabEnv(size=7)
    perception = PerceptionModule()
    memory = EpisodicSemanticMemory()

    obs0 = env.reset()
    diff0 = perception.extract_diff(obs0)
    relations0 = perception.extract_spatial_relations(obs0)

    assert len(relations0) > 0
    # Step agent down
    res1 = env.step(Action.DOWN)
    diff1 = perception.extract_diff(res1.obs)
    relations1 = perception.extract_spatial_relations(res1.obs)

    assert "agent" in diff1.moved_entities
    memory.record_transition(
        step=1,
        action=Action.DOWN,
        prev_obs=obs0,
        next_obs=res1.obs,
        diff=diff1,
        spatial_relations=relations0,
        reward=res1.reward
    )

    assert len(memory.episodes) == 1
    assert memory.episodes[0].action == Action.DOWN
    print("Day 2 perception and memory test passed successfully!")

if __name__ == "__main__":
    test_perception_and_memory()
