import unittest
from world_model_lab.core.types import Action, ConditionType, EffectType, Position
from world_model_lab.environment.grid_lab import GridLabEnv, HiddenRule
from world_model_lab.agents.baseline_agent import BaselineAgent

def test_grid_env_initialization():
    env = GridLabEnv(size=7)
    obs = env.reset()
    assert obs.grid_shape == (7, 7)
    assert len(obs.entities) >= 4
    assert obs.agent_pos.r == 1 and obs.agent_pos.c == 1

def test_hidden_rule_trigger():
    # Hidden rule: Red adjacent to Green -> Blue becomes active
    rule = HiddenRule(
        rule_id="r1",
        condition=ConditionType.ADJACENCY,
        cause_color="red",
        trigger_color="green",
        effect=EffectType.STATE_CHANGE,
        target_color="blue",
        target_new_state="active"
    )
    env = GridLabEnv(size=7, rules=[rule])
    obs = env.reset()

    blue = obs.get_entity_by_color("blue")
    assert blue is not None
    assert blue.state == "dormant"

    # Red is at (3, 2), Green is at (3, 4).
    # Move Green block to (3, 3) in environment so they become adjacent.
    green = next(e for e in env.entities if e.color == "green")
    green.pos = Position(3, 3)

    # Step WAIT to evaluate rules
    res = env.step(Action.WAIT)
    blue_updated = res.obs.get_entity_by_color("blue")
    assert blue_updated.state == "active"
    assert any("STATE_CHANGE:blue:active" in e for e in res.obs.events)

def test_rule_mutation():
    rule = HiddenRule(
        rule_id="r1",
        condition=ConditionType.ADJACENCY,
        cause_color="red",
        trigger_color="green",
        effect=EffectType.STATE_CHANGE,
        target_color="blue",
        target_new_state="active"
    )
    env = GridLabEnv(size=7, rules=[rule])
    env.reset()

    # Mutate target from blue to yellow
    env.mutate_rule(old_target_color="blue", new_target_color="yellow")

    red = env.entities[1]  # red
    green = env.entities[2]  # green
    green.pos = Position(red.pos.r, red.pos.c + 1)  # make adjacent

    res = env.step(Action.WAIT)
    yellow = res.obs.get_entity_by_color("yellow")
    blue = res.obs.get_entity_by_color("blue")
    assert yellow.state == "active"
    assert blue.state == "dormant"

def test_baseline_agent_run():
    env = GridLabEnv(size=7)
    agent = BaselineAgent(seed=42)
    obs = env.reset()

    for _ in range(10):
        action = agent.select_action(obs)
        res = env.step(action)
        obs = res.obs
    assert agent.step_count == 10
