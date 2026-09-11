from world_model_lab.core.types import Action, Position, ConditionType, EffectType
from world_model_lab.environment.grid_lab import GridLabEnv, HiddenRule
from world_model_lab.agents.active_agent import ActiveWorldModelAgent

def test_active_agent_learning_loop():
    # Hidden rule: Red adjacent to Green -> Blue becomes active
    rule = HiddenRule(
        rule_id="r_adj_red_green",
        condition=ConditionType.ADJACENCY,
        cause_color="red",
        trigger_color="green",
        effect=EffectType.STATE_CHANGE,
        target_color="blue",
        target_new_state="active"
    )

    env = GridLabEnv(size=7, rules=[rule])
    agent = ActiveWorldModelAgent(grid_size=7)

    obs = env.reset()
    solved = False
    for step in range(30):
        action = agent.select_action(obs)
        res = env.step(action)
        obs = res.obs
        blue = obs.get_entity_by_color("blue")
        if blue and blue.state == "active":
            solved = True
            break

    state = agent.get_world_model_state()
    print(f"Test ended at step {agent.step_count}. Target active: {solved}")
    print(f"Best Hypothesis: {state['best_hypothesis']} (Confidence: {state['best_confidence']:.3f})")

    assert agent.step_count > 0
    print("Day 4 active loop test passed successfully!")

if __name__ == "__main__":
    test_active_agent_learning_loop()
