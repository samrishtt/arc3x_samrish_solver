import numpy as np
from world_model_lab.adapters.arc3x_bridge import ARC3xBridge
from world_model_lab.core.types import Action
from world_model_lab.agents.active_agent import ActiveWorldModelAgent

def test_arc3x_integration():
    bridge = ARC3xBridge()
    # Create a synthetic 15x15 ARC-AGI-3 frame
    frame = np.zeros((15, 15), dtype=np.int32)
    # Add avatar color at (5, 5)
    frame[5, 5] = 12
    # Add target color at (10, 10)
    frame[10, 10] = 3
    # Add key color at (5, 8)
    frame[5, 8] = 9

    obs = bridge.frame_to_observation(
        frame=frame,
        step=1,
        avatar_pos=(5, 5),
        last_action_int=1
    )

    assert obs.grid_shape == (15, 15)
    assert len(obs.entities) >= 3
    assert obs.agent_pos.r == 5 and obs.agent_pos.c == 5

    # Connect to ActiveWorldModelAgent
    agent = ActiveWorldModelAgent(grid_size=15)
    action = agent.select_action(obs)
    arc_action_int = bridge.action_to_arc(action)

    assert arc_action_int in [1, 2, 3, 4, 5, 7]
    print(f"ARC3x Bridge successfully converted frame to obs, selected action: {action} (ARC int: {arc_action_int})")

if __name__ == "__main__":
    test_arc3x_integration()
