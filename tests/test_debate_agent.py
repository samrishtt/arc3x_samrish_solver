import unittest
from world_model_lab.core.types import Action, Position, Entity, Observation
from world_model_lab.agents.debate_agent import DialecticalDebateAgent
from world_model_lab.environment.grid_lab import GridLabEnv


class TestDialecticalDebateAgent(unittest.TestCase):
    def setUp(self):
        self.env = GridLabEnv(size=7, randomize_layout=True, seed=123)
        self.agent = DialecticalDebateAgent(grid_size=7, seed=123)

    def test_debate_loop_execution(self):
        obs = self.env.reset()
        action = self.agent.select_action(obs)
        self.assertIsInstance(action, Action)
        self.assertGreater(self.agent.total_proposals, 0)
        self.assertGreater(len(self.agent.debate_history), 0)

    def test_critic_vetoes_wall_collision(self):
        # Create an observation where agent is at (0, 0) and attempts to move UP
        agent_ent = Entity(id="agent", color="blue", shape="circle", pos=Position(0, 0), is_agent=True)
        obs = Observation(
            step=1,
            grid_shape=(7, 7),
            entities=[agent_ent],
            agent_pos=Position(0, 0),
            raw_grid=None
        )
        is_obj, reason, severity = self.agent._critic_evaluate(Action.UP, obs, [])
        self.assertTrue(is_obj)
        self.assertEqual(severity, 1.0)
        self.assertIn("Wall collision", reason)

    def test_critic_vetoes_hazard_entity(self):
        # Agent at (3, 3), spike at (3, 4) (to the RIGHT)
        agent_ent = Entity(id="agent", color="blue", shape="circle", pos=Position(3, 3), is_agent=True)
        spike_ent = Entity(id="spike_1", color="red", shape="triangle", pos=Position(3, 4), is_agent=False)
        obs = Observation(
            step=1,
            grid_shape=(7, 7),
            entities=[agent_ent, spike_ent],
            agent_pos=Position(3, 3),
            raw_grid=None
        )
        is_obj, reason, severity = self.agent._critic_evaluate(Action.RIGHT, obs, [])
        self.assertTrue(is_obj)
        self.assertGreaterEqual(severity, 0.9)
        self.assertIn("Hazardous entity", reason)


if __name__ == "__main__":
    unittest.main()
