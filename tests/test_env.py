import asyncio
import unittest

from server.env import SQLDebugEnv
from server.models import SQLDebugAction, ActionType


class TestEnv(unittest.TestCase):
    def test_reset_and_inspect_schema(self):
        async def run():
            env = SQLDebugEnv(task_id="easy_syntax_fix")
            obs, info = await env.reset()
            self.assertFalse(obs.is_done)

            action = SQLDebugAction(action_type=ActionType.INSPECT_SCHEMA)
            obs2, reward, done, info2 = await env.step(action)
            self.assertFalse(done)
            self.assertIsNotNone(obs2.schema_info)
            self.assertGreaterEqual(reward, 0.0)

        asyncio.run(run())

    def test_submit_broken_query_does_not_finish(self):
        async def run():
            env = SQLDebugEnv(task_id="easy_syntax_fix")
            obs, _ = await env.reset()

            action = SQLDebugAction(
                action_type=ActionType.SUBMIT_QUERY,
                query=env.task.broken_query,
            )
            obs2, reward, done, _ = await env.step(action)

            self.assertFalse(done)
            self.assertLessEqual(reward, 0.2)
            self.assertGreaterEqual(reward, -1.0)
            self.assertEqual(obs2.current_query, env.task.broken_query)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

