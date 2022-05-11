import unittest
from rl_utils import PLSEnv, RandomAgent


class MyTestCase(unittest.TestCase):
    def test_valid_solution(self):
        dim = 3
        env = PLSEnv(dim=dim)
        total_rew = 0

        _, rew, done, info = env.step(action=0)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertFalse(done)
        self.assertTrue(info['Feasible'])
        self.assertFalse(info['Solved'])

        _, rew, done, info = env.step(action=4)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertFalse(done)
        self.assertTrue(info['Feasible'])
        self.assertFalse(info['Solved'])

        _, rew, done, info = env.step(action=8)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertFalse(done)
        self.assertTrue(info['Feasible'])
        self.assertFalse(info['Solved'])

        _, rew, done, info = env.step(action=10)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertFalse(done)
        self.assertTrue(info['Feasible'])
        self.assertFalse(info['Solved'])

        _, rew, done, info = env.step(action=14)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertFalse(done)
        self.assertTrue(info['Feasible'])
        self.assertFalse(info['Solved'])

        _, rew, done, info = env.step(action=15)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertFalse(done)
        self.assertTrue(info['Feasible'])
        self.assertFalse(info['Solved'])

        _, rew, done, info = env.step(action=20)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertFalse(done)
        self.assertTrue(info['Feasible'])
        self.assertFalse(info['Solved'])

        _, rew, done, info = env.step(action=21)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertFalse(done)
        self.assertTrue(info['Feasible'])
        self.assertFalse(info['Solved'])

        _, rew, done, info = env.step(action=25)
        total_rew += rew
        self.assertEqual(rew, -1)
        self.assertEqual(total_rew, -dim**2)
        self.assertTrue(done)
        self.assertTrue(info['Feasible'])
        self.assertTrue(info['Solved'])
        env.render()


if __name__ == '__main__':
    unittest.main()
