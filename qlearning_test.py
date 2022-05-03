import unittest

import numpy as np

from qlearning import QLearner

class TestQlearning(unittest.TestCase):
    def test_policy(self):
        test_actions = ['up','down','left','right']
        test_state = 0

        qlearner = QLearner(actions=test_actions)
        qlearner.stop_exploration()
        Q = dict()
        Q[test_state] = [np.array([1.0,0.]),np.array([0,0]),np.array([0,0]),np.array([0,0])]
        qlearner.Q = Q
        self.assertEqual(qlearner.policy(test_state), 0, msg="Actions should equal '0'")

        Q[test_state] = [np.array([0,0.]),np.array([1.,0]),np.array([0,0]),np.array([0,0])]
        self.assertEqual(qlearner.policy(test_state), 1, msg="Actions should be equal '1'")

if __name__ == '__main__':
    unittest.main()