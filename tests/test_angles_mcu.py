import unittest

import numpy as np

import mcu_angles_pipeline


class AnglesMcuTestCase(unittest.TestCase):
    def test_angles_mcu_without_random(self):
        result = mcu_angles_pipeline.pipeline(noise=False, plot=False)
        correct = np.array([0.01072429, 1.01149428, 0.01127911])
        self.assertTrue(np.allclose(result, correct), f"Your result {result}; correct {correct}")


if __name__ == '__main__':
    unittest.main()
