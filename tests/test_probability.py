import unittest
import numpy as np
from math_for_ai.probability import bernoulli_trial_mean, theoretical_standard_error


class ProbabilityTests(unittest.TestCase):
    def test_zero_probability(self):
        self.assertEqual(bernoulli_trial_mean(0, 20), (0, 0))

    def test_one_probability(self):
        self.assertEqual(bernoulli_trial_mean(1, 20), (1, 0))

    def test_reproducible(self):
        self.assertEqual(bernoulli_trial_mean(.3, 100, 42), bernoulli_trial_mean(.3, 100, 42))

    def test_standard_error_formula(self):
        mean, se = bernoulli_trial_mean(.3, 100, 42)
        self.assertAlmostEqual(se, np.sqrt(mean * (1 - mean) / 99))

    def test_theoretical_error(self):
        self.assertAlmostEqual(theoretical_standard_error(.5, 100), .05)
        self.assertAlmostEqual(theoretical_standard_error(.5, 400), .025)

    def test_invalid_probability(self):
        for p in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError): bernoulli_trial_mean(p, 10)
            with self.assertRaises(ValueError): theoretical_standard_error(p, 10)

    def test_invalid_size(self):
        for n in (0, -1, 2.5, True):
            with self.assertRaises(ValueError): bernoulli_trial_mean(.5, n)
            with self.assertRaises(ValueError): theoretical_standard_error(.5, n)
        with self.assertRaises(ValueError): bernoulli_trial_mean(.5, 1)
        self.assertAlmostEqual(theoretical_standard_error(.5, 1), .5)
