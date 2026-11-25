import unittest
import numpy as np
from math_for_ai.logistic import sigmoid, loss_gradient, fit


class LogisticTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack([np.ones(6), [-3, -2, -1, 1, 2, 3]])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_sigmoid(self):
        np.testing.assert_allclose(sigmoid([-1000, 0, 1000]), [0, .5, 1])

    def test_zero_loss(self):
        loss, _ = loss_gradient(self.X, self.y, [0, 0])
        self.assertAlmostEqual(loss, np.log(2))

    def test_extreme_wrong_prediction(self):
        loss, grad = loss_gradient([[1]], [0], [1000])
        self.assertAlmostEqual(loss, 1000)
        np.testing.assert_allclose(grad, [1])

    def test_gradient(self):
        w = np.array([.3, -.7]); eps = 1e-6
        _, grad = loss_gradient(self.X, self.y, w, .1)
        fd = [(loss_gradient(self.X, self.y, w+eps*e, .1)[0] -
               loss_gradient(self.X, self.y, w-eps*e, .1)[0])/(2*eps)
              for e in np.eye(2)]
        np.testing.assert_allclose(grad, fd, atol=1e-8)

    def test_fit(self):
        w, history = fit(self.X, self.y)
        self.assertTrue(np.all(np.diff(history) <= 1e-12))
        self.assertLess(history[-1], history[0])
        np.testing.assert_array_equal(sigmoid(self.X @ w) >= .5, self.y)

    def test_invalid_labels(self):
        with self.assertRaises(ValueError): fit([[1], [2]], [0, 2])

    def test_invalid_steps(self):
        with self.assertRaises(ValueError): fit(self.X, self.y, steps=-1)

    def test_invalid_regularization(self):
        with self.assertRaises(ValueError): fit(self.X, self.y, l2=-1)

    def test_zero_design(self):
        w, history = fit(np.zeros((2, 1)), [0, 1], steps=3, l2=0)
        np.testing.assert_allclose(w, [0]); np.testing.assert_allclose(history, np.log(2))
