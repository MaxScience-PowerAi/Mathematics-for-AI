import unittest
import numpy as np
from math_for_ai import (cosine_similarity, project, least_squares, svd_approximation,
                         quadratic_descent, descent_regime, linear_regression_descent)

class LinearAlgebraTests(unittest.TestCase):
    def test_cosine_parallel(self): self.assertAlmostEqual(cosine_similarity([1, 2], [2, 4]), 1)
    def test_cosine_orthogonal(self): self.assertAlmostEqual(cosine_similarity([1, 0], [0, 2]), 0)
    def test_cosine_antiparallel(self): self.assertAlmostEqual(cosine_similarity([1, 2], [-1, -2]), -1)
    def test_cosine_zero(self):
        with self.assertRaises(ValueError): cosine_similarity([0, 0], [1, 2])
    def test_invalid_vector(self):
        with self.assertRaises(ValueError): cosine_similarity([np.nan], [1])
    def test_projection_residual(self):
        x, u = np.array([3., 2.]), np.array([1., 1.])
        p = project(x, u)
        np.testing.assert_allclose(p, [2.5, 2.5]); self.assertAlmostEqual((x-p) @ u, 0)
    def test_projection_zero(self):
        with self.assertRaises(ValueError): project([1, 2], [0, 0])
    def test_least_squares_exact(self):
        np.testing.assert_allclose(least_squares([[1, 0], [1, 1], [1, 2]], [1, 3, 5]), [1, 2])
    def test_rank_deficiency(self):
        np.testing.assert_allclose(least_squares([[1, 1], [2, 2]], [2, 4]), [1, 1])
    def test_svd_optimal_error(self):
        a = np.diag([5., 3., 1.]); b = svd_approximation(a, 1)
        self.assertEqual(np.linalg.matrix_rank(b), 1); self.assertAlmostEqual(np.linalg.norm(a-b), np.sqrt(10))
    def test_svd_rank_zero(self): np.testing.assert_allclose(svd_approximation(np.eye(2), 0), np.zeros((2, 2)))
    def test_svd_invalid_rank(self):
        with self.assertRaises(ValueError): svd_approximation(np.eye(2), 3)

class OptimizationTests(unittest.TestCase):
    def test_quadratic_exact(self):
        x, costs = quadratic_descent(2, .25, 2, 3)
        np.testing.assert_allclose(x, [2, 1, .5, .25]); np.testing.assert_allclose(costs, x*x)
    def test_one_step(self): np.testing.assert_allclose(quadratic_descent(4, .5, 2, 3)[0], [4, 0, 0, 0])
    def test_boundary(self): np.testing.assert_allclose(quadratic_descent(2, 1, 2, 3)[0], [2, -2, 2, -2])
    def test_divergence(self): self.assertGreater(abs(quadratic_descent(2, 1.2, 2, 4)[0][-1]), 2)
    def test_zero_fixed_point(self): np.testing.assert_allclose(quadratic_descent(0, 1.2, 2, 4)[0], 0)
    def test_regimes(self):
        for eta, label in [(0.25,'direct convergence'),(.5,'one-step convergence'),(.75,'oscillatory convergence'),(1.,'constant-amplitude oscillation'),(1.2,'divergence')]:
            self.assertEqual(descent_regime(eta, 2), label)
    def test_invalid_steps(self):
        with self.assertRaises(ValueError): quadratic_descent(1, .1, 1, -1)
    def test_invalid_eta(self):
        with self.assertRaises(ValueError): descent_regime(0, 2)
    def test_regression(self):
        X = np.column_stack([np.ones(5), np.linspace(-1, 1, 5)]); y = 1 + 2*X[:,1]
        w, losses = linear_regression_descent(X, y, .5, 100)
        np.testing.assert_allclose(w[-1], least_squares(X, y), atol=1e-8)
        self.assertTrue(np.all(np.diff(losses) <= 1e-14))
    def test_regression_invalid_shape(self):
        with self.assertRaises(ValueError): linear_regression_descent([[1], [2]], [1], .1, 2)

if __name__ == '__main__': unittest.main()
