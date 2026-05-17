import math
import unittest

from opsd_alignment.src.gradients import alignment, distillation_gradient, ideal_gradient, student_teacher_kl


class OpsdAlignmentGradientTest(unittest.TestCase):
    def test_ideal_gradient_prefers_above_average_success_tokens(self):
        grad = ideal_gradient([0.75, 0.25], [0.0, 1.0])

        self.assertLess(grad[0], 0)
        self.assertGreater(grad[1], 0)
        self.assertTrue(math.isclose(sum(grad), 0.0, abs_tol=1e-12))

    def test_forward_kl_descent_moves_toward_teacher_preference(self):
        grad = distillation_gradient([0.8, 0.2], [0.2, 0.8], objective="forward_kl")

        self.assertLess(grad[0], 0)
        self.assertGreater(grad[1], 0)
        self.assertTrue(math.isclose(sum(grad), 0.0, abs_tol=1e-12))

    def test_reverse_kl_descent_moves_toward_teacher_distribution(self):
        grad = distillation_gradient([0.8, 0.2], [0.2, 0.8], objective="reverse_kl")

        self.assertEqual(grad, [-0.6000000000000001, 0.6000000000000001])

    def test_jsd_descent_moves_toward_teacher_preference(self):
        grad = distillation_gradient([0.8, 0.2], [0.2, 0.8], objective="jsd", jsd_alpha=0.5)

        self.assertLess(grad[0], 0)
        self.assertGreater(grad[1], 0)
        self.assertTrue(math.isclose(sum(grad), 0.0, abs_tol=1e-12))

    def test_alignment_is_positive_when_directions_match(self):
        g_ideal = ideal_gradient([0.8, 0.2], [0.0, 1.0])
        g_distill = distillation_gradient([0.8, 0.2], [0.2, 0.8], objective="forward_kl")

        self.assertGreater(alignment(g_ideal, g_distill), 0)

    def test_alignment_returns_none_for_near_zero_gradient(self):
        self.assertIsNone(alignment([0.0, 0.0], [1.0, -1.0]))

    def test_student_teacher_kl_is_zero_for_same_distribution(self):
        self.assertTrue(math.isclose(student_teacher_kl([0.4, 0.6], [0.4, 0.6]), 0.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
