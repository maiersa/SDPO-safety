import unittest

from opsd_alignment.src.node_selection import NodeScore, select_diagnostic_nodes


class OpsdAlignmentNodeSelectionTest(unittest.TestCase):
    def test_gkd_magnitude_policy_selects_largest_distillation_updates(self):
        scores = [
            NodeScore(token_position=3, student_entropy=10.0, student_teacher_kl=10.0, gkd_magnitude=0.1),
            NodeScore(token_position=4, student_entropy=0.1, student_teacher_kl=0.1, gkd_magnitude=5.0),
            NodeScore(token_position=5, student_entropy=1.0, student_teacher_kl=1.0, gkd_magnitude=2.0),
        ]

        selected = select_diagnostic_nodes(scores, nodes_per_rollout=2, selection_policy="gkd_magnitude")

        self.assertEqual([score.token_position for score, _ in selected], [4, 5])
        self.assertEqual([reason for _, reason in selected], ["high_gkd_magnitude", "high_gkd_magnitude"])

    def test_kl_entropy_policy_keeps_existing_behavior(self):
        scores = [
            NodeScore(token_position=3, student_entropy=0.1, student_teacher_kl=3.0, gkd_magnitude=0.0),
            NodeScore(token_position=4, student_entropy=5.0, student_teacher_kl=0.2, gkd_magnitude=0.0),
            NodeScore(token_position=5, student_entropy=1.0, student_teacher_kl=2.0, gkd_magnitude=0.0),
        ]

        selected = select_diagnostic_nodes(scores, nodes_per_rollout=2, selection_policy="kl_entropy")

        self.assertEqual([score.token_position for score, _ in selected], [3, 4])
        self.assertEqual([reason for _, reason in selected], ["high_kl", "high_entropy"])


if __name__ == "__main__":
    unittest.main()
