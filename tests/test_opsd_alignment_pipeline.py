import json
import tempfile
import unittest
from pathlib import Path

from opsd_alignment.scripts.compute_gradients_and_alignment import compute_alignment_records


NL = chr(10)


class OpsdAlignmentPipelineTest(unittest.TestCase):
    def test_compute_alignment_records_joins_distribution_and_branch_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            distribution_path = tmp / "distributions.jsonl"
            branch_path = tmp / "branches.jsonl"
            distribution = {
                "question_id": "q1",
                "source": "synthetic",
                "difficulty": "easy",
                "checkpoint": "model_a",
                "rollout_id": "rollout_0",
                "node_id": "node_0",
                "token_position": 3,
                "teacher_context": "full_solution",
                "selection_reason": "high_kl",
                "student_rollout_correct": False,
                "student_entropy": 0.5,
                "student_teacher_kl": 0.2,
                "candidate_token_ids": [10, 20],
                "candidate_tokens": [
                    {"token_id": 10, "token_str": " bad"},
                    {"token_id": 20, "token_str": " good"},
                ],
                "p_student": [0.8, 0.2],
                "p_teacher": [0.2, 0.8],
                "prefix_text": "prefix",
                "question": "question",
                "answer": "answer",
            }
            branches = [
                {
                    "checkpoint": "model_a",
                    "node_id": "node_0",
                    "candidate_token_id": 10,
                    "p_success": 0.0,
                    "num_correct_continuations": 0,
                    "num_forced_rollouts": 4,
                },
                {
                    "checkpoint": "model_a",
                    "node_id": "node_0",
                    "candidate_token_id": 20,
                    "p_success": 1.0,
                    "num_correct_continuations": 4,
                    "num_forced_rollouts": 4,
                },
            ]
            distribution_path.write_text(json.dumps(distribution) + NL, encoding="utf-8")
            branch_path.write_text("".join(json.dumps(record) + NL for record in branches), encoding="utf-8")
            config = {"diagnostic": {"distillation_objective": "forward_kl", "min_gradient_norm": 1e-8}}

            records = compute_alignment_records(
                config,
                distribution_file=str(distribution_path),
                branch_file=str(branch_path),
            )

            self.assertEqual(len(records), 1)
            self.assertGreater(records[0]["alignment"], 0)
            self.assertEqual(records[0]["p_success"], [0.0, 1.0])
            self.assertEqual(records[0]["candidate_tokens"][1]["p_success"], 1.0)

    def test_objective_override_to_jsd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            distribution_path = tmp / "distributions.jsonl"
            branch_path = tmp / "branches.jsonl"
            distribution_path.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "checkpoint": "model_a",
                        "rollout_id": "rollout_0",
                        "node_id": "node_0",
                        "token_position": 3,
                        "teacher_context": "answer_only",
                        "candidate_token_ids": [1, 2],
                        "candidate_tokens": [{"token_id": 1}, {"token_id": 2}],
                        "p_student": [0.7, 0.3],
                        "p_teacher": [0.3, 0.7],
                    }
                )
                + NL,
                encoding="utf-8",
            )
            branch_path.write_text(
                json.dumps({"checkpoint": "model_a", "node_id": "node_0", "candidate_token_id": 1, "p_success": 0.0})
                + NL
                + json.dumps({"checkpoint": "model_a", "node_id": "node_0", "candidate_token_id": 2, "p_success": 1.0})
                + NL,
                encoding="utf-8",
            )
            config = {"diagnostic": {"distillation_objective": "forward_kl", "jsd_alpha": 0.5}}

            records = compute_alignment_records(
                config,
                distribution_file=str(distribution_path),
                branch_file=str(branch_path),
                objective="jsd",
                jsd_alpha=0.25,
            )

            self.assertEqual(records[0]["distillation_objective"], "jsd")
            self.assertEqual(records[0]["jsd_alpha"], 0.25)


if __name__ == "__main__":
    unittest.main()
