import json
import tempfile
import unittest
from pathlib import Path

from opsd_alignment.scripts.build_questions import (
    build_questions,
    extract_boxed_answer,
    extract_gsm8k_final_answer,
    load_gsm8k_questions,
)


NL = chr(10)


class OpsdAlignmentDatasetBuilderTest(unittest.TestCase):
    def test_synthetic_questions_have_expected_schema(self):
        records = build_questions(num_synthetic=20, seed=17)

        self.assertEqual(len(records), 20)
        self.assertEqual({"id", "source", "difficulty", "question", "answer", "reference_solution"}, set(records[0]))
        self.assertEqual(records[0]["source"], "synthetic")

    def test_answer_extractors(self):
        self.assertEqual(extract_gsm8k_final_answer("Compute it. #### 1,234"), "1234")
        self.assertEqual(extract_boxed_answer(r"Therefore \boxed{42}."), "42")

    def test_load_local_gsm8k_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gsm8k.jsonl"
            path.write_text(
                json.dumps({"question": "What is 2+3?", "answer": "2+3=5 #### 5"}) + NL,
                encoding="utf-8",
            )

            records = load_gsm8k_questions(str(path), limit=1, rng=__import__("random").Random(0))

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["answer"], "5")
            self.assertEqual(records[0]["source"], "gsm8k")


if __name__ == "__main__":
    unittest.main()
