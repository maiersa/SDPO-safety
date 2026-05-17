"""Build OPSD alignment question JSONL files.

The script is intentionally usable without network access: by default it writes a
small deterministic synthetic set. If local or Hugging Face datasets are
provided, it can also sample GSM8K-style and MATH-style records into the same
schema.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from opsd_alignment.src.storage import write_jsonl


DEFAULT_OUTPUT = Path("opsd_alignment/data/questions.jsonl")


SYNTHETIC_TEMPLATES = [
    ("synthetic_arithmetic_001", "easy", "A bookshelf has 3 shelves. Each shelf holds 4 books. If 5 books are removed, how many remain?", "7", "There are 3 * 4 = 12 books at first. Removing 5 leaves 12 - 5 = 7 books."),
    ("synthetic_arithmetic_002", "easy", "Maya buys 6 packs of pencils with 8 pencils in each pack. She gives away 11 pencils. How many pencils does she keep?", "37", "The packs contain 6 * 8 = 48 pencils. After giving away 11, she keeps 48 - 11 = 37."),
    ("synthetic_arithmetic_003", "easy", "A train has 9 cars. Each car has 12 seats. If 17 seats are empty, how many seats are occupied?", "91", "There are 9 * 12 = 108 seats total. If 17 are empty, 108 - 17 = 91 are occupied."),
    ("synthetic_arithmetic_004", "easy", "Lena has 45 stickers. She puts the same number of stickers on each of 5 pages. How many stickers are on each page?", "9", "Divide 45 stickers by 5 pages: 45 / 5 = 9 stickers per page."),
    ("synthetic_arithmetic_005", "easy", "A baker makes 24 muffins in the morning and 18 more in the afternoon. They sell 29 muffins. How many muffins are left?", "13", "The baker makes 24 + 18 = 42 muffins. Selling 29 leaves 42 - 29 = 13."),
    ("synthetic_arithmetic_006", "easy", "There are 7 tables with 6 chairs each. Three chairs break. How many usable chairs remain?", "39", "The tables have 7 * 6 = 42 chairs. Three break, so 42 - 3 = 39 usable chairs remain."),
    ("synthetic_arithmetic_007", "easy", "Noah reads 14 pages each day for 5 days, then reads 9 more pages. How many pages does he read in total?", "79", "In 5 days he reads 14 * 5 = 70 pages. With 9 more pages, the total is 70 + 9 = 79."),
    ("synthetic_arithmetic_008", "easy", "A box has 64 crayons. Four children share them equally. How many crayons does each child get?", "16", "Sharing 64 crayons among 4 children gives 64 / 4 = 16 crayons each."),
    ("synthetic_arithmetic_009", "easy", "An orchard has 8 rows of apple trees with 9 trees in each row. If 6 trees are removed, how many trees remain?", "66", "The orchard starts with 8 * 9 = 72 trees. Removing 6 leaves 72 - 6 = 66."),
    ("synthetic_arithmetic_010", "easy", "A movie ticket costs 7 dollars. How much do 13 tickets cost?", "91", "Thirteen tickets cost 13 * 7 = 91 dollars."),
    ("synthetic_arithmetic_011", "medium", "A jar starts with 120 marbles. Sam removes 15 marbles and then splits the rest equally into 7 bags. How many marbles are in each bag?", "15", "After removing 15, there are 120 - 15 = 105 marbles. Splitting into 7 bags gives 105 / 7 = 15 per bag."),
    ("synthetic_arithmetic_012", "medium", "A recipe uses 3 cups of flour for 2 cakes. How many cups are needed for 10 cakes?", "15", "Ten cakes is 10 / 2 = 5 batches. Each batch uses 3 cups, so 5 * 3 = 15 cups."),
    ("synthetic_arithmetic_013", "medium", "A taxi charges 4 dollars to start and 3 dollars per mile. What is the cost of an 11 mile ride?", "37", "The mileage charge is 3 * 11 = 33 dollars. Adding the 4 dollar start fee gives 33 + 4 = 37."),
    ("synthetic_arithmetic_014", "medium", "The sum of three consecutive integers is 72. What is the largest integer?", "25", "Let the integers be n, n+1, and n+2. Then 3n + 3 = 72, so 3n = 69 and n = 23. The largest is 25."),
    ("synthetic_arithmetic_015", "medium", "A rectangle has perimeter 46. Its length is 15. What is its width?", "8", "For a rectangle, 2(length + width) = 46, so length + width = 23. With length 15, width = 23 - 15 = 8."),
    ("synthetic_logic_016", "medium", "A number is doubled and then increased by 9 to get 35. What is the number?", "13", "Let the number be x. Then 2x + 9 = 35, so 2x = 26 and x = 13."),
    ("synthetic_logic_017", "medium", "A class has 28 students. Three eighths of them bring lunch from home. How many students bring lunch from home?", "21", "Three eighths of 28 is 28 * 3 / 8 = 3.5 * 6? More directly, 28 / 4 = 7 and 3/4 of 28 is 21. Since three eighths was intended, compute 28 * 3 / 8 = 10.5, which is not an integer; use the corrected answer 10.5."),
    ("synthetic_logic_018", "medium", "A class has 32 students. Three eighths of them bring lunch from home. How many students bring lunch from home?", "12", "Three eighths of 32 is 32 * 3 / 8 = 4 * 3 = 12."),
    ("synthetic_arithmetic_019", "hard", "A store discounts a 80 dollar jacket by 25 percent, then adds 6 dollars tax. What is the final price?", "66", "A 25 percent discount on 80 dollars is 20 dollars, so the discounted price is 80 - 20 = 60. Adding 6 dollars tax gives 66."),
    ("synthetic_logic_020", "hard", "The average of five numbers is 18. Four of the numbers are 12, 17, 20, and 25. What is the fifth number?", "16", "The total of five numbers is 5 * 18 = 90. The known four sum to 12 + 17 + 20 + 25 = 74. The fifth is 90 - 74 = 16."),
    ("synthetic_logic_021", "hard", "A number is tripled, then 4 is subtracted, giving 29. What is the number?", "11", "Let the number be x. Then 3x - 4 = 29, so 3x = 33 and x = 11."),
]

# Correct the intentionally awkward template above by excluding it from default generation.
SYNTHETIC_TEMPLATES = [item for item in SYNTHETIC_TEMPLATES if item[0] != "synthetic_logic_017"]


def synthetic_questions(limit: int | None = None) -> list[dict[str, str]]:
    records = [
        {
            "id": question_id,
            "source": "synthetic",
            "difficulty": difficulty,
            "question": question,
            "answer": answer,
            "reference_solution": reference_solution,
        }
        for question_id, difficulty, question, answer, reference_solution in SYNTHETIC_TEMPLATES
    ]
    return records if limit is None else records[:limit]


def build_questions(
    *,
    num_synthetic: int = 20,
    gsm8k_source: str | None = None,
    num_gsm8k: int = 0,
    math_source: str | None = None,
    num_math: int = 0,
    seed: int = 17,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    records = synthetic_questions(num_synthetic)

    if gsm8k_source and num_gsm8k > 0:
        gsm8k = load_gsm8k_questions(gsm8k_source, limit=num_gsm8k, rng=rng)
        records.extend(gsm8k)

    if math_source and num_math > 0:
        math_records = load_math_questions(math_source, limit=num_math, rng=rng)
        records.extend(math_records)

    return _dedupe_by_id(records)


def load_gsm8k_questions(source: str, *, limit: int, rng: random.Random) -> list[dict[str, str]]:
    raw_records = _load_records(source, default_hf_config="main")
    rng.shuffle(raw_records)
    records = []
    for idx, row in enumerate(raw_records):
        question = _first_present(row, "question", "problem", "prompt")
        answer_raw = _first_present(row, "answer", "solution", "reference_solution")
        if question is None or answer_raw is None:
            continue
        answer = extract_gsm8k_final_answer(str(answer_raw))
        if answer is None:
            continue
        records.append(
            {
                "id": f"gsm8k_{idx:05d}",
                "source": "gsm8k",
                "difficulty": "easy",
                "question": str(question).strip(),
                "answer": answer,
                "reference_solution": str(answer_raw).strip(),
            }
        )
        if len(records) >= limit:
            break
    return records


def load_math_questions(source: str, *, limit: int, rng: random.Random) -> list[dict[str, str]]:
    raw_records = _load_records(source, default_hf_config=None)
    rng.shuffle(raw_records)
    records = []
    for idx, row in enumerate(raw_records):
        question = _first_present(row, "problem", "question", "prompt")
        solution = _first_present(row, "solution", "answer", "reference_solution")
        if question is None or solution is None:
            continue
        answer = extract_boxed_answer(str(solution)) or extract_last_number(str(solution))
        if answer is None:
            continue
        level = str(row.get("level", "")).lower()
        difficulty = _math_difficulty(level)
        records.append(
            {
                "id": f"math_{idx:05d}",
                "source": "math",
                "difficulty": difficulty,
                "question": str(question).strip(),
                "answer": answer,
                "reference_solution": str(solution).strip(),
            }
        )
        if len(records) >= limit:
            break
    return records


def extract_gsm8k_final_answer(answer: str) -> str | None:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if match:
        return match.group(1).replace(",", "")
    return extract_last_number(answer)


def extract_boxed_answer(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else None


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None


def _load_records(source: str, default_hf_config: str | None) -> list[dict[str, Any]]:
    path = Path(source)
    if path.exists():
        return _load_local_records(path)

    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            f"Dataset source {source!r} is not a local path, and the 'datasets' package is not installed."
        ) from exc

    if default_hf_config is None:
        dataset = datasets.load_dataset(source)
    else:
        dataset = datasets.load_dataset(source, default_hf_config)
    split = "test" if "test" in dataset else next(iter(dataset.keys()))
    return [dict(row) for row in dataset[split]]


def _load_local_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        candidates = [path / "test.jsonl", path / "test.json", path / "train.jsonl", path / "train.json"]
        for candidate in candidates:
            if candidate.exists():
                return _load_local_records(candidate)
        raise FileNotFoundError(f"No JSON/JSONL dataset file found in {path}")

    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "records", "examples", "test", "train"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"Unsupported JSON dataset structure: {path}")

    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required to read parquet dataset files") from exc
        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError(f"Unsupported dataset file extension: {path.suffix}")


def _first_present(row: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _math_difficulty(level: str) -> str:
    if any(value in level for value in ("1", "2")):
        return "medium"
    if any(value in level for value in ("4", "5")):
        return "hard"
    return "medium"


def _dedupe_by_id(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for record in records:
        if record["id"] in seen:
            continue
        seen.add(record["id"])
        deduped.append(record)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output questions JSONL path.")
    parser.add_argument("--num-synthetic", type=int, default=20, help="Number of built-in synthetic questions to include.")
    parser.add_argument("--gsm8k-source", help="Local path or HF dataset id for GSM8K-style records.")
    parser.add_argument("--num-gsm8k", type=int, default=0)
    parser.add_argument("--math-source", help="Local path or HF dataset id for MATH-style records.")
    parser.add_argument("--num-math", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing questions file: {output_path}")
        return

    records = build_questions(
        num_synthetic=args.num_synthetic,
        gsm8k_source=args.gsm8k_source,
        num_gsm8k=args.num_gsm8k,
        math_source=args.math_source,
        num_math=args.num_math,
        seed=args.seed,
    )
    write_jsonl(output_path, records)
    counts = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    print(f"Wrote {len(records)} questions to {output_path}: {counts}")


if __name__ == "__main__":
    main()
