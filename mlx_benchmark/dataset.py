"""Dataset loading and filtering."""

import json
from pathlib import Path

DEFAULT_DATASET = Path(__file__).parent / "data" / "dataset_v2.jsonl"

VALID_TYPES = {"qa", "fill_blank", "mcq", "true_false", "coding", "debug"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def load_dataset(path: str | Path | None = None) -> list[dict]:
    """Load the MLX benchmark dataset from a JSONL file.

    Args:
        path: Path to the JSONL file. Defaults to the bundled dataset.

    Returns:
        List of sample dicts with keys: question, answer, category, subcategory,
        difficulty, type.
    """
    path = Path(path) if path else DEFAULT_DATASET
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def filter_dataset(
    samples: list[dict],
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
    types: list[str] | None = None,
) -> list[dict]:
    """Filter samples by category, difficulty, and/or type."""
    if categories:
        categories_set = set(categories)
        samples = [s for s in samples if s["category"] in categories_set]
    if difficulties:
        difficulties_set = set(difficulties)
        samples = [s for s in samples if s["difficulty"] in difficulties_set]
    if types:
        types_set = set(types)
        samples = [s for s in samples if s["type"] in types_set]
    return samples