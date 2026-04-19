"""Benchmark runner — orchestrates evaluation of a model on the dataset."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .backends import create_backend
from .dataset import filter_dataset, load_dataset
from .judge import evaluate
from .prompts import build_prompt


@dataclass
class BenchResult:
    """A single question evaluation result."""

    question: str
    reference_answer: str
    model_answer: str
    type: str
    category: str
    subcategory: str
    difficulty: str
    correct: bool


@dataclass
class BenchStats:
    """Aggregate benchmark statistics."""

    total: int = 0
    correct: int = 0
    by_type: dict = field(default_factory=lambda: {})
    by_difficulty: dict = field(default_factory=lambda: {})
    by_category: dict = field(default_factory=lambda: {})

    def record(self, result: BenchResult):
        self.total += 1
        self.correct += int(result.correct)
        for group, key in [
            (self.by_type, result.type),
            (self.by_difficulty, result.difficulty),
            (self.by_category, result.category),
        ]:
            entry = group.setdefault(key, {"total": 0, "correct": 0})
            entry["total"] += 1
            entry["correct"] += int(result.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total * 100 if self.total else 0.0


def run_benchmark(
    model: str,
    provider: str = "ollama",
    dataset_path: str | None = None,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    judge_host: str | None = None,
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
    output_dir: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    limit: int | None = None,
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
    types: list[str] | None = None,
    rate_limit_delay: float = 0.5,
    host: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    workers: int = 1,
    verbose: bool = True,
) -> tuple[list[BenchResult], BenchStats]:
    """Run the full benchmark.

    Args:
        model: Model name (e.g. 'llama3.2').
        provider: Backend provider ('ollama', 'anthropic', 'openai', 'groq', 'openrouter').
        dataset_path: Path to dataset JSONL. Defaults to bundled dataset.
        judge_model: Model for LLM judge. Defaults to same as model.
        judge_provider: Provider for judge. Defaults to same as provider.
        judge_host: Host for judge backend. Defaults to same as host.
        judge_api_key: API key for judge backend. Defaults to same as api_key.
        judge_base_url: Base URL for judge backend. Defaults to same as base_url.
        output_dir: Directory to save results JSON. None = don't save.
        max_tokens: Max tokens for model responses.
        temperature: Sampling temperature.
        limit: Max number of samples (for quick runs).
        categories: Filter to these categories.
        difficulties: Filter to these difficulties.
        types: Filter to these question types.
        rate_limit_delay: Seconds between API calls (per worker).
        host: Ollama host URL.
        api_key: API key for cloud providers.
        base_url: Custom base URL for OpenAI-compatible providers.
        workers: Number of concurrent threads for batch processing (default 1 = sequential).
        verbose: Print progress to stdout.

    Returns:
        Tuple of (results list, aggregate stats).
    """
    # Create backends
    model_backend = create_backend(model=model, provider=provider, host=host, api_key=api_key, base_url=base_url)

    judge_provider = judge_provider or provider
    judge_model = judge_model or model
    judge_host = judge_host or host
    judge_api_key = judge_api_key or api_key
    judge_base_url = judge_base_url or base_url
    judge_backend = create_backend(model=judge_model, provider=judge_provider, host=judge_host, api_key=judge_api_key, base_url=judge_base_url)

    # Load and filter dataset
    samples = load_dataset(dataset_path)
    samples = filter_dataset(samples, categories=categories, difficulties=difficulties, types=types)
    if limit:
        samples = samples[:limit]

    if verbose:
        print(f"MLX Benchmark — {model_backend.name}")
        print(f"  Judge:      {judge_backend.name}")
        print(f"  Samples:    {len(samples)}")
        print(f"  Workers:    {workers}")
        print(f"  Types:      {sorted(set(s['type'] for s in samples))}")
        print(f"  Categories: {sorted(set(s['category'] for s in samples))}")
        print()

    stats = BenchStats()
    results: list[BenchResult] = []
    print_lock = threading.Lock()
    completed_count = [0]  # mutable counter shared across threads

    def _process_sample(idx: int, sample: dict) -> BenchResult | None:
        system, user_msg = build_prompt(sample)

        try:
            model_answer = model_backend.generate(
                prompt=user_msg, system=system, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as e:
            if verbose:
                with print_lock:
                    print(f"  [{idx + 1}/{len(samples)}] API error: {e}")
            time.sleep(2)
            return None

        try:
            is_correct = evaluate(judge_backend, sample, model_answer)
        except Exception as e:
            if verbose:
                with print_lock:
                    print(f"  [{idx + 1}/{len(samples)}] Judge error: {e}")
            is_correct = False

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        return BenchResult(
            question=sample["question"],
            reference_answer=sample["answer"],
            model_answer=model_answer,
            type=sample["type"],
            category=sample["category"],
            subcategory=sample.get("subcategory", ""),
            difficulty=sample["difficulty"],
            correct=is_correct,
        )

    if workers > 1:
        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i, sample in enumerate(samples):
                future = executor.submit(_process_sample, i, sample)
                futures_map[future] = i

            # Collect in completion order; print immediately, store for ordered output
            partial: dict[int, BenchResult | None] = {}
            for future in as_completed(futures_map):
                idx = futures_map[future]
                result = future.result()
                partial[idx] = result
                if result is not None:
                    stats.record(result)
                    if verbose:
                        with print_lock:
                            completed_count[0] += 1
                            marker = "✓" if result.correct else "✗"
                            print(
                                f"  [{completed_count[0]}/{len(samples)}] {marker}"
                                f"  {result.type}/{result.difficulty}"
                                f"  acc: {stats.accuracy:.1f}%"
                            )

        # Restore original dataset order for the returned list
        results = [partial[i] for i in range(len(samples)) if partial.get(i) is not None]
    else:
        for i, sample in enumerate(samples):
            result = _process_sample(i, sample)
            if result is None:
                continue
            results.append(result)
            stats.record(result)
            if verbose:
                marker = "✓" if result.correct else "✗"
                print(f"  [{i + 1}/{len(samples)}] {marker}  {result.type}/{result.difficulty}  acc: {stats.accuracy:.1f}%")

    # Print summary
    if verbose and stats.total > 0:
        _print_summary(model_backend.name, stats)

    # Save results
    if output_dir and stats.total > 0:
        _save_results(output_dir, model_backend.name, judge_backend.name, results, stats)

    return results, stats


def _print_summary(model_name: str, stats: BenchStats):
    print("\n" + "=" * 60)
    print(f"  BENCHMARK RESULTS — {model_name}")
    print("=" * 60)
    print(f"  Overall: {stats.correct}/{stats.total} ({stats.accuracy:.1f}%)\n")

    print("  By type:")
    for key in sorted(stats.by_type):
        s = stats.by_type[key]
        pct = s["correct"] / s["total"] * 100
        print(f"    {key:15s}  {s['correct']:3d}/{s['total']:3d}  ({pct:5.1f}%)")

    print("\n  By difficulty:")
    for key in sorted(stats.by_difficulty):
        s = stats.by_difficulty[key]
        pct = s["correct"] / s["total"] * 100
        print(f"    {key:15s}  {s['correct']:3d}/{s['total']:3d}  ({pct:5.1f}%)")

    print("\n  By category:")
    for key in sorted(stats.by_category):
        s = stats.by_category[key]
        pct = s["correct"] / s["total"] * 100
        print(f"    {key:25s}  {s['correct']:3d}/{s['total']:3d}  ({pct:5.1f}%)")
    print()


def _save_results(output_dir: str, model_name: str, judge_name: str, results: list[BenchResult], stats: BenchStats):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model_name.replace("/", "_").replace(".", "-")

    result_file = out_path / f"bench_{slug}_{ts}.json"
    with open(result_file, "w") as f:
        json.dump(
            {
                "model": model_name,
                "judge": judge_name,
                "timestamp": ts,
                "stats": {
                    "total": stats.total,
                    "correct": stats.correct,
                    "accuracy": round(stats.accuracy, 2),
                    "by_type": stats.by_type,
                    "by_difficulty": stats.by_difficulty,
                    "by_category": stats.by_category,
                },
                "results": [
                    {
                        "question": r.question,
                        "reference_answer": r.reference_answer,
                        "model_answer": r.model_answer,
                        "type": r.type,
                        "category": r.category,
                        "subcategory": r.subcategory,
                        "difficulty": r.difficulty,
                        "correct": r.correct,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"  Results saved to: {result_file}")