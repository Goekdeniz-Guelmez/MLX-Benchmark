"""CLI entry point for mlx-bench."""

import argparse
import sys
from pathlib import Path

from .runner import run_benchmark


def _load_config(path: str) -> dict:
    """Load a YAML or JSON config file."""
    raw = Path(path).read_text()
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml
            return yaml.safe_load(raw)
        except ImportError:
            sys.exit(
                "PyYAML is required for --config. Install it with:\n"
                "  pip install pyyaml"
            )
    else:
        import json
        return json.loads(raw)


def _parse_models(args) -> list[dict]:
    """Resolve the model list from --config or --model.

    Returns a list of dicts with keys: name, provider, host, api_key, base_url.
    """
    if args.config:
        config = _load_config(args.config)
        models = []
        for entry in config.get("models", []):
            models.append({
                "name": entry["name"],
                "provider": entry.get("provider", args.provider),
                "host": entry.get("host", args.host),
                "api_key": entry.get("api_key", args.api_key),
                "base_url": entry.get("base_url", args.base_url),
            })
        return models

    # --model flag (comma-separated)
    raw = [m.strip() for m in args.model.split(",")]
    return [
        {
            "name": m,
            "provider": args.provider,
            "host": args.host,
            "api_key": args.api_key,
            "base_url": args.base_url,
        }
        for m in raw
    ]


def _get_judge(args, config: dict | None):
    """Resolve judge settings from config or args."""
    if config and "judge" in config:
        j = config["judge"]
        return {
            "judge_model": j.get("name"),
            "judge_provider": j.get("provider", args.judge_provider),
            "judge_host": j.get("host", args.host),
            "judge_api_key": j.get("api_key", args.api_key),
            "judge_base_url": j.get("base_url", args.base_url),
        }
    return {
        "judge_model": args.judge_model,
        "judge_provider": args.judge_provider,
        "judge_host": args.host,
        "judge_api_key": args.api_key,
        "judge_base_url": args.base_url,
    }


def main():
    parser = argparse.ArgumentParser(
        prog="mlx-bench",
        description="Benchmark LLMs on Apple MLX knowledge and coding tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Benchmark a local Ollama model
  mlx-bench --model llama3.2

  # Benchmark multiple models from config
  mlx-bench --config models.yml

  # Benchmark with a different judge model
  mlx-bench --model llama3.2 --judge-model mistral

  # Use Anthropic Claude as the model
  mlx-bench --provider anthropic --model claude-sonnet-4-20250514

  # Filter by type/difficulty/category
  mlx-bench --model llama3.2 --types coding debug --difficulties medium hard very-hard

  # Quick test run
  mlx-bench --config models.yml --limit 10

  # Generate LaTeX table and PNG chart from existing results
  mlx-bench --latex --plot

  # Generate LaTeX from specific result files
  mlx-bench --latex --results results/bench_ollama_llama3-2_*.json

  # Run benchmark and also generate exports
  mlx-bench --config models.yml --latex --plot
""",
    )

    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument(
        "--model",
        help="Model name (comma-separated for multiple). Provider set by --provider.",
    )
    model_group.add_argument(
        "--config",
        help="Path to YAML/JSON config file listing models with per-model providers.",
    )
    parser.add_argument("--latex", action="store_true",
                        help="Generate a LaTeX table from benchmark results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate a PNG bar chart from benchmark results")
    parser.add_argument("--results", nargs="*", default=None,
                        help="Result JSON files to export (default: latest in --output-dir)")

    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "anthropic", "openai", "groq", "openrouter"],
                        help="Default provider (default: ollama)")
    parser.add_argument("--judge-model", default=None,
                        help="Model for LLM judge (default: same as --model)")
    parser.add_argument("--judge-provider", default=None,
                        choices=["ollama", "anthropic", "openai", "groq", "openrouter"],
                        help="Provider for the judge model (default: same as --provider)")
    parser.add_argument("--dataset", default=None,
                        help="Path to dataset JSONL (default: bundled dataset_v2.jsonl)")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory for result JSON files (default: ./results)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens for model responses (default: 1024)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples for quick runs")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Filter to specific categories")
    parser.add_argument("--difficulties", nargs="*", default=None,
                        choices=["easy", "medium", "hard", "very-hard"],
                        help="Filter to specific difficulties")
    parser.add_argument("--types", nargs="*", default=None,
                        choices=["qa", "fill_blank", "mcq", "true_false", "coding", "debug"],
                        help="Filter to specific question types")
    parser.add_argument("--rate-limit", type=float, default=None,
                        help="Delay in seconds between API calls (default: 0.5)")
    parser.add_argument("--host", default=None,
                        help="Ollama host URL (default: http://localhost:11434)")
    parser.add_argument("--api-key", default=None,
                        help="API key for cloud providers (falls back to env vars)")
    parser.add_argument("--base-url", default=None,
                        help="Custom base URL for OpenAI-compatible providers")

    args = parser.parse_args()

    has_model = args.model or args.config
    has_export = args.latex or args.plot

    if not has_model and not has_export:
        parser.error("one of --model, --config, --latex, or --plot is required")

    # --- Standalone export mode: no benchmark run needed ---
    if has_export and not has_model:
        from .export import load_result_files, find_result_files, generate_latex_table, generate_plot

        if args.results:
            paths = [Path(r) for r in args.results]
        else:
            paths = find_result_files(args.output_dir)
            if not paths:
                sys.exit(f"No result files found in {args.output_dir}")

        result_data = load_result_files(paths)
        if not result_data:
            sys.exit("No valid result data found.")

        if args.latex:
            latex = generate_latex_table(result_data, output_dir=args.output_dir)
            print(f"  LaTeX table saved to {Path(args.output_dir) / 'bench_results.tex'}")
        if args.plot:
            generate_plot(result_data, output_dir=args.output_dir)
        return

    # --- Benchmark run (optionally followed by export) ---
    config = None
    if args.config:
        config = _load_config(args.config)

    # Resolve defaults (config file can override)
    defaults = {
        "max_tokens": (args.max_tokens if args.max_tokens is not None
                       else config.get("max_tokens", 1024) if config else 1024),
        "temperature": (args.temperature if args.temperature is not None
                       else config.get("temperature", 0.0) if config else 0.0),
        "rate_limit_delay": (args.rate_limit if args.rate_limit is not None
                             else config.get("rate_limit", 0.5) if config else 0.5),
    }

    models = _parse_models(args)
    judge = _get_judge(args, config)

    for i, model_cfg in enumerate(models):
        if len(models) > 1:
            print(f"\n{'=' * 60}")
            print(f"  Benchmarking model {i + 1}/{len(models)}: {model_cfg['name']}")
            print(f"{'=' * 60}\n")

        run_benchmark(
            model=model_cfg["name"],
            provider=model_cfg["provider"],
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_tokens=defaults["max_tokens"],
            temperature=defaults["temperature"],
            limit=args.limit,
            categories=args.categories,
            difficulties=args.difficulties,
            types=args.types,
            rate_limit_delay=defaults["rate_limit_delay"],
            host=model_cfg["host"],
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            **judge,
            verbose=True,
        )

    # --- Export after benchmark run ---
    if has_export:
        from .export import load_result_files, find_result_files, generate_latex_table, generate_plot

        paths = find_result_files(args.output_dir, latest=True)
        if not paths:
            sys.exit(f"No result files found in {args.output_dir}")

        result_data = load_result_files(paths)

        if args.latex:
            generate_latex_table(result_data, output_dir=args.output_dir)
            print(f"  LaTeX table saved to {Path(args.output_dir) / 'bench_results.tex'}")
        if args.plot:
            generate_plot(result_data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()