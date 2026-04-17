---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - mlx
  - apple
  - apple-silicon
  - benchmark
  - evaluation
  - llm-benchmark
  - coding
  - debugging
  - machine-learning
  - deep-learning
  - unified-memory
  - metal
size_categories:
  - n<1K
---

# MLX Benchmark Dataset

## Table of Contents

- [Dataset Description](#dataset-description)
- [Supported Tasks](#supported-tasks)
- [Dataset Structure](#dataset-structure)
- [Data Fields](#data-fields)
- [Data Statistics](#data-statistics)
- [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
- [Curation Rationale](#curation-rationale)
- [Source Data](#source-data)
- [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
- [Social Impact of Dataset](#social-impact-of-dataset)
- [Discussion of Biases](#discussion-of-biases)
- [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
- [Citation](#citation)

---

## Dataset Description

- **Repository:** [Goekdeniz-Guelmez/MLX-Benchmark](https://github.com/Goekdeniz-Guelmez/MLX-Benchmark)
- **Paper:** *Coming soon*
- **Point of Contact:** Gökdeniz Gülmez

### Dataset Summary

The **MLX Benchmark Dataset** is a curated evaluation benchmark consisting of **520 questions** designed to measure large language model (LLM) proficiency in Apple's **MLX** machine learning framework. MLX is an array framework for machine learning on Apple Silicon that leverages unified memory architecture, and this dataset is the first comprehensive benchmark specifically targeting MLX knowledge and coding ability.

The dataset covers the full breadth of the MLX ecosystem — from core array operations (`mlx.core`) and neural network building blocks (`mlx.nn`), to higher-level libraries like `mlx-lm` (language model inference and LoRA fine-tuning), `mlx-vlm` (vision-language models), and `mlx-embeddings` (embedding models). Questions span six distinct task types (knowledge QA, multiple choice, true/false, fill-in-the-blank, full code generation, and debugging) across four difficulty levels (easy, medium, hard, very-hard).

This dataset is intended for benchmarking LLMs on their ability to understand, write, and debug MLX code — a domain that is underrepresented in existing benchmarks despite MLX's growing adoption in the Apple Silicon ML ecosystem.

### Why MLX?

Apple's MLX framework introduces paradigms that differ significantly from PyTorch and JAX:

- **Unified memory** — no explicit device transfers between CPU and GPU
- **Lazy evaluation** — operations build a computation graph that is only executed when explicitly materialized with `mx.eval()`
- **Function transforms** — `mx.grad()`, `mx.vmap()`, and `mx.compile()` as first-class composable primitives
- **Metal-accelerated backends** — automatic GPU dispatch via streams

These differences mean that models trained on general code corpora often struggle with MLX-specific patterns. This benchmark directly measures that gap.

---

## Supported Tasks

| Task | Type | Description |
|------|------|-------------|
| **Knowledge QA** | `question-answering` | Free-form questions about MLX APIs, behavior, and semantics |
| **Multiple Choice** | `question-answering` | Select the correct answer from 4 options (A/B/C/D) |
| **True/False** | `question-answering` | Determine whether a statement about MLX is correct |
| **Fill-in-the-Blank** | `text-generation` | Complete partial MLX code snippets |
| **Code Generation** | `text-generation` | Write complete MLX programs from scratch |
| **Debugging** | `text-generation` | Identify and fix bugs in MLX code |

---

## Dataset Structure

### Data Instances

Each line in the JSONL file is a single benchmark question. Example entries for each type:

**QA** (`type: "qa"`):
```json
{
  "question": "How do you create a 3x4 matrix of zeros in MLX?",
  "answer": "mx.zeros([3, 4])",
  "category": "mlx_core",
  "subcategory": "array_creation",
  "difficulty": "easy",
  "type": "qa"
}
```

**Fill-in-the-Blank** (`type: "fill_blank"`):
```json
{
  "question": "Complete this code to compute gradients:\n\ndef loss(w, x, y):\n    return mx.mean((w @ x - y) ** 2)\n\n# Get the gradient function:\ngrad_fn = ___\n# Compute gradient w.r.t. w:\ngrads = ___",
  "answer": "grad_fn = mx.grad(loss)\ngrads = grad_fn(w, x, y)",
  "category": "coding",
  "subcategory": "autodiff",
  "difficulty": "easy",
  "type": "fill_blank"
}
```

**Multiple Choice** (`type: "mcq"`):
```json
{
  "question": "Which of these is NOT a valid MLX function transform?\nA) mx.grad\nB) mx.vmap\nC) mx.jit\nD) mx.compile",
  "answer": "C) mx.jit — MLX uses mx.compile for JIT compilation, not mx.jit (which is a JAX convention). mx.grad, mx.vmap, and mx.compile are all valid MLX transforms.",
  "category": "mlx_core",
  "subcategory": "transforms",
  "difficulty": "medium",
  "type": "mcq"
}
```

**True/False** (`type: "true_false"`):
```json
{
  "question": "True or False: MLX arrays must be explicitly moved to GPU with a .to() call before GPU operations.",
  "answer": "False. MLX uses unified memory — there is no device-level array placement. All arrays live in shared memory, and device selection happens per-operation via the stream parameter.",
  "category": "mlx_core",
  "subcategory": "unified_memory",
  "difficulty": "easy",
  "type": "true_false"
}
```

**Code Generation** (`type: "coding"`):
```json
{
  "question": "Write a complete MLX training loop for a 2-layer MLP classifier on 10 classes with AdamW optimizer.",
  "answer": "import mlx.core as mx\nimport mlx.nn as nn\nimport mlx.optimizers as optim\n\nclass MLP(nn.Module):\n    def __init__(self, input_dim, hidden_dim, output_dim):\n        super().__init__()\n        self.fc1 = nn.Linear(input_dim, hidden_dim)\n        self.fc2 = nn.Linear(hidden_dim, output_dim)\n\n    def __call__(self, x):\n        return self.fc2(nn.relu(self.fc1(x)))\n\nmodel = MLP(128, 256, 10)\noptimizer = optim.AdamW(learning_rate=1e-3)\n\ndef loss_fn(model, x, y):\n    logits = model(x)\n    return nn.losses.cross_entropy(logits, y).mean()\n\nloss_and_grad = mx.value_and_grad(loss_fn)\n\nfor epoch in range(10):\n    x = mx.random.normal([32, 128])\n    y = mx.random.randint(0, 10, [32])\n    loss, grads = loss_and_grad(model, x, y)\n    optimizer.update(model, grads)\n    mx.eval(model.parameters(), optimizer.state, loss)\n    print(f\"Epoch {epoch}: loss={loss.item():.4f}\")",
  "category": "coding",
  "subcategory": "training_loop",
  "difficulty": "medium",
  "type": "coding"
}
```

**Debugging** (`type: "debug"`):
```json
{
  "question": "What is wrong with this MLX training loop?\n\nfor x, y in dataloader:\n    loss, grads = loss_and_grad(model, x, y)\n    optimizer.update(model, grads)\n    print(f'loss: {loss.item()}')",
  "answer": "Missing mx.eval() call. The optimizer.update() only builds a lazy computation graph — values are never materialized. Fix:\n    optimizer.update(model, grads)\n    mx.eval(model.parameters(), optimizer.state, loss)\n    print(f'loss: {loss.item()}')",
  "category": "debugging",
  "subcategory": "lazy_eval",
  "difficulty": "medium",
  "type": "debug"
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The benchmark question or prompt. May contain multi-line code snippets for `fill_blank`, `coding`, and `debug` types. |
| `answer` | `string` | The reference (ground-truth) answer. For `mcq`, includes the correct letter and explanation. For `coding`, contains complete executable code. For `debug`, identifies the bug and provides the fix. |
| `category` | `string` | Top-level topic category (see [Categories](#categories) below). |
| `subcategory` | `string` | Fine-grained topic within the category (e.g., `array_creation`, `attention`, `lora_finetuning`). |
| `difficulty` | `string` | One of `easy`, `medium`, `hard`, `very-hard`. |
| `type` | `string` | Question format: `qa`, `mcq`, `true_false`, `fill_blank`, `coding`, or `debug`. |

### Data Splits

This dataset has a **single split** — all 520 questions are intended to be used together as a benchmark. There is no train/validation/test split because the dataset is designed for **evaluation**, not training. Using these questions for training would contaminate the benchmark.

---

## Data Statistics

### Overall

| Metric | Value |
|--------|-------|
| Total questions | 520 |
| Question types | 6 |
| Categories | 11 |
| Subcategories | 90+ |
| Difficulty levels | 4 |

### By Question Type

| Type | Count | Percentage |
|------|-------|-----------|
| `qa` | 432 | 83.1% |
| `coding` | 33 | 6.3% |
| `debug` | 21 | 4.0% |
| `mcq` | 12 | 2.3% |
| `true_false` | 12 | 2.3% |
| `fill_blank` | 10 | 1.9% |

### By Difficulty

| Difficulty | Count | Percentage |
|-----------|-------|-----------|
| `easy` | 180 | 34.6% |
| `medium` | 181 | 34.8% |
| `hard` | 109 | 21.0% |
| `very-hard` | 50 | 9.6% |

### By Category

| Category | Count | Description |
|----------|-------|-------------|
| `mlx_core` | 188 | Core array operations, transforms, lazy evaluation, unified memory |
| `mlx_nn` | 73 | Neural network layers, modules, activations, losses |
| `mlx_lm_lora` | 55 | LoRA fine-tuning with mlx-lm |
| `mlx_lm` | 61 | Language model loading, inference, generation, server |
| `coding` | 35 | General code writing tasks (training loops, attention, custom layers) |
| `mlx_embeddings` | 21 | Embedding model usage and inference |
| `debugging` | 21 | Debugging MLX code (lazy eval pitfalls, shape errors, etc.) |
| `mlx_optimizers` | 19 | Optimizer usage, LR scheduling, training algorithms |
| `mlx_vlm` | 19 | Vision-language model inference and usage |
| `conceptual` | 13 | Conceptual understanding of MLX design philosophy |
| `mlx_embeddings_lora` | 15 | LoRA fine-tuning with mlx-embeddings |

### Type x Difficulty Cross-Tabulation

| Type | Easy | Medium | Hard | Very-Hard |
|------|------|--------|------|-----------|
| `qa` | 164 | 148 | 89 | 31 |
| `coding` | 2 | 10 | 9 | 12 |
| `debug` | 2 | 7 | 5 | 7 |
| `fill_blank` | 5 | 5 | 0 | 0 |
| `mcq` | 2 | 5 | 5 | 0 |
| `true_false` | 5 | 6 | 1 | 0 |

### Category x Difficulty Cross-Tabulation

| Category | Easy | Medium | Hard | Very-Hard |
|----------|------|--------|------|-----------|
| `mlx_core` | 92 | 54 | 33 | 9 |
| `mlx_nn` | 33 | 31 | 9 | 0 |
| `mlx_lm_lora` | 19 | 14 | 16 | 6 |
| `mlx_lm` | 14 | 21 | 15 | 11 |
| `coding` | 3 | 11 | 9 | 12 |
| `mlx_embeddings` | 4 | 11 | 5 | 1 |
| `debugging` | 2 | 7 | 5 | 7 |
| `mlx_optimizers` | 4 | 9 | 6 | 0 |
| `mlx_vlm` | 4 | 9 | 3 | 3 |
| `conceptual` | 1 | 6 | 6 | 0 |
| `mlx_embeddings_lora` | 4 | 8 | 2 | 1 |

---

## Categories

The dataset covers 11 categories spanning the entire MLX ecosystem:

### `mlx_core` (188 questions)
Core array operations and framework fundamentals. Subcategories include:
- **array_creation** — `mx.zeros`, `mx.ones`, `mx.array`, `mx.arange`, `mx.linspace`
- **array_manipulation** — reshaping, slicing, concatenation, broadcasting
- **math_ops** — element-wise operations, reductions, linear algebra
- **transforms** — `mx.grad`, `mx.vmap`, `mx.compile` and their composition
- **lazy_eval** / **lazy_evaluation** — understanding when computation is triggered, `mx.eval()` semantics
- **unified_memory** — the absence of device transfers, stream-based dispatch
- **autodiff** — automatic differentiation, gradient computation, `value_and_grad`
- **data_format** — dtype handling, casting, half-precision support

### `mlx_nn` (73 questions)
Neural network building blocks. Subcategories include:
- **layers** — `Linear`, `Conv1d`, `Conv2d`, `Embedding`, `RMSNorm`, `LayerNorm`
- **activations** — `relu`, `gelu`, `silu`, `tanh`
- **normalization** — batch norm, layer norm, RMS norm
- **loss_functions** / **losses** — cross-entropy, MSE, custom losses
- **initialization** — weight init strategies, parameter management
- **modules** — custom `nn.Module` definitions, `__call__` convention

### `mlx_lm_lora` (55 questions)
LoRA fine-tuning with mlx-lm. Subcategories include:
- **lora_finetuning** — full LoRA training workflows
- **lora_params** — rank, alpha, dropout configuration
- **adapter_methods** — different adapter architectures
- **training_algorithms** — training recipes, hyperparameter selection

### `mlx_lm` (61 questions)
Language model inference and serving. Subcategories include:
- **loading** — loading models and tokenizers from Hugging Face
- **generation** — text generation, sampling strategies, chat templates
- **cli** — `mlx_lm.generate`, `mlx_lm.lora`, `mlx_lm.server` CLI tools
- **server** — OpenAI-compatible server deployment
- **inference** — inference configuration, quantization, batching
- **supported_models** — which architectures are supported

### `coding` (35 questions)
Full code writing tasks that require producing complete, executable MLX programs. Subcategories include:
- **training_loop** — writing training loops with proper `mx.eval()` calls
- **attention** — implementing scaled dot-product, multi-head attention
- **custom_layers** — building custom `nn.Module` subclasses
- **training_algorithms** — implementing RL, DPO, or other training recipes

### `debugging` (21 questions)
Identifying and fixing bugs in MLX code. Common bug patterns include:
- Missing `mx.eval()` calls (the #1 MLX pitfall)
- Incorrect `mx.grad` usage (forgetting to call the returned function)
- Shape mismatches in lazy-evaluated graphs
- Wrong stream/device assumptions

### `mlx_embeddings` (21 questions)
Embedding model usage with mlx-embeddings.

### `mlx_embeddings_lora` (15 questions)
LoRA fine-tuning for embedding models.

### `mlx_optimizers` (19 questions)
Optimizer usage and learning rate scheduling. Covers `Adam`, `AdamW`, `SGD`, `Lion`, and `ScheduleFree` optimizers, as well as LR schedulers (`cosine_decay`, `step_decay`, `exponential_decay`).

### `mlx_vlm` (19 questions)
Vision-language model inference. Covers image prompting, multi-modal generation, and supported VLM architectures.

### `conceptual` (13 questions)
Conceptual understanding questions about MLX's design philosophy, unified memory architecture, and how it differs from PyTorch/JAX.

---

## Difficulty Levels

| Level | Description |
|-------|-------------|
| **easy** | Single API calls, basic array operations, straightforward knowledge questions. A model with basic MLX familiarity should answer correctly. Example: "How do you create a 3x4 matrix of zeros in MLX?" |
| **medium** | Multi-step reasoning, combining 2-3 MLX concepts, understanding lazy evaluation semantics, configuring LoRA training. Requires working knowledge of the framework. |
| **hard** | Complex code generation, multi-concept integration, debugging subtle lazy evaluation issues, understanding MLX internals. Requires deep familiarity with MLX patterns. |
| **very-hard** | Multi-file code generation, advanced training algorithms (DPO, RLHF), implementing custom attention mechanisms, non-obvious debugging scenarios. Requires expert-level MLX proficiency. |

The `very-hard` difficulty was introduced to push frontier models — questions at this level typically require producing 50+ lines of correct, idiomatic MLX code or identifying multi-step bugs in non-trivial training pipelines.

---

## Dataset Creation

### Curation Rationale

Existing LLM benchmarks (HumanEval, MBPP, MMLU, etc.) do not cover Apple's MLX framework. General coding benchmarks test Python proficiency broadly but miss MLX-specific paradigms like:

1. **Lazy evaluation** — forgetting `mx.eval()` is the most common MLX bug, and no other framework has this exact semantics
2. **Unified memory** — no `.to(device)` calls, which trips up PyTorch-trained models
3. **Function transform composition** — `mx.grad(mx.vmap(f))` works differently than JAX
4. **LoRA fine-tuning ecosystem** — `mlx-lm`, `mlx-embeddings`, `mlx-vlm` have their own CLI and Python APIs

This dataset fills that gap by providing a structured, multi-difficulty evaluation specifically for MLX knowledge and coding ability.

### Source Data

Questions were authored by MLX practitioners and framework contributors, drawing from:
- The [official MLX documentation](https://ml-explore.github.io/mlx/)
- The [mlx-lm repository](https://github.com/ml-explore/mlx-lm)
- The [mlx-examples repository](https://github.com/ml-explore/mlx-examples)
- Real-world MLX training code and common pitfalls encountered in practice
- The MLX GitHub issues and discussions

#### Data Collection and Processing

1. Questions were manually authored with reference answers verified against MLX source code and documentation
2. Each question was tagged with category, subcategory, difficulty, and type
3. Answers for `mcq` and `true_false` types were validated for unambiguous correctness
4. Code answers for `coding`, `fill_blank`, and `debug` types were verified for executability
5. The dataset was reviewed for duplicate or near-duplicate questions

### Personal and Sensitive Information

This dataset does not contain any personal, sensitive, or private information. All questions are about the MLX framework and its APIs. No user data, model outputs, or private code is included.

---

## Considerations for Using the Data

### Social Impact of Dataset

This dataset enables more rigorous evaluation of LLMs on Apple's MLX framework, which may help:

- Developers choose models for MLX-related tasks (code assistance, documentation, training)
- Framework maintainers understand where LLMs fail on MLX-specific concepts
- Researchers study how well LLMs transfer knowledge from PyTorch/JAX to MLX

Potential negative impacts: models fine-tuned on this dataset could overfit to the benchmark questions, making scores unreliable. We discourage using this dataset for training.

### Discussion of Biases

- **Coverage bias**: The dataset over-represents `mlx_core` (188/520 = 36.2%) and under-represents `mlx_vlm` (19/520 = 3.7%), reflecting the relative maturity and documentation coverage of these sub-ecosystems.
- **Type bias**: 83.1% of questions are `qa` type, while `fill_blank` (1.9%) and `mcq` (2.3%) are underrepresented. This reflects the nature of MLX knowledge — most questions naturally take a free-form QA format.
- **Difficulty bias**: Easy and medium questions together make up 69.4% of the dataset. Very-hard questions (9.6%) are concentrated in `coding` and `debug` types, which may skew difficulty-by-type comparisons.
- **Framework version bias**: Questions are based on MLX as of early 2025. API changes in future MLX releases may make some answers outdated.

### Other Known Limitations

1. **No execution-based evaluation**: The benchmark evaluates answers via LLM judge (for `qa`, `fill_blank`, `coding`, `debug`) or exact/pattern matching (for `mcq`, `true_false`). Code answers are not executed, so subtle runtime bugs may be missed.
2. **Single correct answer**: Most questions have one canonical answer. Alternative correct solutions (especially in `coding` tasks) may be marked incorrect by a strict judge.
3. **English only**: All questions are in English.
4. **Static dataset**: The dataset does not evolve with MLX releases. Users should verify answers against the latest MLX documentation.
5. **Small size**: At 520 questions, the dataset may not cover all edge cases or rare MLX features. Confidence intervals on accuracy scores are relatively wide.

---

## Evaluation

### Using the Benchmark CLI

The dataset is bundled with the `mlx-bench` CLI tool:

```bash
pip install mlx-benchmark

# Benchmark a local Ollama model
mlx-bench --model llama3.2

# Benchmark with a cloud provider
mlx-bench --provider anthropic --model claude-sonnet-4-20250514

# Filter by difficulty or type
mlx-bench --model llama3.2 --difficulties hard very-hard --types coding debug

# Generate LaTeX table and PNG chart
mlx-bench --latex --plot
```

### Using the Python API

```python
from mlx_benchmark import run_benchmark

results, stats = run_benchmark(
    model="llama3.2",
    provider="ollama",
    types=["coding", "debug"],
    difficulties=["hard", "very-hard"],
)

print(f"Overall accuracy: {stats.accuracy:.1f}%")
print(f"By difficulty: {stats.by_difficulty}")
print(f"By type: {stats.by_type}")
print(f"By category: {stats.by_category}")
```

### Evaluation Methodology

- **`mcq` and `true_false`**: Evaluated via exact matching (letter extraction for MCQ, keyword matching for true/false). No LLM judge needed.
- **`qa`, `fill_blank`, `coding`, `debug`**: Evaluated via an LLM judge that compares the model's answer against the reference answer. The judge is prompted to be strict — only minor formatting differences are tolerated.
- **Scoring**: Each question is scored as correct/incorrect. Aggregate accuracy is computed overall and per breakdown (type, difficulty, category).

### Exporting Results for Publication

```bash
# LaTeX table (booktabs format, two tables: by difficulty and by type)
mlx-bench --latex --results results/bench_*.json

# PNG bar chart (grouped bars: Overall, Easy, Medium, Hard per model)
mlx-bench --plot --results results/bench_*.json
```

---

## Additional Information

### Dataset Version

- **v2** — 520 questions, 6 types, 11 categories, 4 difficulty levels (easy, medium, hard, very-hard)

### Licensing Information

This dataset is released under the **MIT License**.

### Maintenance

The dataset is maintained by [Gökdeniz Gülmez](https://github.com/Goekdeniz-Guelmez). Bug reports, feature requests, and contributions should be directed to the [GitHub repository](https://github.com/Goekdeniz-Guelmez/MLX-Benchmark).

### Citation

```bibtex
@misc{mlx-benchmark-dataset,
  author = {G\"{u}lmez, G\"{o}kdeniz},
  title = {{MLX Benchmark}: Evaluating {LLMs} on {Apple MLX} Framework Knowledge and Coding Tasks},
  year = {2025},
  url = {https://github.com/Goekdeniz-Guelmez/MLX-Benchmark}
}
```