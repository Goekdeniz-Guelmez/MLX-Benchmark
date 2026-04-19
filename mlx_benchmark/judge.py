"""Answer evaluation — fast heuristic matching + LLM-as-judge fallback."""

import re

from .backends.base import Backend

JUDGE_SYSTEM = (
    "You are a strict technical evaluator for MLX (Apple's ML framework) answers. "
    "Compare the MODEL_ANSWER to the REFERENCE_ANSWER and determine if the model's "
    "answer is correct. Be strict but fair — minor formatting differences are OK, "
    "but the core technical content must be right.\n\n"
    "Reply with ONLY 'CORRECT' or 'INCORRECT' followed by a brief reason."
)


def quick_match(sample: dict, model_answer: str) -> float | None:
    """Fast heuristic scoring for structured question types.

    Returns:
        1.0 (correct), 0.0 (incorrect), or None (fall through to LLM judge).
    """
    if not model_answer:
        return 0.0
    stype = sample["type"]
    ref = sample["answer"].strip()
    ans = model_answer.strip()

    # True/False — match the leading boolean keyword
    if stype == "true_false":
        ref_bool = ref.lower().startswith("true")
        ans_bool = ans.lower().startswith("true")
        return 1.0 if ref_bool == ans_bool else 0.0

    # MCQ — match the leading letter (A/B/C/D)
    if stype == "mcq":
        ref_letter = re.match(r"\s*([A-D])", ref, re.IGNORECASE)
        ans_letter = re.match(r"\s*([A-D])", ans, re.IGNORECASE)
        if ref_letter and ans_letter:
            return 1.0 if ref_letter.group(1).upper() == ans_letter.group(1).upper() else 0.0
        return None  # can't parse → fall through

    # All other types need semantic judgment
    return None


def llm_judge(backend: Backend, question: str, model_answer: str, reference_answer: str) -> bool:
    """Use an LLM backend to judge answer correctness.

    Returns:
        True if the judge deems the answer correct, False otherwise.
    """
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"MODEL_ANSWER:\n{model_answer}\n\n"
        f"REFERENCE_ANSWER:\n{reference_answer}"
    )
    response = backend.generate(prompt=prompt, system=JUDGE_SYSTEM, max_tokens=128, temperature=0.0)
    return bool(response) and response.strip().upper().startswith("CORRECT")


def evaluate(backend: Backend, sample: dict, model_answer: str) -> bool:
    """Evaluate a model answer against a reference.

    Uses fast heuristics for structured types (true_false, mcq) and falls
    back to LLM-as-judge for open-ended types (qa, coding, debug, fill_blank).
    """
    score = quick_match(sample, model_answer)
    if score is not None:
        return score == 1.0
    return llm_judge(backend, sample["question"], model_answer, sample["answer"])