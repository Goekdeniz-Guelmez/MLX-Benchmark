"""Prompt construction for each question type."""

SYSTEM_PROMPT = (
    "You are an expert in Apple's MLX framework for Apple Silicon. "
    "Answer concisely and accurately. When asked to write code, only output "
    "the code (no prose explanation). When asked a true/false question, start "
    "your answer with 'True' or 'False'. When asked a multiple-choice question, "
    "start your answer with the letter of your choice (e.g. 'A)')."
)

TYPE_SUFFIXES = {
    "coding": "\n\nOutput only the code — no explanation.",
    "fill_blank": "\n\nOutput only the completed code — no explanation.",
    "debug": "\n\nFirst state the bug, then show the corrected code.",
    "true_false": "\n\nStart your answer with 'True:' or 'False:'.",
    "mcq": "\n\nStart your answer with the letter of your choice (e.g. 'A)').",
}


def build_prompt(sample: dict) -> tuple[str, str]:
    """Build system prompt and user message for a dataset sample.

    Returns:
        (system_prompt, user_message) tuple.
    """
    question = sample["question"]
    suffix = TYPE_SUFFIXES.get(sample["type"], "")
    return SYSTEM_PROMPT, question + suffix