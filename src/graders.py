"""Deterministic channel parsing, 4-gram loop audit, CoT spillover audit, and GSM8K/CSQA/IFEval verifiers."""
from __future__ import annotations

from collections.abc import Sequence
import random
import re
from typing import Any
from langdetect import DetectorFactory

# Seed langdetect global RNG before importing instruction_following_eval (Defect D3)
DetectorFactory.seed = 0

try:
    from instruction_following_eval import evaluation_lib as _ifeval_mod  # noqa: E402
except ImportError:
    from instruction_following_eval import evaluation_main as _ifeval_mod  # noqa: E402
evaluation_main = _ifeval_mod if hasattr(_ifeval_mod, "InputExample") else __import__(
    "instruction_following_eval.evaluation_lib", fromlist=["InputExample"]
)

INHIBITION_IDS: frozenset[str] = frozenset({
    "punctuation:no_comma",
    "keywords:forbidden_words",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "language:response_language",
})

STRUCTURAL_IDS: frozenset[str] = frozenset({
    "detectable_format:json_format",
    "detectable_format:constrained_response",
    "detectable_format:title",
    "detectable_content:postscript",
    "detectable_content:number_placeholders",
    "startend:end_checker",
    "startend:quotation",
    "combination:repeat_prompt",
    "keywords:existence",
    "change_case:capital_word_frequency",
})

SPECIAL_END_TOKENS: Sequence[str] = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "</s>",
)

# Patch D16: Post-</think> CoT spillover detector on truncated runs
SPILLOVER_RE = re.compile(
    r"^\s*(?:wait[,:\s]|let's\s+think|step\s+1\b|hmm[,:\s])", re.IGNORECASE
)

NUM_TOKEN_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?")
BOXED_CONTENT_RE = re.compile(r"\\boxed\{\s*([^{}]+(?:\{[^{}]*\}[^{}]*)*)\s*\}")
HASH_ANS_RE = re.compile(r"####\s*\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
CUE_ANS_RE = re.compile(
    r"(?:final\s+answer|the\s+answer|answer|equals|total\s+is)"
    r"\s*\*{0,2}\s*(?:is|:|=)\s*\*{0,2}\s*\$?\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)

MC_BOXED_RE = re.compile(r"\\boxed\{\s*\(?([A-E])\)?\s*\}", re.IGNORECASE)
MC_EXPLICIT_RE = re.compile(
    r"(?:final\s+answer|correct\s+(?:answer|option|choice)|best\s+(?:answer|option|choice)|answer|option|choice)"
    r"(?:\s+here)?(?:\s+would\s+be|\s+is|\*{0,2}\s*:)\s*\*{0,2}\s*\(?([A-E])\)?(?![a-zA-Z])",
    re.IGNORECASE,
)
MC_LEADING_RE = re.compile(r"^\s*\*{0,2}\(?([A-E])\)?\*{0,2}(?:[\s.):,-]|$)")
MC_PAREN_RE = re.compile(r"\(([A-E])\)")


def strip_special_end_tokens(text: str) -> str:
    cleaned = text
    for tok in SPECIAL_END_TOKENS:
        cleaned = cleaned.replace(tok, "")
    return cleaned


def split_think_and_answer(
    raw_text: str, expect_think_close: bool, is_prompted_cot: bool = False
) -> tuple[str, str, bool]:
    """Returns (think_trace, clean_final_answer, think_exhausted)."""
    cleaned = strip_special_end_tokens(raw_text)
    if is_prompted_cot:
        if "Final Answer:" in cleaned:
            cot_part, final_part = cleaned.rsplit("Final Answer:", 1)
            return cot_part.strip(), final_part.strip(), False
        return "", cleaned.strip(), False

    if "</think>" in cleaned:
        think_part, ans_part = cleaned.rsplit("</think>", 1)
        think_part = re.sub(r"^\s*<think>\s*", "", think_part)
        return think_part, ans_part.lstrip("\n").strip(), False

    if expect_think_close or cleaned.lstrip().startswith("<think>"):
        return re.sub(r"^\s*<think>\s*", "", cleaned), "", True

    return "", cleaned.strip(), False


def has_4gram_repetition_loop(text: str, min_repeats: int = 4) -> bool:
    """Detects degenerate greedy decoding loops (>= 4 consecutive or >= 6 total 4-gram repeats)."""
    tokens = text.lower().split()
    if len(tokens) < 16:
        return False
    counts: dict[tuple[str, str, str, str], int] = {}
    for i in range(len(tokens) - 3):
        gram = (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= 6:
            return True
    for i in range(len(tokens) - 4 * min_repeats + 1):
        unit = tokens[i : i + 4]
        if all(tokens[i + 4 * r : i + 4 * (r + 1)] == unit for r in range(1, min_repeats)):
            return True
    return False


def has_cot_spillover(final_answer: str, truncated: bool = True) -> bool:
    """Detects post-</think> CoT discourse spillover into a_final when truncated=True (Patch D16)."""
    if not truncated or not final_answer:
        return False
    return bool(SPILLOVER_RE.search(final_answer))


def _parse_numeric_token(s: str) -> float | None:
    s = s.strip().replace(",", "").rstrip(".")
    if not s or s in ("-", "+"):
        return None
    if "/" in s:
        num_s, den_s = s.split("/", 1)
        den = float(den_s)
        return float(num_s) / den if abs(den) > 1e-12 else None
    try:
        return float(s)
    except ValueError:
        return None


def grade_gsm8k(final_answer: str, gold: float) -> int:
    if not final_answer:
        return 0
    boxed_matches = BOXED_CONTENT_RE.findall(final_answer)
    if boxed_matches:
        inner = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{[^{}]*\}", " ", boxed_matches[-1])
        inner = inner.replace("\\$", "").replace("$", "").replace("\\%", "")
        frac = re.search(r"\\d?frac\{\s*([-+]?\d+)\s*\}\{\s*([-+]?\d+)\s*\}", inner)
        if frac and abs(float(frac.group(2))) > 1e-12:
            val = float(frac.group(1)) / float(frac.group(2))
            return int(abs(val - gold) < 1e-4)
        nums = NUM_TOKEN_RE.findall(inner)
        if nums:
            val = _parse_numeric_token(nums[-1])
            if val is not None:
                return int(abs(val - gold) < 1e-4)

    for pattern in (HASH_ANS_RE, CUE_ANS_RE):
        matches = pattern.findall(final_answer)
        if matches:
            val = _parse_numeric_token(matches[-1])
            if val is not None:
                return int(abs(val - gold) < 1e-4)

    nums = NUM_TOKEN_RE.findall(final_answer)
    if not nums:
        return 0
    val = _parse_numeric_token(nums[-1])
    return int(val is not None and abs(val - gold) < 1e-4)


def grade_csqa(final_answer: str, gold_letter: str) -> int:
    if not final_answer:
        return 0
    gold_u = gold_letter.strip().upper()
    for pattern in (MC_BOXED_RE, MC_EXPLICIT_RE):
        matches = pattern.findall(final_answer)
        if matches:
            return int(matches[-1].upper() == gold_u)
    m_lead = MC_LEADING_RE.match(final_answer)
    if m_lead:
        return int(m_lead.group(1).upper() == gold_u)
    paren_matches = MC_PAREN_RE.findall(final_answer)
    if paren_matches:
        return int(paren_matches[-1].upper() == gold_u)
    cleaned = re.sub(r"\bA\s+[a-z]{2,}\b", " ", final_answer)
    letters = re.findall(r"\b([A-E])\b", cleaned)
    return int(bool(letters) and letters[-1].upper() == gold_u)


def grade_ifeval_item(item: dict[str, Any], final_answer: str) -> dict[str, Any]:
    """Runs official google-research instruction_following_eval strict & loose verifiers."""
    DetectorFactory.seed = 0
    random.seed(0)

    inst_ids: list[str] = list(item["instruction_id_list"])
    inp = evaluation_main.InputExample(
        key=item["key"],
        instruction_id_list=inst_ids,
        prompt=item["prompt"],
        kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in item["kwargs"]],
    )
    if not final_answer:
        n_inst = len(inst_ids)
        return {
            "prompt_strict": 0,
            "prompt_loose": 0,
            "inst_strict_list": [False] * n_inst,
            "inst_loose_list": [False] * n_inst,
            "inhibition_strict_all": 0 if any(i in INHIBITION_IDS for i in inst_ids) else None,
            "structural_strict_all": 0 if any(i in STRUCTURAL_IDS for i in inst_ids) else None,
        }

    out_strict = evaluation_main.test_instruction_following_strict(
        inp, {item["prompt"]: final_answer}
    )
    out_loose = evaluation_main.test_instruction_following_loose(
        inp, {item["prompt"]: final_answer}
    )
    strict_flags = list(out_strict.follow_instruction_list)
    inh_indices = [idx for idx, iid in enumerate(inst_ids) if iid in INHIBITION_IDS]
    str_indices = [idx for idx, iid in enumerate(inst_ids) if iid in STRUCTURAL_IDS]

    return {
        "prompt_strict": int(out_strict.follow_all_instructions),
        "prompt_loose": int(out_loose.follow_all_instructions),
        "inst_strict_list": strict_flags,
        "inst_loose_list": list(out_loose.follow_instruction_list),
        "inhibition_strict_all": (
            int(all(strict_flags[idx] for idx in inh_indices)) if inh_indices else None
        ),
        "structural_strict_all": (
            int(all(strict_flags[idx] for idx in str_indices)) if str_indices else None
        ),
    }


def grade_item(item: dict[str, Any], final_answer: str) -> dict[str, Any]:
    """Unified deterministic dispatcher across GSM8K, CommonsenseQA, and IFEval."""
    ds = item["dataset"]
    if ds == "gsm8k":
        return {"z": grade_gsm8k(final_answer, float(item["gold"]))}
    if ds == "csqa":
        return {"z": grade_csqa(final_answer, str(item["gold"]))}
    if ds == "ifeval":
        res = grade_ifeval_item(item, final_answer)
        res["z"] = res["prompt_strict"]
        return res
    raise ValueError(f"Unsupported dataset: {ds}")
