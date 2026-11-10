"""Deterministic zero-overlap dataset builder for C-BARR."""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
from typing import Any
from datasets import load_dataset

# Exhaustive 25-type IFEval routing partition: 10 Planning + 15 Precision (5 Inhibition + 10 Structural)
PLANNING_IDS: frozenset[str] = frozenset({
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:nth_paragraph_first_word",
    "detectable_format:number_bullet_lists",
    "detectable_format:multiple_sections",
    "detectable_format:number_highlighted_sections",
    "keywords:frequency",
    "keywords:letter_frequency",
    "combination:two_responses",
})

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

PRECISION_IDS: frozenset[str] = INHIBITION_IDS | STRUCTURAL_IDS
assert len(PLANNING_IDS) == 10 and len(PRECISION_IDS) == 15
assert not (PLANNING_IDS & PRECISION_IDS) and not (INHIBITION_IDS & STRUCTURAL_IDS)

# Upstream google/IFEval prompt-kwargs mismatch keys documented in google-research Issue #3103
KNOWN_UPSTREAM_DEFECT_KEYS: frozenset[int] = frozenset({30, 337, 2078, 3311})


def classify_ifeval_slice(inst_ids: list[str]) -> str:
    has_plan = any(i in PLANNING_IDS for i in inst_ids)
    has_prec = any(i in PRECISION_IDS for i in inst_ids)
    if has_plan and has_prec:
        return "ifeval_conflict"
    if has_plan:
        return "ifeval_planning"
    return "ifeval_precision"


def classify_precision_subtypes(inst_ids: list[str]) -> dict[str, bool]:
    return {
        "has_inhibition": any(i in INHIBITION_IDS for i in inst_ids),
        "has_structural": any(i in STRUCTURAL_IDS for i in inst_ids),
    }


def build_all_splits(out_dir: str = "data", seed: int = 42) -> dict[str, str]:
    rng = random.Random(seed)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. GSM8K: seed/dev from 'train' (7,473), eval_150 from 'test' (1,319)
    gsm_train = list(load_dataset("openai/gsm8k", "main", split="train"))
    gsm_test = list(load_dataset("openai/gsm8k", "main", split="test"))
    rng.shuffle(gsm_train)
    rng.shuffle(gsm_test)

    def parse_gsm(ex: dict[str, Any], idx: int, split_tag: str) -> dict[str, Any]:
        rationale, gold_str = ex["answer"].split("####")
        steps = len([ln for ln in rationale.strip().splitlines() if ln.strip()])
        gold_val = float(gold_str.strip().replace(",", ""))
        sub = "gsm8k_easy" if steps <= 2 else ("gsm8k_hard" if steps >= 4 else "gsm8k_mid")
        return {
            "uid": f"gsm8k_{split_tag}_{idx}",
            "dataset": "gsm8k",
            "slice": sub,
            "prompt": ex["question"].strip(),
            "gold": gold_val,
            "steps": steps,
            "rationale_gold": rationale.strip(),
        }

    gsm_tr_parsed = [parse_gsm(e, i, "tr") for i, e in enumerate(gsm_train)]
    gsm_te_parsed = [parse_gsm(e, i, "te") for i, e in enumerate(gsm_test)]

    seed_gsm = gsm_tr_parsed[:15]
    dev_gsm_easy = [x for x in gsm_tr_parsed[15:] if x["slice"] == "gsm8k_easy"][:10]
    dev_gsm_hard = [x for x in gsm_tr_parsed[15:] if x["slice"] == "gsm8k_hard"][:10]
    eval_gsm_easy = [x for x in gsm_te_parsed if x["slice"] == "gsm8k_easy"][:75]
    eval_gsm_hard = [x for x in gsm_te_parsed if x["slice"] == "gsm8k_hard"][:75]

    # 2. CommonsenseQA: seed/dev from 'train', eval_150 from 'validation'
    csqa_train = list(load_dataset("tau/commonsense_qa", split="train"))
    csqa_val = list(load_dataset("tau/commonsense_qa", split="validation"))
    rng.shuffle(csqa_train)
    rng.shuffle(csqa_val)

    def parse_csqa(ex: dict[str, Any], idx: int, split_tag: str) -> dict[str, Any]:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        opts = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
        return {
            "uid": f"csqa_{split_tag}_{idx}",
            "dataset": "csqa",
            "slice": "csqa",
            "prompt": f"{ex['question'].strip()}\n{opts}",
            "gold": ex["answerKey"].strip().upper(),
        }

    seed_csqa = [parse_csqa(e, i, "tr") for i, e in enumerate(csqa_train[:15])]
    dev_csqa = [parse_csqa(e, i, "tr") for i, e in enumerate(csqa_train[15:35], start=15)]
    eval_csqa = [parse_csqa(e, i, "val") for i, e in enumerate(csqa_val[:150])]

    # 3. IFEval: partition 541 prompts into seed (30), dev (20), eval (150),
    #    calibration-disjoint diag_491, and strictly eval-disjoint diag_341_strict
    ifeval_raw = list(load_dataset("google/IFEval", split="train"))
    assert len(ifeval_raw) == 541, f"Expected 541 IFEval rows, got {len(ifeval_raw)}"
    rng.shuffle(ifeval_raw)

    ifeval_all: list[dict[str, Any]] = []
    for e in ifeval_raw:
        inst_ids = list(e["instruction_id_list"])
        sl = classify_ifeval_slice(inst_ids)
        sub_flags = classify_precision_subtypes(inst_ids)
        key_int = int(e["key"])
        ifeval_all.append({
            "uid": f"ifeval_{key_int}",
            "key": key_int,
            "dataset": "ifeval",
            "slice": sl,
            "prompt": e["prompt"],
            "instruction_id_list": inst_ids,
            "kwargs": e["kwargs"],
            "has_inhibition": sub_flags["has_inhibition"],
            "has_structural": sub_flags["has_structural"],
            "has_upstream_defect": key_int in KNOWN_UPSTREAM_DEFECT_KEYS,
        })

    plan_pool = [x for x in ifeval_all if x["slice"] == "ifeval_planning"]
    prec_pool = [x for x in ifeval_all if x["slice"] == "ifeval_precision"]
    conf_pool = [x for x in ifeval_all if x["slice"] == "ifeval_conflict"]

    assert len(plan_pool) >= 85, f"Insufficient pure-planning prompts: {len(plan_pool)}"
    assert len(prec_pool) >= 85, f"Insufficient pure-precision prompts: {len(prec_pool)}"
    assert len(conf_pool) >= 30, f"Insufficient conflict prompts: {len(conf_pool)}"

    seed_ifeval = plan_pool[:15] + prec_pool[:15]
    dev_ifeval = plan_pool[15:25] + prec_pool[15:25]
    eval_ifeval = plan_pool[25:85] + prec_pool[25:85] + conf_pool[:30]

    cal_uids = {x["uid"] for x in (seed_ifeval + dev_ifeval)}
    eval_ifeval_uids = {x["uid"] for x in eval_ifeval}

    diag_ifeval_491 = [x for x in ifeval_all if x["uid"] not in cal_uids]
    diag_ifeval_341_strict = [
        x for x in ifeval_all if x["uid"] not in (cal_uids | eval_ifeval_uids)
    ]

    splits: dict[str, list[dict[str, Any]]] = {
        "seed_60.jsonl": seed_gsm + seed_csqa + seed_ifeval,
        "dev_60.jsonl": dev_gsm_easy + dev_gsm_hard + dev_csqa + dev_ifeval,
        "eval_450.jsonl": eval_gsm_easy + eval_gsm_hard + eval_csqa + eval_ifeval,
        "ifeval_diag_491.jsonl": diag_ifeval_491,
        "ifeval_diag_341_strict.jsonl": diag_ifeval_341_strict,
    }

    assert len(splits["seed_60.jsonl"]) == 60
    assert len(splits["dev_60.jsonl"]) == 60
    assert len(splits["eval_450.jsonl"]) == 450
    assert len(splits["ifeval_diag_491.jsonl"]) == 491
    assert len(splits["ifeval_diag_341_strict.jsonl"]) == 341

    u_seed = {x["uid"] for x in splits["seed_60.jsonl"]}
    u_dev = {x["uid"] for x in splits["dev_60.jsonl"]}
    u_eval = {x["uid"] for x in splits["eval_450.jsonl"]}
    u_diag_strict = {x["uid"] for x in splits["ifeval_diag_341_strict.jsonl"]}
    assert not (u_seed & u_dev)
    assert not (u_seed & u_eval)
    assert not (u_dev & u_eval)
    assert not ((u_seed | u_dev | u_eval) & u_diag_strict)

    manifest: dict[str, str] = {}
    for fname, rows in splits.items():
        fpath = out / fname
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        fpath.write_text(payload, encoding="utf-8")
        manifest[fname] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    (out / "manifest_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    checksums = build_all_splits()
    print("Frozen C-BARR splits with SHA-256 manifest:", json.dumps(checksums, indent=2))
