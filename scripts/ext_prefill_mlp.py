"""Optional Workstream W7: 2-Layer Prefill Hidden-State MLP Probe vs. Gate-2 Escalation."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_matrix import resolve_device_and_dtype
from src.engine import run_gate1_lexical
from src.metrics import evaluate_system_table
from src.routers import _get_slow_rec, select_tiered_budget

_PRECISION_RULE_PREFIXES = ("detectable_format:json", "detectable_format:title", "detectable_format:constrained", "change_case:", "punctuation:")
_PRECISION_RULE_IDS = {
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "detectable_content:postscript",
    "detectable_content:number_placeholders",
    "detectable_format:json_format",
    "detectable_format:title",
    "detectable_format:constrained_response",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "change_case:capital_word_frequency",
    "punctuation:no_comma",
    "startend:end_checker",
    "language:response_language",
    "combination:repeat_prompt",
}
_PLANNING_RULE_IDS = {
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:nth_paragraph_first_word",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "keywords:frequency",
    "startend:quotation",
    "combination:two_responses",
}

_IFEVAL_PARAPHRASE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bYour\s+(?:entire\s+)?(?:response|answer|reply)\s+must\b", re.IGNORECASE), "The generated output ought to"),
    (re.compile(r"\bwithout\s+using\b", re.IGNORECASE), "avoiding any inclusion of"),
    (re.compile(r"\b(?:do not|don't|never)\s+(?:use|include|mention)\b", re.IGNORECASE), "steer clear of employing"),
    (re.compile(r"\bat least\s+(\d+)\s+words?\b", re.IGNORECASE), r"a minimum length of \1 words"),
    (re.compile(r"\b(?:less|fewer)\s+than\s+(\d+)\s+words?\b", re.IGNORECASE), r"strictly under \1 words"),
    (re.compile(r"\bat least\s+(\d+)\s+sentences?\b", re.IGNORECASE), r"no fewer than \1 sentences"),
    (re.compile(r"\bat least\s+(\d+)\s+paragraphs?\b", re.IGNORECASE), r"no fewer than \1 paragraphs"),
    (re.compile(r"\bin\s+(?:all\s+)?(?:capital|uppercase)\s+letters?\b", re.IGNORECASE), "exclusively in uppercase script"),
    (re.compile(r"\bin\s+(?:all\s+)?lowercase(?:\s+letters?)?\b", re.IGNORECASE), "exclusively in lowercase script"),
    (re.compile(r"\bno\s+commas?\b", re.IGNORECASE), "omitting every comma character"),
    (re.compile(r"\bin\s+(?:valid\s+)?json\s+format\b", re.IGNORECASE), "serialized in structured object notation"),
    (re.compile(r"\bbullet\s+points?\b", re.IGNORECASE), "itemized list markers"),
    (re.compile(r"\bdouble\s+(?:quotation\s+marks|quotes)\b", re.IGNORECASE), "enclosing speech delimiters"),
)


def paraphrase_ifeval_prompt(prompt: str) -> str:
    """Applies deterministic IFEval++-style lexical paraphrasing preserving constraint semantics."""
    out = prompt
    for pat, repl in _IFEVAL_PARAPHRASE_RULES:
        out = pat.sub(repl, out)
    return out


def _constraint_targets(row: dict[str, Any]) -> tuple[int, int]:
    """Returns ground-truth (is_precision, is_planning) labels from item schema or Gate-1 fallback."""
    item = row.get("item", row)
    inst_ids = item.get("instruction_id_list")
    if isinstance(inst_ids, list) and inst_ids:
        y_prec = int(any(i in _PRECISION_RULE_IDS for i in inst_ids))
        y_plan = int(any(i in _PLANNING_RULE_IDS for i in inst_ids))
        return y_prec, y_plan
    g1 = run_gate1_lexical(str(item.get("prompt", "")))
    return int(g1.phi_prec), int(g1.phi_plan)


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class PrefillMLPProbe(nn.Module):
    """Lightweight 2-layer MLP classifier over frozen mid-layer prefill hidden states."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultiHeadPrefillProbe(nn.Module):
    """Shared prefill encoder with Gate-2 escalation head and auxiliary [phi_prec, phi_plan] head."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.escalation_head = nn.Linear(hidden_dim, 1)
        self.constraint_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.escalation_head(h).squeeze(-1), self.constraint_head(h)


def _mock_prefill_features(
    rows: list[dict[str, Any]],
    dim: int = 64,
    semantic_rows: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Deterministic mock embeddings capturing prompt identity + invariant constraint subspace."""
    feats = np.zeros((len(rows), dim), dtype=np.float32)
    ref_rows = semantic_rows if semantic_rows is not None else rows
    for i, (r, ref_r) in enumerate(zip(rows, ref_rows)):
        h = hashlib.sha256(r["item"]["prompt"].encode("utf-8")).digest()
        seed_int = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed_int)
        feats[i] = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        if dim >= 4:
            y_prec, y_plan = _constraint_targets(ref_r)
            feats[i, 0] += 2.2 * (1.0 if y_prec else -1.0)
            feats[i, 1] += 2.2 * (1.0 if y_plan else -1.0)
    return feats


def _extract_hf_prefill_features(
    model_id: str,
    rows: list[dict[str, Any]],
    layer_idx: int | None = None,
    prompt_override: list[str] | None = None,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    device, dtype = resolve_device_and_dtype("auto", "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=False
    ).to(device)
    model.eval()

    n_layers = int(getattr(model.config, "num_hidden_layers", 32))
    target_layer = n_layers // 2 if layer_idx is None else layer_idx

    vecs: list[np.ndarray] = []
    prompts = (
        prompt_override
        if prompt_override is not None
        else [r["item"]["prompt"] for r in rows]
    )
    with torch.no_grad():
        for p_txt in prompts:
            msgs = [{"role": "user", "content": p_txt}]
            formatted = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inp = tok(formatted, return_tensors="pt").to(device)
            out = model(**inp, output_hidden_states=True, use_cache=False)
            h_vec = out.hidden_states[target_layer][0, -1, :].float().cpu().numpy()
            vecs.append(h_vec)
    return np.vstack(vecs).astype(np.float32)


def _load_or_extract_cached_features(
    cache_path: pathlib.Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    extractor: Any,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    """Loads per-split prefill vectors from disk .npz cache when UIDs match; otherwise extracts and writes .npz."""
    expected_keys = [
        f"{split_name}::{r['item'].get('uid', idx)}"
        for split_name, split_rows in rows_by_split.items()
        for idx, r in enumerate(split_rows)
    ]
    if use_cache and cache_path.exists():
        try:
            npz = np.load(cache_path, allow_pickle=False)
            uids = list(npz["uids"].astype(str))
            feats = npz["features"].astype(np.float32)
            idx_map = {k: i for i, k in enumerate(uids)}
            if all(k in idx_map for k in expected_keys):
                out: dict[str, np.ndarray] = {}
                for split_name, split_rows in rows_by_split.items():
                    idxs = [
                        idx_map[f"{split_name}::{r['item'].get('uid', i)}"]
                        for i, r in enumerate(split_rows)
                    ]
                    out[split_name] = feats[idxs]
                return out
        except Exception:
            pass

    extracted = extractor(rows_by_split)
    if use_cache:
        all_vecs = np.vstack([extracted[k] for k in rows_by_split]).astype(np.float32)
        np.savez(
            cache_path,
            uids=np.array(expected_keys, dtype=np.str_),
            features=all_vecs,
        )
    return extracted


def _select_mlp_run(row: dict[str, Any], prob: float, tau: float, tau_h_high: float) -> dict[str, Any]:
    """Routes each item via Gate-1 regex cascade + zero-generation prefill MLP escalation."""
    prompt_str = row["item"]["prompt"]
    g1 = run_gate1_lexical(prompt_str)
    if g1.phi_prec == 1 and g1.phi_plan == 0 and g1.phi_sym == 0:
        return row["b0"]
    # Prefill MLP pays zero Gate-2 probe decoding tax (1.0 total call), so evaluate direct slow caps
    b_dir = select_tiered_budget(prompt_str, g1, None, tau_h_high)
    if g1.phi_plan == 1 or prob >= tau:
        return _get_slow_rec(row, b_dir, direct=True, prefix="m2")
    return row["b0"]


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": 100.0 * prec, "recall": 100.0 * rec, "f1": 100.0 * f1, "n_pos": tp + fn}


def train_and_evaluate_prefill_mlp(
    seed_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    x_seed: np.ndarray,
    x_dev: np.ndarray,
    x_eval: np.ndarray,
    tau_h_high: float,
    seed: int = 42,
    epochs: int = 200,
    x_eval_para: np.ndarray | None = None,
    train_multihead: bool = True,
) -> dict[str, Any]:
    """Fits multi-head prefill probe on seed_60, tunes thresholds on dev_60, evaluates on eval_450 & paraphrases."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize using seed split statistics only (zero leakage)
    mu = x_seed.mean(axis=0, keepdims=True)
    sigma = np.maximum(1e-5, x_seed.std(axis=0, keepdims=True))
    xs = torch.tensor((x_seed - mu) / sigma, dtype=torch.float32)
    xd = torch.tensor((x_dev - mu) / sigma, dtype=torch.float32)
    xe = torch.tensor((x_eval - mu) / sigma, dtype=torch.float32)
    xep = (
        torch.tensor((x_eval_para - mu) / sigma, dtype=torch.float32)
        if x_eval_para is not None
        else None
    )

    y_seed = torch.tensor(
        [
            float(
                (int(r["m2_b128"]["z"]) > int(r["fast_done"]["z"]))
                or (int(r["fast_done"]["z"]) == 0 and int(r["probe"]["I_inertia"]) == 1)
            )
            for r in seed_rows
        ],
        dtype=torch.float32,
    )
    yc_seed = torch.tensor([_constraint_targets(r) for r in seed_rows], dtype=torch.float32)

    if train_multihead:
        mlp: nn.Module = MultiHeadPrefillProbe(in_dim=xs.shape[1], hidden_dim=128)
    else:
        mlp = PrefillMLPProbe(in_dim=xs.shape[1], hidden_dim=128)

    n_pos = max(1.0, float(y_seed.sum().item()))
    n_neg = max(1.0, float((1.0 - y_seed).sum().item()))
    esc_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32))
    pos_c = torch.clamp(yc_seed.sum(dim=0), min=1.0)
    neg_c = torch.clamp(len(yc_seed) - yc_seed.sum(dim=0), min=1.0)
    con_criterion = nn.BCEWithLogitsLoss(pos_weight=neg_c / pos_c)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)

    mlp.train()
    for _ in range(epochs):
        opt.zero_grad()
        if train_multihead:
            esc_logits, con_logits = mlp(xs)
            loss = esc_criterion(esc_logits, y_seed) + 0.5 * con_criterion(con_logits, yc_seed)
        else:
            loss = esc_criterion(mlp(xs), y_seed)
        loss.backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        if train_multihead:
            e_dev, c_dev = mlp(xd)
            e_eval, c_eval = mlp(xe)
            p_dev = torch.sigmoid(e_dev).numpy()
            p_eval = torch.sigmoid(e_eval).numpy()
            pc_dev = torch.sigmoid(c_dev).numpy()
            pc_eval = torch.sigmoid(c_eval).numpy()
            pc_eval_para = torch.sigmoid(mlp(xep)[1]).numpy() if xep is not None else None
        else:
            p_dev = torch.sigmoid(mlp(xd)).numpy()
            p_eval = torch.sigmoid(mlp(xe)).numpy()
            pc_dev = pc_eval = pc_eval_para = None

    best_tau = 0.50
    best_u = -1e9
    for tau in np.linspace(0.15, 0.85, 29):
        u_vals = []
        for r, prob in zip(dev_rows, p_dev):
            chosen = _select_mlp_run(r, float(prob), float(tau), tau_h_high)
            u_vals.append(float(chosen["z"]) - 0.05 * (float(chosen["T_total"]) / 250.0))
        mean_u = float(np.mean(u_vals))
        if mean_u > best_u:
            best_u = mean_u
            best_tau = float(tau)

    eval_items = [r["item"] for r in eval_rows]
    eval_runs = [
        _select_mlp_run(r, float(prob), best_tau, tau_h_high)
        for r, prob in zip(eval_rows, p_eval)
    ]
    eval_tab = evaluate_system_table(eval_items, eval_runs)
    out_payload: dict[str, Any] = {
        "best_tau": best_tau,
        "dev_best_u005": best_u,
        "eval_table": {k: v for k, v in eval_tab.items() if k not in ("z_vec", "u_vec")},
    }

    if train_multihead and pc_dev is not None and pc_eval is not None:
        yc_dev = [_constraint_targets(r) for r in dev_rows]
        yc_eval = [_constraint_targets(r) for r in eval_rows]
        best_taus_c = []
        for col in (0, 1):
            y_true_d = [y[col] for y in yc_dev]
            t_best, f1_best = 0.5, -1.0
            for t in np.linspace(0.25, 0.75, 11):
                f1 = _binary_metrics(y_true_d, [int(p >= t) for p in pc_dev[:, col]])["f1"]
                if f1 > f1_best:
                    f1_best, t_best = f1, float(t)
            best_taus_c.append(t_best)

        g1_orig = [run_gate1_lexical(r["item"]["prompt"]) for r in eval_rows]
        g1_para = [run_gate1_lexical(paraphrase_ifeval_prompt(r["item"]["prompt"])) for r in eval_rows]
        out_payload["gate1_semantic_probe"] = {
            "tau_prec": best_taus_c[0],
            "tau_plan": best_taus_c[1],
            "regex_orig": {
                "prec": _binary_metrics([y[0] for y in yc_eval], [g.phi_prec for g in g1_orig]),
                "plan": _binary_metrics([y[1] for y in yc_eval], [g.phi_plan for g in g1_orig]),
            },
            "mlp_orig": {
                "prec": _binary_metrics([y[0] for y in yc_eval], [int(p >= best_taus_c[0]) for p in pc_eval[:, 0]]),
                "plan": _binary_metrics([y[1] for y in yc_eval], [int(p >= best_taus_c[1]) for p in pc_eval[:, 1]]),
            },
        }
        para_report: dict[str, Any] = {
            "regex_para": {
                "prec": _binary_metrics([y[0] for y in yc_eval], [g.phi_prec for g in g1_para]),
                "plan": _binary_metrics([y[1] for y in yc_eval], [g.phi_plan for g in g1_para]),
            }
        }
        if pc_eval_para is not None:
            para_report["mlp_para"] = {
                "prec": _binary_metrics([y[0] for y in yc_eval], [int(p >= best_taus_c[0]) for p in pc_eval_para[:, 0]]),
                "plan": _binary_metrics([y[1] for y in yc_eval], [int(p >= best_taus_c[1]) for p in pc_eval_para[:, 1]]),
            }
        out_payload["ifevalpp_paraphrase_stress"] = para_report

    return out_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional W7: Multi-head prefill hidden-state MLP probe.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--layer", type=int, default=-1, help="-1 uses mid-layer L//2")
    parser.add_argument("--no-cache", action="store_true", help="Disable .npz disk caching of prefill hidden states")
    parser.add_argument("--mock", action="store_true", help="Use deterministic synthetic embeddings")
    args = parser.parse_args()

    tag = args.model.replace("/", "__")
    model_dir = pathlib.Path(args.out_dir) / tag
    seed_rows = _load_jsonl(model_dir / "seed_60_cache.jsonl")
    dev_rows = _load_jsonl(model_dir / "dev_60_cache.jsonl")
    eval_rows = _load_jsonl(model_dir / "eval_450_cache.jsonl")
    eval_para_rows = [
        {"item": {**r["item"], "prompt": paraphrase_ifeval_prompt(r["item"]["prompt"])}}
        for r in eval_rows
    ]
    summary_path = model_dir / "matrix_summary.json"
    tau_h_high = 0.85
    if summary_path.exists():
        tau_h_high = float(
            json.loads(summary_path.read_text(encoding="utf-8"))["calibrated_params"]["tau_h_high"]
        )

    rows_by_split = {
        "seed": seed_rows,
        "dev": dev_rows,
        "eval": eval_rows,
        "eval_para": eval_para_rows,
    }
    if args.mock:
        cache_path = model_dir / "prefill_hidden_mock.npz"

        def _mock_extract(splits: dict[str, list[dict[str, Any]]]) -> dict[str, np.ndarray]:
            return {
                "seed": _mock_prefill_features(splits["seed"]),
                "dev": _mock_prefill_features(splits["dev"]),
                "eval": _mock_prefill_features(splits["eval"]),
                "eval_para": _mock_prefill_features(splits["eval_para"], semantic_rows=splits["eval"]),
            }

        feats = _load_or_extract_cached_features(
            cache_path, rows_by_split, _mock_extract, use_cache=not args.no_cache
        )
    else:
        layer_arg = None if args.layer < 0 else args.layer
        layer_tag = f"L{args.layer}"
        cache_path = model_dir / f"prefill_hidden_{layer_tag}.npz"

        def _hf_extract(splits: dict[str, list[dict[str, Any]]]) -> dict[str, np.ndarray]:
            combined = splits["seed"] + splits["dev"] + splits["eval"] + splits["eval_para"]
            all_x = _extract_hf_prefill_features(args.model, combined, layer_idx=layer_arg)
            n1 = len(splits["seed"])
            n2 = n1 + len(splits["dev"])
            n3 = n2 + len(splits["eval"])
            return {
                "seed": all_x[:n1],
                "dev": all_x[n1:n2],
                "eval": all_x[n2:n3],
                "eval_para": all_x[n3:],
            }

        feats = _load_or_extract_cached_features(
            cache_path, rows_by_split, _hf_extract, use_cache=not args.no_cache
        )

    res = train_and_evaluate_prefill_mlp(
        seed_rows,
        dev_rows,
        eval_rows,
        feats["seed"],
        feats["dev"],
        feats["eval"],
        tau_h_high=tau_h_high,
        x_eval_para=feats["eval_para"],
        train_multihead=True,
    )
    res["model"] = args.model
    res["mode"] = "mock" if args.mock else f"layer_{args.layer}"
    res["feature_cache_path"] = str(cache_path) if not args.no_cache else None
    out_path = model_dir / "prefill_mlp_summary.json"
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Prefill MLP probe evaluation to {out_path}")


if __name__ == "__main__":
    main()
