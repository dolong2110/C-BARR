"""Optional Workstream W7: 2-Layer Prefill Hidden-State MLP Probe vs. Gate-2 Escalation."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
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


def _mock_prefill_features(rows: list[dict[str, Any]], dim: int = 64) -> np.ndarray:
    feats = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        h = hashlib.sha256(r["item"]["prompt"].encode("utf-8")).digest()
        seed_int = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed_int)
        feats[i] = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    return feats


def _extract_hf_prefill_features(
    model_id: str, rows: list[dict[str, Any]], layer_idx: int | None = None
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
    with torch.no_grad():
        for r in rows:
            msgs = [{"role": "user", "content": r["item"]["prompt"]}]
            formatted = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inp = tok(formatted, return_tensors="pt").to(device)
            out = model(**inp, output_hidden_states=True, use_cache=False)
            h_vec = out.hidden_states[target_layer][0, -1, :].float().cpu().numpy()
            vecs.append(h_vec)
    return np.vstack(vecs).astype(np.float32)


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
) -> dict[str, Any]:
    """Fits 2-layer prefill probe on seed_60, tunes tau* on dev_60 for U_0.05, evaluates on eval_450."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize using seed split statistics only (zero leakage)
    mu = x_seed.mean(axis=0, keepdims=True)
    sigma = np.maximum(1e-5, x_seed.std(axis=0, keepdims=True))
    xs = torch.tensor((x_seed - mu) / sigma, dtype=torch.float32)
    xd = torch.tensor((x_dev - mu) / sigma, dtype=torch.float32)
    xe = torch.tensor((x_eval - mu) / sigma, dtype=torch.float32)

    # Same binary supervision target used by CBARRCalibrator on Gate 2
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

    mlp = PrefillMLPProbe(in_dim=xs.shape[1], hidden_dim=128)
    n_pos = max(1.0, float(y_seed.sum().item()))
    n_neg = max(1.0, float((1.0 - y_seed).sum().item()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)

    mlp.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(mlp(xs), y_seed)
        loss.backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        p_dev = torch.sigmoid(mlp(xd)).numpy()
        p_eval = torch.sigmoid(mlp(xe)).numpy()

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
    return {
        "best_tau": best_tau,
        "dev_best_u005": best_u,
        "eval_table": {k: v for k, v in eval_tab.items() if k not in ("z_vec", "u_vec")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional W7: Prefill hidden-state MLP probe.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--layer", type=int, default=-1, help="-1 uses mid-layer L//2")
    parser.add_argument("--mock", action="store_true", help="Use deterministic synthetic embeddings")
    args = parser.parse_args()

    tag = args.model.replace("/", "__")
    model_dir = pathlib.Path(args.out_dir) / tag
    seed_rows = _load_jsonl(model_dir / "seed_60_cache.jsonl")
    dev_rows = _load_jsonl(model_dir / "dev_60_cache.jsonl")
    eval_rows = _load_jsonl(model_dir / "eval_450_cache.jsonl")
    summary_path = model_dir / "matrix_summary.json"
    tau_h_high = 0.85
    if summary_path.exists():
        tau_h_high = float(
            json.loads(summary_path.read_text(encoding="utf-8"))["calibrated_params"]["tau_h_high"]
        )

    if args.mock:
        x_seed = _mock_prefill_features(seed_rows)
        x_dev = _mock_prefill_features(dev_rows)
        x_eval = _mock_prefill_features(eval_rows)
    else:
        layer_arg = None if args.layer < 0 else args.layer
        all_rows = seed_rows + dev_rows + eval_rows
        all_x = _extract_hf_prefill_features(args.model, all_rows, layer_idx=layer_arg)
        x_seed = all_x[: len(seed_rows)]
        x_dev = all_x[len(seed_rows) : len(seed_rows) + len(dev_rows)]
        x_eval = all_x[len(seed_rows) + len(dev_rows) :]

    res = train_and_evaluate_prefill_mlp(
        seed_rows, dev_rows, eval_rows, x_seed, x_dev, x_eval, tau_h_high=tau_h_high
    )
    res["model"] = args.model
    res["mode"] = "mock" if args.mock else f"layer_{args.layer}"
    out_path = model_dir / "prefill_mlp_summary.json"
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Prefill MLP probe evaluation to {out_path}")


if __name__ == "__main__":
    main()
