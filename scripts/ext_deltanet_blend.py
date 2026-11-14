"""Optional Workstream W7: Gated DeltaNet State Blending on E_hurt (Qwen3.5 Hybrid Layers)."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_matrix import resolve_device_and_dtype
from src.graders import grade_item
from src.metrics import wilson_ci


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def blend_deltanet_states(
    cache_think: Any, cache_prompt: Any, alpha: float
) -> tuple[Any, int]:
    """Interpolates Gated DeltaNet recurrent_states: (1 - alpha) * S_think + alpha * S_prompt.

    Leaves softmax GQA key/value caches untouched. Returns (cache_think, n_blended_layers).
    """
    n_blended = 0
    layers_t = getattr(cache_think, "layers", [])
    layers_p = getattr(cache_prompt, "layers", [])
    for idx in range(min(len(layers_t), len(layers_p))):
        lt = layers_t[idx]
        lp = layers_p[idx]
        has_rec_t = bool(
            getattr(lt, "is_recurrent_states_initialized", [False])[0]
            if getattr(lt, "recurrent_states", None)
            else False
        )
        has_rec_p = bool(
            getattr(lp, "is_recurrent_states_initialized", [False])[0]
            if getattr(lp, "recurrent_states", None)
            else False
        )
        if has_rec_t and has_rec_p:
            s_t = lt.recurrent_states[0]
            s_p = lp.recurrent_states[0]
            if s_t.shape == s_p.shape:
                blended = ((1.0 - alpha) * s_t.float() + alpha * s_p.float()).to(s_t.dtype)
                s_t.copy_(blended)
                n_blended += 1
    return cache_think, n_blended


def generate_with_blended_deltanet(
    model: Any,
    tok: Any,
    eos_ids: set[int],
    prompt_str: str,
    t_think: str,
    alpha: float,
    max_ans_tokens: int = 1024,
) -> dict[str, Any]:
    """Prefills prompt + t_think, blends DeltaNet recurrent state toward prompt state, and decodes."""
    prompt_msgs = [{"role": "user", "content": prompt_str}]
    prompt_no_think = tok.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    think_prefix = tok.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if not think_prefix.rstrip().endswith("<think>"):
        think_prefix += "<think>\n"
    prompt_with_think = f"{think_prefix}{t_think}\n</think>\n"

    device = next(model.parameters()).device
    ids_p = tok.encode(prompt_no_think, add_special_tokens=False)
    ids_f = tok.encode(prompt_with_think, add_special_tokens=False)
    if len(ids_f) < 2:
        raise ValueError("Full think prefix unexpectedly shorter than 2 tokens")

    # Prefill two parallel passes: prompt-only (S_prompt) and prompt + <think>...</think> (S_think).
    # Pass config=model.config so transformers initializes LinearAttentionLayer for Qwen3.5 DeltaNet layers.
    # Note: we stop inp_f_prefix one token before the end so the final step runs AFTER state blending,
    # ensuring the very first generated answer logits reflect the interpolated DeltaNet memory.
    inp_p = torch.tensor([ids_p], dtype=torch.long, device=device)
    inp_f_prefix = torch.tensor([ids_f[:-1]], dtype=torch.long, device=device)
    last_tok = torch.tensor([[ids_f[-1]]], dtype=torch.long, device=device)

    with torch.no_grad():
        cache_p = DynamicCache(config=model.config)
        _ = model(input_ids=inp_p, past_key_values=cache_p, use_cache=True)

        cache_t = DynamicCache(config=model.config)
        _ = model(input_ids=inp_f_prefix, past_key_values=cache_t, use_cache=True)

        _, n_blended = blend_deltanet_states(cache_t, cache_p, alpha=alpha)

        # Prime the blended recurrent cache with the closing token and switch to 1-token greedy decode
        pos = torch.tensor([len(ids_f) - 1], dtype=torch.long, device=device)
        out = model(
            input_ids=last_tok,
            past_key_values=cache_t,
            cache_position=pos,
            use_cache=True,
        )

        generated: list[int] = []
        next_tok_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        cur_pos = len(ids_f)
        past = out.past_key_values
        for _ in range(max_ans_tokens):
            generated.append(next_tok_id)
            if next_tok_id in eos_ids:
                break
            if len(generated) >= max_ans_tokens:
                break
            step_in = torch.tensor([[next_tok_id]], dtype=torch.long, device=device)
            step_pos = torch.tensor([cur_pos], dtype=torch.long, device=device)
            out = model(
                input_ids=step_in,
                past_key_values=past,
                cache_position=step_pos,
                use_cache=True,
            )
            past = out.past_key_values
            cur_pos += 1
            next_tok_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())

    text = tok.decode(generated, skip_special_tokens=True).strip()
    return {
        "a_final": text,
        "n_blended_layers": n_blended,
        "gen_tokens": len(generated),
    }


def run_mock_blend_check(alphas: list[float]) -> dict[str, Any]:
    """Exercises exact DynamicCache LinearAttentionLayer recurrent-state blending on CPU."""
    try:
        from transformers.cache_utils import LinearAttentionLayer as _RecurrentLayerCls
    except ImportError:
        from transformers.cache_utils import LinearAttentionAndFullAttentionLayer as _RecurrentLayerCls

    cache_p = DynamicCache()
    cache_t = DynamicCache()
    layer_p = _RecurrentLayerCls()
    layer_t = _RecurrentLayerCls()
    layer_p.update_recurrent_state(torch.zeros((1, 4, 8, 8)))
    layer_t.update_recurrent_state(torch.ones((1, 4, 8, 8)) * 2.0)
    cache_p.layers.append(layer_p)
    cache_t.layers.append(layer_t)

    results = {}
    for a in alphas:
        # Clone state for clean sweep test
        ct_copy = DynamicCache()
        lt_copy = _RecurrentLayerCls()
        lt_copy.update_recurrent_state(layer_t.recurrent_states[0].clone())
        ct_copy.layers.append(lt_copy)
        _, n_blended = blend_deltanet_states(ct_copy, cache_p, alpha=a)
        mean_norm = float(ct_copy.layers[0].recurrent_states[0].mean().item())
        results[f"alpha_{a:.2f}"] = {"n_blended_layers": n_blended, "blended_mean_val": mean_norm}
    return {"mode": "mock", "sweep": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional W7: Interpolate Qwen3.5 Gated DeltaNet recurrent states on E_hurt."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max-items", type=int, default=0, help="0 means all E_hurt items")
    parser.add_argument("--mock", action="store_true", help="Run CPU DynamicCache unit sweep only")
    args = parser.parse_args()

    tag = args.model.replace("/", "__")
    model_dir = pathlib.Path(args.out_dir) / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "ext_deltanet_blend.json"

    if args.mock:
        res = run_mock_blend_check(args.alphas)
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"Saved mock DeltaNet blend check to {out_path}")
        return

    cache_path = model_dir / "diag_ehurt_cache.jsonl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Run scripts/run_matrix.py first to populate {cache_path}")

    rows = _load_jsonl(cache_path)
    e_hurt = [
        r for r in rows if int(r["b0"]["z"]) == 1 and int(r["b1b"]["z"]) == 0 and r["b1b"]["t_think"]
    ]
    if args.max_items > 0:
        e_hurt = e_hurt[: args.max_items]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    device, dtype = resolve_device_and_dtype("auto", "auto")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=False
    ).to(device)
    model.eval()

    eos_raw = tok.eos_token_id
    eos_ids = set(eos_raw if isinstance(eos_raw, list) else [eos_raw])
    for special in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(special)
        if tid is not None and tid != tok.unk_token_id:
            eos_ids.add(int(tid))

    alpha_summary: dict[str, Any] = {}
    for a in args.alphas:
        all_pass, inh_pass, str_pass = [], [], []
        n_blended = 0
        for row in e_hurt:
            out = generate_with_blended_deltanet(
                model, tok, eos_ids, row["item"]["prompt"], row["b1b"]["t_think"], alpha=a
            )
            n_blended = max(n_blended, int(out["n_blended_layers"]))
            g = grade_item(row["item"], out["a_final"])
            all_pass.append(int(g["z"]))
            if g.get("inhibition_strict_all") is not None:
                inh_pass.append(int(g["inhibition_strict_all"]))
            if g.get("structural_strict_all") is not None:
                str_pass.append(int(g["structural_strict_all"]))

        _, lo, hi = wilson_ci(sum(all_pass), len(all_pass))
        if e_hurt and n_blended == 0:
            print(
                f"[WARN] alpha={a:.2f}: blended 0 recurrent layers (model may lack Gated DeltaNet cache layers).",
                flush=True,
            )
        alpha_summary[f"{a:.2f}"] = {
            "alpha": a,
            "n_ehurt": len(e_hurt),
            "n_blended_layers": n_blended,
            "overall_recovery_pct": 100.0 * float(np.mean(all_pass)) if all_pass else 0.0,
            "overall_wilson_ci95": [100.0 * lo, 100.0 * hi],
            "inhibition_recovery_pct": 100.0 * float(np.mean(inh_pass)) if inh_pass else 0.0,
            "structural_recovery_pct": 100.0 * float(np.mean(str_pass)) if str_pass else 0.0,
        }

    payload = {"model": args.model, "n_ehurt": len(e_hurt), "alphas": alpha_summary}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved DeltaNet blending results to {out_path}")


if __name__ == "__main__":
    main()
