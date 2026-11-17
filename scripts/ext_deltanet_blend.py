"""Optional Workstream W7: Gated DeltaNet State Blending on E_hurt (Qwen3.5 Hybrid Layers)."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gpu_pool import select_worker_gpu_slots
from scripts.run_matrix import resolve_device_and_dtype
from src.engine import ensure_qwen35_device_safe_kernels
from src.graders import grade_item
from src.metrics import wilson_ci


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _is_state_initialized(layer: Any, flag_attr: str, buf_attr: str) -> bool:
    buf = getattr(layer, buf_attr, None)
    flags = getattr(layer, flag_attr, None)
    if not buf or not flags:
        return False
    return bool(flags[0] if isinstance(flags, (list, tuple)) else flags.get(0, False))


def restore_memory_states(
    cache_think: Any,
    cache_prompt: Any,
    alpha: float,
    blend_conv: bool = False,
    evict_think_kv: bool = False,
) -> dict[str, int]:
    """Interpolates DeltaNet recurrent/conv states and optionally evicts <think> from GQA KV caches."""
    n_blended_rec = 0
    n_blended_conv = 0
    n_evicted_kv = 0
    layers_t = getattr(cache_think, "layers", [])
    layers_p = getattr(cache_prompt, "layers", [])
    for idx in range(min(len(layers_t), len(layers_p))):
        lt = layers_t[idx]
        lp = layers_p[idx]
        if _is_state_initialized(lt, "is_recurrent_states_initialized", "recurrent_states") and (
            _is_state_initialized(lp, "is_recurrent_states_initialized", "recurrent_states")
        ):
            s_t = lt.recurrent_states[0]
            s_p = lp.recurrent_states[0]
            if s_t.shape == s_p.shape:
                blended = ((1.0 - alpha) * s_t.float() + alpha * s_p.float()).to(s_t.dtype)
                s_t.copy_(blended)
                n_blended_rec += 1
        if blend_conv and _is_state_initialized(lt, "is_conv_states_initialized", "conv_states") and (
            _is_state_initialized(lp, "is_conv_states_initialized", "conv_states")
        ):
            c_t = lt.conv_states[0]
            c_p = lp.conv_states[0]
            if c_t.shape == c_p.shape:
                blended_c = ((1.0 - alpha) * c_t.float() + alpha * c_p.float()).to(c_t.dtype)
                c_t.copy_(blended_c)
                n_blended_conv += 1
        if (
            evict_think_kv
            and getattr(lt, "keys", None) is not None
            and getattr(lp, "keys", None) is not None
            and getattr(lt, "values", None) is not None
            and getattr(lp, "values", None) is not None
        ):
            lt.keys = lp.keys.clone()
            lt.values = lp.values.clone()
            n_evicted_kv += 1
    return {
        "n_blended_layers": n_blended_rec,
        "n_blended_conv": n_blended_conv,
        "n_evicted_kv": n_evicted_kv,
    }


def blend_deltanet_states(
    cache_think: Any, cache_prompt: Any, alpha: float
) -> tuple[Any, int]:
    """Interpolates Gated DeltaNet recurrent_states: (1 - alpha) * S_think + alpha * S_prompt.

    Leaves softmax GQA key/value caches untouched. Returns (cache_think, n_blended_layers).
    """
    stats = restore_memory_states(cache_think, cache_prompt, alpha=alpha)
    return cache_think, stats["n_blended_layers"]


def generate_with_blended_deltanet(
    model: Any,
    tok: Any,
    eos_ids: set[int],
    prompt_str: str,
    t_think: str,
    alpha: float,
    max_ans_tokens: int = 1024,
    blend_conv: bool = False,
    evict_think_kv: bool = False,
) -> dict[str, Any]:
    """Prefills prompt + t_think, blends DeltaNet/conv states and optionally evicts <think> KV, then decodes."""
    prompt_msgs = [{"role": "user", "content": prompt_str}]
    prompt_no_think = tok.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    think_prefix = tok.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if not think_prefix.rstrip().endswith("<think>"):
        think_prefix += "<think>\n"
    post_close_ws = "\n\n" if prompt_no_think.endswith("</think>\n\n") else "\n"
    prompt_with_think = f"{think_prefix}{t_think}\n</think>{post_close_ws}"

    device = next(model.parameters()).device
    if device.type != "cuda":
        ensure_qwen35_device_safe_kernels()
    ids_p = tok.encode(prompt_no_think, add_special_tokens=False)
    ids_f = tok.encode(prompt_with_think, add_special_tokens=False)
    if len(ids_f) < 2 or not ids_p:
        raise ValueError("Full think prefix unexpectedly shorter than 2 tokens")

    # Stop both parallel prefills right before the final post-</think> separator token so
    # recurrent, conv, and full-attention KV caches align at the exact </think> boundary before step 0.
    if len(ids_p) >= 2:
        inp_p = torch.tensor([ids_p[:-1]], dtype=torch.long, device=device)
        evict_start_pos = len(ids_p) - 1
        evict_last_id = ids_p[-1]
    else:
        inp_p = torch.tensor([ids_p], dtype=torch.long, device=device)
        evict_start_pos = len(ids_p)
        evict_last_id = ids_f[-1]
    inp_f_prefix = torch.tensor([ids_f[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        cache_p = DynamicCache(config=model.config)
        _ = model(input_ids=inp_p, past_key_values=cache_p, use_cache=True)

        cache_t = DynamicCache(config=model.config)
        _ = model(input_ids=inp_f_prefix, past_key_values=cache_t, use_cache=True)

        mem_stats = restore_memory_states(
            cache_t,
            cache_p,
            alpha=alpha,
            blend_conv=blend_conv,
            evict_think_kv=evict_think_kv,
        )
        n_blended = mem_stats["n_blended_layers"]

        # When full-attention KV caches were evicted back to the clean prompt cache (length inp_p),
        # resume cache_position at evict_start_pos with evict_last_id; otherwise use ids_f[-1].
        if mem_stats["n_evicted_kv"] > 0:
            start_pos = evict_start_pos
            last_tok_id = evict_last_id
        else:
            start_pos = len(ids_f) - 1
            last_tok_id = ids_f[-1]
        last_tok = torch.tensor([[last_tok_id]], dtype=torch.long, device=device)
        pos = torch.tensor([start_pos], dtype=torch.long, device=device)
        out = model(
            input_ids=last_tok,
            past_key_values=cache_t,
            cache_position=pos,
            use_cache=True,
        )

        generated: list[int] = []
        next_tok_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        cur_pos = start_pos + 1
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
        "n_blended_conv": mem_stats["n_blended_conv"],
        "n_evicted_kv": mem_stats["n_evicted_kv"],
        "gen_tokens": len(generated),
    }


def run_mock_blend_check(
    alphas: list[float],
    blend_conv: bool = False,
    evict_think_kv: bool = False,
) -> dict[str, Any]:
    """Exercises exact DynamicCache LinearAttentionLayer + DynamicLayer restoration on CPU."""
    try:
        from transformers.cache_utils import DynamicLayer, LinearAttentionLayer as _RecurrentLayerCls
    except ImportError:
        from transformers.cache_utils import (
            DynamicLayer,
            LinearAttentionAndFullAttentionLayer as _RecurrentLayerCls,
        )

    cache_p = DynamicCache()
    cache_t = DynamicCache()
    layer_p = _RecurrentLayerCls()
    layer_t = _RecurrentLayerCls()
    layer_p.update_recurrent_state(torch.zeros((1, 4, 8, 8)))
    layer_t.update_recurrent_state(torch.ones((1, 4, 8, 8)) * 2.0)
    layer_p.update_conv_state(torch.zeros((1, 8, 4)), state_idx=0)
    layer_t.update_conv_state(torch.ones((1, 8, 4)) * 2.0, state_idx=0)
    cache_p.layers.append(layer_p)
    cache_t.layers.append(layer_t)

    if evict_think_kv:
        kv_p = DynamicLayer()
        kv_p.update(torch.zeros((1, 2, 4, 8)), torch.zeros((1, 2, 4, 8)))
        kv_t = DynamicLayer()
        kv_t.update(torch.ones((1, 2, 9, 8)), torch.ones((1, 2, 9, 8)))
        cache_p.layers.append(kv_p)
        cache_t.layers.append(kv_t)

    results = {}
    for a in alphas:
        ct_copy = DynamicCache()
        lt_copy = _RecurrentLayerCls()
        lt_copy.update_recurrent_state(layer_t.recurrent_states[0].clone())
        lt_copy.update_conv_state(layer_t.conv_states[0].clone(), state_idx=0)
        ct_copy.layers.append(lt_copy)
        if evict_think_kv:
            kv_copy = DynamicLayer()
            kv_copy.update(cache_t.layers[1].keys.clone(), cache_t.layers[1].values.clone())
            ct_copy.layers.append(kv_copy)

        stats = restore_memory_states(
            ct_copy,
            cache_p,
            alpha=a,
            blend_conv=blend_conv,
            evict_think_kv=evict_think_kv,
        )
        mean_norm = float(ct_copy.layers[0].recurrent_states[0].mean().item())
        entry: dict[str, Any] = {
            "n_blended_layers": stats["n_blended_layers"],
            "n_blended_conv": stats["n_blended_conv"],
            "n_evicted_kv": stats["n_evicted_kv"],
            "blended_mean_val": mean_norm,
        }
        if blend_conv:
            entry["conv_mean_val"] = float(ct_copy.layers[0].conv_states[0].mean().item())
        if evict_think_kv:
            entry["evicted_kv_seq_len"] = int(ct_copy.layers[1].get_seq_length())
        results[f"alpha_{a:.2f}"] = entry
    return {
        "mode": "mock",
        "blend_conv_state": blend_conv,
        "evict_think_kv": evict_think_kv,
        "sweep": results,
    }


def _blend_key(uid: str, alpha: float, blend_conv: bool, evict_kv: bool) -> tuple[str, str, bool, bool]:
    return (str(uid), f"{float(alpha):.4f}", bool(blend_conv), bool(evict_kv))


def _load_blend_cache(path: pathlib.Path) -> dict[tuple[str, str, bool, bool], dict[str, Any]]:
    out: dict[tuple[str, str, bool, bool], dict[str, Any]] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = _blend_key(
            rec["uid"],
            float(rec["alpha"]),
            bool(rec.get("blend_conv_state", False)),
            bool(rec.get("evict_think_kv", False)),
        )
        out[k] = rec
    return out


def _filter_uncached_ehurt(
    e_hurt: list[dict[str, Any]],
    cache_map: dict[tuple[str, str, bool, bool], dict[str, Any]],
    alphas: list[float],
    blend_conv: bool,
    evict_kv: bool,
) -> list[dict[str, Any]]:
    """Returns E_hurt items missing at least one (uid, alpha, blend_conv, evict_kv) entry."""
    return [
        row
        for row in e_hurt
        if any(
            _blend_key(str(row["item"]["uid"]), a, blend_conv, evict_kv) not in cache_map
            for a in alphas
        )
    ]


def _terminate_procs(procs: list[tuple[int, subprocess.Popen[Any]]]) -> None:
    """Gracefully terminates active worker subprocesses so sibling partial shards can be harvested."""
    alive = [p for _, p in procs if p.poll() is None]
    if not alive:
        return
    for p in alive:
        try:
            p.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if all(p.poll() is not None for p in alive):
            return
        time.sleep(0.1)
    for p in alive:
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass


def _harvest_partial_blend_shards(
    blend_cache_path: pathlib.Path, shard_dir: pathlib.Path
) -> dict[tuple[str, str, bool, bool], dict[str, Any]]:
    """Merges any completed or interrupted deltanet_blend_shard_*.jsonl files into canonical cache."""
    merged = _load_blend_cache(blend_cache_path)
    if not shard_dir.is_dir():
        return merged
    shard_files = sorted(shard_dir.glob("deltanet_blend_shard_*.jsonl"))
    if not shard_files:
        return merged
    before_n = len(merged)
    for sp in shard_files:
        if sp.is_file():
            merged.update(_load_blend_cache(sp))
            sp.unlink(missing_ok=True)
    try:
        if not any(shard_dir.iterdir()):
            shard_dir.rmdir()
    except OSError:
        pass
    if len(merged) > before_n or not blend_cache_path.is_file():
        tmp_c = blend_cache_path.with_suffix(".jsonl.tmp")
        with tmp_c.open("w", encoding="utf-8") as fh:
            for rec in merged.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_c.replace(blend_cache_path)
    return merged


def _detect_auto_slots(model_id: str, n_items: int) -> list[str]:
    if not torch.cuda.is_available() or n_items <= 0:
        return []
    raw_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cands = [x.strip() for x in raw_vis.split(",") if x.strip()] if raw_vis else None
    wait_sec = int(os.environ.get("CBARR_WAIT_GPU_SEC", "3600"))
    slots = select_worker_gpu_slots(
        model_id,
        candidate_ids=cands,
        workers_per_gpu=os.environ.get("CBARR_WORKERS_PER_GPU", "auto"),
        max_gpus=8,
        wait_timeout_sec=wait_sec,
    )
    return slots[:n_items]


def _summarize_and_save_blend(
    out_path: pathlib.Path,
    model_id: str,
    e_hurt: list[dict[str, Any]],
    alphas: list[float],
    blend_conv: bool,
    evict_kv: bool,
    cache_map: dict[tuple[str, str, bool, bool], dict[str, Any]],
) -> None:
    alpha_summary: dict[str, Any] = {}
    for idx_a, a in enumerate(alphas, start=1):
        all_pass, inh_pass, str_pass = [], [], []
        n_blended = 0
        n_blended_conv = 0
        n_evicted_kv = 0
        for row in e_hurt:
            uid = str(row["item"]["uid"])
            rec = cache_map[_blend_key(uid, a, blend_conv, evict_kv)]
            n_blended = max(n_blended, int(rec["n_blended_layers"]))
            n_blended_conv = max(n_blended_conv, int(rec.get("n_blended_conv", 0)))
            n_evicted_kv = max(n_evicted_kv, int(rec.get("n_evicted_kv", 0)))
            g = rec["grade"]
            all_pass.append(int(g["z"]))
            if g.get("inhibition_strict_all") is not None:
                inh_pass.append(int(g["inhibition_strict_all"]))
            if g.get("structural_strict_all") is not None:
                str_pass.append(int(g["structural_strict_all"]))

        _, lo, hi = wilson_ci(sum(all_pass), len(all_pass))
        if e_hurt and n_blended == 0 and n_evicted_kv == 0:
            print(
                f"[WARN] alpha={a:.2f}: modified 0 recurrent/KV layers (check model cache architecture).",
                flush=True,
            )
        alpha_summary[f"{a:.2f}"] = {
            "alpha": a,
            "n_ehurt": len(e_hurt),
            "n_blended_layers": n_blended,
            "n_blended_conv": n_blended_conv,
            "n_evicted_kv": n_evicted_kv,
            "overall_recovery_pct": 100.0 * float(np.mean(all_pass)) if all_pass else 0.0,
            "overall_wilson_ci95": [100.0 * lo, 100.0 * hi],
            "inhibition_recovery_pct": 100.0 * float(np.mean(inh_pass)) if inh_pass else 0.0,
            "structural_recovery_pct": 100.0 * float(np.mean(str_pass)) if str_pass else 0.0,
        }
        print(
            f"[deltanet_blend] ({idx_a}/{len(alphas)}) alpha={a:.2f}: "
            f"recovery={alpha_summary[f'{a:.2f}']['overall_recovery_pct']:.1f}% "
            f"({sum(all_pass)}/{len(e_hurt)})",
            flush=True,
        )

    payload = {
        "model": model_id,
        "n_ehurt": len(e_hurt),
        "blend_conv_state": blend_conv,
        "evict_think_kv": evict_kv,
        "alphas": alpha_summary,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved DeltaNet blending results to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional W7: Interpolate DeltaNet recurrent/conv states and evict <think> KV on E_hurt."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max-items", type=int, default=0, help="0 means all E_hurt items")
    parser.add_argument(
        "--blend-conv-state",
        action="store_true",
        help="Also linearly interpolate Gated DeltaNet 1D conv_states alongside recurrent_states.",
    )
    parser.add_argument(
        "--evict-think-kv",
        action="store_true",
        help="Evict <think> tokens from full-attention GQA KV caches back to the prompt KV cache.",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default="auto",
        help="Number of parallel GPU workers ('auto' or positive integer).",
    )
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2, help="Auto-retry partial GPU OOM failures.")
    parser.add_argument("--retry-wait-sec", type=int, default=15)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mock", action="store_true", help="Run CPU DynamicCache unit sweep only")
    args = parser.parse_args()

    tag = args.model.replace("/", "__")
    model_dir = pathlib.Path(args.out_dir) / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "ext_deltanet_blend.json"
    blend_cache_path = model_dir / "ext_deltanet_blend_cache.jsonl"

    if args.mock:
        res = run_mock_blend_check(
            args.alphas,
            blend_conv=args.blend_conv_state,
            evict_think_kv=args.evict_think_kv,
        )
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

    shard_dir = model_dir / "shards"
    is_worker = bool(args.worker or args.num_shards > 1)
    canon_cache = (
        _load_blend_cache(blend_cache_path)
        if is_worker
        else _harvest_partial_blend_shards(blend_cache_path, shard_dir)
    )
    uncached_items = _filter_uncached_ehurt(
        e_hurt, canon_cache, args.alphas, args.blend_conv_state, args.evict_think_kv
    )
    if not is_worker and not uncached_items:
        print(
            f"[deltanet_blend] All {len(e_hurt) * len(args.alphas)} (item, alpha) pairs cached; summarizing immediately.",
            flush=True,
        )
        _summarize_and_save_blend(
            out_path,
            args.model,
            e_hurt,
            args.alphas,
            args.blend_conv_state,
            args.evict_think_kv,
            canon_cache,
        )
        return

    # Parent supervisor dispatches uncached items across live free-VRAM GPU worker subprocesses with retry.
    if not is_worker:
        has_cuda = bool(torch.cuda.is_available())
        req_multi = args.shards != "auto" and int(args.shards) > 1
        if has_cuda or req_multi:
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=args.model,
                    local_files_only=os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"),
                )
                prewarmed = True
            except Exception:
                prewarmed = False

            max_attempts = max(0, int(args.max_retries)) + 1
            merged = canon_cache
            for attempt in range(1, max_attempts + 1):
                merged = _harvest_partial_blend_shards(blend_cache_path, shard_dir)
                uncached_items = _filter_uncached_ehurt(
                    e_hurt, merged, args.alphas, args.blend_conv_state, args.evict_think_kv
                )
                if not uncached_items:
                    break
                if args.shards == "auto":
                    slot_gpus = _detect_auto_slots(args.model, len(uncached_items))
                else:
                    req_n = max(1, min(len(uncached_items), int(args.shards)))
                    auto_s = _detect_auto_slots(args.model, req_n)
                    slot_gpus = (
                        (auto_s * ((req_n // max(1, len(auto_s))) + 1))[:req_n]
                        if auto_s
                        else ([] if has_cuda else ["cpu"] * req_n)
                    )

                if not slot_gpus:
                    raise RuntimeError(
                        f"No CUDA GPU with sufficient free VRAM became available for {args.model} DeltaNet Blend."
                    )

                eff_shards = len(slot_gpus)
                print(
                    f"[deltanet_blend] Attempt {attempt}/{max_attempts}: spawning {eff_shards} worker(s) on GPUs {slot_gpus} "
                    f"for {len(uncached_items)}/{len(e_hurt)} remaining items x {len(args.alphas)} alphas...",
                    flush=True,
                )
                shard_dir.mkdir(parents=True, exist_ok=True)
                procs: list[tuple[int, subprocess.Popen[Any]]] = []
                failed_rc = 0
                try:
                    for s_idx in range(eff_shards):
                        env = os.environ.copy()
                        if slot_gpus[s_idx] != "cpu":
                            env["CUDA_VISIBLE_DEVICES"] = slot_gpus[s_idx]
                        for tvar in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                            env.setdefault(tvar, "4")
                        env.setdefault("TOKENIZERS_PARALLELISM", "false")
                        if prewarmed:
                            env.setdefault("HF_HUB_OFFLINE", "1")
                        cmd = [
                            sys.executable,
                            str(pathlib.Path(__file__).resolve()),
                            "--worker",
                            "--model",
                            args.model,
                            "--out-dir",
                            args.out_dir,
                            "--alphas",
                            *[str(a) for a in args.alphas],
                            "--max-items",
                            str(args.max_items),
                            "--shard-idx",
                            str(s_idx),
                            "--num-shards",
                            str(eff_shards),
                        ]
                        if args.blend_conv_state:
                            cmd.append("--blend-conv-state")
                        if args.evict_think_kv:
                            cmd.append("--evict-think-kv")
                        procs.append((s_idx, subprocess.Popen(cmd, env=env)))

                    pending = list(procs)
                    while pending:
                        still_running: list[tuple[int, subprocess.Popen[Any]]] = []
                        for s_idx, p in pending:
                            rc = p.poll()
                            if rc is None:
                                still_running.append((s_idx, p))
                            elif rc != 0:
                                failed_rc = rc
                                print(
                                    f"[deltanet_blend] Worker shard={s_idx}/{eff_shards} failed (exit={rc}); "
                                    "terminating sibling workers and harvesting partial shards.",
                                    flush=True,
                                )
                                _terminate_procs(
                                    [item for item in pending if item[0] != s_idx]
                                )
                                still_running = []
                                break
                        pending = still_running
                        if pending:
                            time.sleep(1.0)
                finally:
                    _terminate_procs(procs)
                    merged = _harvest_partial_blend_shards(blend_cache_path, shard_dir)

                if failed_rc == 0:
                    break
                if attempt < max_attempts:
                    wait_s = max(0, int(args.retry_wait_sec))
                    print(
                        f"[deltanet_blend] Retrying remaining items after {wait_s}s live GPU rescan...",
                        flush=True,
                    )
                    if wait_s > 0:
                        time.sleep(wait_s)
                else:
                    raise RuntimeError(
                        f"deltanet_blend shard worker exited with status {failed_rc} after {max_attempts} attempts"
                    )

            merged = _harvest_partial_blend_shards(blend_cache_path, shard_dir)
            _summarize_and_save_blend(
                out_path,
                args.model,
                e_hurt,
                args.alphas,
                args.blend_conv_state,
                args.evict_think_kv,
                merged,
            )
            return

    my_items = uncached_items[args.shard_idx :: args.num_shards] if is_worker else uncached_items
    worker_cache_path = (
        (model_dir / "shards" / f"deltanet_blend_shard_{args.shard_idx}_of_{args.num_shards}.jsonl")
        if is_worker
        else blend_cache_path
    )
    tag_lbl = f"shard={args.shard_idx}/{args.num_shards}" if is_worker else "single"
    if not my_items:
        print(f"[deltanet_blend {tag_lbl}] 0 uncached items assigned; exiting cleanly.", flush=True)
        return
    worker_cache_path.parent.mkdir(parents=True, exist_ok=True)
    known = dict(canon_cache)
    known.update(_load_blend_cache(worker_cache_path))

    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, local_files_only=offline)
        device, dtype = resolve_device_and_dtype("auto", "auto")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=False, local_files_only=offline
        ).to(device)
    except Exception:
        if offline:
            raise
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, local_files_only=True)
        device, dtype = resolve_device_and_dtype("auto", "auto")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=False, local_files_only=True
        ).to(device)
    model.eval()

    eos_raw = tok.eos_token_id
    eos_ids = set(eos_raw if isinstance(eos_raw, list) else [eos_raw])
    for special in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(special)
        if tid is not None and tid != tok.unk_token_id:
            eos_ids.add(int(tid))

    print(
        f"[deltanet_blend {tag_lbl}] Model={args.model} | items={len(my_items)} remaining ({len(e_hurt)} total) | "
        f"alphas={args.alphas} on device={device} ({dtype})",
        flush=True,
    )
    log_every = 5 if len(my_items) <= 25 else 10
    t0 = time.perf_counter()
    with worker_cache_path.open("a", encoding="utf-8") as fh:
        for i_row, row in enumerate(my_items, start=1):
            uid = str(row["item"]["uid"])
            for a in args.alphas:
                key = _blend_key(uid, a, args.blend_conv_state, args.evict_think_kv)
                if key in known:
                    continue
                out = generate_with_blended_deltanet(
                    model,
                    tok,
                    eos_ids,
                    row["item"]["prompt"],
                    row["b1b"]["t_think"],
                    alpha=a,
                    blend_conv=args.blend_conv_state,
                    evict_think_kv=args.evict_think_kv,
                )
                g = grade_item(row["item"], out["a_final"])
                rec = {
                    "uid": uid,
                    "alpha": float(a),
                    "blend_conv_state": bool(args.blend_conv_state),
                    "evict_think_kv": bool(args.evict_think_kv),
                    "n_blended_layers": int(out["n_blended_layers"]),
                    "n_blended_conv": int(out.get("n_blended_conv", 0)),
                    "n_evicted_kv": int(out.get("n_evicted_kv", 0)),
                    "a_final": out["a_final"],
                    "grade": g,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                known[key] = rec
            if i_row % log_every == 0 or i_row == len(my_items):
                print(
                    f"[deltanet_blend {tag_lbl}] completed {i_row}/{len(my_items)} items "
                    f"({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )

    if not is_worker:
        _summarize_and_save_blend(
            out_path,
            args.model,
            e_hurt,
            args.alphas,
            args.blend_conv_state,
            args.evict_think_kv,
            known,
        )


if __name__ == "__main__":
    main()
