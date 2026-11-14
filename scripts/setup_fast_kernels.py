"""Safe installer & live CUDA self-test for Qwen3.5 Gated DeltaNet fast kernels (`fla` & `causal-conv1d`)."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.logfmt import banner, hl


def should_attempt_kernels(cc: tuple[int, int] | None, mode: str) -> tuple[bool, str]:
    """Determines whether fast Triton/CUDA DeltaNet kernels should be attempted for a GPU."""
    if mode in ("off", "check-only"):
        return False, f"mode={mode}"
    if cc is None:
        return False, "no CUDA device"
    if mode == "always":
        return True, f"forced on sm_{cc[0]}{cc[1]}"
    # auto: Triton/FLA kernels require Volta/Turing+ (sm >= 7.5); Pascal P100 (sm_60) is unsupported
    if cc >= (7, 5):
        return True, f"supported sm_{cc[0]}{cc[1]}"
    return False, f"sm_{cc[0]}{cc[1]} < sm_75 (PyTorch fallback required)"


def _pip_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pip", *args],
        capture_output=True,
        text=True,
    )


def _write_pip_constraints(path: pathlib.Path) -> pathlib.Path:
    """Pins already-installed torch & transformers so kernel installs never downgrade container packages."""
    lines: list[str] = []
    for pkg in ("torch", "transformers"):
        try:
            ver = importlib.metadata.version(pkg)
            # Strip local CUDA suffix (+cu124) only if exact pin fails? pip matches local version tags directly.
            lines.append(f"{pkg}=={ver}\n")
        except importlib.metadata.PackageNotFoundError:
            pass
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _uninstall_pkg(dist_name: str) -> None:
    _pip_cmd("uninstall", "-y", "-q", dist_name)


def _verify_fla_on_gpu(use_bf16: bool) -> tuple[bool, str]:
    """Runs a fresh-subprocess live CUDA verification of both raw FLA and transformers Qwen3.5 hooks."""
    if importlib.util.find_spec("fla") is None:
        return False, "not installed"
    code = f"""
import torch
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _raw_chunk,
    fused_recurrent_gated_delta_rule as _raw_rec,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
dt = torch.bfloat16 if {use_bf16!r} else torch.float16
q = torch.randn(1, 16, 4, 64, device="cuda", dtype=dt)
k = torch.randn(1, 16, 4, 64, device="cuda", dtype=dt)
v = torch.randn(1, 16, 4, 64, device="cuda", dtype=dt)
g = -torch.rand(1, 16, 4, device="cuda", dtype=torch.float32)
beta = torch.rand(1, 16, 4, device="cuda", dtype=dt)
ro, rst = _raw_chunk(
    q, k, v, g=g, beta=beta,
    initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
)
assert torch.isfinite(ro).all()
ro1, _ = _raw_rec(
    q[:, :1], k[:, :1], v[:, :1], g=g[:, :1], beta=beta[:, :1],
    initial_state=rst, output_final_state=True, use_qk_l2norm_in_kernel=True,
)
assert torch.isfinite(ro1).all()
out, st = torch_chunk_gated_delta_rule(
    q, k, v, g=g, beta=beta,
    initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
)
assert torch.isfinite(out).all()
out1, _ = torch_recurrent_gated_delta_rule(
    q[:, :1], k[:, :1], v[:, :1], g=g[:, :1], beta=beta[:, :1],
    initial_state=st, output_final_state=True, use_qk_l2norm_in_kernel=True,
)
assert torch.isfinite(out1).all()
torch.cuda.synchronize()
print("OK")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode == 0 and "OK" in res.stdout:
        return True, "verified CUDA pass"
    err = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["unknown error"]
    return False, err[0][:200]


def _verify_causal_conv1d_on_gpu(use_bf16: bool) -> tuple[bool, str]:
    """Runs a fresh-subprocess live CUDA check of raw causal_conv1d and transformers hooks."""
    if importlib.util.find_spec("causal_conv1d") is None:
        return False, "not installed"
    code = f"""
import torch
from causal_conv1d import causal_conv1d_fn as _raw_fn, causal_conv1d_update as _raw_upd
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
dt = torch.bfloat16 if {use_bf16!r} else torch.float16
x = torch.randn(1, 32, 16, device="cuda", dtype=dt)
w = torch.randn(32, 4, device="cuda", dtype=dt)
ry = _raw_fn(x, w, None, activation="silu")
assert torch.isfinite(ry).all()
y = causal_conv1d_fn(x, w, None, activation="silu")
assert torch.isfinite(y).all()
state = torch.zeros(1, 32, 4, device="cuda", dtype=dt)
x1 = torch.randn(1, 32, 1, device="cuda", dtype=dt)
y1 = causal_conv1d_update(x1, state, w, None, "silu")
assert torch.isfinite(y1).all()
torch.cuda.synchronize()
print("OK")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode == 0 and "OK" in res.stdout:
        return True, "verified CUDA pass"
    err = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["unknown error"]
    return False, err[0][:200]


def setup_and_verify_kernels(
    mode: str = "auto",
    allow_source_build: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Installs (if eligible) and GPU-verifies `flash-linear-attention` and `causal-conv1d`."""
    has_cuda = torch.cuda.is_available()
    cc: tuple[int, int] | None = None
    gpu_name = "CPU"
    if has_cuda:
        cc = tuple(int(x) for x in torch.cuda.get_device_capability(0)[:2])
        gpu_name = torch.cuda.get_device_name(0)
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported() and cc is not None and cc[0] >= 8)
    dtype_name = "bfloat16" if use_bf16 else ("float16" if has_cuda else "float32")
    n_cpus = os.cpu_count() or 2
    has_nvcc = shutil.which("nvcc") is not None

    attempt, reason = should_attempt_kernels(cc, mode)
    report: dict[str, Any] = {
        "mode": mode,
        "gpu": gpu_name,
        "cc": f"sm_{cc[0]}{cc[1]}" if cc else None,
        "dtype": dtype_name,
        "attempt_install": attempt,
        "decision_reason": reason,
        "fla_active": importlib.util.find_spec("fla") is not None,
        "causal_conv1d_active": importlib.util.find_spec("causal_conv1d") is not None,
        "actions": [],
    }
    if dry_run or mode == "off":
        return report

    if attempt:
        constraint_path = _write_pip_constraints(pathlib.Path("/tmp/cbarr_pip_constraints.txt"))
        if not report["fla_active"]:
            # Install pure-Python + Triton FLA constrained against active torch/transformers
            res_fla = _pip_cmd(
                "install",
                "-q",
                "--constraint",
                str(constraint_path),
                "flash-linear-attention>=0.4.2",
            )
            if res_fla.returncode != 0:
                # Fallback: install einops + --no-deps so minor metadata pins never block compatible Triton FLA
                _pip_cmd("install", "-q", "einops")
                res_fla = _pip_cmd("install", "-q", "--no-deps", "flash-linear-attention>=0.4.2")
            report["actions"].append(f"install_fla={'ok' if res_fla.returncode == 0 else 'failed'}")

        if not report["causal_conv1d_active"]:
            # Always try prebuilt binary wheels first (~2s) to avoid 20-min nvcc compile on 2-CPU Colab VMs
            res_cc = _pip_cmd(
                "install",
                "-q",
                "--only-binary=:all:",
                "--no-build-isolation",
                "--constraint",
                str(constraint_path),
                "causal-conv1d>=1.4.0",
            )
            if res_cc.returncode == 0:
                report["actions"].append("install_causal_conv1d_wheel=ok")
            elif has_nvcc and (allow_source_build or n_cpus >= 8):
                res_src = _pip_cmd(
                    "install",
                    "-q",
                    "--no-build-isolation",
                    "--constraint",
                    str(constraint_path),
                    "causal-conv1d>=1.4.0",
                )
                report["actions"].append(
                    f"build_causal_conv1d_src={'ok' if res_src.returncode == 0 else 'failed'}"
                )
            else:
                report["actions"].append(
                    f"skip_causal_conv1d_src(nvcc={has_nvcc},cpus={n_cpus})"
                )

    # Self-test any installed kernel directly on active GPU & rollback cleanly if incompatible
    if has_cuda:
        if importlib.util.find_spec("fla") is not None:
            ok_fla, msg_fla = _verify_fla_on_gpu(use_bf16=use_bf16)
            report["fla_active"] = ok_fla
            report["fla_check"] = msg_fla
            if not ok_fla and attempt:
                _uninstall_pkg("flash-linear-attention")
                report["actions"].append("rolled_back_fla")
        else:
            report["fla_active"] = False
            report["fla_check"] = "not installed (PyTorch fallback active)"

        if importlib.util.find_spec("causal_conv1d") is not None:
            ok_cc, msg_cc = _verify_causal_conv1d_on_gpu(use_bf16=use_bf16)
            report["causal_conv1d_active"] = ok_cc
            report["causal_conv1d_check"] = msg_cc
            if not ok_cc and attempt:
                _uninstall_pkg("causal-conv1d")
                report["actions"].append("rolled_back_causal_conv1d")
        else:
            report["causal_conv1d_active"] = False
            report["causal_conv1d_check"] = "not installed (PyTorch fallback active)"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and verify Qwen3.5 Gated DeltaNet kernels.")
    parser.add_argument(
        "--mode",
        choices=("auto", "always", "off", "check-only"),
        default="auto",
        help="Kernel provisioning policy (default: auto for sm >= 7.5).",
    )
    parser.add_argument(
        "--allow-source-build",
        action="store_true",
        help="Allow compiling causal-conv1d from source even when cpu_count < 8.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON only without highlighted banner.",
    )
    args = parser.parse_args()
    rep = setup_and_verify_kernels(mode=args.mode, allow_source_build=args.allow_source_build)
    if not args.json:
        fla_tag = hl("ACTIVE (Triton)", "green") if rep["fla_active"] else hl("PYTORCH FALLBACK", "yellow")
        cc_tag = (
            hl("ACTIVE (CUDA)", "green")
            if rep["causal_conv1d_active"]
            else hl("PYTORCH FALLBACK", "yellow")
        )
        hdr_style = "green" if (rep["fla_active"] and rep["causal_conv1d_active"]) else "yellow"
        print(banner(f"DELTANET KERNEL STATUS ({rep['gpu']} | dtype={rep['dtype']})", hdr_style))
        print(f"  flash-linear-attention (fla) : {fla_tag} [{rep.get('fla_check', rep['decision_reason'])}]")
        print(f"  causal-conv1d                : {cc_tag} [{rep.get('causal_conv1d_check', rep['decision_reason'])}]")
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
