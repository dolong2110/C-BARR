#!/usr/bin/env python3
"""Generates unified, 2-part split, and 4-part concurrent C-BARR Colab/Kaggle notebooks."""

from __future__ import annotations

import ast
import json
import pathlib
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


@dataclass(frozen=True)
class NotebookSpec:
    filename: str
    mode: str  # "full" | "part1" | "part2" | "part1a_4b" | "part1b_smollm3" | "part2_scale" | "part3_analyze"
    title: str
    summary_bullets: tuple[str, ...]
    models_primary: tuple[str, ...]
    models_scale: tuple[str, ...]
    auto_include_9b: bool
    run_preflight_tests: bool
    run_live_smoke_expr: str
    run_calibrate_only: bool
    run_primary_matrix: bool
    run_scale_sweep: bool
    run_w7_extensions: bool
    run_phase7_analysis: bool
    unpack_synced_on_start: bool
    sync_include_ledger: bool
    expected_models_check: tuple[str, ...]
    sync_label_prefix: str
    bundle_filename: str


ALL_4_CORE_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3.5-4B",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
)


SPECS: tuple[NotebookSpec, ...] = (
    NotebookSpec(
        filename="C_BARR_Colab_Kaggle.ipynb",
        mode="full",
        title="C-BARR: Colab & Kaggle End-to-End GPU Runner",
        summary_bullets=(
            "Runs the complete **C-BARR** pipeline end-to-end (Preflight -> Calibration -> Phase 4/5 Primary -> Phase 6 Scale Sweep -> Phase 7 Analysis) in one session.",
            "**Intra-Model `N`-GPU Auto-Sharding (`MAX_SHARDS_PER_MODEL = 4`)**: Automatically shards each model across `min(GPU_COUNT, MAX_SHARDS_PER_MODEL)` local GPUs (`2`, `3`, `4+` GPUs), keeping 0% GPU idle time and merging deterministically before calibration/evaluation (bit-for-bit identical to 1 GPU).",
            "**Cross-Session GitHub Artifact Sync (`scripts/sync_results.py`)**: Compresses compact JSONL caches (`*.jsonl.gz`) and syncs per-model summaries + `EXPERIMENT_LOG.md`.",
            "**Hardware-Aware Precision (`DTYPE = \"auto\"`) & VRAM Guard**: Uses `bfloat16` on Ampere+ (`L4`/`A100`), `float16` on `T4`/`P100`, and enables `Qwen/Qwen3.5-9B` only when `VRAM >= 22 GB`.",
        ),
        models_primary=("Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"),
        models_scale=("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"),
        auto_include_9b=True,
        run_preflight_tests=True,
        run_live_smoke_expr="HAS_CUDA",
        run_calibrate_only=True,
        run_primary_matrix=True,
        run_scale_sweep=True,
        run_w7_extensions=False,
        run_phase7_analysis=True,
        unpack_synced_on_start=True,
        sync_include_ledger=True,
        expected_models_check=(),
        sync_label_prefix="full",
        bundle_filename="cbarr_results_bundle.zip",
    ),
    NotebookSpec(
        filename="C_BARR_Part1_Primary_Colab_Kaggle.ipynb",
        mode="part1",
        title="C-BARR (Part 1 of 2): Preflight + Primary Models (4B & SmolLM3-3B)",
        summary_bullets=(
            "**Session 1 of 2 (Sequential 2-Part Split)**: Runs Gate 0, Step 0a Smoke, Step 0b Calibration, and Phase 4/5 (`strict341 -> diag491`) for `Qwen/Qwen3.5-4B` and `HuggingFaceTB/SmolLM3-3B`.",
            "**Intra-Model `N`-GPU Auto-Sharding**: Uses all available local GPUs (`min(GPU_COUNT, MAX_SHARDS_PER_MODEL)`) on `4B` and `SmolLM3-3B` sequentially so no GPU sits idle (~1.5–2.5h on `GPU T4 x2`).",
            "**Auto GitHub Hand-off (`scripts/sync_results.py`)**: Compresses per-model caches (`*.jsonl.gz`, ~1.5–2 MB/model) and pushes summaries + `EXPERIMENT_LOG.md` after each stage so **Part 2** can pull and combine everything.",
        ),
        models_primary=("Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"),
        models_scale=("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"),
        auto_include_9b=True,
        run_preflight_tests=True,
        run_live_smoke_expr="HAS_CUDA",
        run_calibrate_only=True,
        run_primary_matrix=True,
        run_scale_sweep=False,
        run_w7_extensions=False,
        run_phase7_analysis=False,
        unpack_synced_on_start=False,
        sync_include_ledger=True,
        expected_models_check=(),
        sync_label_prefix="part1-primary",
        bundle_filename="cbarr_part1_bundle.zip",
    ),
    NotebookSpec(
        filename="C_BARR_Part2_Scale_Analyze_Colab_Kaggle.ipynb",
        mode="part2",
        title="C-BARR (Part 2 of 2): Pull Part 1 + Scale Sweep + Combined Phase 7",
        summary_bullets=(
            "**Session 2 of 2 (Sequential 2-Part Split)**: Clones/pulls `dolong2110/C-BARR`, unpacks **Part 1**'s pushed caches (`4B` + `SmolLM3-3B`), runs **Phase 6 Scale Sweep** (`0.8B` & `2B`, plus `9B` if `VRAM >= 22 GB`, `N`-GPU sharded), optional **W7**, and **Combined Phase 7**.",
            "**Unified Paper Outputs & GitHub Push**: Phase 7 runs across **all** Part 1 + Part 2 models together (`stats_summary.json`, `paper_tables.tex`, anonymized `paper/figures/fig{1,2,3}_*.pdf`), then pushes the combined artifacts + `EXPERIMENT_LOG.md` to GitHub.",
        ),
        models_primary=("Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"),
        models_scale=("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"),
        auto_include_9b=True,
        run_preflight_tests=True,
        run_live_smoke_expr="False",
        run_calibrate_only=False,
        run_primary_matrix=False,
        run_scale_sweep=True,
        run_w7_extensions=False,
        run_phase7_analysis=True,
        unpack_synced_on_start=True,
        sync_include_ledger=True,
        expected_models_check=ALL_4_CORE_MODELS,
        sync_label_prefix="part2-combined",
        bundle_filename="cbarr_results_bundle.zip",
    ),
    NotebookSpec(
        filename="C_BARR_Part1a_Qwen4B_Colab_Kaggle.ipynb",
        mode="part1a_4b",
        title="C-BARR Worker 1/3: Gate 0 + Calibrate + Qwen/Qwen3.5-4B (Primary)",
        summary_bullets=(
            "**Balanced Concurrent Split (Worker 1 of 3 GPU Workers)**: Runs Gate 0, Step 0a Smoke, Step 0b Calibration, and `Qwen/Qwen3.5-4B` (`strict341 -> diag491`).",
            "**Intra-Model `N`-GPU Auto-Sharding**: Automatically shards items across up to `MAX_SHARDS_PER_MODEL = 4` local GPUs (`T4 x2` -> ~1.5–2h, `4× GPU` -> ~45–60 min), bit-for-bit identical to 1 GPU.",
            "**Conflict-Free Concurrent GitHub Sync (`--no-ledger`)**: Pushes **only** `outputs/Qwen__Qwen3.5-4B/*` so you can run Worker 1a, Worker 1b, and Worker 2 **simultaneously** on separate Kaggle/Colab sessions with zero git conflicts before running **Part 3 Combiner**.",
        ),
        models_primary=("Qwen/Qwen3.5-4B",),
        models_scale=(),
        auto_include_9b=False,
        run_preflight_tests=True,
        run_live_smoke_expr="HAS_CUDA",
        run_calibrate_only=True,
        run_primary_matrix=True,
        run_scale_sweep=False,
        run_w7_extensions=False,
        run_phase7_analysis=False,
        unpack_synced_on_start=False,
        sync_include_ledger=False,
        expected_models_check=(),
        sync_label_prefix="worker-4b",
        bundle_filename="cbarr_part1a_qwen4b_bundle.zip",
    ),
    NotebookSpec(
        filename="C_BARR_Part1b_SmolLM3_Colab_Kaggle.ipynb",
        mode="part1b_smollm3",
        title="C-BARR Worker 2/3: HuggingFaceTB/SmolLM3-3B (Primary)",
        summary_bullets=(
            "**Balanced Concurrent Split (Worker 2 of 3 GPU Workers)**: Dedicated runner for `HuggingFaceTB/SmolLM3-3B` (`strict341 -> diag491`, ~45–60 min on `T4 x2` or ~25 min on `4× GPU`).",
            "**Intra-Model `N`-GPU Auto-Sharding**: Splits items across up to `MAX_SHARDS_PER_MODEL = 4` local GPUs and merges deterministically.",
            "**Conflict-Free Concurrent GitHub Sync (`--no-ledger`)**: Pushes **only** `outputs/HuggingFaceTB__SmolLM3-3B/*` so it can run side-by-side with `Part1a_Qwen4B` and `Part2_Scale` with zero git conflicts.",
        ),
        models_primary=("HuggingFaceTB/SmolLM3-3B",),
        models_scale=(),
        auto_include_9b=False,
        run_preflight_tests=False,
        run_live_smoke_expr="False",
        run_calibrate_only=False,
        run_primary_matrix=True,
        run_scale_sweep=False,
        run_w7_extensions=False,
        run_phase7_analysis=False,
        unpack_synced_on_start=False,
        sync_include_ledger=False,
        expected_models_check=(),
        sync_label_prefix="worker-smollm3",
        bundle_filename="cbarr_part1b_smollm3_bundle.zip",
    ),
    NotebookSpec(
        filename="C_BARR_Part2_Scale_Colab_Kaggle.ipynb",
        mode="part2_scale",
        title="C-BARR Worker 3/3: Scale Sweep (Qwen3.5-0.8B & 2B [+9B if >=22 GB])",
        summary_bullets=(
            "**Balanced Concurrent Split (Worker 3 of 3 GPU Workers)**: Dedicated runner for Phase 6 Scale Sweep (`Qwen/Qwen3.5-0.8B` & `Qwen/Qwen3.5-2B` with `strict341`, plus `Qwen/Qwen3.5-9B` automatically if `VRAM >= 22 GB`).",
            "**Intra-Model `N`-GPU Auto-Sharding**: Splits each scale model cleanly across up to `MAX_SHARDS_PER_MODEL = 4` local GPUs (~45 min combined on `T4 x2`).",
            "**Conflict-Free Concurrent GitHub Sync (`--no-ledger`)**: Pushes **only** `outputs/Qwen__Qwen3.5-{0.8B,2B,9B}/*` immediately after each model completes so it can run simultaneously alongside `Part1a_Qwen4B` and `Part1b_SmolLM3`.",
        ),
        models_primary=(),
        models_scale=("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"),
        auto_include_9b=True,
        run_preflight_tests=False,
        run_live_smoke_expr="False",
        run_calibrate_only=False,
        run_primary_matrix=False,
        run_scale_sweep=True,
        run_w7_extensions=False,
        run_phase7_analysis=False,
        unpack_synced_on_start=False,
        sync_include_ledger=False,
        expected_models_check=(),
        sync_label_prefix="worker-scale",
        bundle_filename="cbarr_part2_scale_bundle.zip",
    ),
    NotebookSpec(
        filename="C_BARR_Part3_W7_Analyze_Colab_Kaggle.ipynb",
        mode="part3_analyze",
        title="C-BARR Combiner (Step 4): Unpack All Workers + Optional W7 + Combined Phase 7",
        summary_bullets=(
            "**Final Combiner for the 3-Worker Split**: Pulls `dolong2110/C-BARR`, unpacks all worker models (`Qwen3.5-4B`, `SmolLM3-3B`, `Qwen3.5-0.8B`, `Qwen3.5-2B`, `+9B` if present) via `scripts/sync_results.py --mode unpack`, optionally runs **W7** (`RUN_W7_EXTENSIONS = True` on `4B`), and runs **Combined Phase 7**.",
            "**Runs on CPU (~2 min) when `RUN_W7_EXTENSIONS = False`**: Does not require a GPU quota slot unless you toggle `RUN_W7_EXTENSIONS = True`.",
            "**Unified Paper Outputs & Push**: Generates `outputs/stats_summary.json`, `outputs/paper_tables.tex`, anonymized `paper/figures/*.pdf`, and unified `EXPERIMENT_LOG.md`, then pushes everything cleanly to GitHub.",
        ),
        models_primary=("Qwen/Qwen3.5-4B", "HuggingFaceTB/SmolLM3-3B"),
        models_scale=("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"),
        auto_include_9b=True,
        run_preflight_tests=True,
        run_live_smoke_expr="False",
        run_calibrate_only=False,
        run_primary_matrix=False,
        run_scale_sweep=False,
        run_w7_extensions=False,
        run_phase7_analysis=True,
        unpack_synced_on_start=True,
        sync_include_ledger=True,
        expected_models_check=ALL_4_CORE_MODELS,
        sync_label_prefix="part3-combined",
        bundle_filename="cbarr_results_bundle.zip",
    ),
)


def _to_source_lines(text: str) -> list[str]:
    lines = text.strip("\n").splitlines()
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _md_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _to_source_lines(text),
    }


def _code_cell(cell_id: str, code: str) -> dict:
    clean = code.strip("\n")
    ast.parse(clean, filename=f"{cell_id}.py")
    for line in clean.splitlines():
        s = line.lstrip()
        if s.startswith("!") or s.startswith("%"):
            raise ValueError(f"Shell magic disallowed in cell {cell_id}: {line}")
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_source_lines(clean),
    }


def _build_intro_cell(spec: NotebookSpec) -> dict:
    badge_url = (
        f"https://colab.research.google.com/github/dolong2110/C-BARR/blob/master/notebooks/{spec.filename}"
    )
    bullets_md = "\n".join(f"- {b}" for b in spec.summary_bullets)
    text = f"""# {spec.title}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({badge_url})

{bullets_md}

- **Crash-Safe Resume**: Every completed item is flushed per UID into `OUT_DIR/<model>/*_cache.jsonl`. On Colab, mounting Google Drive preserves all progress across disconnects; on Kaggle, HF model caches go into `/kaggle/temp/hf_cache` so `/kaggle/working` stays within quota."""
    return _md_cell("cbarr-intro", text)


def _build_env_detect_cell() -> dict:
    code = '''import os
import pathlib
import platform
import shutil
import subprocess
import sys

# Detect runtime platform (Google Colab, Kaggle, or standalone Linux)
IS_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or pathlib.Path("/kaggle/working").exists()
PLATFORM = "colab" if IS_COLAB else ("kaggle" if IS_KAGGLE else "linux")

# Toggle whether Colab mounts Google Drive so intermediate caches survive disconnects
USE_GOOGLE_DRIVE = True
DRIVE_ROOT = None

if IS_COLAB:
    WORKSPACE_ROOT = pathlib.Path("/content")
    if USE_GOOGLE_DRIVE:
        try:
            from google.colab import drive

            drive.mount("/content/drive", force_remount=False)
            DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive/C-BARR")
            DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[WARN] Google Drive mount skipped ({exc}); using ephemeral /content storage.")
elif IS_KAGGLE:
    WORKSPACE_ROOT = pathlib.Path("/kaggle/working")
else:
    WORKSPACE_ROOT = pathlib.Path.cwd()

REPO_DIR = WORKSPACE_ROOT / "C-BARR"
OUT_DIR = (DRIVE_ROOT / "outputs") if DRIVE_ROOT else (REPO_DIR / "outputs")
FIG_DIR = REPO_DIR / "paper" / "figures"

# Store large HF weight downloads outside /kaggle/working so output quota is preserved for caches & tables
if IS_KAGGLE:
    hf_tmp = pathlib.Path("/kaggle/temp/hf_cache")
    hf_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_tmp)
elif IS_COLAB:
    os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

# Avoid matplotlib font-cache permission warnings on ephemeral containers
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import torch

HAS_CUDA = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if HAS_CUDA else 0
VRAM_GB = 0.0
GPU_NAME = "None (CPU)"
SM_CAP = (0, 0)
if HAS_CUDA:
    props = torch.cuda.get_device_properties(0)
    GPU_NAME = props.name
    VRAM_GB = props.total_memory / (1024**3)
    SM_CAP = torch.cuda.get_device_capability(0)

disk = shutil.disk_usage(str(WORKSPACE_ROOT))
print(f"Platform       : {PLATFORM} ({platform.machine()})")
print(f"Python / Torch : {sys.version.split()[0]} / {torch.__version__}")
print(f"GPU            : {GPU_NAME} x{GPU_COUNT} | VRAM={VRAM_GB:.1f} GB | SM={SM_CAP} | BF16={HAS_CUDA and torch.cuda.is_bf16_supported() and SM_CAP[0] >= 8}")
print(f"Workspace / Out: {REPO_DIR} -> {OUT_DIR}")
print(f"HF_HOME        : {os.environ.get('HF_HOME', 'default (~/.cache/huggingface)')}")
print(f"Free Disk      : {disk.free / (1024**3):.1f} GB")
if not HAS_CUDA:
    print("[WARN] No CUDA GPU detected! CPU/Phase-7 mode only unless GPU Accelerator is enabled.")'''
    return _code_cell("cbarr-env-detect", code)


def _build_config_cell(spec: NotebookSpec) -> dict:
    primary_list = list(spec.models_primary)
    scale_list = list(spec.models_scale)
    expected_list = list(spec.expected_models_check)
    code = f'''import datetime as dt
import json
import time

# === Experiment & Pipeline Configuration ({spec.mode.upper()}) ===
os.environ.setdefault("CBARR_COLOR", "1")  # Highlight key metrics/gates in Colab/Kaggle cells
REPO_URL = "https://github.com/dolong2110/C-BARR.git"
REPO_BRANCH = "master"
NOTEBOOK_NAME = {spec.filename!r}
PIPELINE_MODE = {spec.mode!r}

# Primary evaluation + causal diagnostic models (strict341 -> diag491)
MODELS_PRIMARY = {primary_list!r}

# Cross-scale models (run with --strict-diag for Table 2)
MODELS_SCALE = {scale_list!r}
# Automatically include Qwen3.5-9B (~18.5 GB) when GPU has >= 22 GB VRAM (L4/A100)
AUTO_INCLUDE_9B = {spec.auto_include_9b!r}
FORCE_9B = False
if (AUTO_INCLUDE_9B and VRAM_GB >= 22.0) or FORCE_9B:
    if "Qwen/Qwen3.5-9B" not in MODELS_SCALE:
        MODELS_SCALE.append("Qwen/Qwen3.5-9B")

# Pipeline stage toggles
RUN_PREFLIGHT_TESTS = {spec.run_preflight_tests!r}
RUN_LIVE_SMOKE = {spec.run_live_smoke_expr}
RUN_CALIBRATE_ONLY = {spec.run_calibrate_only!r}
RUN_PRIMARY_MATRIX = {spec.run_primary_matrix!r}
RUN_SCALE_SWEEP = {spec.run_scale_sweep!r}
INCLUDE_B512 = False       # Contingency R3 auxiliary B=512 sweep
RUN_W7_EXTENSIONS = {spec.run_w7_extensions!r}  # Optional W7: DeltaNet state blend + Prefill MLP probe
RUN_PHASE7_ANALYSIS = {spec.run_phase7_analysis!r}
EXPECTED_MODELS_FOR_PHASE7 = {expected_list!r}

# Cross-session & intra-model multi-GPU orchestration
AUTO_PARALLEL_MULTI_GPU = True       # Auto-shards items across up to MAX_SHARDS_PER_MODEL local GPUs
MAX_SHARDS_PER_MODEL = 4             # Appropriate shard cap per model (set 6-8 on 8x GPU servers)
UNPACK_SYNCED_RESULTS_ON_START = {spec.unpack_synced_on_start!r}  # Unpacks *.jsonl.gz from git repo into OUT_DIR
SYNC_RESULTS_TO_GITHUB = True        # Commits & pushes compact summaries + *.jsonl.gz via scripts/sync_results.py
SYNC_INCLUDE_LEDGER = {spec.sync_include_ledger!r}  # False in concurrent worker notebooks to prevent git conflicts on EXPERIMENT_LOG.md
SYNC_LABEL_PREFIX = {spec.sync_label_prefix!r}
BUNDLE_FILENAME = {spec.bundle_filename!r}

# Training, Calibration & Routing Hyperparameters (logged + forwarded to run_matrix.py)
SEED = 42
LAM_UTILITY = 0.05
NORM_TOKENS = 250.0
LOGREG_C = 1.0
TAU_MIN = 0.15
TAU_MAX = 0.85
TAU_STEPS = 29

# Hardware & cache behavior
DTYPE = "auto"                        # "auto" picks BF16 on L4/A100 and FP16 on T4/P100
INSTALL_FAST_DELTANET_KERNELS = "auto"  # "auto" installs + GPU-verifies Triton/wheel DeltaNet kernels on sm >= 7.5
AUTO_PRUNE_HF_CACHE_BELOW_GB = 18.0   # Prune old downloaded weights only when disk drops below threshold
PREFETCH_MODEL_WEIGHTS = False        # Set True to pre-download all active HF checkpoints in Cell 5

# Per-loop Ledger & GitHub Auto-Push
RECORD_EXPERIMENT_LOG = True
EXP_LOG_REL_PATH = "EXPERIMENT_LOG.md"
AUTO_PUSH_RESULTS_TO_GITHUB = True    # Commits + pushes results when GITHUB_TOKEN is set
CHECKPOINT_EVERY_ITEMS = 150          # Single-GPU runs push intermediate progress every N new eval_450 items (0=off)
RUN_ID = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
PIPELINE_START_TIME = time.perf_counter()


def _load_secret(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    if IS_COLAB:
        try:
            from google.colab import userdata

            s = userdata.get(name)
            if s:
                return str(s).strip()
        except Exception:
            pass
    if IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient

            s = UserSecretsClient().get_secret(name)
            if s:
                return str(s).strip()
        except Exception:
            pass
    return ""


GITHUB_TOKEN = _load_secret("GITHUB_TOKEN")  # Needed for private repos and GitHub auto-push
if GITHUB_TOKEN:
    os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN
elif AUTO_PUSH_RESULTS_TO_GITHUB and REPO_URL.startswith("https://"):
    print(
        "[INFO] GITHUB_TOKEN secret not found: results and compact caches will be "
        "committed locally (add GITHUB_TOKEN secret with Contents Read/Write to enable automatic GitHub push)."
    )

HF_TOKEN = _load_secret("HF_TOKEN")          # Optional; improves HF Hub download limits
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    try:
        from huggingface_hub import login

        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Hugging Face authentication configured cleanly.")
    except Exception as exc:
        print(f"[WARN] huggingface_hub login skipped: {{exc}}")

ALL_ACTIVE_MODELS = list(
    dict.fromkeys(
        (MODELS_PRIMARY if RUN_PRIMARY_MATRIX or RUN_CALIBRATE_ONLY else [])
        + (MODELS_SCALE if RUN_SCALE_SWEEP else [])
        + ([MODELS_PRIMARY[0]] if (RUN_W7_EXTENSIONS and MODELS_PRIMARY) else [])
    )
)

N_ACTIVE_SHARDS = (
    min(GPU_COUNT, max(1, int(MAX_SHARDS_PER_MODEL)))
    if (AUTO_PARALLEL_MULTI_GPU and HAS_CUDA and GPU_COUNT >= 2)
    else 1
)

MATRIX_PARAM_ARGS = [
    "--seed", str(SEED),
    "--lam-utility", str(LAM_UTILITY),
    "--norm-tokens", str(NORM_TOKENS),
    "--reg-c", str(LOGREG_C),
    "--tau-min", str(TAU_MIN),
    "--tau-max", str(TAU_MAX),
    "--tau-steps", str(TAU_STEPS),
]
NOTEBOOK_PARAMS = {{
    "platform": PLATFORM,
    "notebook": NOTEBOOK_NAME,
    "pipeline_mode": PIPELINE_MODE,
    "run_id": RUN_ID,
    "gpu_count": GPU_COUNT,
    "active_shards_per_model": N_ACTIVE_SHARDS,
    "dtype": DTYPE,
    "seed": SEED,
    "lam_utility": LAM_UTILITY,
    "norm_tokens": NORM_TOKENS,
    "logreg_C": LOGREG_C,
    "tau_grid": [TAU_MIN, TAU_MAX, TAU_STEPS],
    "models_primary": MODELS_PRIMARY if RUN_PRIMARY_MATRIX else [],
    "models_scale": MODELS_SCALE if RUN_SCALE_SWEEP else [],
    "include_b512": INCLUDE_B512,
    "run_w7_extensions": RUN_W7_EXTENSIONS,
    "run_phase7_analysis": RUN_PHASE7_ANALYSIS,
    "sync_include_ledger": SYNC_INCLUDE_LEDGER,
    "install_fast_kernels": str(INSTALL_FAST_DELTANET_KERNELS),
    "auto_push_github": AUTO_PUSH_RESULTS_TO_GITHUB,
    "checkpoint_every_items": CHECKPOINT_EVERY_ITEMS,
}}
print("Run ID       :", RUN_ID)
print("Notebook Mode:", f"{{PIPELINE_MODE}} ({{NOTEBOOK_NAME}}) | shards/model={{N_ACTIVE_SHARDS}}")
print("Active models:", ALL_ACTIVE_MODELS if ALL_ACTIVE_MODELS else "(combiner / Phase 7 only)")'''
    return _code_cell("cbarr-config", code)


def _build_clone_install_cell() -> dict:
    code = '''import concurrent.futures
import importlib.metadata
import re
import threading


def _sanitize_git_err(msg: str, token: str) -> str:
    return msg.replace(token, "***") if token else msg


def _git_clone_url() -> str:
    if GITHUB_TOKEN and REPO_URL.startswith("https://"):
        return REPO_URL.replace("https://", f"https://x-access-token:{GITHUB_TOKEN}@", 1)
    return REPO_URL


# 1. Clone or pull latest C-BARR repository
if not (REPO_DIR / ".git").exists():
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "clone", "--branch", REPO_BRANCH, "--single-branch", _git_clone_url(), str(REPO_DIR)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(_sanitize_git_err(proc.stderr or proc.stdout, GITHUB_TOKEN))
    print(f"Cloned {REPO_URL} -> {REPO_DIR}")
else:
    # Preserve unpacked/local caches while restoring the tracked smoke gate fixture if overwritten
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "checkout", "--", "outputs/smoke_test_gate3_report.json"],
        check=False,
    )
    proc = subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull", "--rebase", "--autostash", "origin", REPO_BRANCH],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(_sanitize_git_err(proc.stderr or proc.stdout, GITHUB_TOKEN))
    print(f"Updated existing repo at {REPO_DIR}")

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Per-step & per-loop experiment ledger + thread-safe log-tee helpers
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TIMINGS_FILE = LOG_DIR / f"timings_{RUN_ID}.jsonl"
PARAMS_FILE = LOG_DIR / f"params_{RUN_ID}.json"
RUN_LOG_FILE = LOG_DIR / f"cbarr_{RUN_ID}.log"
# Concurrent worker notebooks log locally so simultaneous pushes never race EXPERIMENT_LOG.md
LEDGER_MD_PATH = (REPO_DIR / EXP_LOG_REL_PATH) if SYNC_INCLUDE_LEDGER else (LOG_DIR / f"experiment_log_{RUN_ID}.md")
PARAMS_FILE.write_text(json.dumps(NOTEBOOK_PARAMS, indent=2) + "\\n", encoding="utf-8")
_ANSI_ESCAPE_RE = re.compile(r"\\x1b\\[[0-9;]*[A-Za-z]")
_IO_LOCK = threading.Lock()


def _highlight_line(line: str) -> str:
    """Applies ANSI highlights to key Colab/Kaggle cell output lines if not already styled."""
    if "\\x1b[" in line:
        return line
    lower = line.lower()
    if any(k in lower for k in ("error", "fail", "traceback", "exception")):
        return f"\\033[1;31m{line}\\033[0m"
    if any(
        k in lower
        for k in (
            "[step",
            "[gate",
            "[sync]",
            "[merge]",
            "pass",
            "completed",
            "calibration",
            "section 6",
            "experiment_log",
            "pushed",
            "delta oaa",
        )
    ):
        return f"\\033[1;32m{line}\\033[0m"
    if any(k in lower for k in ("warn", "executing:", "checkpoint")):
        return f"\\033[1;33m{line}\\033[0m"
    return line


def run_and_tee(
    cmd: list[str],
    extra_env: dict[str, str] | None = None,
    prefix: str = "",
    worker_log_path: pathlib.Path | None = None,
) -> None:
    """Streams highlighted subprocess output live while saving an ANSI-stripped copy to log files."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    tag = f"[{prefix}] " if prefix else ""
    header = f"\\n{tag}$ {' '.join(str(x) for x in cmd)}\\n"
    with _IO_LOCK:
        with RUN_LOG_FILE.open("a", encoding="utf-8") as log_fh:
            log_fh.write(header)
        if worker_log_path is not None:
            worker_log_path.parent.mkdir(parents=True, exist_ok=True)
            with worker_log_path.open("a", encoding="utf-8") as w_fh:
                w_fh.write(header)

    proc = subprocess.Popen(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        stripped = raw_line.rstrip("\\n")
        last_segment = stripped.split("\\r")[-1]
        clean = _ANSI_ESCAPE_RE.sub("", last_segment)
        with _IO_LOCK:
            with RUN_LOG_FILE.open("a", encoding="utf-8") as log_fh:
                log_fh.write(f"{tag}{clean}\\n")
            if worker_log_path is not None:
                with worker_log_path.open("a", encoding="utf-8") as w_fh:
                    w_fh.write(f"{clean}\\n")
            print(f"{tag}{_highlight_line(last_segment)}", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def matrix_step_flags(allow_git_step_push: bool = True) -> list[str]:
    """Builds per-step checkpoint & optional GitHub auto-push flags for scripts/run_matrix.py."""
    PARAMS_FILE.write_text(json.dumps(NOTEBOOK_PARAMS, indent=2) + "\\n", encoding="utf-8")
    if not allow_git_step_push or not SYNC_INCLUDE_LEDGER:
        return ["--step-checkpoint-every", "0"]
    flags = [
        "--run-id", RUN_ID,
        "--timings-file", str(TIMINGS_FILE),
        "--log-file", str(RUN_LOG_FILE),
        "--params-json", f"@{PARAMS_FILE}",
        "--step-checkpoint-every", str(CHECKPOINT_EVERY_ITEMS),
    ]
    if RECORD_EXPERIMENT_LOG:
        flags.extend(["--exp-log", str(LEDGER_MD_PATH)])
        if AUTO_PUSH_RESULTS_TO_GITHUB:
            flags.extend(["--git-push", "--git-branch", REPO_BRANCH])
    return flags


def _append_stage_timing(stage: str, status: str, dur_sec: float) -> None:
    rec = {"stage": stage, "status": status, "duration_sec": round(float(dur_sec), 2)}
    with _IO_LOCK:
        with TIMINGS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\\n")


def record_loop_checkpoint(
    loop_name: str,
    status: str = "RUNNING",
    stage_dur_sec: float | None = None,
    stage_ok: bool = True,
    push: bool = True,
    notes: str = "",
) -> None:
    """Records stage duration telemetry, upserts markdown ledger, and optionally pushes."""
    if stage_dur_sec is not None:
        _append_stage_timing(loop_name, "PASS" if stage_ok else "FAIL", stage_dur_sec)
    if not RECORD_EXPERIMENT_LOG:
        return
    PARAMS_FILE.write_text(json.dumps(NOTEBOOK_PARAMS, indent=2) + "\\n", encoding="utf-8")
    total_sec = round(time.perf_counter() - PIPELINE_START_TIME, 2)
    cmd = [
        sys.executable,
        "scripts/log_experiment.py",
        "--run-id", RUN_ID,
        "--status", status,
        "--loop-name", loop_name,
        "--command", f"notebook:{NOTEBOOK_NAME}",
        "--out-dir", str(OUT_DIR),
        "--timings-file", str(TIMINGS_FILE),
        "--log-file", str(RUN_LOG_FILE),
        "--params-json", f"@{PARAMS_FILE}",
        "--md-log", str(LEDGER_MD_PATH),
        "--total-sec", f"{total_sec:.2f}",
        "--upsert",
    ]
    smoke_rpt = REPO_DIR / "outputs" / "smoke_test_gate3_report.json"
    if smoke_rpt.exists():
        cmd.extend(["--smoke-report", str(smoke_rpt)])
    if notes:
        cmd.extend(["--notes", notes])
    if push and SYNC_INCLUDE_LEDGER and AUTO_PUSH_RESULTS_TO_GITHUB:
        cmd.extend(["--git-push", "--git-branch", REPO_BRANCH])
    subprocess.run(cmd, check=False)


def sync_results_to_github(
    label: str,
    models: list[str] | None = None,
    include_phase7: bool = False,
) -> None:
    """Commits compact summaries & *.jsonl.gz caches (excluding shared ledger in worker mode)."""
    if not SYNC_RESULTS_TO_GITHUB:
        return
    cmd = [
        sys.executable,
        "scripts/sync_results.py",
        "--mode", "push",
        "--repo-dir", str(REPO_DIR),
        "--out-dir", str(OUT_DIR),
        "--label", f"{SYNC_LABEL_PREFIX}:{label}:{RUN_ID}",
        "--git-branch", REPO_BRANCH,
    ]
    if models:
        cmd.extend(["--models", *models])
    if not include_phase7:
        cmd.append("--no-phase7")
    if not SYNC_INCLUDE_LEDGER:
        cmd.append("--no-ledger")
    if AUTO_PUSH_RESULTS_TO_GITHUB:
        cmd.append("--git-push")
    else:
        cmd.append("--no-git-push")
    run_and_tee(cmd)


def run_logged_loop(loop_name: str, fn, *args, push: bool = True, **kwargs):
    """Executes one pipeline loop, guarantees PASS/FAILED ledger logging, and re-raises on error."""
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:
        dt_sec = time.perf_counter() - t0
        record_loop_checkpoint(
            loop_name,
            status=f"FAILED (at {loop_name})",
            stage_dur_sec=dt_sec,
            stage_ok=False,
            push=push,
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise
    dt_sec = time.perf_counter() - t0
    record_loop_checkpoint(
        loop_name,
        status=f"RUNNING (after {loop_name})",
        stage_dur_sec=dt_sec,
        stage_ok=True,
        push=push,
    )
    return out


# 2. Install/upgrade only missing runtime packages (preserve container CUDA PyTorch!)
def _pkg_ver(name: str) -> tuple[int, ...] | None:
    try:
        v = importlib.metadata.version(name)
        nums = [int(x) for x in re.findall(r"\\d+", v)[:2]]
        return tuple(nums)
    except importlib.metadata.PackageNotFoundError:
        return None


required_pkgs = [
    "transformers>=4.53.0",
    "accelerate",
    "datasets",
    "matplotlib",
    "scipy",
    "scikit-learn",
    "nltk",
    "immutabledict",
    "langdetect",
    "absl-py",
]
torch_v = _pkg_ver("torch")
if torch_v is None or torch_v < (2, 6):
    required_pkgs.insert(0, "torch>=2.6")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", *required_pkgs], check=True)

# Optional fast CUDA/Triton Gated DeltaNet kernels with live CUDA self-test & safe rollback
kernel_mode = (
    "always"
    if INSTALL_FAST_DELTANET_KERNELS is True
    else ("off" if INSTALL_FAST_DELTANET_KERNELS is False else str(INSTALL_FAST_DELTANET_KERNELS))
)
if HAS_CUDA and kernel_mode != "off":
    subprocess.run(
        [sys.executable, "scripts/setup_fast_kernels.py", "--mode", kernel_mode],
        check=False,
    )

import importlib.util as _ilu

has_fla = _ilu.find_spec("fla") is not None
has_cc1d = _ilu.find_spec("causal_conv1d") is not None
print(f"Environment provisioning complete (fla={has_fla}, causal_conv1d={has_cc1d}).")'''
    return _code_cell("cbarr-clone-install", code)


def _build_download_data_cell() -> dict:
    code = '''from langdetect import DetectorFactory
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Patch D3: Pin langdetect seed globally
DetectorFactory.seed = 0

# 1. Verify official google-research instruction_following_eval is present (clone sparse if missing)
ifeval_dir = REPO_DIR / "instruction_following_eval"
if not ifeval_dir.exists():
    tmp_gr = pathlib.Path("/tmp/gr_sparse")
    if tmp_gr.exists():
        shutil.rmtree(tmp_gr)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/google-research/google-research.git",
            str(tmp_gr),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_gr), "sparse-checkout", "set", "instruction_following_eval"],
        check=True,
    )
    shutil.copytree(tmp_gr / "instruction_following_eval", ifeval_dir)
    shutil.rmtree(tmp_gr)

# 2. Download NLTK packages required by IFEval verifiers
for nltk_res in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng"):
    nltk.download(nltk_res, quiet=True)

# 3. Verify frozen dataset splits against manifest_sha256.json (never re-randomize frozen splits!)
from scripts.run_matrix import REQUIRED_DATA_SPLITS, verify_manifest_integrity

data_dir = REPO_DIR / "data"
if not (data_dir / "manifest_sha256.json").exists() or not all(
    (data_dir / f).exists() for f in REQUIRED_DATA_SPLITS
):
    print("Frozen splits missing; building deterministically via data/freeze_datasets.py (seed=42)...")
    subprocess.run([sys.executable, str(data_dir / "freeze_datasets.py")], check=True)

manifest = verify_manifest_integrity(data_dir)
print(f"Verified {len(manifest)} frozen dataset files via SHA-256 manifest:")
for fname in REQUIRED_DATA_SPLITS:
    print(f"  OK {fname}: {manifest[fname][:16]}...")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 4. Unpack previously pushed *.jsonl.gz caches + summaries when combining results
if UNPACK_SYNCED_RESULTS_ON_START:
    run_and_tee(
        [
            sys.executable,
            "scripts/sync_results.py",
            "--mode",
            "unpack",
            "--repo-dir",
            str(REPO_DIR),
            "--out-dir",
            str(OUT_DIR),
        ]
    )
    unpacked_models = sorted(
        d.name
        for d in OUT_DIR.iterdir()
        if d.is_dir() and d.name not in {"logs", "__pycache__"} and (d / "eval_450_cache.jsonl").exists()
    )
    print(f"Models currently ready in OUT_DIR: {unpacked_models}")

# 5. Pre-download Hugging Face tokenizers (and optionally full checkpoints)
for m_id in ALL_ACTIVE_MODELS:
    print(f"Checking tokenizer cache for {m_id}...")
    AutoTokenizer.from_pretrained(m_id)
    if PREFETCH_MODEL_WEIGHTS:
        print(f"Pre-downloading weights for {m_id}...")
        AutoModelForCausalLM.from_pretrained(m_id)'''
    return _code_cell("cbarr-download-data", code)


def _build_gate0_smoke_cell() -> dict:
    code = '''if RUN_PREFLIGHT_TESTS:
    print("=== [Gate 0] Running Offline Preflight Unit Tests ===")
    run_logged_loop(
        "Gate 0: Preflight Tests",
        run_and_tee,
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    )

if RUN_LIVE_SMOKE:
    print("=== [Step 0a] Running Live GPU 12-Item Smoke Gate ===")
    smoke_models = [m for m in MODELS_PRIMARY if ("9B" not in m or VRAM_GB >= 22.0)] or ALL_ACTIVE_MODELS[:1]
    if smoke_models:
        run_logged_loop(
            "Step 0a: Live GPU 12-Item Smoke",
            run_and_tee,
            [sys.executable, "scripts/run_smoke_test.py", "--live-gpu", "--models", *smoke_models],
        )'''
    return _code_cell("cbarr-gate0-smoke", code)


def _build_calibrate_cell() -> dict:
    code = '''if RUN_CALIBRATE_ONLY:
    cal_model = MODELS_PRIMARY[0] if MODELS_PRIMARY else (ALL_ACTIVE_MODELS[0] if ALL_ACTIVE_MODELS else "Qwen/Qwen3.5-4B")
    print(f"=== [Step 0b] Running Zero-Leakage Calibration on {cal_model} (shards={N_ACTIVE_SHARDS}) ===")
    if N_ACTIVE_SHARDS <= 1:
        run_logged_loop(
            f"Step 0b: Calibration ({cal_model})",
            run_and_tee,
            [
                sys.executable,
                "scripts/run_matrix.py",
                "--model",
                cal_model,
                "--data-dir",
                "data",
                "--out-dir",
                str(OUT_DIR),
                "--dtype",
                DTYPE,
                "--calibrate-only",
                *MATRIX_PARAM_ARGS,
                *matrix_step_flags(),
            ],
        )
    else:
        t_cal = time.perf_counter()

        def _cal_shard(s_idx: int) -> None:
            slug = cal_model.replace("/", "__")
            w_log = LOG_DIR / f"cbarr_{RUN_ID}_gpu{s_idx}_{slug}_calib.log"
            run_and_tee(
                [
                    sys.executable,
                    "scripts/run_matrix.py",
                    "--model",
                    cal_model,
                    "--data-dir",
                    "data",
                    "--out-dir",
                    str(OUT_DIR),
                    "--dtype",
                    DTYPE,
                    "--calibrate-only",
                    "--shard-idx",
                    str(s_idx),
                    "--num-shards",
                    str(N_ACTIVE_SHARDS),
                    "--step-checkpoint-every",
                    "0",
                    *MATRIX_PARAM_ARGS,
                ],
                extra_env={"CUDA_VISIBLE_DEVICES": str(s_idx)},
                prefix=f"GPU{s_idx}:calib",
                worker_log_path=w_log,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_ACTIVE_SHARDS) as pool:
            futs = [pool.submit(_cal_shard, i) for i in range(N_ACTIVE_SHARDS)]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

        run_and_tee(
            [
                sys.executable,
                "scripts/run_matrix.py",
                "--model",
                cal_model,
                "--data-dir",
                "data",
                "--out-dir",
                str(OUT_DIR),
                "--dtype",
                DTYPE,
                "--calibrate-only",
                "--merge-shards",
                *MATRIX_PARAM_ARGS,
                *matrix_step_flags(),
            ]
        )
        record_loop_checkpoint(
            f"Step 0b: Calibration ({cal_model})",
            status=f"RUNNING (after Step 0b: {cal_model})",
            stage_dur_sec=time.perf_counter() - t_cal,
            stage_ok=True,
        )'''
    return _code_cell("cbarr-calibrate", code)


def _build_primary_matrix_cell() -> dict:
    code = '''def _prune_hf_hub_if_low_disk() -> None:
    hf_hub = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface")) / "hub"
    if not hf_hub.exists():
        return
    free_gb = shutil.disk_usage(str(hf_hub)).free / (1024**3)
    if free_gb < AUTO_PRUNE_HF_CACHE_BELOW_GB:
        print(f"[Disk Guard] Free disk {free_gb:.1f} GB < {AUTO_PRUNE_HF_CACHE_BELOW_GB:.1f} GB; pruning HF weight cache...")
        for item in hf_hub.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                shutil.rmtree(item, ignore_errors=True)


def run_model_matrix_sharded(
    model_id: str,
    stage_prefix: str,
    strict_diag: bool,
    include_b512: bool = False,
) -> None:
    """Runs one model across 1 GPU or N_ACTIVE_SHARDS GPUs in parallel, then merges deterministically."""
    mode_lbl = "strict341" if strict_diag else "diag491"
    loop_name = f"{stage_prefix}: {model_id} ({mode_lbl})"
    base_cmd = [
        sys.executable,
        "scripts/run_matrix.py",
        "--model",
        model_id,
        "--data-dir",
        "data",
        "--out-dir",
        str(OUT_DIR),
        "--dtype",
        DTYPE,
        *MATRIX_PARAM_ARGS,
    ]
    if strict_diag:
        base_cmd.append("--strict-diag")
    if include_b512:
        base_cmd.append("--include-b512")

    if N_ACTIVE_SHARDS <= 1:
        cmd = [*base_cmd, *matrix_step_flags()]
        print("Executing:", " ".join(cmd), flush=True)
        run_logged_loop(loop_name, run_and_tee, cmd)
        return

    print(
        f"=== [{N_ACTIVE_SHARDS}-GPU Intra-Model Sharded] {loop_name} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    slug = model_id.replace("/", "__")

    def _run_shard(shard_idx: int) -> None:
        w_log = LOG_DIR / f"cbarr_{RUN_ID}_gpu{shard_idx}_{slug}_{mode_lbl}.log"
        s_cmd = [
            *base_cmd,
            "--shard-idx",
            str(shard_idx),
            "--num-shards",
            str(N_ACTIVE_SHARDS),
            "--step-checkpoint-every",
            "0",
        ]
        run_and_tee(
            s_cmd,
            extra_env={"CUDA_VISIBLE_DEVICES": str(shard_idx)},
            prefix=f"GPU{shard_idx}:{model_id.split('/')[-1]}",
            worker_log_path=w_log,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_ACTIVE_SHARDS) as pool:
            futs = {pool.submit(_run_shard, s_i): s_i for s_i in range(N_ACTIVE_SHARDS)}
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

        merge_cmd = [*base_cmd, "--merge-shards", *matrix_step_flags()]
        print("Merging shards:", " ".join(merge_cmd), flush=True)
        run_and_tee(merge_cmd)
    except Exception as exc:
        record_loop_checkpoint(
            loop_name,
            status=f"FAILED (at {loop_name})",
            stage_dur_sec=time.perf_counter() - t0,
            stage_ok=False,
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise

    record_loop_checkpoint(
        loop_name,
        status=f"RUNNING (after {loop_name})",
        stage_dur_sec=time.perf_counter() - t0,
        stage_ok=True,
    )


def run_models_batch(
    models: list[str],
    stage_prefix: str,
    strict_diag: bool,
    include_b512: bool = False,
    sync_each_model: bool = True,
) -> None:
    """Runs each model across all N_ACTIVE_SHARDS GPUs sequentially with immediate per-model GitHub sync."""
    if not models:
        return
    mode_lbl = "strict341" if strict_diag else "diag491"
    for idx, model_id in enumerate(models):
        if idx > 0:
            _prune_hf_hub_if_low_disk()
        run_model_matrix_sharded(
            model_id,
            stage_prefix=stage_prefix,
            strict_diag=strict_diag,
            include_b512=include_b512,
        )
        if sync_each_model:
            short_slug = model_id.split("/")[-1].lower()
            sync_results_to_github(
                label=f"{short_slug}-{mode_lbl}",
                models=[model_id],
                include_phase7=False,
            )


if RUN_PRIMARY_MATRIX and MODELS_PRIMARY:
    print(f"\\n=== [Phase 4/5 Stage A] Primary Models strict341: {MODELS_PRIMARY} ===")
    run_models_batch(MODELS_PRIMARY, "Phase 4/5", strict_diag=True, include_b512=INCLUDE_B512)

    print(f"\\n=== [Phase 4/5 Stage B] Primary Models diag491: {MODELS_PRIMARY} ===")
    run_models_batch(MODELS_PRIMARY, "Phase 4/5", strict_diag=False, include_b512=INCLUDE_B512)'''
    return _code_cell("cbarr-primary-matrix", code)


def _build_scale_w7_cell() -> dict:
    code = '''if RUN_SCALE_SWEEP and MODELS_SCALE:
    print(f"\\n=== [Phase 6] Scale Sweep (strict341): {MODELS_SCALE} ===")
    run_models_batch(MODELS_SCALE, "Phase 6", strict_diag=True, include_b512=INCLUDE_B512)

if RUN_W7_EXTENSIONS:
    w7_model = MODELS_PRIMARY[0] if MODELS_PRIMARY else (ALL_ACTIVE_MODELS[0] if ALL_ACTIVE_MODELS else "Qwen/Qwen3.5-4B")
    print(f"\\n=== [Optional W7] Running DeltaNet Blend & Prefill MLP on {w7_model} ===")
    run_logged_loop(
        f"W7: DeltaNet Blend ({w7_model})",
        run_and_tee,
        [sys.executable, "scripts/ext_deltanet_blend.py", "--model", w7_model, "--out-dir", str(OUT_DIR)],
    )
    run_logged_loop(
        f"W7: Prefill MLP Probe ({w7_model})",
        run_and_tee,
        [sys.executable, "scripts/ext_prefill_mlp.py", "--model", w7_model, "--out-dir", str(OUT_DIR)],
    )
    sync_results_to_github(label="w7-extensions", models=[w7_model], include_phase7=False)'''
    return _code_cell("cbarr-scale-w7", code)


def _build_analyze_export_cell() -> dict:
    code = '''import zipfile

if EXPECTED_MODELS_FOR_PHASE7:
    missing_models = [
        m for m in EXPECTED_MODELS_FOR_PHASE7
        if not (OUT_DIR / m.replace("/", "__") / "eval_450_cache.jsonl").exists()
    ]
    if missing_models:
        print(
            f"[WARN] Missing expected worker model caches in OUT_DIR: {missing_models}. "
            "Ensure worker notebooks finished pushing to GitHub before running combined Phase 7."
        )
    else:
        print(f"[OK] All expected models verified in {OUT_DIR}:")
        for d in sorted(p.name for p in OUT_DIR.iterdir() if p.is_dir() and (p / "eval_450_cache.jsonl").exists()):
            print(f"  - {d}")

if RUN_PHASE7_ANALYSIS:
    print("=== [Phase 7] Statistical Inference, LaTeX Tables & Vector Figures ===")
    t_p7 = time.perf_counter()
    try:
        run_and_tee(
            [
                sys.executable,
                "scripts/analyze_and_plot.py",
                "--out-dir",
                str(OUT_DIR),
                "--fig-dir",
                str(FIG_DIR),
            ]
        )
        print("=== [Audit] Double-Blind Anonymity Check ===")
        run_and_tee(["bash", "scripts/check_anonymity.sh"])
    except Exception as exc:
        record_loop_checkpoint(
            "Phase 7: Analysis & Gate-4 Audit",
            status="FAILED (at Phase 7)",
            stage_dur_sec=time.perf_counter() - t_p7,
            stage_ok=False,
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise

    record_loop_checkpoint(
        "Phase 7: Analysis & Gate-4 Audit",
        status="PASSED",
        stage_dur_sec=time.perf_counter() - t_p7,
        stage_ok=True,
        push=False,
    )

    stats_path = OUT_DIR / "stats_summary.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for m_name, m_data in stats.get("models", {}).items():
        g4 = m_data["gate_audit"]["gate4_primary"]
        boot = m_data.get("paired_bootstrap_95ci", {}).get("M2_minus_B1b", {})
        oaa_d = boot.get("oaa_250", {})
        print(
            f"[{m_name}] M2>=B1b Macro={g4['m2_macro_ge_b1b']} | "
            f"TokRed={g4['token_reduction_pct']:.1f}% | "
            f"p_holm={g4['m2_p_holm_vs_b1b']:.4g} | "
            f"Delta OAA_250={100*oaa_d.get('diff_b_minus_a', 0):.1f} pp "
            f"[{100*oaa_d.get('ci95_lo', 0):.1f}, {100*oaa_d.get('ci95_hi', 0):.1f}]"
        )
    sync_results_to_github(label="final-phase7", include_phase7=True)
else:
    record_loop_checkpoint(
        f"{SYNC_LABEL_PREFIX}: Worker Complete",
        status=f"PASSED ({PIPELINE_MODE.upper()})",
        stage_dur_sec=0.0,
        stage_ok=True,
        push=False,
    )
    sync_results_to_github(
        label="complete",
        models=ALL_ACTIVE_MODELS if ALL_ACTIVE_MODELS else None,
        include_phase7=False,
    )

# Package compact downloadable bundle (tables, stats JSONs, per-model summaries, *.jsonl.gz, ledger, figures)
bundle_zip = WORKSPACE_ROOT / BUNDLE_FILENAME
active_slugs = {m.replace("/", "__") for m in ALL_ACTIVE_MODELS} if (not SYNC_INCLUDE_LEDGER and ALL_ACTIVE_MODELS) else None
with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    if LEDGER_MD_PATH.exists():
        zf.write(LEDGER_MD_PATH, arcname=EXP_LOG_REL_PATH if SYNC_INCLUDE_LEDGER else LEDGER_MD_PATH.name)
    for log_artifact in (RUN_LOG_FILE, TIMINGS_FILE, PARAMS_FILE):
        if log_artifact.exists():
            zf.write(log_artifact, arcname=f"outputs/logs/{log_artifact.name}")
    if RUN_PHASE7_ANALYSIS:
        for top_file in ("stats_summary.json", "paper_tables.tex"):
            p = OUT_DIR / top_file
            if p.exists():
                zf.write(p, arcname=f"outputs/{top_file}")
    for sub in sorted(OUT_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if active_slugs is not None and sub.name not in active_slugs:
            continue
        for fname in (
            "calibration_summary.json",
            "matrix_summary.json",
            "matrix_summary_strict341.json",
            "matrix_summary_diag491.json",
            "tables_rows.tex",
            "tables_rows_strict341.tex",
            "tables_rows_diag491.tex",
            "ext_deltanet_blend.json",
            "prefill_mlp_summary.json",
        ):
            fp = sub / fname
            if fp.exists():
                zf.write(fp, arcname=f"outputs/{sub.name}/{fname}")
    repo_out = REPO_DIR / "outputs"
    if repo_out.is_dir():
        for gz_fp in sorted(repo_out.glob("*/*_cache.jsonl.gz")):
            if active_slugs is not None and gz_fp.parent.name not in active_slugs:
                continue
            zf.write(gz_fp, arcname=f"outputs/{gz_fp.parent.name}/{gz_fp.name}")
    if RUN_PHASE7_ANALYSIS and FIG_DIR.is_dir():
        for pdf in sorted(FIG_DIR.glob("*.pdf")):
            zf.write(pdf, arcname=f"paper/figures/{pdf.name}")

print(f"Packaged results bundle: {bundle_zip} ({bundle_zip.stat().st_size / 1024:.1f} KB)")
if IS_COLAB:
    try:
        from google.colab import files

        files.download(str(bundle_zip))
    except Exception as exc:
        print(f"[INFO] Download file manually from Colab Files pane: {bundle_zip} ({exc})")'''
    return _code_cell("cbarr-analyze-export", code)


def build_notebook(spec: NotebookSpec) -> dict:
    cells = [
        _build_intro_cell(spec),
        _build_env_detect_cell(),
        _build_config_cell(spec),
        _build_clone_install_cell(),
        _build_download_data_cell(),
        _build_gate0_smoke_cell(),
        _build_calibrate_cell(),
        _build_primary_matrix_cell(),
        _build_scale_w7_cell(),
        _build_analyze_export_cell(),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        nb = build_notebook(spec)
        out_path = NOTEBOOKS_DIR / spec.filename
        out_path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] Wrote {out_path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
