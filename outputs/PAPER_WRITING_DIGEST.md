# C-BARR Final Paper-Writing Statistical & Mechanistic Digest

- **Analyzer Code Version**: `v0.5.27` (`8165453`)
- **Evaluated Backbones (5)**: `HuggingFaceTB/SmolLM3-3B`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`
- **Bootstrap Resamples**: `n_boot=10000` (`seed=42`)

## 1. Cross-Backbone Headline Accuracy, Efficiency & Pathology (`N=450`)

| Model | B0 Macro (Tok) | B1b Macro (Tok, Loop%, Exh%) | M2 Macro (Tok, Loop%, Spill%) | M2_Prec Macro (IF-Prec) | M2_B512 Macro | Oracle_UB Macro | Tok Sav.% | Delta Macro vs B1b [95% CI] | Clean-446 Max Delta | Gate 4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-0.8B` | 52.8% (346t) | 2.7% (1997t, 96.4%, 96.0%) | **53.6%** (436t, 16.7%, 0.00%) | 56.1% (71.7%) | -- | 73.4% | 78.2% | +50.9 [+45.8, +56.1] pp | 0.24 pp | PASS |
| `Qwen/Qwen3.5-2B` | 67.7% (333t) | 13.7% (1879t, 82.4%, 80.2%) | **69.6%** (382t, 7.6%, 0.22%) | 69.6% (75.0%) | -- | 87.2% | 79.7% | +55.9 [+50.6, +61.2] pp | 0.08 pp | PASS |
| `Qwen/Qwen3.5-4B` | 84.1% (312t) | 30.2% (1735t, 58.4%, 63.8%) | **84.6%** (360t, 2.7%, 0.22%) | 85.4% (91.7%) | 86.1% | 94.2% | 79.2% | +54.3 [+49.2, +59.3] pp | 0.36 pp | PASS |
| `Qwen/Qwen3.5-9B` | 87.5% (310t) | 38.3% (1668t, 51.3%, 53.1%) | **86.0%** (380t, 2.4%, 0.22%) | 86.3% (88.3%) | -- | 93.9% | 77.2% | +47.7 [+42.4, +52.9] pp | 0.40 pp | PASS |
| `HuggingFaceTB/SmolLM3-3B` | 67.2% (249t) | 69.4% (977t, 32.0%, 11.3%) | **68.5%** (371t, 9.6%, 0.22%) | 69.9% (83.3%) | 67.9% | 88.2% | 62.0% | -0.9 [-5.7, +3.7] pp | 0.48 pp | PARITY/CHECK |

## 2. Pre-Registered 4-Pair Confirmatory Holm-Bonferroni (`M2` vs. `B1b, B0, B2a, M1c`)

| Model | Comparison | Wins (`M2` / Base) | Delta Macro [95% CI] | Delta U_0.05 [95% CI] | Tok Sav.% | `p_exact` | `p_holm_4` (exact / mid-p) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-0.8B` | `M2_vs_B1b` | 237 / 2 | +50.9 [+45.8, +56.1] pp | +83.4 [+78.5, +88.4] pp | +78.2% | 6.49e-68 | 2.6e-67 / 1.31e-67 |
| `Qwen/Qwen3.5-0.8B` | `M2_vs_B0` | 23 / 23 | +0.8 [-3.8, +5.4] pp | -1.8 [-4.9, +1.3] pp | -26.1% | 1 | 1 / 1 |
| `Qwen/Qwen3.5-0.8B` | `M2_vs_B2a` | 112 / 51 | +15.2 [+9.4, +21.0] pp | +10.8 [+5.4, +16.2] pp | -46.9% | 2.01e-06 | 4.01e-06 / 2.89e-06 |
| `Qwen/Qwen3.5-0.8B` | `M2_vs_M1c` | 73 / 0 | +21.7 [+17.6, +26.0] pp | +24.8 [+20.6, +29.3] pp | +49.7% | 2.12e-22 | 6.35e-22 / 3.18e-22 |
| `Qwen/Qwen3.5-2B` | `M2_vs_B1b` | 242 / 5 | +55.9 [+50.6, +61.2] pp | +82.6 [+77.4, +87.9] pp | +79.7% | 6.64e-65 | 2.66e-64 / 1.36e-64 |
| `Qwen/Qwen3.5-2B` | `M2_vs_B0` | 12 / 7 | +1.9 [-1.1, +5.0] pp | +0.1 [-1.8, +2.0] pp | -14.8% | 0.359 | 0.359 / 0.263 |
| `Qwen/Qwen3.5-2B` | `M2_vs_B2a` | 65 / 41 | +8.6 [+3.6, +13.6] pp | +5.3 [+0.8, +9.9] pp | +0.1% | 0.025 | 0.05 / 0.0398 |
| `Qwen/Qwen3.5-2B` | `M2_vs_M1c` | 58 / 0 | +19.8 [+16.1, +23.6] pp | +18.3 [+14.5, +22.4] pp | +41.7% | 6.94e-18 | 2.08e-17 / 1.04e-17 |
| `Qwen/Qwen3.5-4B` | `M2_vs_B1b` | 226 / 3 | +54.3 [+49.2, +59.3] pp | +77.0 [+71.6, +82.4] pp | +79.2% | 4.64e-63 | 1.86e-62 / 9.4e-63 |
| `Qwen/Qwen3.5-4B` | `M2_vs_B0` | 6 / 5 | +0.4 [-2.2, +3.1] pp | -0.7 [-2.2, +0.7] pp | -15.5% | 1 | 1 / 1 |
| `Qwen/Qwen3.5-4B` | `M2_vs_B2a` | 32 / 28 | +2.2 [-2.3, +6.6] pp | -0.1 [-3.6, +3.3] pp | -16.2% | 0.699 | 1 / 1 |
| `Qwen/Qwen3.5-4B` | `M2_vs_M1c` | 66 / 0 | +22.4 [+18.6, +26.1] pp | +21.1 [+17.0, +25.3] pp | +47.0% | 2.71e-20 | 8.13e-20 / 4.07e-20 |
| `Qwen/Qwen3.5-9B` | `M2_vs_B1b` | 194 / 5 | +47.7 [+42.4, +52.9] pp | +67.8 [+62.5, +73.2] pp | +77.2% | 6.32e-51 | 2.53e-50 / 1.3e-50 |
| `Qwen/Qwen3.5-9B` | `M2_vs_B0` | 16 / 20 | -1.5 [-4.5, +1.4] pp | -2.3 [-5.0, +0.4] pp | -22.5% | 0.618 | 0.618 / 0.511 |
| `Qwen/Qwen3.5-9B` | `M2_vs_B2a` | 52 / 24 | +7.6 [+3.1, +12.1] pp | +3.8 [-0.1, +7.5] pp | -48.1% | 0.00176 | 0.00352 / 0.00253 |
| `Qwen/Qwen3.5-9B` | `M2_vs_M1c` | 106 / 3 | +27.2 [+23.0, +31.4] pp | +36.5 [+31.7, +41.5] pp | +64.3% | 6.65e-28 | 2e-27 / 1.03e-27 |
| `HuggingFaceTB/SmolLM3-3B` | `M2_vs_B1b` | 40 / 47 | -0.9 [-5.7, +3.7] pp | +10.6 [+6.1, +15.0] pp | +62.0% | 0.52 | 0.961 / 0.817 |
| `HuggingFaceTB/SmolLM3-3B` | `M2_vs_B0` | 30 / 19 | +1.3 [-2.7, +5.2] pp | -0.0 [-3.2, +3.0] pp | -49.2% | 0.152 | 0.61 / 0.476 |
| `HuggingFaceTB/SmolLM3-3B` | `M2_vs_B2a` | 46 / 36 | +2.9 [-2.0, +7.7] pp | +0.1 [-3.9, +4.0] pp | -39.5% | 0.32 | 0.961 / 0.817 |
| `HuggingFaceTB/SmolLM3-3B` | `M2_vs_M1c` | 24 / 17 | +1.9 [-1.6, +5.5] pp | +6.7 [+3.6, +9.8] pp | +40.7% | 0.349 | 0.961 / 0.817 |

## 3. Proposition 1 Conformal Degradation Risk & Hindsight `Oracle_UB` Retention

| Model | Dev Deg. vs B1b | CRC Bound | Hoeffding UCB (`M=29`) | Eval Deg. vs B1b (`k/450`) | Eval Deg. vs B0 (`k/450`) | Within Bounds | Oracle Macro Ret.% | Tok Ratio vs Oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-0.8B` | 0.0% | 1.6% | 23.0% | 0.44% (2/450) | 5.11% (23/450) | CRC=True, Hoef=True | 0.0% | 0.00x |
| `Qwen/Qwen3.5-2B` | 0.0% | 1.6% | 23.0% | 1.11% (5/450) | 1.56% (7/450) | CRC=True, Hoef=True | 0.0% | 0.00x |
| `Qwen/Qwen3.5-4B` | 1.7% | 3.3% | 24.7% | 0.67% (3/450) | 1.11% (5/450) | CRC=True, Hoef=True | 0.0% | 0.00x |
| `Qwen/Qwen3.5-9B` | 1.7% | 3.3% | 24.7% | 1.11% (5/450) | 4.44% (20/450) | CRC=True, Hoef=True | 0.0% | 0.00x |
| `HuggingFaceTB/SmolLM3-3B` | 11.7% | 13.1% | 34.7% | 10.44% (47/450) | 4.22% (19/450) | CRC=True, Hoef=True | 0.0% | 0.00x |

## 4. Section 6 Counterfactual Controls & Pooled Exact `k/n` Audit

### Strict Disjoint Pool (`diag_341_strict`)
| Stratum | Models | `A1_strip` (`k/n`) | `B2a_post` (`k/n`, 95% CI) | `B1_pre` (`k/n`) | `B2b_raw` (`k/n`) | `A2_swap` (`k/n`) | `B3_cut` (`k/n`) | `p_midp(B2a vs B1b)` | `p_midp(B2a vs A1)` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `deltanet_models` | 4 | 100.0% (854/854) | **90.3%** (771/854, [88.1, 92.1]) | 88.8% (758/854) | 47.9% (409/854) | 64.9% (554/854) | 83.0% (709/854) | 8.05e-233 | 1.03e-25 |
| `dense_models` | 1 | 100.0% (58/58) | **63.8%** (37/58, [50.9, 74.9]) | 62.1% (36/58) | 34.5% (20/58) | 55.2% (32/58) | 60.3% (35/58) | 7.28e-12 | 4.77e-07 |
| `all_models` | 5 | 100.0% (912/912) | **88.6%** (808/912, [86.4, 90.5]) | 87.1% (794/912) | 47.0% (429/912) | 64.3% (586/912) | 81.6% (744/912) | 5.86e-244 | 4.93e-32 |

### Full Calibration-Disjoint Pool (`diag_491`)
| Stratum | Models | `A1_strip` (`k/n`) | `B2a_post` (`k/n`, 95% CI) | `B1_pre` (`k/n`) | `B2b_raw` (`k/n`) | `A2_swap` (`k/n`) | `B3_cut` (`k/n`) | `p_midp(B2a vs B1b)` | `p_midp(B2a vs A1)` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `deltanet_models` | 1 | 100.0% (336/336) | **96.1%** (323/336, [93.5, 97.7]) | 95.5% (321/336) | 49.4% (166/336) | 70.2% (236/336) | 86.3% (290/336) | 5.85e-98 | -- |
| `dense_models` | 1 | 100.0% (80/80) | **62.5%** (50/80, [51.5, 72.3]) | 58.8% (47/80) | 33.8% (27/80) | 60.0% (48/80) | 62.5% (50/80) | 8.88e-16 | 9.31e-10 |
| `all_models` | 2 | 100.0% (416/416) | **89.7%** (373/416, [86.4, 92.2]) | 88.5% (368/416) | 46.4% (193/416) | 68.3% (284/416) | 81.7% (340/416) | 5.2e-113 | -- |

## 5. Gate-2 Probe Horizon Ablation (`K in {1, 4, 8, 16, 32}`)

| Model | `K=1` Macro / U_0.05 (Tok) | `K=4` Macro / U_0.05 (Tok) | `K=8` Macro / U_0.05 (Tok) | `K=16` Macro / U_0.05 (Tok) | `K=32` Macro / U_0.05 (Tok) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-0.8B` | 55.4% / 49.4% (429t) | 54.1% / 47.4% (439t) | 54.9% / 48.4% (435t) | 55.4% / 49.4% (428t) | 53.6% / 47.5% (436t) |
| `Qwen/Qwen3.5-2B` | 69.8% / 63.4% (410t) | 69.6% / 63.7% (381t) | 69.6% / 63.7% (381t) | 70.3% / 63.9% (403t) | 69.6% / 63.7% (382t) |
| `Qwen/Qwen3.5-4B` | 84.7% / 78.0% (356t) | 84.4% / 77.5% (369t) | 84.7% / 78.0% (356t) | 84.5% / 78.0% (358t) | 84.6% / 77.7% (360t) |
| `Qwen/Qwen3.5-9B` | 85.2% / 77.5% (382t) | 85.8% / 78.1% (385t) | 85.3% / 78.0% (379t) | 85.7% / 77.9% (382t) | 86.0% / 78.4% (380t) |
| `HuggingFaceTB/SmolLM3-3B` | 68.4% / 64.9% (366t) | 68.3% / 64.4% (358t) | 67.2% / 63.5% (335t) | 67.2% / 63.4% (331t) | 68.5% / 64.6% (371t) |

## 6. Workstream W7 Mechanistic Extensions & Paraphrase Probe

### `HuggingFaceTB/SmolLM3-3B`
- **Dense Softmax `<think>` KV Cache Eviction (`alpha=0`)** (`n=80`): `alpha=0.00`: **86.2%** (Inhib=94.9%, Struct=86.0%)
- **Prefill MLP Probe (`layer_-1`, `tau=0.850`)**: Eval Macro=66.8%, OAA_250=69.6%, U_0.05=63.4%, Tok=308 | Canonical Prec F1: Regex=86.0% vs MLP=81.1% | `IFEval++` Paraphrase Prec F1 (Rec): Regex=78.9% (69.0%) vs **MLP=81.7% (87.4%)**
- **Stage-2 Completion Bridge Replay (`replayed=139`)**: Classic Macro=68.5% (OAA_250=72.0%) vs SolveFirst Macro=67.0% (OAA_250=69.6%)

### `Qwen/Qwen3.5-4B`
- **DeltaNet Recurrent State Blend (`conv=False, evict_kv=False`)** (`n=336`): `alpha=0.00`: **93.2%** (Inhib=95.5%, Struct=94.8%); `alpha=0.25`: **95.2%** (Inhib=97.8%, Struct=97.4%); `alpha=0.50`: **95.2%** (Inhib=98.5%, Struct=96.1%); `alpha=0.75`: **94.0%** (Inhib=97.8%, Struct=94.8%); `alpha=1.00`: **91.7%** (Inhib=94.8%, Struct=93.5%)
- **DeltaNet Joint Recurrent + 1D Conv + `<think>` KV Eviction** (`n=336`): `alpha=0.00`: **85.7%** (Inhib=96.3%, Struct=90.3%); `alpha=0.25`: **88.1%** (Inhib=96.3%, Struct=92.3%); `alpha=0.50`: **89.9%** (Inhib=95.5%, Struct=93.5%); `alpha=0.75`: **91.7%** (Inhib=95.5%, Struct=93.5%); `alpha=1.00`: **95.8%** (Inhib=98.5%, Struct=98.7%)
- **Prefill MLP Probe (`layer_-1`, `tau=0.525`)**: Eval Macro=84.1%, OAA_250=84.9%, U_0.05=77.8%, Tok=357 | Canonical Prec F1: Regex=86.0% vs MLP=81.2% | `IFEval++` Paraphrase Prec F1 (Rec): Regex=78.9% (69.0%) vs **MLP=82.6% (81.6%)**
- **Stage-2 Completion Bridge Replay (`replayed=17`)**: Classic Macro=84.6% (OAA_250=84.9%) vs SolveFirst Macro=84.3% (OAA_250=84.4%)


## 7. Appendix B: OOD `IFBench_test`, 3-Seed `R6`, and Cross-Model L2 Sensitivity

- **IFBench_test OOD (`Qwen/Qwen3.5-4B`, `N=300`)**: B0=28.7% (386t) | B1b=2.0% (2034t) | **M2=31.3%** (OAA_250=31.3%, 443t, `p_exact=1.73e-24`)
- **3-Seed Sampled R6**: *Pending execution (`--part r6`).*

### Gate-2 L2 Regularization Sensitivity (`C in {0.1, 0.25 (prod), 0.5, 1.0}`)
| Model | `C=0.10` Macro (Esc, Tok) | `C=0.25 (prod)` Macro (Esc, Tok) | `C=0.50` Macro (Esc, Tok) | `C=1.00` Macro (Esc, Tok) |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-0.8B` | 54.3% (16.7%, 393t) | 53.6% (31.7%, 436t) | 53.6% (25.0%, 419t) | 53.6% (28.3%, 426t) |
| `Qwen/Qwen3.5-2B` | 70.5% (28.3%, 398t) | 69.6% (16.7%, 382t) | 70.9% (31.7%, 393t) | 71.3% (43.3%, 395t) |
| `Qwen/Qwen3.5-4B` | 84.8% (20.0%, 359t) | 84.6% (20.0%, 360t) | 84.6% (20.0%, 358t) | 84.6% (20.0%, 359t) |
| `Qwen/Qwen3.5-9B` | 85.9% (50.0%, 382t) | 86.0% (43.3%, 380t) | 85.9% (41.7%, 379t) | 86.0% (33.3%, 366t) |
| `HuggingFaceTB/SmolLM3-3B` | 68.5% (50.0%, 381t) | 68.5% (46.7%, 371t) | 68.7% (46.7%, 377t) | 68.5% (48.3%, 383t) |

## 8. End-to-End Wall-Clock Latency, Compute Footprint & Reproducibility Checklist

| Model | Dtype | Seed | `B0` Mean / P95 | `B1b` Mean / P95 (`Sum min`) | `M2` Mean / P95 (`Sum min`) | Speedup (`B1b / M2`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3.5-0.8B` | `torch.bfloat16` | `42` | 9.41s / 27.59s | 54.17s / 57.52s (406.3m) | 11.46s / 33.80s (85.9m) | **4.73x** |
| `Qwen/Qwen3.5-2B` | `torch.bfloat16` | `42` | 9.90s / 30.26s | 55.82s / 62.94s (418.6m) | 10.99s / 30.94s (82.4m) | **5.08x** |
| `Qwen/Qwen3.5-4B` | `torch.bfloat16` | `42` | 11.47s / 29.68s | 63.62s / 77.06s (477.2m) | 12.75s / 35.97s (95.6m) | **4.99x** |
| `Qwen/Qwen3.5-9B` | `torch.bfloat16` | `42` | 12.16s / 31.09s | 65.31s / 86.08s (489.9m) | 14.22s / 35.46s (106.6m) | **4.59x** |
| `HuggingFaceTB/SmolLM3-3B` | `torch.bfloat16` | `42` | 6.65s / 20.30s | 26.04s / 55.16s (195.3m) | 9.50s / 28.05s (71.2m) | **2.74x** |
