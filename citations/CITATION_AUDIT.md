# C-BARR Scholarly Citation & Claim Audit

This report audits all scholarly papers downloaded in `citations/` (`48/49` PDFs, `78.57 MB` total; indexed in [`MANIFEST.json`](./MANIFEST.json)) against:
1. [`paper/references.bib`](../paper/references.bib) and [`paper/main.tex`](../paper/main.tex),
2. [`README.md`](../README.md), and
3. [`docs/NLP Research_ C-BARR .docx`](../docs/NLP%20Research_%20C-BARR%20.docx) (prose, tables, and inline superscript mappings into Tab 8 `Works Cited`).

---

## 1. Executive Summary

- **PDF Corpus Downloaded (`citations/`)**: `48` of `49` targeted scholarly PDFs downloaded (`82,391,712` bytes / `78.57 MB`) and verified intact via full-text `PDFKit` extraction (`48/48` readable, `5–47` pages each). One entry (`43_openreview_query_transfer_Pc7eA7Zb33`, ICLR submission OpenReview `Pc7eA7Zb33`) returned `HTTP 403 Forbidden` across OpenReview PDF/attachment endpoints and was left undownloaded per anti-bot policy.
- **`paper/references.bib` (23 entries)**:
  - **Authors**: `23/23` (`100%`) match the official arXiv metadata and PDF page-1 headers in exact author count and spelling.
  - **Camera-Ready Title & Venue Fixes Applied (8 entries updated)**:
    - Updated `zeng2025pac` title to its NeurIPS 2026 camera-ready title (*PAC Reasoning: Controlling the Performance Loss for Efficient Reasoning*).
    - Upgraded 7 entries from `@article` (`arXiv preprint`) to their accepted conference/journal venues and publication years: `yang2024gateddeltanet` (**ICLR 2025**), `yan2026entrocut` (**ICASSP 2026**), `wang2026conformal` (**ICML 2026**), `zeng2026glimprouter` (**Findings of ACL 2026**), `kwon2025reasonif` (**Findings of ACL 2026**, pp. 29149–29164, year `2026`), `zhao2026bridging` (**TACL 2026**, Vol. 14, pp. 1329–1347, `2026.tacl-1.60`), and `dong2025ifevalpp` (**ACL 2026**, `2026.acl-long.354`, year `2026`).
- **`paper/main.tex` (3 precision fixes applied)**:
  - Separated `ding2026thinkjev` (typed multi-budget decision calibration) from `yan2026entrocut` (uncalibrated open-ended intra-trace entropy truncation) in Section 2 (`\ref` wording in Related Work).
  - Adjusted Proposition 1 (`wang2026conformal`) phrasing so the risk-controlled early-stopping framing cites `Conformal Thinking` accurately while stating the standard split conformal expectation bound.
  - Added missing `\cite{aggarwal2025optimalthinkingbench}` attribution specifically for `OAA_250` and normalized `AUC_OAA` in Section 4 alongside `\cite{nguyen2026early}`.
- **`README.md`**: Audited; zero inaccurate external citations or misattributed quantitative claims.
- **`docs/NLP Research_ C-BARR .docx`**:
  - Core empirical claims (Kumar 2026 Table 2.2 + activation patching `32%–58%` vs. `14%–40%`, Li et al. 2025 `Qwen2.5-7B-Instruct` router, EntroCut Table 1, DEER Table 1, DART `32%–73%`, Thinking Inertia `~15 pp`, s1 `50.0% -> 56.7%`, TALE `~67%`) **match the primary PDFs**.
  - Identified **10 inline superscript mis-mappings** in Tabs 1–7 (`^75`, `^64/65`, `^4`, `^38`, `^52`, `^27`, `^84`, `^19`, `^95/96`, plus internal self-citations `^1`/`^16`), **3 numerical/terminological nuances** (`B-PAC` 81.01% routing call reduction vs. 41.37% token savings; `DEER` vs. `EntroCut` probe strings and compression ranges; `OAA_t` strict `< t` vs. `<= t`), and **3 noisy pasted appendices** (restarted superscript sequences `1..51` / `1..54` and a secondary `Works cited` block containing personal webpage links and unrelated scrape URLs that must be removed before sharing).

---

## 2. Updates Applied to `paper/references.bib` and `paper/main.tex`

### 2.1 `paper/references.bib` Corrections

| Key | Previous Value | Verified & Updated Value | Source Evidence |
| :--- | :--- | :--- | :--- |
| `yang2024gateddeltanet` | `@article`, `arXiv preprint arXiv:2412.06464`, `year={2024}` | `@inproceedings`, `International Conference on Learning Representations (ICLR)`, `year={2025}` | `02_yang2024_gateddeltanet.pdf` p. 1 (*Published as a conference paper at ICLR 2025*) |
| `zeng2025pac` | `title={On the Provable Performance Guarantee of Efficient Reasoning Models}` | `title={{PAC} Reasoning: Controlling the Performance Loss for Efficient Reasoning}` | `09_zeng2025_pac_reasoning.pdf` p. 1 & arXiv v3 (*Accepted at NeurIPS 2026 poster*) |
| `yan2026entrocut` | `@article`, `arXiv preprint arXiv:2601.22617` | `@inproceedings`, `Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)` | arXiv `2601.22617` comment (*Accepted by ICASSP26*) |
| `wang2026conformal` | `@article`, `arXiv preprint arXiv:2602.03814` | `@inproceedings`, `Proceedings of the International Conference on Machine Learning (ICML)` | arXiv `2602.03814v2` comment (*ICML 2026*) |
| `zeng2026glimprouter` | `@article`, `arXiv preprint arXiv:2601.05110` | `@inproceedings`, `Findings of the Association for Computational Linguistics: ACL 2026` | arXiv `2601.05110v3` comment (*Accepted to ACL 2026 Findings*) |
| `kwon2025reasonif` | `@article`, `arXiv preprint arXiv:2510.15211`, `year={2025}` | `@inproceedings`, `Findings of the Association for Computational Linguistics: ACL 2026`, `pages={29149--29164}`, `year={2026}` | `10_kwon2025_reasonif.pdf` p. 1 venue footer |
| `zhao2026bridging` | `arXiv preprint arXiv:2606.03624` | `Transactions of the Association for Computational Linguistics (TACL)`, `vol={14}`, `pp. 1329--1347`, `doi={10.1162/tacl.a.730}`, `2026.tacl-1.60` | ACL Anthology `2026.tacl-1.60` |
| `dong2025ifevalpp` | `@article`, `arXiv preprint arXiv:2512.14754`, `year={2025}` | `@inproceedings`, `Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)`, `2026.acl-long.354`, `year={2026}` | arXiv `2512.14754v3` (*ACL 2026 main oral*) & ACL Anthology |

### 2.2 Remaining `.bib` Entries Verified Unchanged

All remaining 15 entries in `paper/references.bib` (`muennighoff2025s1` EMNLP 2025, `ma2025nothinking` arXiv, `li2025pitfalls` NeurIPS 2025, `kumar2026constraint` arXiv, `aggarwal2025optimalthinkingbench` arXiv, `zhou2023ifeval` arXiv, `lei2026thinkinginertia` NeurIPS 2026, `lee2026dart` Findings EMNLP 2026, `zhu2026savethinking` Findings EMNLP 2026 `2026.findings-emnlp.761`, `yu2026bpac` ICML 2026, `leviathan2025promptrep` arXiv, `venkateswaran2026spotlight` EACL 2026, `ding2026thinkjev` arXiv, `nguyen2026early` arXiv, `zhu2026leakage` arXiv) match their PDF headers, arXiv metadata, and confirmed venues.

### 2.3 `paper/main.tex` Corrections & Claim Checks

- **Section 2 (Related Work)**: Split `\cite{ding2026thinkjev,yan2026entrocut}` because `ThinkJev` calibrates closed-set *typed decisions* over a cached thought (`22_ding2026_thinkjev.pdf` Abstract), whereas `EntroCut` is an uncalibrated entropy-guided adaptive `<think>` truncation rule (`15_yan2026_entrocut.pdf` Section 2.3).
- **Section 2 (`SpotLight`)**: Verified *pre-softmax* characterization against `06_venkateswaran2026_spotlight.pdf` (EACL 2026, p. 3752, Section 2): SpotLight adds an additive bias directly to attention logits `L = QK^T` prior to softmax.
- **Section 3 (Proposition 1)**: Refined citation phrasing around `\frac{n \hat{R}_n(\tau) + 1}{n + 1}` so `wang2026conformal` is cited for risk-controlled early-stopping formulations while stating the standard split conformal expectation bound cleanly.
- **Section 4 (Metrics)**: Added `\cite{aggarwal2025optimalthinkingbench}` directly on `OAA_250` and normalized `AUC_OAA`, reserving `\cite{nguyen2026early}` for efficiency-reliability and hindsight-oracle auditing.

---

## 3. Audit of `docs/NLP Research_ C-BARR .docx`

### 3.1 Quantitative & Methodological Claims Verified Against PDFs

| Claim in `.docx` | Primary Source PDF | Audit Verdict |
| :--- | :--- | :--- |
| `Kumar (2026)` Table 2.2 (9 models, Qwen3 1.7B–32B & Hunyuan 1.8B–A13B ON/OFF prompt & instr scores); `Hunyuan-7B` `63.40% OFF vs. 63.40% ON`, `23.5%` flip (`11.8%` Helped / `11.6%` Hurt); `5 Planning / 9 Precision / 11 Neutral` | `18_kumar2026_constraint_shifts.pdf` (Tables 1–2, §4.2) | **Exact match** across all 9 models and percentages. (`834` = IFEval instruction instances / Table 5 total pairs.) |
| `Kumar (2026)` prefill activation-patching restoration: `32%–58%` Precision vs. `14%–40%` Planning mean layer-restoration (gap ~30 pp at 14B) | `18_kumar2026_constraint_shifts.pdf` (p. 1 Abstract, §4.6, Fig. 5) | **Exact match** verbatim. |
| `Li et al. (2025)` trains `Qwen2.5-7B-Instruct` classifier (3 epochs, LR `1e-5`), evaluates `541` IFEval prompts & `1,150` ComplexBench prompts, measures Constraint Attention drops on `LOSE` instances | `05_li2025_pitfalls.pdf` (p. 9, Abstract, §4) | **Exact match**. |
| `EntroCut` (`DS-1.5B`: AIME24 `15,591 -> 8,295` / `29.2% vs. 28.8%`, MATH500 `4,984 -> 3,341` / `83.1% vs. 81.6%`; `DS-7B`: MATH500 `4,097 -> 3,046` / `92.1% vs. 91.1%`, AMC23 `6,434 -> 5,216` / `90.3% vs. 89.2%`; threshold `0.15–0.225`) | `15_yan2026_entrocut.pdf` (Table 1, §2.3, §3.1) | **Exact match**. Note: `18.9% -- 47.6%` math token savings in Table 1.1 matches EntroCut's DS-7B AMC23 `-18.9%` to DS-1.5B AIME25 `-47.6%` (`15,099 -> 7,912`). |
| `DEER` (`DS-7B` `DEER-Ent`: GSM8K `1,484 -> 876` / `89.6% -> 90.8%`, MATH `3,858 -> 2,261` / `87.4% -> 89.2%`, GPQA `10,247 -> 5,037` / `23.7% -> 29.6%`; threshold `0.95`) | `37_iclr2026_deer_or_8df90a14.pdf` (Table 1) | **Exact match** on all Table 1 `DEER-Ent` numbers. |
| `DART`: `K = 2` no-think drafts (`T=0.6` local, `T=0.7` API), `32%–73%` thinking-token reduction | `19_lee2026_dart.pdf` (Abstract, §3) | **Exact match**. |
| `Thinking Inertia`: ~`15 pp` mean accuracy drop under NL suppression; offline metrics `ETR`, `QRel.`, `EIR` | `20_lei2026_thinking_inertia.pdf` | **Exact match**. |
| `s1`: `50.0% -> 56.7%` on AIME24 (`1K` vs. w/o BF); `TALE-EP`: `~67%` token reduction | `03_muennighoff2025_s1.pdf` (Table 1); `24_han2025_tale.pdf` | **Exact match**. |
| `Prompt Repetition`: `5 wins, 1 loss, 22 ties` under step-by-step reasoning; `47/70` wins, `0` losses without reasoning | `12_leviathan2025_promptrep.pdf` (§2, Figs. 1 & 4) | **Exact match**. |
| `ThinkSwitcher` (`20%–30%` savings, EMNLP 2025 Findings); `Thinkless` (`50%–90%` long-thinking reduction); `SelfBudgeter` (`61%` avg math compression); `TRAAC` (COLM 2026) | `30_liang2025_thinkswitcher.pdf`, `28_fang2025_thinkless.pdf`, `27_chen2025_selfbudgeter.pdf`, `42_openreview_traac_ZSeKA0LrJg.pdf` | **Exact match**. |

### 3.2 Subtle Claim Nuances in `.docx`

1. **`B-PAC` (`14_yu2026_bpac.pdf`, p. 1)**: Table 1.1 summarizes *"up to 81% less thinking-model use"*. Precisely, `81.01%` (`100% - 18.99%`) is the reduction in *expert/thinking-model routing calls* (`ECP = 18.99%`) on Magpie at tolerance `\epsilon = 0.08`, whereas *token savings* reported for that same setting is `41.37%`.
2. **`DEER` vs. `EntroCut` Mechanics (`p[1027]`)**: `.docx` attributes the exact probe string `"</think>\n\nSo the final answer is"` to both methods. In the PDFs, `EntroCut` uses `"</think>/n/nSo the final answer is"`, whereas `DEER` (Fig. 3) switches using `"</think>"` paired with `"**Final Answer**\n\boxed{..."` indirection; DEER's overall CoT reduction across all evaluated models/benchmarks is `19.1%–80.1%` (vs. `18.9%–47.6%` in Table 1.1 from EntroCut).
3. **Minor Terminology / Formula Notes**:
   - `OptimalThinkingBench` (`08_aggarwal2025_optimalthinkingbench.pdf` §4) defines `OAA_t` using strict inequality `ThinkTokens_i < t`, whereas `src/metrics.py` uses `ti <= threshold_t` (identical except at exact equality `t = 250`).
   - `44_openreview_instaboost_xDyJVMnab8.pdf` capitalizes the method as `INSTABOOST` (written `InstaBoost` in `.docx`).

### 3.3 Broken Inline Superscripts & Structural Issues in `.docx`

OOXML inspection confirmed that Tab 8 `Works Cited` contains **103** numbered entries (`TAB8_NUM=1..103` at paragraphs `2128..2230`, so inline `[^97..^103]` align 1:1). However, several inline superscripts in Tabs 1–7 resolve to **wrong Tab 8 entries** (typically scraping/re-indexing artifacts):

| Inline Location in `.docx` | Superscript Used | Actual `Tab 8` Target | Intended Source |
| :--- | :--- | :--- | :--- |
| Table 2.2 (8 of 9 rows) & `p[1232]` | `[^75]` (x10) | `ruisv/bllm-qwen3.5-4b` (HF model card) | `Tab8[7]` / `Tab8[85]` (Kumar 2026) |
| Section 2 (`p[1027]`) DEER & EntroCut | `[^64]`, `[^65]` | `litert-community/Qwen3.5-0.8B`, `jaredpalmer/kev-0.8b` | `Tab8[9]` (DEER), `Tab8[8]` (EntroCut) |
| Table 1.1 (`p[984]`) ThinkJev row | `[^4]` | `Qwen/Qwen3.5-0.8B` (HF model card) | `Tab8[62]` / `Tab8[63]` (ThinkJev) |
| Table 1.1 (`p[943]`) TALE row | `[^38]` | `BAGEN: Are LLM Agents Budget-Aware?` (`2606.00198`) | `Tab8[39]` (TALE, `2412.18547`) |
| Table 1.1 (`p[956]`) RL length range | `[^52]` | `Neha Anna John - CatalyzeX` author page | `Tab8[44]` / `Tab8[53]` (Thinkless `50%–90%`) |
| Table 3.3 (`p[1353]`) PyTorch fallback | `[^27]` | `Do LLMs Trust the Accuser or the Accusation?` (`2026.emnlp-main.1628`) | `Tab8[22]` (`transformers` linear-attention fallback issue) |
| Section 2 (`p[1234]`) DeltaNet decay | `[^84]` | `IFEval++` (`2512.14754`) | `Tab8[68]` / `Tab8[82]` (Gated Delta Networks) |
| Table 3.2 (`p[1339]`, `p[1349]`) IFEval | `[^19]` | `How to Deploy Qwen3.5-4B with vLLM` (Overshoot AI) | `Tab8[15]` (IFEval `2311.07911`) |
| Sections 3/6 (`p[1353]`, `p[1354]`, `p[3276]`) | `[^95]`, `[^96]` | Third-party unsubmitted repo script (`m10_clean_repro.sh`) | Official NLTK `punkt_tab` docs (also noted by `.docx` Tab 8 warning on `p[3765]`) |
| Throughout Tabs 1–7 (`47x`, `13x`) | `[^1]`, `[^16]` | Internal C-BARR draft/spec documents | Remove before external release |

**Pasted Appendix Noise in `.docx`**: Three appended sections restart or duplicate citation sequences and should be stripped before sharing the `.docx` externally:
1. Pasted *Prior-Art Audit* restarts superscript numbering `1..51` (colliding with Tab 8 `1..103`).
2. Pasted secondary `Works cited` block contains unrelated web-scrape artifacts and **two personal resume/portfolio links** (double-blind PII risk if distributed externally).
3. Pasted *Venue Ranking* block uses a third independent numbering sequence `1..54`.

---

## 4. Complete 49-Entry Corpus Inventory (`citations/`)

| # | Slug | Identifier | `.bib` Key | Status / Size | Pages | Verified Venue |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | `01_zhou2023_ifeval` | `arXiv:2311.07911` | `zhou2023ifeval` | OK (`0.44 MB`) | 43 | arXiv preprint (2023) |
| 02 | `02_yang2024_gateddeltanet` | `arXiv:2412.06464` | `yang2024gateddeltanet` | OK (`0.42 MB`) | 22 | **ICLR 2025** (`.bib` updated) |
| 03 | `03_muennighoff2025_s1` | `arXiv:2501.19393` | `muennighoff2025s1` | OK (`1.00 MB`) | 47 | EMNLP 2025 |
| 04 | `04_ma2025_nothinking` | `arXiv:2504.09858` | `ma2025nothinking` | OK (`0.78 MB`) | 33 | arXiv preprint (2025) |
| 05 | `05_li2025_pitfalls` | `arXiv:2505.11423` | `li2025pitfalls` | OK (`2.98 MB`) | 29 | NeurIPS 2025 (Spotlight) |
| 06 | `06_venkateswaran2026_spotlight` | `arXiv:2505.12025` | `venkateswaran2026spotlight` | OK (`1.55 MB`) | 19 | EACL 2026 (pp. 3752+) |
| 07 | `07_zhu2026_savethinking` | `arXiv:2505.15276` | `zhu2026savethinking` | OK (`1.00 MB`) | 13 | Findings EMNLP 2026 (`2026.findings-emnlp.761`) |
| 08 | `08_aggarwal2025_optimalthinkingbench` | `arXiv:2508.13141` | `aggarwal2025optimalthinkingbench` | OK (`2.06 MB`) | 30 | arXiv preprint (2025) |
| 09 | `09_zeng2025_pac_reasoning` | `arXiv:2510.09133` | `zeng2025pac` | OK (`1.11 MB`) | 41 | **NeurIPS 2026** (`.bib` title updated) |
| 10 | `10_kwon2025_reasonif` | `arXiv:2510.15211` | `kwon2025reasonif` | OK (`0.51 MB`) | 16 | **Findings ACL 2026** (`.bib` updated) |
| 11 | `11_dong2025_ifevalpp` | `arXiv:2512.14754` | `dong2025ifevalpp` | OK (`0.90 MB`) | 29 | **ACL 2026** (`2026.acl-long.354`, `.bib` updated) |
| 12 | `12_leviathan2025_promptrep` | `arXiv:2512.14982` | `leviathan2025promptrep` | OK (`1.73 MB`) | 12 | arXiv preprint (2025) |
| 13 | `13_zeng2026_glimprouter` | `arXiv:2601.05110` | `zeng2026glimprouter` | OK (`0.58 MB`) | 15 | **Findings ACL 2026** (`.bib` updated) |
| 14 | `14_yu2026_bpac` | `arXiv:2601.22446` | `yu2026bpac` | OK (`15.88 MB`) | 25 | ICML 2026 |
| 15 | `15_yan2026_entrocut` | `arXiv:2601.22617` | `yan2026entrocut` | OK (`1.34 MB`) | 5 | **ICASSP 2026** (`.bib` updated) |
| 16 | `16_wang2026_conformal_thinking` | `arXiv:2602.03814` | `wang2026conformal` | OK (`0.87 MB`) | 15 | **ICML 2026** (`.bib` updated) |
| 17 | `17_zhao2026_bridging` | `arXiv:2606.03624` | `zhao2026bridging` | OK (`0.63 MB`) | 18 | **TACL 2026** (`2026.tacl-1.60`, `.bib` updated) |
| 18 | `18_kumar2026_constraint_shifts` | `arXiv:2606.09662` | `kumar2026constraint` | OK (`1.58 MB`) | 16 | arXiv preprint (2026) |
| 19 | `19_lee2026_dart` | `arXiv:2606.23181` | `lee2026dart` | OK (`0.51 MB`) | 17 | Findings EMNLP 2026 |
| 20 | `20_lei2026_thinking_inertia` | `arXiv:2610.11765` | `lei2026thinkinginertia` | OK (`1.62 MB`) | 41 | NeurIPS 2026 |
| 21 | `21_zhu2026_thinking_leakage` | `arXiv:2610.14270` | `zhu2026leakage` | OK (`0.38 MB`) | 24 | arXiv preprint (2026) |
| 22 | `22_ding2026_thinkjev` | `arXiv:2610.37732` | `ding2026thinkjev` | OK (`0.72 MB`) | 31 | arXiv preprint (2026) |
| 23 | `23_nguyen2026_early_stopping` | `arXiv:2611.00166` | `nguyen2026early` | OK (`2.18 MB`) | 19 | arXiv preprint (2026) |
| 24 | `24_han2025_tale` | `arXiv:2412.18547` | — | OK (`1.06 MB`) | 14 | Findings ACL 2025 (`.docx` Tab8[39]) |
| 25 | `25_chen2024_overthinking_o1` | `arXiv:2412.21187` | — | OK (`3.08 MB`) | 17 | arXiv preprint (`.docx` Tab8[33]) |
| 26 | `26_aggarwal2025_l1_lcpo` | `arXiv:2503.04697` | — | OK (`1.27 MB`) | 27 | COLM 2025 (`.docx` Tab8[40,42]) |
| 27 | `27_chen2025_selfbudgeter` | `arXiv:2505.11274` | — | OK (`0.63 MB`) | 22 | arXiv preprint (`.docx` Tab8[48]) |
| 28 | `28_fang2025_thinkless` | `arXiv:2505.13379` | — | OK (`0.95 MB`) | 12 | arXiv preprint (`.docx` Tab8[44,53]) |
| 29 | `29_zhang2025_adaptthink` | `arXiv:2505.13417` | — | OK (`0.79 MB`) | 15 | arXiv preprint (`.docx` Tab8[46,47]) |
| 30 | `30_liang2025_thinkswitcher` | `arXiv:2505.14183` | — | OK (`0.84 MB`) | 17 | Findings EMNLP 2025 (`.docx` Tab8[56–58]) |
| 31 | `31_fu2025_mathif` | `arXiv:2505.14810` | — | OK (`0.63 MB`) | 19 | arXiv preprint (`.docx` Scaling Reasoning) |
| 32 | `32_wang2025_wait_removal` | `arXiv:2506.08343` | — | OK (`1.59 MB`) | 24 | Findings EMNLP 2025 (`NOWAIT`, `.docx` Tab8[41,89]) |
| 33 | `33_verih2025` | `arXiv:2511.04694` | — | OK (`0.57 MB`) | 26 | arXiv preprint (`.docx` Instruction Ladder) |
| 34 | `34_traac_or_rlcot_2601_06052` | `arXiv:2601.06052` | — | OK (`0.59 MB`) | 15 | arXiv preprint (`.docx` Tab8[31]) |
| 35 | `35_gpac2026` | `arXiv:2601.22790` | — | OK (`1.06 MB`) | 23 | arXiv preprint (`G-PAC`, `.docx` Prior-Art^50) |
| 36 | `36_kumar2026_circuit_condensation` | `arXiv:2608.27254` | — | OK (`0.68 MB`) | 31 | arXiv preprint (`.docx` related) |
| 37 | `37_iclr2026_deer_or_8df90a14` | ICLR 2026 PDF | — | OK (`3.73 MB`) | 41 | ICLR 2026 (`DEER`, `.docx` Tab8[9]) |
| 38 | `38_iclr2026_mago_dfdc9c54` | ICLR 2026 PDF | — | OK (`4.41 MB`) | 16 | ICLR 2026 (`MAGO`, `.docx` Tab8[45]) |
| 39 | `39_acl2026_ccc_622` | `2026.findings-acl.622` | — | OK (`0.45 MB`) | 13 | Findings ACL 2026 (`CCC`, `.docx` Tab8[50–51]) |
| 40 | `40_emnlp2026_accuser_1628` | `2026.emnlp-main.1628` | — | OK (`0.52 MB`) | 12 | EMNLP 2026 (`.docx` spurious `Tab8[27]`) |
| 41 | `41_aaai2026_switchable_40799` | `arXiv:2508.10026` | — | OK (`2.59 MB`) | 10 | AAAI 2026 (`SABER`, `.docx` Tab8[55]) |
| 42 | `42_openreview_traac_ZSeKA0LrJg` | `arXiv:2510.01581` / OpenReview | — | OK (`0.54 MB`) | 27 | COLM 2026 (`TRAAC`, `.docx` Tab8[49]) |
| 43 | `43_openreview_query_transfer_Pc7eA7Zb33` | OpenReview `Pc7eA7Zb33` | — | **403 Blocked** (`0.00 MB`) | — | Hosted only behind OpenReview 403 anti-bot challenge |
| 44 | `44_openreview_instaboost_xDyJVMnab8` | `arXiv:2506.13734` | — | OK (`0.81 MB`) | 44 | `INSTABOOST` (`.docx` Prior-Art) |
| 45 | `45_pmlr2026_yuehhan26a` | `arXiv:2603.05706` | — | OK (`1.28 MB`) | 40 | PMLR / CoT controllability (`.docx` Prior-Art) |
| 46 | `46_hf_dhruv1000_logra_ifbench` | HF draft PDF | — | OK (`0.37 MB`) | 8 | Unsubmitted HF draft (`.docx` Tab8[25]) |
| 47 | `47_cobbe2021_gsm8k` | `arXiv:2110.14168` | — | OK (`2.80 MB`) | 22 | GSM8K benchmark (used in C-BARR suite) |
| 48 | `48_talmor2019_commonsenseqa` | `arXiv:1811.00937` | — | OK (`1.28 MB`) | 10 | CommonsenseQA benchmark (used in C-BARR suite) |
| 49 | `49_lin2026_bagen` | `arXiv:2606.00198` | — | OK (`5.28 MB`) | 36 | arXiv preprint (`BAGEN`, `.docx` spurious `Tab8[38]`) |
