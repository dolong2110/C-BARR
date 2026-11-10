"""Template-normalized routing, fast entropy/inertia probing, and C-BARR interventions."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
import time
from typing import Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.graders import strip_special_end_tokens

UNIT_NUM_RE = re.compile(
    r"\b\d+\s*(?:words?|sentences?|paragraphs?|sections?|bullets?|"
    r"bullet points?|items?|lines?|characters?|letters?|times?|commas?|asterisks?|placeholders?)\b"
    r"|\b(?:18|19|20)\d0s\b"
    r"|\b\d+(?:st|nd|rd|th)\s+paragraph\b",
    re.IGNORECASE,
)
STANDALONE_NUM_RE = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)")
REPEAT_PROMPT_RE = re.compile(
    r"(?:repeat the (?:request|prompt)|first[, ]+repeat)", re.IGNORECASE
)

# Aligned across all 15 PRECISION_IDS (5 Inhibition + 10 Structural)
PRECISION_RE = re.compile(
    r"(without using|do not (?:use|include|mention|contain)|refrain from|no commas?|forbidden|"
    r"\bjson\b|all lowercase|lowercase letters|no capital letters|"
    r"all (?:capital|uppercase)|capital letters only|"
    r"(?:words with all capital letters|all[- ]capital words)|"
    r"(?:end|finish) your (?:entire )?response|exact phrase|no other words should follow|"
    r"wrap your (?:entire )?response|double quotation marks|double quotes|"
    r"answer with one of the following|constrained to|my answer is (?:yes|no|maybe)|"
    r"double angular brackets|<<[^>]*>>|"
    r"postscript|\bp\.p?\.s\.|"
    r"placeholders?|square brackets|"
    r"include (?:the )?keywords?|"
    r"entire response should be in [a-z]+|no other language is allowed|"
    r"repeat the (?:request|prompt)|first[, ]+repeat)",
    re.IGNORECASE,
)

# Aligned across all 10 PLANNING_IDS
PLANNING_RE = re.compile(
    r"((?:at least|less than|fewer than|more than|around|exactly|at most|\d+\+)\s*\d*\s*"
    r"(?:words?|sentences?|paragraphs?|sections?|bullets?|bullet points?|times?)|"
    r"(?:contain(?:s)?|have|has|include(?:s)?|write in|there should be)\s+"
    r"(?:at least |less than |more than |around |exactly )?\d+\+?\s+"
    r"(?:words?|sentences?|paragraphs?|sections?|bullets?|bullet points?)|"
    r"\b\d+\s*-?\s*word\b|"
    r"markdown divider|\*\s*\*\s*\*|"
    r"(?:\d+(?:st|nd|rd|th)\s+paragraph|paragraph\s+\d+)\s+must\s+start\s+with|"
    r"bullet points?|multiple sections|section\s+x|highlighted sections|"
    r"(?:letter|word)\s+['\"]?[a-z0-9_]+['\"]?\s+should\s+appear|"
    r"two (?:different )?responses|\*{6})",
    re.IGNORECASE,
)

MATH_CUE_RE = re.compile(
    r"(=|%|\$|how many|how much|calculate|total|altogether|remaining|"
    r"average|profit|difference|per hour|each day|ratio|twice|half)",
    re.IGNORECASE,
)

INERTIA_MARKERS: Sequence[str] = (
    "step", "because", "first", "let's", "therefore", "wait", "since"
)
REFLECTION_SPLIT_RE = re.compile(r"\b(?:Wait|Alternatively|Hmm)\b|\n\n")
P_TERM = "\nOkay, I think I have finished thinking."


@dataclass(frozen=True)
class Gate1Output:
    phi_sym: int
    phi_plan: int
    phi_prec: int
    anchor_clause: str


def run_gate1_lexical(prompt: str) -> Gate1Output:
    masked = UNIT_NUM_RE.sub(" [UNIT_COUNT] ", prompt)
    nums = STANDALONE_NUM_RE.findall(masked.replace(",", ""))
    phi_sym = int(len(nums) >= 2 and bool(MATH_CUE_RE.search(masked)))
    phi_plan = int(bool(PLANNING_RE.search(prompt)))
    phi_prec = int(bool(PRECISION_RE.search(prompt)))

    clauses: list[str] = []
    if REPEAT_PROMPT_RE.search(prompt):
        clauses.append(
            "Repeat the user's complete original request word-for-word without change before answering."
        )
    for sent in re.split(r"(?<=[.!?])\s+|\n+", prompt.strip()):
        if REPEAT_PROMPT_RE.search(sent):
            continue
        if PRECISION_RE.search(sent) or PLANNING_RE.search(sent):
            clauses.append(sent.strip())

    joined = " ".join(clauses)
    words = joined.split()
    if len(words) > 30:
        joined = " ".join(words[:30])
    return Gate1Output(
        phi_sym=phi_sym, phi_plan=phi_plan, phi_prec=phi_prec, anchor_clause=joined
    )


def _build_reanchor_block(
    truncated: bool, anchor_clause: str, anchor_locus: str
) -> tuple[str, str]:
    """Returns (injected_close_block, visible_prefix_kept_in_scored_answer)."""
    term = P_TERM if truncated else ""
    if not anchor_clause:
        return f"{term}\n</think>\n", ""

    reminder_line = f"[Reminder: strictly follow: {anchor_clause}]"
    if anchor_locus == "post_stripped":
        return f"{term}\n</think>\n{reminder_line}\n", ""
    if anchor_locus == "pre_close":
        return f"{term}\nConstraint check: strictly follow: {anchor_clause}\n</think>\n", ""
    if anchor_locus == "post_raw":
        return f"{term}\n</think>\n{reminder_line}\n", f"{reminder_line}\n"
    raise ValueError(f"Unsupported anchor_locus: {anchor_locus}")


class CBARRDualModeEngine:
    """Normalized text-only inference wrapper for Qwen3.5 (0.8B-9B) and SmolLM3-3B."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        freeze_smollm3_date: str = "01 November 2026",
    ):
        self.model_id = model_id
        self.is_smollm3 = "smollm3" in model_id.lower()
        self.freeze_smollm3_date = freeze_smollm3_date
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # AutoModelForCausalLM loads Qwen3_5ForCausalLM on Qwen/Qwen3.5-* checkpoints,
        # dropping the ViT vision tower and MTP heads to save GPU memory.
        try:
            loaded = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            loaded = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model = loaded.to(device).eval()

        raw_eos: list[int] = []
        for src in (
            self.tok.eos_token_id,
            getattr(self.model.generation_config, "eos_token_id", None),
        ):
            if isinstance(src, int):
                raw_eos.append(src)
            elif isinstance(src, (list, tuple)):
                raw_eos.extend(int(x) for x in src if x is not None)
        for tok_str in ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "</s>"):
            tid = self.tok.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id:
                raw_eos.append(tid)
        self.eos_ids: set[int] = set(raw_eos)

    def build_prompt_prefix(
        self,
        user_prompt: str,
        enable_thinking: bool,
        system_prompt: str | None = None,
    ) -> str:
        msgs: list[dict[str, str]] = []
        if system_prompt is not None:
            sys_txt = system_prompt
            if self.is_smollm3 and "/think" not in sys_txt and "/no_think" not in sys_txt:
                sys_txt = f"{sys_txt}\n{'/think' if enable_thinking else '/no_think'}"
            msgs.append({"role": "system", "content": sys_txt})
        elif self.is_smollm3:
            msgs.append({"role": "system", "content": "/think" if enable_thinking else "/no_think"})
        msgs.append({"role": "user", "content": user_prompt})

        try:
            rendered = self.tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            rendered = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        if self.is_smollm3 and self.freeze_smollm3_date:
            rendered = re.sub(
                r"Today Date: [^\n]+", f"Today Date: {self.freeze_smollm3_date}", rendered
            )

        if enable_thinking and system_prompt is None and not rendered.rstrip().endswith("<think>"):
            rendered += "<think>\n"
        return rendered

    @torch.no_grad()
    def _decode_raw(
        self, prompt_str: str, max_new_tokens: int, log_entropies: bool = False
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        inp = self.tok(prompt_str, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=sorted(self.eos_ids) if self.eos_ids else None,
            pad_token_id=self.tok.pad_token_id or next(iter(self.eos_ids), 0),
            return_dict_in_generate=True,
            output_scores=log_entropies,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        gen_ids = out.sequences[0, inp.input_ids.shape[1] :].tolist()
        hit_eos = bool(gen_ids and gen_ids[-1] in self.eos_ids)

        content_len = len(gen_ids)
        while content_len > 0 and gen_ids[content_len - 1] in self.eos_ids:
            content_len -= 1

        step_entropies: list[float] = []
        if log_entropies and out.scores:
            for j in range(min(content_len, len(out.scores))):
                log_p = torch.log_softmax(out.scores[j][0].float(), dim=-1)
                p = torch.exp(log_p)
                step_entropies.append(float(-torch.sum(p * log_p).item()))

        return {
            "raw_text": self.tok.decode(gen_ids, skip_special_tokens=False),
            "total_gen_tokens": len(gen_ids),
            "content_tokens": content_len,
            "hit_eos": hit_eos,
            "step_entropies": step_entropies,
            "latency_ms": latency_ms,
        }

    def run_direct_non_thinking(self, prompt: str, max_new_tokens: int = 1024) -> dict[str, Any]:
        """Executes B0 (Always Non-Thinking) or Gate-1 pure-precision bypass in 1.0 call (Patch D14: 1024)."""
        pref = self.build_prompt_prefix(prompt, enable_thinking=False)
        res = self._decode_raw(pref, max_new_tokens=max_new_tokens, log_entropies=False)
        a_final = strip_special_end_tokens(res["raw_text"]).strip()
        return {
            "t_think": "",
            "a_final": a_final,
            "T_probe": 0,
            "T_think": 0,
            "T_wasted": 0,
            "T_inj": 0,
            "T_final": res["total_gen_tokens"],
            "T_total": res["total_gen_tokens"],
            "calls": 1,
            "truncated": False,
            "think_exhausted": False,
            "latency_ms": res["latency_ms"],
        }

    def run_prompted_mode(
        self, prompt: str, mode: str, max_new_tokens: int = 2048
    ) -> dict[str, Any]:
        """Executes B1a (Prompted Step-by-Step CoT) or B2a (Self-Selective Prompting)."""
        if mode == "B1a_cot":
            sys_p = (
                "Think step by step to solve the user's request, then write 'Final Answer:' "
                "followed strictly by your final response."
            )
        elif mode == "B2a_self_select":
            sys_p = (
                "Decide whether this task benefits from step-by-step reasoning. If yes, reason "
                "briefly and then write 'Final Answer:' followed by your response. If direct "
                "answering is better (such as strict formatting constraints), write 'Final Answer:' "
                "immediately followed by your response."
            )
        else:
            raise ValueError(f"Unsupported prompted mode: {mode}")

        pref = self.build_prompt_prefix(prompt, enable_thinking=False, system_prompt=sys_p)
        res = self._decode_raw(pref, max_new_tokens=max_new_tokens, log_entropies=False)
        cleaned = strip_special_end_tokens(res["raw_text"])
        if "Final Answer:" in cleaned:
            cot_part, ans_part = cleaned.rsplit("Final Answer:", 1)
            t_cot_toks = len(self.tok.encode(cot_part, add_special_tokens=False))
            t_final_toks = max(0, res["total_gen_tokens"] - t_cot_toks)
            return {
                "t_think": cot_part.strip(),
                "a_final": ans_part.strip(),
                "T_probe": 0,
                "T_think": t_cot_toks,
                "T_wasted": 0,
                "T_inj": 0,
                "T_final": t_final_toks,
                "T_total": res["total_gen_tokens"],
                "calls": 1,
                "truncated": not res["hit_eos"],
                "think_exhausted": False,
                "latency_ms": res["latency_ms"],
            }
        return {
            "t_think": "",
            "a_final": cleaned.strip(),
            "T_probe": 0,
            "T_think": 0,
            "T_wasted": 0,
            "T_inj": 0,
            "T_final": res["total_gen_tokens"],
            "T_total": res["total_gen_tokens"],
            "calls": 1,
            "truncated": not res["hit_eos"],
            "think_exhausted": False,
            "latency_ms": res["latency_ms"],
        }

    def run_gate2_fast_probe(
        self, prompt: str, g1: Gate1Output | None = None, probe_tokens: int = 32
    ) -> dict[str, Any]:
        """Runs 32-token non-thinking uncertainty probe without corrupting IFEval prefixes."""
        g1 = g1 or run_gate1_lexical(prompt)
        is_mc = "(A)" in prompt and "(B)" in prompt
        is_pure_math = bool(g1.phi_sym and not g1.phi_prec and not g1.phi_plan)
        ans_prefix = "Answer: " if is_mc else ("#### " if is_pure_math else "")

        base_pref = self.build_prompt_prefix(prompt, enable_thinking=False) + ans_prefix
        res = self._decode_raw(base_pref, max_new_tokens=probe_tokens, log_entropies=True)
        ents = res["step_entropies"]
        clean_fast = strip_special_end_tokens(res["raw_text"]).strip()
        low = clean_fast.lower()

        concise_probe = is_mc or is_pure_math
        inertia = int(
            (concise_probe and res["content_tokens"] > 16)
            or any(w in low for w in INERTIA_MARKERS)
        )
        return {
            "base_pref": base_pref,
            "ans_prefix": ans_prefix,
            "probe_raw_text": res["raw_text"],
            "fast_answer": (ans_prefix + clean_fast).strip(),
            "hit_eos": res["hit_eos"],
            "T_probe": res["total_gen_tokens"],
            "m_content": res["content_tokens"],
            "H_first": ents[0] if ents else 0.0,
            "H_mean": float(np.mean(ents)) if ents else 0.0,
            "H_max": float(np.max(ents)) if ents else 0.0,
            "step_entropies": ents,
            "I_inertia": inertia,
            "latency_ms": res["latency_ms"],
        }

    def complete_fast_probe(
        self, probe_out: dict[str, Any], max_ans_tokens: int = 1024
    ) -> dict[str, Any]:
        """Completes a non-escalated Gate-2 probe if 32 tokens did not yet reach EOS (Patch D14: 1024)."""
        if probe_out["hit_eos"] or max_ans_tokens <= probe_out["T_probe"]:
            return {
                "t_think": "",
                "a_final": probe_out["fast_answer"],
                "T_probe": 0,
                "T_think": 0,
                "T_wasted": 0,
                "T_inj": 0,
                "T_final": probe_out["T_probe"],
                "T_total": probe_out["T_probe"],
                "calls": 1,
                "truncated": not probe_out["hit_eos"],
                "think_exhausted": False,
                "latency_ms": probe_out["latency_ms"],
            }
        rem = max_ans_tokens - probe_out["T_probe"]
        cont_pref = probe_out["base_pref"] + probe_out["probe_raw_text"]
        s_rem = self._decode_raw(cont_pref, max_new_tokens=rem, log_entropies=False)
        combined = probe_out["ans_prefix"] + strip_special_end_tokens(
            probe_out["probe_raw_text"] + s_rem["raw_text"]
        )
        tot_toks = probe_out["T_probe"] + s_rem["total_gen_tokens"]
        return {
            "t_think": "",
            "a_final": combined.strip(),
            "T_probe": 0,
            "T_think": 0,
            "T_wasted": 0,
            "T_inj": 0,
            "T_final": tot_toks,
            "T_total": tot_toks,
            "calls": 1,
            "truncated": not s_rem["hit_eos"],
            "think_exhausted": False,
            "latency_ms": probe_out["latency_ms"] + s_rem["latency_ms"],
        }

    def run_capped_think_and_anchor(
        self,
        prompt: str,
        budget_b: int,
        anchor_clause: str = "",
        max_ans_tokens: int = 1024,
        allow_budget_force: bool = True,
        anchor_locus: str = "post_stripped",
    ) -> dict[str, Any]:
        """Executes capped <think> (B in {64, 128, 250}) + P_TERM + Post/Pre-</think> re-anchoring."""
        pref = self.build_prompt_prefix(prompt, enable_thinking=True)
        s1 = self._decode_raw(pref, max_new_tokens=budget_b, log_entropies=False)
        raw1 = s1["raw_text"]
        clean_raw1 = re.sub(r"^\s*<think>\s*", "", raw1)

        if "</think>" in clean_raw1:
            t_think, t_after = clean_raw1.split("</think>", 1)
            t_think_toks = len(self.tok.encode(t_think, add_special_tokens=False))
            if not anchor_clause and s1["hit_eos"]:
                t_final_clean = strip_special_end_tokens(t_after).lstrip("\n").strip()
                t_final_toks = max(0, s1["total_gen_tokens"] - t_think_toks)
                return {
                    "t_think": t_think,
                    "a_final": t_final_clean,
                    "T_probe": 0,
                    "T_think": t_think_toks,
                    "T_wasted": 0,
                    "T_inj": 0,
                    "T_final": t_final_toks,
                    "T_total": s1["total_gen_tokens"],
                    "calls": 1,
                    "truncated": False,
                    "think_exhausted": False,
                    "latency_ms": s1["latency_ms"],
                }
            if not anchor_clause and not s1["hit_eos"]:
                after_clean_toks = max(0, s1["total_gen_tokens"] - t_think_toks)
                rem_ans = max(1, max_ans_tokens - after_clean_toks)
                s2 = self._decode_raw(pref + clean_raw1, max_new_tokens=rem_ans, log_entropies=False)
                a_final = strip_special_end_tokens(t_after + s2["raw_text"]).lstrip("\n").strip()
                tot = s1["total_gen_tokens"] + s2["total_gen_tokens"]
                return {
                    "t_think": t_think,
                    "a_final": a_final,
                    "T_probe": 0,
                    "T_think": t_think_toks,
                    "T_wasted": 0,
                    "T_inj": 0,
                    "T_final": after_clean_toks + s2["total_gen_tokens"],
                    "T_total": tot,
                    "calls": 1,
                    "truncated": False,
                    "think_exhausted": False,
                    "latency_ms": s1["latency_ms"] + s2["latency_ms"],
                }

            t_wasted = max(0, s1["total_gen_tokens"] - t_think_toks)
            close_block, vis_prefix = _build_reanchor_block(False, anchor_clause, anchor_locus)
            cont_prefix = pref + t_think + close_block
            truncated = False
        else:
            if not allow_budget_force:
                return {
                    "t_think": strip_special_end_tokens(clean_raw1),
                    "a_final": "",
                    "T_probe": 0,
                    "T_think": s1["total_gen_tokens"],
                    "T_wasted": 0,
                    "T_inj": 0,
                    "T_final": 0,
                    "T_total": s1["total_gen_tokens"],
                    "calls": 1,
                    "truncated": True,
                    "think_exhausted": True,
                    "latency_ms": s1["latency_ms"],
                }
            t_think = strip_special_end_tokens(clean_raw1)
            t_think_toks = s1["total_gen_tokens"]
            t_wasted = 0
            close_block, vis_prefix = _build_reanchor_block(True, anchor_clause, anchor_locus)
            cont_prefix = pref + t_think + close_block
            truncated = True

        t_inj_toks = len(self.tok.encode(close_block, add_special_tokens=False))
        s2 = self._decode_raw(cont_prefix, max_new_tokens=max_ans_tokens, log_entropies=False)
        s2_clean = strip_special_end_tokens(s2["raw_text"]).lstrip("\n").strip()
        a_final = f"{vis_prefix}{s2_clean}".strip() if vis_prefix else s2_clean
        tot = t_think_toks + t_wasted + t_inj_toks + s2["total_gen_tokens"]
        return {
            "t_think": t_think,
            "a_final": a_final,
            "T_probe": 0,
            "T_think": t_think_toks,
            "T_wasted": t_wasted,
            "T_inj": t_inj_toks,
            "T_final": s2["total_gen_tokens"],
            "T_total": tot,
            "calls": 1,
            "truncated": truncated,
            "think_exhausted": False,
            "latency_ms": s1["latency_ms"] + s2["latency_ms"],
        }

    def run_diagnostic_control(
        self, prompt: str, cached_t_think: str, math_filler_pool_ids: list[int], mode: str
    ) -> str:
        """Executes Causal Diagnostic Controls (Trace Stripping vs. Post-Think Re-Anchoring + A2/B1/B3)."""
        pref = self.build_prompt_prefix(prompt, enable_thinking=True)
        g1 = run_gate1_lexical(prompt)
        reminder_line = f"[Reminder: strictly follow: {g1.anchor_clause}]"

        if mode == "A1_strip":
            clamped = self.build_prompt_prefix(prompt, enable_thinking=False)
        elif mode == "B2a_post_anchor":
            clamped = f"{pref}{cached_t_think}\n</think>\n{reminder_line}\n"
        elif mode == "B2b_post_raw":
            clamped = f"{pref}{cached_t_think}\n</think>\n{reminder_line}\n"
        elif mode == "B1_pre_anchor":
            clamped = (
                f"{pref}{cached_t_think}\n"
                f"Constraint check: strictly follow: {g1.anchor_clause}\n</think>\n"
            )
        elif mode == "A2_math_swap":
            n_toks = len(self.tok.encode(cached_t_think, add_special_tokens=False))
            repeats = (n_toks // max(1, len(math_filler_pool_ids))) + 1
            pooled = (math_filler_pool_ids * repeats)[:n_toks]
            matched_math = self.tok.decode(pooled, skip_special_tokens=True)
            clamped = f"{pref}{matched_math}\n</think>\n"
        elif mode == "B3_early_cut_no_anchor":
            m = REFLECTION_SPLIT_RE.search(cached_t_think[32:]) if len(cached_t_think) > 32 else None
            prefix_think = (
                cached_t_think[: 32 + m.start()] if m else cached_t_think[: len(cached_t_think) // 2]
            )
            clamped = f"{pref}{prefix_think}{P_TERM}\n</think>\n"
        else:
            raise ValueError(f"Unknown diagnostic control mode: {mode}")

        # Patch D14: max_new_tokens=1024
        out = self._decode_raw(clamped, max_new_tokens=1024, log_entropies=False)
        res_clean = strip_special_end_tokens(out["raw_text"]).lstrip("\n").strip()
        if mode == "B2b_post_raw":
            return f"{reminder_line}\n{res_clean}"
        return res_clean


# Backward-compatible alias for scripts/tests referencing DualModeEngine
DualModeEngine = CBARRDualModeEngine
