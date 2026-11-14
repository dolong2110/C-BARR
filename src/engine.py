"""Template-normalized routing, fast entropy/inertia probing, and C-BARR interventions."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import re
import time
from typing import Any
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from src.graders import strip_special_end_tokens

# Word-form and ordinal helpers calibrated on seed_60 / dev_60 IFEval + GSM8K rules
_NUM_WORD = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"once|twice|thrice|double|triple|half|a couple|several)"
)
_ORDINAL_WORD = (
    r"(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
)

# Unit-bound numerals (e.g., "300 words", "1990s", "2nd paragraph") are masked inside formatting
# clauses before counting numbers for phi_sym so length/structural rules don't mimic math problems.
UNIT_NUM_RE = re.compile(
    r"\b\d+\s*(?:words?|sentences?|paragraphs?|sections?|bullets?|"
    r"bullet points?|items?|lines?|characters?|letters?|times?|commas?|asterisks?|placeholders?)\b"
    r"|\b(?:18|19|20)\d0s\b"
    r"|\b\d+(?:st|nd|rd|th)\s+paragraph\b",
    re.IGNORECASE,
)
_ALWAYS_MASK_NUM_RE = re.compile(
    r"\b(?:18|19|20)\d0s\b|\b\d+(?:st|nd|rd|th)\s+paragraph\b|\bsection\s+\d+\b",
    re.IGNORECASE,
)
STANDALONE_NUM_RE = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)")
REPEAT_PROMPT_RE = re.compile(
    r"(?:repeat the (?:request|prompt)|first[, ]+repeat)", re.IGNORECASE
)

# Capital-word frequency rules belong to Precision (change_case:capital_word_frequency)
# and should not falsely trigger structural Planning rules.
CAPITAL_WORD_FREQ_RE = re.compile(
    r"(?:\bwords?\s+(?:with|in)\s+all[- ]?(?:capital letters|caps)|"
    r"\ball[- ]?(?:capital|caps)\s+words?\b)",
    re.IGNORECASE,
)
_CAP_FREQ_STRIP_RE = re.compile(
    r"(?:(?:use|include|at least|at most|less than|fewer than|more than|only|\d+|some|a few)\s+)*"
    r"(?:words?\s+(?:with|in)\s+all[- ]?(?:capital letters|caps)|all[- ]?(?:capital|caps)\s+words?)"
    r"(?:\s+(?:in your response|should appear|at least|at most|less than|fewer than|more than|only|than|times|words|\d+|,))*",
    re.IGNORECASE,
)

# Aligned across all 15 PRECISION_IDS (5 Inhibition + 10 Structural)
PRECISION_RE = re.compile(
    r"(without using|(?:do not|don't|never)\s+(?:use|include|mention|contain|say|write)|"
    r"refrain from|no commas?|forbidden|"
    r"\bjson\b|all lowercase|lowercase letters|no capital letters|"
    r"all (?:capital|uppercase)|capital letters only|"
    r"words?\s+(?:with|in)\s+all[- ]?(?:capital letters|caps)|all[- ]?(?:capital|caps)\s+words?|"
    r"(?:end|finish)\s+(?:your|the)\s+(?:entire\s+)?(?:response|answer|reply|song|letter|story|essay|poem|email|message|text)(?:\s+with)?|"
    r"exact phrase|no (?:other|additional) words should follow|very end of your (?:entire )?response|"
    r"wrap your (?:entire )?response|double quotation marks|double quotes|"
    r"answer with one of the following|constrained to|my answer is (?:yes|no|maybe)|"
    r"double angular brackets|<<[^>]*>>|"
    r"post\s*script|\bp\.\s*p?\.?\s*s\.?(?!\w)|"
    r"placeholders?|square brackets|"
    r"include (?:the )?keywords?|"
    r"entire response should be in [a-z]+|no other language is allowed|"
    r"repeat the (?:request|prompt)|first[, ]+repeat)",
    re.IGNORECASE,
)

# Aligned across all 10 PLANNING_IDS (supports digits, spelled-out numbers, ordinals, and ranges)
PLANNING_RE = re.compile(
    rf"((?:at least|less than|fewer than|more than|around|approximately|exactly|at most|\d+\+)\s*"
    rf"(?:{_NUM_WORD})(?:\s+(?:[a-z-]+)){{0,2}}\s*"
    rf"(?:words?|sentences?|paragraphs?|sections?|bullets?|bullet points?|times?)|"
    rf"(?:appear|use|used|using|repeat(?:ed)?|occur(?:s)?|mention(?:ed)?|include(?:d)?|say|write)\s+"
    rf"[^.!?\n]{{0,35}}\b(?:at least|less than|fewer than|more than|at most|exactly)\s+(?:once|twice|thrice)\b|"
    rf"(?:contain(?:s)?|have|has|include(?:s)?|write(?:\s+in)?|there should be|divide(?:d)? into|consist(?:s)? of)\s+"
    rf"(?:at least |less than |fewer than |more than |around |exactly |about )?"
    rf"(?:{_NUM_WORD})(?:\s+(?:[a-z-]+)){{0,2}}\s+"
    rf"(?:words?|sentences?|paragraphs?|sections?|bullets?|bullet points?)|"
    rf"\b(?:{_NUM_WORD}|\d+\+)\s*(?:-\s*)?(?:word|sentence|paragraph)\b|"
    rf"(?:number\s+of\s+(?:words|sentences|paragraphs|sections)[^.!?\n]{{0,40}}in\s+the\s+range\s+of\s+\d+\s+to\s+\d+|"
    rf"between\s+(?:{_NUM_WORD})\s+and\s+(?:{_NUM_WORD})\s+(?:words?|sentences?|paragraphs?|sections?))|"
    r"markdown divider|\*\s*\*\s*\*|"
    rf"(?:{_ORDINAL_WORD}\s+paragraph|paragraph\s+(?:{_NUM_WORD}))\s+must\s+start\s+with|"
    r"bullet points?|multiple sections|section\s+x|highlighted sections?|"
    rf"italicize\s+(?:at\s+least\s+)?(?:{_NUM_WORD})[^.!?\n]{{0,30}}sections?|"
    r"(?:letter|word)\s+['\"]?[a-z0-9_]+['\"]?\s+(?:should\s+appear|at\s+least|less\s+than|more\s+than)|"
    r"two (?:different )?responses|\*{6})",
    re.IGNORECASE,
)

MATH_CUE_RE = re.compile(
    r"(=|%|\$|how many|how much|how long|how old|how far|calculate|total|altogether|remaining|"
    r"average|profit|difference|per\s+(?:hour|day|week|month|year|minute|gallon|pound|item|mile|box|pack|dozen)|"
    r"each\s+(?:day|week|month|year|hour|minute|person|item|box)|"
    r"ratio|twice|thrice|triple|quadruple|half|"
    r"spent|spend|cost|costs|earned|earn|paid|save[sd]?|bought|weighs|weight|mph|km/h|"
    r"(?:years?|months?)\s+(?:old|younger|older)|times?\s+as\s+(?:many|much|long|old|fast))",
    re.IGNORECASE,
)

INERTIA_MARKERS: Sequence[str] = (
    "step", "because", "first", "let's", "therefore", "wait", "since"
)
REFLECTION_SPLIT_RE = re.compile(r"\b(?:Wait|Alternatively|Hmm)\b|\n\n")
# Deliberation-closing phrase spliced before </think> when a budget cap truncates <think>
P_TERM = "\nOkay, I think I have finished thinking."


@dataclass(frozen=True)
class Gate1Output:
    phi_sym: int
    phi_plan: int
    phi_prec: int
    anchor_clause: str


def _strip_cap_word_freq_for_plan(text: str) -> str:
    """Removes pure capital-word-frequency phrases before running PLANNING_RE."""
    if not CAPITAL_WORD_FREQ_RE.search(text):
        return text
    return _CAP_FREQ_STRIP_RE.sub(" ", text)


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def run_gate1_lexical(prompt: str) -> Gate1Output:
    """Runs Gate 1: sub-millisecond regex constraint screening and balanced anchor-clause extraction."""
    # Evaluate planning/precision first; mask structural unit counts only inside non-question constraint clauses
    # so GSM8K word problems mentioning "450 words" or "15 items" keep their numeric signals for phi_sym.
    plan_text = _strip_cap_word_freq_for_plan(prompt)
    phi_plan = int(bool(PLANNING_RE.search(plan_text)))
    phi_prec = int(bool(PRECISION_RE.search(prompt)))

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", prompt.strip()) if s.strip()]
    masked_sents: list[str] = []
    for sent in sents:
        s_clean = _ALWAYS_MASK_NUM_RE.sub(" [UNIT_COUNT] ", sent)
        is_question_math = bool(re.search(r"\bhow\s+(?:many|much|long|old|far|tall)\b", sent, re.IGNORECASE))
        is_constraint_sent = bool(PLANNING_RE.search(_strip_cap_word_freq_for_plan(sent)) or PRECISION_RE.search(sent))
        if is_constraint_sent and not is_question_math:
            s_clean = UNIT_NUM_RE.sub(" [UNIT_COUNT] ", s_clean)
        masked_sents.append(s_clean)
    masked = " ".join(masked_sents) if masked_sents else UNIT_NUM_RE.sub(" [UNIT_COUNT] ", prompt)
    nums = STANDALONE_NUM_RE.findall(masked.replace(",", ""))
    phi_sym = int(len(nums) >= 2 and bool(MATH_CUE_RE.search(masked)))

    plan_clauses: list[str] = []
    prec_clauses: list[str] = []
    ordered_clauses: list[str] = []

    # Normalize "repeat the prompt" cleanly so injecting the reminder doesn't repeat itself recursively
    if REPEAT_PROMPT_RE.search(prompt):
        rep_clause = (
            "Repeat the user's complete original request word-for-word without change before answering."
        )
        prec_clauses.append(rep_clause)
        ordered_clauses.append(rep_clause)
    for sent in sents:
        if REPEAT_PROMPT_RE.search(sent):
            continue
        has_prec = bool(PRECISION_RE.search(sent))
        has_plan = bool(PLANNING_RE.search(_strip_cap_word_freq_for_plan(sent)))
        if has_prec or has_plan:
            ordered_clauses.append(sent)
            if has_plan and not has_prec:
                plan_clauses.append(sent)
            elif has_prec and not has_plan:
                prec_clauses.append(sent)
            else:
                plan_clauses.append(sent)
                prec_clauses.append(sent)

    # CRGC-style structured dual-constraint bridge on conflict prompts so neither rule gets clipped at 30 words
    if phi_plan == 1 and phi_prec == 1 and plan_clauses and prec_clauses and " ".join(plan_clauses) != " ".join(prec_clauses):
        p_txt = " ".join(plan_clauses)
        r_txt = " ".join(prec_clauses)
        # Allocate 26 content words (+4 label tokens = <=30 total words) proportional to need with guaranteed floor
        p_len = len(p_txt.split())
        r_len = len(r_txt.split())
        p_cap = min(p_len, 13 + max(0, 13 - r_len))
        r_cap = min(r_len, 26 - p_cap)
        joined = f"Structure: {_truncate_words(p_txt, p_cap)} | Format: {_truncate_words(r_txt, r_cap)}"
    else:
        joined = _truncate_words(" ".join(ordered_clauses), 30)

    return Gate1Output(
        phi_sym=phi_sym, phi_plan=phi_plan, phi_prec=phi_prec, anchor_clause=joined
    )


def _build_reanchor_block(
    truncated: bool,
    anchor_clause: str,
    anchor_locus: str,
    post_think_bridge: str | None = None,
) -> tuple[str, str]:
    """Returns (injected_close_block, visible_prefix_kept_in_scored_answer)."""
    term = P_TERM if truncated else ""
    if not anchor_clause:
        return f"{term}\n</think>\n", ""

    reminder_line = f"[Reminder: strictly follow: {anchor_clause}]"
    # pre_close: place check inside <think> just before </think> to ablate tag-boundary locus
    if anchor_locus == "pre_close":
        return f"{term}\nConstraint check: strictly follow: {anchor_clause}\n</think>\n", ""
    # post_stripped: deliver reminder right after </think> without polluting assistant's visible buffer
    if anchor_locus == "post_stripped":
        if post_think_bridge:
            return f"{term}\n</think>{post_think_bridge}", ""
        return f"{term}\n</think>\n{reminder_line}\n", ""
    # post_raw: leak reminder into the scored answer to quantify wrapper corruption on strict rules
    if anchor_locus == "post_raw":
        block = f"{term}\n</think>{post_think_bridge}" if post_think_bridge else f"{term}\n</think>\n{reminder_line}\n"
        return block, f"{reminder_line}\n"
    raise ValueError(f"Unsupported anchor_locus: {anchor_locus}")


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Applies flat additive OpenAI/vLLM-style presence penalty to tokens generated after prompt_len."""

    def __init__(self, prompt_len: int, penalty: float):
        self.prompt_len = int(prompt_len)
        self.penalty = float(penalty)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.penalty == 0.0 or input_ids.shape[1] <= self.prompt_len:
            return scores
        gen_slice = input_ids[:, self.prompt_len :]
        for b_idx in range(gen_slice.shape[0]):
            seen = torch.unique(gen_slice[b_idx])
            scores[b_idx, seen] = scores[b_idx, seen] - self.penalty
        return scores


class CBARRDualModeEngine:
    """Normalized text-only inference wrapper for Qwen3.5 (0.8B-9B) and SmolLM3-3B."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        freeze_smollm3_date: str = "01 November 2026",
        temperature: float = 0.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        sampling_seed: int = 42,
    ):
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {temperature}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must lie in (0.0, 1.0], got {top_p}")
        self.model_id = model_id
        self.is_smollm3 = "smollm3" in model_id.lower()
        self.freeze_smollm3_date = freeze_smollm3_date
        self.device = device
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.presence_penalty = float(presence_penalty)
        self.sampling_seed = int(sampling_seed)
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
        # SmolLM3 uses /think or /no_think tags in the system prompt instead of enable_thinking
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
            # Older/model-specific tokenizers without enable_thinking kwarg
            rendered = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        # Pin SmolLM3's system date header so prompts/caches remain reproducible across calendar days
        if self.is_smollm3 and self.freeze_smollm3_date:
            rendered = re.sub(
                r"Today Date: [^\n]+", f"Today Date: {self.freeze_smollm3_date}", rendered
            )

        # Guarantee an explicit opening <think> tag before budgeted generation starts
        if enable_thinking and system_prompt is None and not rendered.rstrip().endswith("<think>"):
            rendered += "<think>\n"
        return rendered

    def _build_post_think_user_bridge(self, anchor_clause: str) -> str | None:
        """Wraps a post-</think> reminder inside a turn-delimited user message so greedy decoding never echoes it."""
        if not anchor_clause or not hasattr(self.tok, "apply_chat_template"):
            return None
        sentinel_a = "__CBARR_PREV_ASST__"
        sentinel_u = "__CBARR_REANCHOR__"
        msgs = [
            {"role": "user", "content": "__CBARR_ORIG_REQ__"},
            {"role": "assistant", "content": sentinel_a},
            {"role": "user", "content": sentinel_u},
        ]
        try:
            rendered = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            rendered = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        if sentinel_a not in rendered or sentinel_u not in rendered:
            return None
        bridge = rendered.split(sentinel_a, 1)[1]
        reminder_msg = (
            f"[Reminder: strictly follow: {anchor_clause}]\n"
            "Apply to the original request above cleanly; do not repeat this note."
        )
        return bridge.replace(sentinel_u, reminder_msg, 1)

    def _build_reanchor_for_engine(
        self, truncated: bool, anchor_clause: str, anchor_locus: str
    ) -> tuple[str, str]:
        """Shared re-anchor builder used by both M2 (Phase 4-6) and Section 6 diagnostic controls."""
        bridge = (
            self._build_post_think_user_bridge(anchor_clause)
            if anchor_clause and anchor_locus in ("post_stripped", "post_raw")
            else None
        )
        return _build_reanchor_block(truncated, anchor_clause, anchor_locus, post_think_bridge=bridge)

    @torch.inference_mode()
    def _decode_generate_core(
        self,
        inp_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        prompt_len_for_penalty: int,
        max_new_tokens: int,
        log_entropies: bool,
        seed_key: str,
        past_key_values: Any = None,
        return_past: bool = False,
    ) -> dict[str, Any]:
        """Executes HF model.generate from token IDs with optional KV/DeltaNet cache reuse."""
        t0 = time.perf_counter()
        cur_len = int(inp_ids.shape[1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": sorted(self.eos_ids) if self.eos_ids else None,
            "pad_token_id": self.tok.pad_token_id or next(iter(self.eos_ids), 0),
            "return_dict_in_generate": True,
            "output_scores": log_entropies,
        }
        if past_key_values is not None:
            gen_kwargs["past_key_values"] = past_key_values
        if return_past or past_key_values is not None:
            gen_kwargs["use_cache"] = True
        temperature = float(getattr(self, "temperature", 0.0))
        top_p = float(getattr(self, "top_p", 1.0))
        presence_penalty = float(getattr(self, "presence_penalty", 0.0))
        sampling_seed = int(getattr(self, "sampling_seed", 42))
        if temperature > 0.0:
            # Hash (sampling_seed, seed_key) without max_new_tokens so resume and S1 slicing match bit-for-bit.
            digest = hashlib.sha256(f"{sampling_seed}:{seed_key}".encode("utf-8")).digest()
            torch.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if presence_penalty != 0.0:
                gen_kwargs["logits_processor"] = LogitsProcessorList(
                    [PresencePenaltyLogitsProcessor(prompt_len_for_penalty, presence_penalty)]
                )
        else:
            gen_kwargs["do_sample"] = False

        out = self.model.generate(
            input_ids=inp_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        gen_ids = out.sequences[0, cur_len:].tolist()
        hit_eos = bool(gen_ids and gen_ids[-1] in self.eos_ids)

        # Measure entropy only across non-EOS content tokens so stopping punctuation doesn't skew stats
        content_len = len(gen_ids)
        while content_len > 0 and gen_ids[content_len - 1] in self.eos_ids:
            content_len -= 1

        step_entropies: list[float] = []
        if log_entropies and out.scores:
            for j in range(min(content_len, len(out.scores))):
                log_p = torch.log_softmax(out.scores[j][0].float(), dim=-1)
                p = torch.exp(log_p)
                step_entropies.append(float(-torch.sum(p * log_p).item()))

        payload: dict[str, Any] = {
            "raw_text": self.tok.decode(gen_ids, skip_special_tokens=False),
            "gen_ids": gen_ids,
            "total_gen_tokens": len(gen_ids),
            "content_tokens": content_len,
            "hit_eos": hit_eos,
            "step_entropies": step_entropies,
            "latency_ms": latency_ms,
        }
        if return_past:
            payload["_kv_state"] = {
                "sequences": out.sequences,
                "past_key_values": getattr(out, "past_key_values", None),
                "prompt_len_for_penalty": prompt_len_for_penalty,
                "gen_ids": gen_ids,
            }
        return payload

    @torch.inference_mode()
    def _decode_raw(
        self, prompt_str: str, max_new_tokens: int, log_entropies: bool = False
    ) -> dict[str, Any]:
        inp = self.tok(prompt_str, return_tensors="pt").to(self.device)
        prompt_len = int(inp["input_ids"].shape[1])
        mask = inp.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(inp["input_ids"])
        return self._decode_generate_core(
            inp_ids=inp["input_ids"],
            attention_mask=mask,
            prompt_len_for_penalty=prompt_len,
            max_new_tokens=max_new_tokens,
            log_entropies=log_entropies,
            seed_key=prompt_str,
        )

    @torch.inference_mode()
    def _decode_raw_stateful(
        self, prompt_str: str, max_new_tokens: int, log_entropies: bool = False
    ) -> dict[str, Any]:
        """Runs _decode_raw while preserving Stage-1 past_key_values when using the native backend."""
        if type(self)._decode_raw is not CBARRDualModeEngine._decode_raw:
            return self._decode_raw(prompt_str, max_new_tokens=max_new_tokens, log_entropies=log_entropies)
        inp = self.tok(prompt_str, return_tensors="pt").to(self.device)
        prompt_len = int(inp["input_ids"].shape[1])
        mask = inp.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(inp["input_ids"])
        return self._decode_generate_core(
            inp_ids=inp["input_ids"],
            attention_mask=mask,
            prompt_len_for_penalty=prompt_len,
            max_new_tokens=max_new_tokens,
            log_entropies=log_entropies,
            seed_key=prompt_str,
            return_past=True,
        )

    @torch.inference_mode()
    def _decode_continue_from_state(
        self,
        kv_state: dict[str, Any] | None,
        fallback_prompt_str: str,
        append_str: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        """Continues decoding from a live Stage-1 KV/DeltaNet cache, or falls back cleanly to _decode_raw."""
        if (
            kv_state is None
            or kv_state.get("past_key_values") is None
            or type(self)._decode_raw is not CBARRDualModeEngine._decode_raw
        ):
            return self._decode_raw(fallback_prompt_str, max_new_tokens=max_new_tokens, log_entropies=False)
        seq = kv_state["sequences"]
        if append_str:
            add_ids = self.tok.encode(append_str, add_special_tokens=False)
            if add_ids:
                add_t = torch.tensor([add_ids], dtype=seq.dtype, device=seq.device)
                seq = torch.cat([seq, add_t], dim=1)
        attn_mask = torch.ones_like(seq)
        return self._decode_generate_core(
            inp_ids=seq,
            attention_mask=attn_mask,
            prompt_len_for_penalty=int(kv_state["prompt_len_for_penalty"]),
            max_new_tokens=max_new_tokens,
            log_entropies=False,
            seed_key=fallback_prompt_str,
            past_key_values=kv_state["past_key_values"],
            return_past=False,
        )

    def _slice_precomputed_s1(
        self, b1b_s1: dict[str, Any] | None, budget_b: int
    ) -> dict[str, Any] | None:
        """Slices deterministic greedy Stage-1 tokens from B1b (B=2048) down to budget_b."""
        if not b1b_s1:
            return None
        full_ids = b1b_s1.get("gen_ids")
        if not full_ids:
            return None
        b1b_hit_eos = bool(b1b_s1.get("hit_eos", False))
        if len(full_ids) < budget_b and not b1b_hit_eos:
            return None
        s1_ids = list(full_ids[:budget_b])
        hit_eos = bool(b1b_hit_eos and len(full_ids) <= budget_b)
        content_len = len(s1_ids)
        while content_len > 0 and s1_ids[content_len - 1] in self.eos_ids:
            content_len -= 1
        full_lat = float(b1b_s1.get("latency_ms", 0.0))
        s1_lat = full_lat * (len(s1_ids) / max(1, len(full_ids)))
        return {
            "raw_text": self.tok.decode(s1_ids, skip_special_tokens=False),
            "total_gen_tokens": len(s1_ids),
            "content_tokens": content_len,
            "hit_eos": hit_eos,
            "step_entropies": [],
            "latency_ms": s1_lat,
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

    def _prefix_inertia_flag(
        self,
        res: dict[str, Any],
        clean_fast: str,
        concise_probe: bool,
        horizon_k: int,
    ) -> int:
        """Computes Thinking Inertia on the first <= horizon_k probe tokens for GlimpRouter ablation."""
        gen_ids = res.get("gen_ids")
        if isinstance(gen_ids, list) and gen_ids:
            sub_ids = [t for t in gen_ids[:horizon_k] if t not in self.eos_ids]
            sub_txt = strip_special_end_tokens(self.tok.decode(sub_ids, skip_special_tokens=False)).lower()
            n_toks = len(sub_ids)
        else:
            words = clean_fast.lower().split()
            sub_txt = " ".join(words[:horizon_k])
            n_toks = min(int(res["content_tokens"]), horizon_k)
        return int((concise_probe and n_toks > min(16, horizon_k)) or any(w in sub_txt for w in INERTIA_MARKERS))

    def run_gate2_fast_probe(
        self, prompt: str, g1: Gate1Output | None = None, probe_tokens: int = 32
    ) -> dict[str, Any]:
        """Runs <=32-token non-thinking uncertainty probe with sub-trajectory GlimpRouter horizons."""
        g1 = g1 or run_gate1_lexical(prompt)
        # Only prepend format prefixes on clear MC / math queries; empty prefix keeps IFEval clean
        is_mc = "(A)" in prompt and "(B)" in prompt
        is_pure_math = bool(g1.phi_sym and not g1.phi_prec and not g1.phi_plan)
        ans_prefix = "Answer: " if is_mc else ("#### " if is_pure_math else "")

        base_pref = self.build_prompt_prefix(prompt, enable_thinking=False) + ans_prefix
        use_stateful = type(self)._decode_raw is CBARRDualModeEngine._decode_raw
        res = (
            self._decode_raw_stateful(base_pref, max_new_tokens=probe_tokens, log_entropies=True)
            if use_stateful
            else self._decode_raw(base_pref, max_new_tokens=probe_tokens, log_entropies=True)
        )
        ents = res["step_entropies"]
        clean_fast = strip_special_end_tokens(res["raw_text"]).strip()
        low = clean_fast.lower()

        # Detect Thinking Inertia: verbose non-thinking starts on MC/math or CoT discourse cues
        concise_probe = is_mc or is_pure_math
        inertia = int(
            (concise_probe and res["content_tokens"] > 16)
            or any(w in low for w in INERTIA_MARKERS)
        )

        def _h_stats(k: int) -> tuple[float, float]:
            sub = ents[:k]
            if not sub:
                return 0.0, 0.0
            return float(np.mean(sub)), float(np.max(sub))

        h4_mean, h4_max = _h_stats(4)
        h8_mean, h8_max = _h_stats(8)
        h16_mean, h16_max = _h_stats(16)
        h_first = float(ents[0]) if ents else 0.0

        return {
            "base_pref": base_pref,
            "ans_prefix": ans_prefix,
            "probe_raw_text": res["raw_text"],
            "fast_answer": (ans_prefix + clean_fast).strip(),
            "hit_eos": res["hit_eos"],
            "T_probe": res["total_gen_tokens"],
            "m_content": res["content_tokens"],
            "H_first": h_first,
            "H_1": h_first,
            "H_4_mean": h4_mean,
            "H_4_max": h4_max,
            "H_8_mean": h8_mean,
            "H_8_max": h8_max,
            "H_16_mean": h16_mean,
            "H_16_max": h16_max,
            "H_mean": float(np.mean(ents)) if ents else 0.0,
            "H_max": float(np.max(ents)) if ents else 0.0,
            "step_entropies": ents,
            "I_inertia": inertia,
            "I_inertia_4": self._prefix_inertia_flag(res, clean_fast, concise_probe, 4),
            "I_inertia_8": self._prefix_inertia_flag(res, clean_fast, concise_probe, 8),
            "I_inertia_16": self._prefix_inertia_flag(res, clean_fast, concise_probe, 16),
            "latency_ms": res["latency_ms"],
            "_kv_state": res.get("_kv_state"),
        }

    def complete_fast_probe(
        self, probe_out: dict[str, Any], max_ans_tokens: int = 1024
    ) -> dict[str, Any]:
        """Completes a non-escalated Gate-2 probe if 32 tokens did not yet reach EOS (Patch D14: 1024)."""
        kv_state = probe_out.pop("_kv_state", None)
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
        s_rem = self._decode_continue_from_state(
            kv_state=kv_state,
            fallback_prompt_str=cont_pref,
            append_str="",
            max_new_tokens=rem,
        )
        if kv_state is not None and kv_state.get("gen_ids") is not None and s_rem.get("gen_ids") is not None:
            raw_combined = self.tok.decode(
                list(kv_state["gen_ids"]) + list(s_rem["gen_ids"]),
                skip_special_tokens=False,
            )
        else:
            raw_combined = probe_out["probe_raw_text"] + s_rem["raw_text"]
        combined = probe_out["ans_prefix"] + strip_special_end_tokens(raw_combined)
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
        precomputed_b1b_s1: dict[str, Any] | None = None,
        out_s1_cache: dict[str, Any] | None = None,
        s2_memo: dict[tuple[str, int], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Executes capped <think> (B in {64, 128, 250}) + P_TERM + Post/Pre-</think> re-anchoring."""
        pref = self.build_prompt_prefix(prompt, enable_thinking=True)
        s1 = self._slice_precomputed_s1(precomputed_b1b_s1, budget_b)
        can_reuse_s1_kv = (
            s1 is None
            and precomputed_b1b_s1 is None
            and s2_memo is None
            and out_s1_cache is None
            and type(self)._decode_raw is CBARRDualModeEngine._decode_raw
        )
        if s1 is None:
            s1 = (
                self._decode_raw_stateful(pref, max_new_tokens=budget_b, log_entropies=False)
                if can_reuse_s1_kv
                else self._decode_raw(pref, max_new_tokens=budget_b, log_entropies=False)
            )
        s1_kv_state = s1.pop("_kv_state", None)
        if out_s1_cache is not None:
            out_s1_cache.clear()
            out_s1_cache.update(s1)
        raw1 = s1["raw_text"]
        clean_raw1 = re.sub(r"^\s*<think>\s*", "", raw1)

        if "</think>" in clean_raw1:
            t_think, t_after = clean_raw1.split("</think>", 1)
            t_think_toks = len(self.tok.encode(t_think, add_special_tokens=False))
            # Branch 1: model closed </think> and finished answer cleanly with no re-anchor needed
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
            # Branch 2: closed </think> inside budget_b, but answer hit budget_b before EOS -> continue answer
            if not anchor_clause and not s1["hit_eos"]:
                after_clean_toks = max(0, s1["total_gen_tokens"] - t_think_toks)
                rem_ans = max(1, max_ans_tokens - after_clean_toks)
                cont2_pref = pref + clean_raw1
                memo2_key = (cont2_pref, int(rem_ans))
                if s2_memo is not None and memo2_key in s2_memo:
                    s2 = s2_memo[memo2_key]
                else:
                    s2 = self._decode_continue_from_state(
                        kv_state=s1_kv_state if not raw1.lstrip().startswith("<think>") else None,
                        fallback_prompt_str=cont2_pref,
                        append_str="",
                        max_new_tokens=rem_ans,
                    )
                    if s2_memo is not None:
                        s2_memo[memo2_key] = s2
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

            # Branch 3: closed </think> naturally, but we discard post-</think> draft and re-anchor
            s1_kv_state = None
            t_wasted = max(0, s1["total_gen_tokens"] - t_think_toks)
            close_block, vis_prefix = self._build_reanchor_for_engine(
                False, anchor_clause, anchor_locus
            )
            cont_prefix = pref + t_think + close_block
            truncated = False
        else:
            # Unbudgeted B1b mode returns think_exhausted=True when B=2048 runs out inside <think>
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
            # Budgeted M2 mode: force-close <think> via P_TERM + optional constraint reminder
            t_think = strip_special_end_tokens(clean_raw1)
            t_think_toks = s1["total_gen_tokens"]
            t_wasted = 0
            close_block, vis_prefix = self._build_reanchor_for_engine(
                True, anchor_clause, anchor_locus
            )
            cont_prefix = pref + t_think + close_block
            truncated = True

        t_inj_toks = len(self.tok.encode(close_block, add_special_tokens=False))
        memo_key = (cont_prefix, int(max_ans_tokens))
        if s2_memo is not None and memo_key in s2_memo:
            s2 = s2_memo[memo_key]
        else:
            can_cont_trunc = (
                s1_kv_state is not None
                and truncated
                and not s1["hit_eos"]
                and not raw1.lstrip().startswith("<think>")
            )
            s2 = self._decode_continue_from_state(
                kv_state=s1_kv_state if can_cont_trunc else None,
                fallback_prompt_str=cont_prefix,
                append_str=close_block,
                max_new_tokens=max_ans_tokens,
            )
            if s2_memo is not None:
                s2_memo[memo_key] = s2
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

        # Each mode holds E_hurt item & cached_t_think fixed to isolate recency vs. interference:
        if mode == "A1_strip":
            # Delete <think> completely (0 distance, 0 interference)
            clamped = self.build_prompt_prefix(prompt, enable_thinking=False)
        elif mode in ("B2a_post_anchor", "B2b_post_raw"):
            # Keep <think> intact; restore post-</think> recency cleanly (B2b keeps visible wrapper in a_final)
            locus = "post_stripped" if mode == "B2a_post_anchor" else "post_raw"
            close_block, _ = self._build_reanchor_for_engine(False, g1.anchor_clause, locus)
            clamped = f"{pref}{cached_t_think}{close_block}"
        elif mode == "B1_pre_anchor":
            # Test pre-</think> injection locus before the closing tag
            clamped = (
                f"{pref}{cached_t_think}\n"
                f"Constraint check: strictly follow: {g1.anchor_clause}\n</think>\n"
            )
        elif mode == "A2_math_swap":
            # Match token length of cached_t_think using off-task GSM8K math rationales
            n_toks = len(self.tok.encode(cached_t_think, add_special_tokens=False))
            repeats = (n_toks // max(1, len(math_filler_pool_ids))) + 1
            pooled = (math_filler_pool_ids * repeats)[:n_toks]
            matched_math = self.tok.decode(pooled, skip_special_tokens=True)
            clamped = f"{pref}{matched_math}\n</think>\n"
        elif mode == "B3_early_cut_no_anchor":
            # Truncate at first reflection pivot (or midpoint) without reminder to halve context distance
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
