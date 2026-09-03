"""transformers backend: local generation plus scoring-position cell masses.

For EDFL scoring, the backend runs one forward pass per prompt and reads the
next-token distribution. For benchmark generation, it can also call
``model.generate`` when logprobs are not requested.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .openai_backend import TextResult

_CACHE: Dict[str, Any] = {}


def _load(model_id: str, device: Optional[str], dtype: Optional[str]):
    key = f"{model_id}|{device}|{dtype}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("local backend needs torch and transformers") from exc

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    td = getattr(torch, dtype) if dtype else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=td).to(device).eval()
    _CACHE[key] = (tok, model, device, torch)
    return _CACHE[key]


class LocalBackend:
    """Reads the scoring position from local weights.

    ``chat_template`` decides the frame, and the frame decides which token form
    the cells take: through a template a model wants ``"A"``, on a raw prompt the
    same model wants ``" A"``. Getting it wrong collapses ``d``, so this is not a
    cosmetic switch -- see ``ValidityReport.d_evidence``.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        opts = dict(getattr(cfg, "options", None) or {})
        self.model_id = opts.get("model_id") or getattr(cfg, "base_url", None)
        if not self.model_id:
            raise ValueError("local backend needs options={'model_id': ...}")
        self.batch = int(opts.get("batch", 16))
        self.topk = int(opts.get("topk", 64))
        self.chat_template = bool(opts.get("chat_template", True))
        self.max_input_tokens = opts.get("max_input_tokens")
        self.tok, self.model, self.device, self._torch = _load(
            self.model_id, opts.get("device"), opts.get("dtype"))

    def _render(self, prompt: str, instructions: str) -> str:
        if not self.chat_template:
            return prompt
        msgs = ([{"role": "system", "content": instructions}] if instructions else [])
        msgs.append({"role": "user", "content": prompt})
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        except Exception:
            # Older jinja2 cannot compile the template; ChatML by hand is exact
            # for the Qwen family and harmless elsewhere as a raw prompt.
            sysmsg = f"<|im_start|>system\n{instructions}<|im_end|>\n" if instructions else ""
            return f"{sysmsg}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _score_batch(self, *, prompts: Sequence[str], instructions: str = "") -> List[TextResult]:
        torch, out = self._torch, []
        rendered = [self._render(p, instructions) for p in prompts]
        with torch.no_grad():
            for i in range(0, len(rendered), self.batch):
                enc = self.tok(rendered[i:i + self.batch], return_tensors="pt",
                               padding=True, add_special_tokens=False).to(self.device)
                lp = torch.nn.functional.log_softmax(
                    self.model(**enc).logits[:, -1, :].float(), dim=-1)
                vals, idx = lp.topk(self.topk, dim=-1)
                for v, ix in zip(vals.tolist(), idx.tolist()):
                    top = [{"token": self.tok.decode([t]), "logprob": val}
                           for val, t in zip(v, ix)]
                    out.append(TextResult(
                        text=top[0]["token"], response_id=None,
                        logprobs=[{"token": top[0]["token"],
                                   "logprob": top[0]["logprob"], "top_logprobs": top}]))
        return out

    def _generate_batch(
        self,
        *,
        prompts: Sequence[str],
        instructions: str = "",
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[TextResult]:
        torch, out = self._torch, []
        rendered = [self._render(p, instructions) for p in prompts]
        with torch.no_grad():
            for i in range(0, len(rendered), self.batch):
                enc = self.tok(
                    rendered[i:i + self.batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=bool(self.max_input_tokens),
                    max_length=self.max_input_tokens,
                    add_special_tokens=False,
                ).to(self.device)
                input_len = enc["input_ids"].shape[1]
                kwargs = {
                    "max_new_tokens": int(max_output_tokens),
                    "pad_token_id": self.tok.pad_token_id,
                    "eos_token_id": self.tok.eos_token_id,
                    "repetition_penalty": float(repetition_penalty),
                }
                if float(temperature) > 0.0:
                    kwargs.update({"do_sample": True, "temperature": float(temperature)})
                    if top_p is not None:
                        kwargs["top_p"] = float(top_p)
                    if top_k is not None:
                        kwargs["top_k"] = int(top_k)
                else:
                    kwargs.update({"do_sample": False})
                seq = self.model.generate(**enc, **kwargs)
                for row in seq:
                    text = self.tok.decode(row[input_len:], skip_special_tokens=True)
                    out.append(TextResult(text=text, response_id=None, logprobs=None))
        return out

    def call_text_batch(self, *, prompts: Sequence[str], instructions: str = "",
                        include_logprobs: bool = False, max_output_tokens: int = 64,
                        temperature: float = 0.0, repetition_penalty: float = 1.0,
                        top_p: Optional[float] = None, top_k: Optional[int] = None,
                        **_kw: Any) -> List[TextResult]:
        if include_logprobs:
            return self._score_batch(prompts=prompts, instructions=instructions)
        return self._generate_batch(
            prompts=prompts,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
        )

    def call_text(self, *, prompt: str, **kw: Any) -> TextResult:
        return self.call_text_batch(prompts=[prompt], **kw)[0]

    def reset_state(self) -> None:
        return None
