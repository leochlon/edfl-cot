"""transformers backend: one forward pass per prompt, exact cell masses.

The hosted path reads a top-k list, so a cell whose mass is low falls off the
table and the readout is discarded -- 59 of 64 draws on one measured item. Here
the softmax is complete, so ``d`` is measured rather than checked and the
instrument-failure branch of ``_readout_from_logprobs`` is unreachable.
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

    def call_text_batch(self, *, prompts: Sequence[str], instructions: str = "",
                        max_output_tokens: int = 1, temperature: float = 0.0,
                        **_kw: Any) -> List[TextResult]:
        """Score one position, or generate when more than one token is asked for.

        ``open_answer`` requests 384 tokens and agentic callers ask for hundreds;
        this backend used to swallow ``max_output_tokens`` in ``**_kw`` and
        return a single token regardless, so those paths came back truncated to
        one character with no error raised.
        """
        if int(max_output_tokens) > 1:
            return self._generate(prompts, instructions, int(max_output_tokens),
                                  float(temperature))
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

    def _generate(self, prompts, instructions, max_new, temperature):
        torch, out = self._torch, []
        rendered = [self._render(p, instructions) for p in prompts]
        with torch.no_grad():
            for i in range(0, len(rendered), self.batch):
                enc = self.tok(rendered[i:i + self.batch], return_tensors="pt",
                               padding=True, add_special_tokens=False).to(self.device)
                g = self.model.generate(
                    **enc, max_new_tokens=max_new, do_sample=temperature > 0,
                    temperature=temperature or None,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
                for row in g[:, enc["input_ids"].shape[1]:]:
                    out.append(TextResult(
                        text=self.tok.decode(row, skip_special_tokens=True),
                        response_id=None, logprobs=None))
        return out

    def call_text(self, *, prompt: str, **kw: Any) -> TextResult:
        return self.call_text_batch(prompts=[prompt], **kw)[0]

    def reset_state(self) -> None:
        return None
