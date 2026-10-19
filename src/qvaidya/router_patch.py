"""MoE router compatibility patch (Section 3.2.1).

Depending on the ``transformers`` version, the Llama-4 MoE block and its
router return either a single tensor or a tuple ``(output, router_logits)``,
and callers that expect one form break on the other once the model is wrapped
by bitsandbytes / PEFT. The patch below unpacks router outputs defensively so
the forward pass works with both conventions.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_router_patch() -> bool:
    """Patch ``Llama4TextMoe.forward`` to tolerate tensor or tuple router returns."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from transformers.models.llama4 import modeling_llama4 as m
    except ImportError:
        logger.warning("transformers has no Llama-4 implementation; router patch skipped.")
        return False

    moe_cls = getattr(m, "Llama4TextMoe", None)
    if moe_cls is None:
        logger.warning("Llama4TextMoe not found; router patch skipped.")
        return False

    original_forward = moe_cls.forward

    @functools.wraps(original_forward)
    def forward(self, hidden_states, *args, **kwargs):
        router = self.router
        router_forward = router.forward

        def safe_router(x, *a, **kw):
            out = router_forward(x, *a, **kw)
            if isinstance(out, (tuple, list)) and len(out) == 1:
                return out[0]
            return out

        router.forward = safe_router
        try:
            return original_forward(self, hidden_states, *args, **kwargs)
        finally:
            router.forward = router_forward

    moe_cls.forward = forward

    _PATCHED = True
    logger.info("Applied Llama-4 MoE router compatibility patch.")
    return True
