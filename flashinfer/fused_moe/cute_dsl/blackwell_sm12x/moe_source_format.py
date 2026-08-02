"""Checkpoint source-format conventions shared by the b12x MoE paths.

Three checkpoint scale conventions are supported:

- ``"modelopt"``: per-expert global scales are direct dequant multipliers
  (ModelOpt's ``weight_scale_2`` / ``input_scale``).
- ``"compressed_tensors"``: per-expert global scales are stored as the
  reciprocal (``weight_global_scale = 448 * 6 / amax``); they must be
  inverted to become multipliers.
- ``"fp4_e8m0_k32"`` (alias ``"mxfp4"``): MXFP4 checkpoints — e2m1 values
  with plain (unswizzled) per-32-group ``float8_e8m0fnu`` block scales and
  NO global scales (pass ``None``; per-expert globals are synthesized
  during W4A16 preparation, see ``_e8m0_expert_scales_to_nvfp4``).
  W4A16-only. The fused ``w13`` rows are expected in kernel-native
  gate-first order (``[gate; up]``, vLLM's MXFP4 layout), so no reorder is
  applied.

Translation happens at weight-preparation time only. The nvfp4 (W4A4)
launch path treats ``source_format`` as pass-through provenance metadata
and requires tensors already in kernel convention (see
``launch_sm120_moe``).

Note: ``_source_global_scale`` validates compressed-tensors global scales
as finite and positive before taking the reciprocal. This is intentional
hardening over the pre-extraction W4A16 helper, which silently produced
``inf`` dequant scales for zero/nonfinite checkpoint scales.
"""

from __future__ import annotations

import torch

_SOURCE_FORMATS = {
    "modelopt": "modelopt",
    "compressed_tensors": "compressed_tensors",
    "compressed-tensors": "compressed_tensors",
    "ct": "compressed_tensors",
    "fp4_e8m0_k32": "fp4_e8m0_k32",
    "mxfp4": "fp4_e8m0_k32",
}


def _normalize_source_format(source_format: str) -> str:
    try:
        return _SOURCE_FORMATS[source_format.lower()]
    except KeyError as exc:
        raise ValueError(
            "source_format must be one of 'modelopt', 'compressed_tensors' "
            f"or 'fp4_e8m0_k32', got {source_format!r}"
        ) from exc


def _validate_alphas_for_source_format(
    w1_alpha: torch.Tensor | None,
    w2_alpha: torch.Tensor | None,
    *,
    source_format: str,
) -> None:
    """Enforce the per-format global-scale (alpha) presence contract.

    ``source_format`` must already be normalized. MXFP4 checkpoints carry no
    global scales, so ``fp4_e8m0_k32`` requires both alphas to be ``None``;
    every other format requires both to be tensors.
    """
    if source_format == "fp4_e8m0_k32":
        if w1_alpha is not None or w2_alpha is not None:
            raise ValueError(
                "source_format='fp4_e8m0_k32' (MXFP4) carries no global "
                "scales; pass w1_alpha=None and w2_alpha=None (per-expert "
                "globals are synthesized during W4A16 preparation)."
            )
    elif w1_alpha is None or w2_alpha is None:
        raise ValueError(
            "w1_alpha and w2_alpha are required for "
            f"source_format={source_format!r}; only 'fp4_e8m0_k32' (MXFP4) "
            "checkpoints omit global scales."
        )


def _source_global_scale(
    global_scale: torch.Tensor, *, source_format: str
) -> torch.Tensor:
    if source_format == "compressed_tensors":
        gs_float = global_scale.to(torch.float32)
        if not bool(torch.isfinite(gs_float).all()) or not bool((gs_float > 0).all()):
            raise ValueError(
                "compressed_tensors global scales must be finite and positive "
                "(the reciprocal of a zero/nonfinite scale would corrupt the "
                "dequant scale)."
            )
        return (1.0 / gs_float).contiguous()
    return global_scale.contiguous()
