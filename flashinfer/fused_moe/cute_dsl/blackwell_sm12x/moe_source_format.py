"""Checkpoint source-format conventions shared by the b12x MoE paths.

Two checkpoint scale conventions are supported:

- ``"modelopt"``: per-expert global scales are direct dequant multipliers
  (ModelOpt's ``weight_scale_2`` / ``input_scale``).
- ``"compressed_tensors"``: per-expert global scales are stored as the
  reciprocal (``weight_global_scale = 448 * 6 / amax``); they must be
  inverted to become multipliers.

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
}


def _normalize_source_format(source_format: str) -> str:
    try:
        return _SOURCE_FORMATS[source_format.lower()]
    except KeyError as exc:
        raise ValueError(
            "source_format must be one of 'modelopt' or 'compressed_tensors', "
            f"got {source_format!r}"
        ) from exc


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
