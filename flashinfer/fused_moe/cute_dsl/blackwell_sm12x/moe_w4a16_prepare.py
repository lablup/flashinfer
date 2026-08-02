"""Local NVFP4/BF16 weight preparation for the CuTeDSL W4A16 path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .moe_w4a16_host import (
    W4A16PackedBuffers,
    make_w4a16_packed_buffers as _make_w4a16_packed_buffers,
    normalize_expert_block_scales,
    reorder_w13_to_gate_up,
    unswizzle_expert_scales,
    validate_w4a16_packed_inputs,
)
from .moe_source_format import (  # noqa: F401  (re-exported for compatibility)
    _SOURCE_FORMATS,
    _normalize_source_format,
    _source_global_scale,
)


_PACKED_TILE_SIZE = 16
_PACKED_TILE_N_SIZE = 64
_PACK_FACTOR_4BIT = 8


@dataclass(frozen=True)
class W4A16PackedWeights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    source_format: str = "modelopt"


def _make_workspace(
    device: torch.device, *, max_blocks_per_sm: int = 4
) -> torch.Tensor:
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    return torch.zeros(
        (sms * int(max_blocks_per_sm) + 2,),
        dtype=torch.int32,
        device=device,
    )


def _scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def _permute_packed_scales(
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    scale_perm, scale_perm_single = _scale_perms()
    if group_size < size_k and group_size != -1:
        scales = scales.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        scales = scales.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return scales.reshape((-1, size_n)).contiguous()


def _e8m0_expert_scales_to_nvfp4(
    blockscale: torch.Tensor,
    *,
    num_experts: int,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate plain MXFP4 ``e8m0`` 32-group scales into the
    ``(e4m3 16-group, fp32 per-expert global)`` convention consumed by the
    W4A16 packing path.

    e8m0 scales are pure powers of two, so with the per-expert global chosen
    as ``2**(emax - 8)`` every block ratio is a power of two in ``(0, 256]``
    — exactly representable in ``e4m3``. The downstream packing
    (``_process_nvfp4_packed_scales``) flushes ratios below ``2**-6`` to
    zero, so the translation is lossless for per-expert scale spreads up to
    ``2**14``; blocks more than 14 octaves below their expert's max scale
    (not observed in real checkpoints) flush to zero there, mirroring what
    the modelopt path does to equally extreme block scales. Each 32-group
    scale is duplicated into its two 16-group halves.
    """
    if cols % 32 != 0:
        raise ValueError(
            f"fp4_e8m0_k32 block scales require cols % 32 == 0, got {cols}"
        )
    if blockscale.dtype == torch.uint8:
        exp_bytes = blockscale
    elif blockscale.dtype == torch.float8_e8m0fnu:
        exp_bytes = blockscale.view(torch.uint8)
    else:
        raise TypeError(
            "fp4_e8m0_k32 block scales must be torch.float8_e8m0fnu (or its "
            f"uint8 view), got {blockscale.dtype}"
        )
    expected = (int(num_experts), int(rows), int(cols) // 32)
    if tuple(exp_bytes.shape) != expected:
        raise ValueError(
            f"expected plain fp4_e8m0_k32 block scales of shape {expected}, "
            f"got {tuple(exp_bytes.shape)}"
        )
    two = torch.tensor(2.0, dtype=torch.float32, device=exp_bytes.device)
    exp = exp_bytes.to(torch.int32) - 127
    emax = exp.amax(dim=(1, 2))
    global_scale = torch.pow(two, (emax - 8).to(torch.float32)).contiguous()
    # e4m3 bottoms out at 2**-9 (subnormal); clamp so no block flushes to 0.
    block_exp = (exp - emax.view(-1, 1, 1) + 8).clamp_min(-9)
    block = torch.pow(two, block_exp.to(torch.float32)).to(torch.float8_e4m3fn)
    return block.repeat_interleave(2, dim=2).contiguous(), global_scale


def _nvfp4_compute_scale_factor(
    packed_scales: torch.Tensor,
    a_dtype: torch.dtype,
) -> float:
    if a_dtype == torch.float16:
        return 1.0
    ws_float = packed_scales.float() * (2**7)
    nonzero_mask = ws_float > 0
    if bool(nonzero_mask.any().item()):
        max_val = ws_float[nonzero_mask].max()
        max_scalar = float(max_val.item())
        if max_scalar < 448 * (2**7):
            return float(2 ** math.floor(math.log2((448 * (2**7)) / max_scalar)))
    return 1.0


def _process_nvfp4_packed_scales(
    packed_scales: torch.Tensor,
    *,
    scale_factor: float,
) -> torch.Tensor:
    packed_scales = packed_scales.to(torch.float16)
    packed_scales = packed_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        packed_scales.size(0),
        -1,
    )
    if scale_factor > 1.0:
        packed_scales = (packed_scales.float() * scale_factor).to(torch.float16)
    packed_scales = packed_scales * (2**7)
    packed_scales[packed_scales < 2] = 0
    packed_scales = packed_scales.view(torch.int16) << 1
    packed_scales = packed_scales.view(torch.float8_e4m3fn)
    return packed_scales[:, 1::2].contiguous()


def _process_nvfp4_packed_global_scale(
    global_scale: torch.Tensor,
    *,
    a_dtype: torch.dtype,
) -> torch.Tensor:
    if a_dtype == torch.float16:
        target_exponent = 5
    elif a_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise TypeError(f"unsupported W4A16 activation dtype {a_dtype}")
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def _repack_4bit_no_perm(
    qweight_i32: torch.Tensor, *, size_k: int, size_n: int
) -> torch.Tensor:
    """Pack 4-bit weights into the W4A16 A16 kernel layout."""
    if qweight_i32.dtype != torch.int32:
        raise TypeError("qweight_i32 must be torch.int32")
    if tuple(qweight_i32.shape) != (size_k // _PACK_FACTOR_4BIT, size_n):
        raise ValueError(
            f"expected qweight shape {(size_k // _PACK_FACTOR_4BIT, size_n)}, "
            f"got {tuple(qweight_i32.shape)}"
        )
    if size_k % _PACKED_TILE_SIZE != 0 or size_n % _PACKED_TILE_N_SIZE != 0:
        raise ValueError(
            f"W4A16 repack requires K,N multiples of 16,64; got {size_k},{size_n}"
        )

    k_tiles = size_k // _PACKED_TILE_SIZE
    n_tiles = size_n // _PACKED_TILE_N_SIZE
    tiles = (qweight_i32.to(torch.int64) & 0xFFFFFFFF).view(
        k_tiles,
        2,
        n_tiles,
        _PACKED_TILE_N_SIZE,
    )
    flat = tiles.permute(0, 2, 1, 3).reshape(k_tiles, n_tiles, 2 * _PACKED_TILE_N_SIZE)

    device = qweight_i32.device
    out_pos = torch.arange(128, device=device, dtype=torch.long)
    th_id = out_pos // 4
    warp_id = out_pos % 4
    tc_col = th_id // 4
    tc_row = (th_id % 4) * 2
    offsets = torch.tensor([0, 1, 8, 9], device=device, dtype=torch.long)
    pack_idx = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device, dtype=torch.long)

    elem = tc_row[:, None] + offsets[None, :]
    row = elem // _PACK_FACTOR_4BIT
    pos = elem % _PACK_FACTOR_4BIT
    col1 = (warp_id * 16 + tc_col)[:, None].expand(-1, 4)
    col2 = col1 + 8
    source_index = torch.cat(
        [row * _PACKED_TILE_N_SIZE + col1, row * _PACKED_TILE_N_SIZE + col2],
        dim=1,
    )[:, pack_idx]
    source_shift = torch.cat([pos, pos], dim=1)[:, pack_idx] * 4

    result = torch.zeros((k_tiles, n_tiles, 128), device=device, dtype=torch.int64)
    for slot in range(8):
        gathered = flat.gather(
            2,
            source_index[:, slot].view(1, 1, 128).expand(k_tiles, n_tiles, 128),
        )
        nibble = (gathered >> source_shift[:, slot].view(1, 1, 128)) & 0xF
        result |= nibble << (slot * 4)

    return result.to(torch.int32).reshape(k_tiles, n_tiles * 128).contiguous()


def _repack_weight(weight: torch.Tensor, *, size_k: int, size_n: int) -> torch.Tensor:
    pieces = []
    for expert in range(weight.shape[0]):
        qweight = weight[expert].view(torch.int32).T.contiguous()
        pieces.append(_repack_4bit_no_perm(qweight, size_k=size_k, size_n=size_n))
    return torch.stack(pieces, dim=0).contiguous()


def _permute_nvfp4_scales(
    scales: torch.Tensor,
    global_scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    a_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = scales.to(a_dtype)
    combined_scale_factor = _nvfp4_compute_scale_factor(scales, a_dtype)
    pieces = []
    for expert in range(scales.shape[0]):
        packed_scales = _permute_packed_scales(
            scales[expert].T,
            size_k=size_k,
            size_n=size_n,
            group_size=16,
        )
        pieces.append(
            _process_nvfp4_packed_scales(
                packed_scales,
                scale_factor=combined_scale_factor,
            )
        )
    packed_scales = torch.stack(pieces, dim=0).contiguous()
    packed_global = _process_nvfp4_packed_global_scale(
        global_scales,
        a_dtype=a_dtype,
    ).to(torch.float32)
    packed_global = packed_global / combined_scale_factor
    return packed_scales, packed_global.contiguous()


def prepare_w4a16_packed_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor | None,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor | None,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    source_format: str = "modelopt",
) -> W4A16PackedWeights:
    source_format = _normalize_source_format(source_format)
    is_e8m0 = source_format == "fp4_e8m0_k32"
    if is_e8m0:
        if w13_global_scale is not None or w2_global_scale is not None:
            raise ValueError(
                "fp4_e8m0_k32 checkpoints carry no global scales; pass "
                "w13_global_scale=None and w2_global_scale=None (per-expert "
                "globals are synthesized from the e8m0 block scales)"
            )
        # Shapes are needed before validation because the synthesized fp32
        # globals participate in it.
        e8m0_num_experts = int(w13_fp4.shape[0])
        e8m0_hidden = int(w2_fp4.shape[1])
        e8m0_intermediate = int(w2_fp4.shape[2] * 2)
        e8m0_w13_rows = int(w13_fp4.shape[1])
        w13_scale_plain, w13_global_scale = _e8m0_expert_scales_to_nvfp4(
            w13_blockscale,
            num_experts=e8m0_num_experts,
            rows=e8m0_w13_rows,
            cols=e8m0_hidden,
        )
        w2_scale_plain, w2_global_scale = _e8m0_expert_scales_to_nvfp4(
            w2_blockscale,
            num_experts=e8m0_num_experts,
            rows=e8m0_hidden,
            cols=e8m0_intermediate,
        )
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated

    w13 = w13_fp4
    if is_e8m0:
        # Plain per-expert scales; the fused w13 rows are already in
        # kernel-native gate-first order (vLLM's MXFP4 layout), so neither
        # unswizzling nor the up_gate -> gate_up reorder applies.
        w13_scale = w13_scale_plain
        w2_scale = w2_scale_plain
    else:
        w13_scale = unswizzle_expert_scales(
            normalize_expert_block_scales(
                w13_blockscale,
                num_experts=num_experts,
                rows=w13_rows,
                cols=hidden_size,
            ),
            rows=w13_rows,
            cols=hidden_size,
        )
        if is_gated:
            w13, w13_scale = reorder_w13_to_gate_up(
                w13,
                w13_scale,
                intermediate_size=intermediate_size,
            )

        w2_scale = unswizzle_expert_scales(
            normalize_expert_block_scales(
                w2_blockscale,
                num_experts=num_experts,
                rows=hidden_size,
                cols=intermediate_size,
            ),
            rows=hidden_size,
            cols=intermediate_size,
        )

    packed_w13 = _repack_weight(w13.contiguous(), size_k=hidden_size, size_n=w13_rows)
    packed_w2 = _repack_weight(
        w2_fp4.contiguous(), size_k=intermediate_size, size_n=hidden_size
    )
    w13_global_scale = _source_global_scale(
        w13_global_scale,
        source_format=source_format,
    )
    w2_global_scale = _source_global_scale(
        w2_global_scale,
        source_format=source_format,
    )
    packed_w13_scale, packed_w13_global_scale = _permute_nvfp4_scales(
        w13_scale,
        w13_global_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        a_dtype=params_dtype,
    )
    packed_w2_scale, packed_w2_global_scale = _permute_nvfp4_scales(
        w2_scale,
        w2_global_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        a_dtype=params_dtype,
    )

    return W4A16PackedWeights(
        w13=packed_w13,
        w13_scale=packed_w13_scale,
        w13_global_scale=packed_w13_global_scale,
        w2=packed_w2,
        w2_scale=packed_w2_scale,
        w2_global_scale=packed_w2_global_scale,
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format=source_format,
    )


def make_w4a16_packed_buffers(
    prepared: W4A16PackedWeights,
    *,
    m: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    route_num_experts: int | None = None,
) -> W4A16PackedBuffers:
    return _make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=dtype,
        device=device,
        route_num_experts=route_num_experts,
    )


__all__ = [
    "W4A16PackedBuffers",
    "W4A16PackedWeights",
    "make_w4a16_packed_buffers",
    "prepare_w4a16_packed_weights",
]
