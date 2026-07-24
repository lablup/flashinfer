"""Triton kernel for compacting MoE routing IDs (ported from b12x).

Remaps global expert IDs to dense local indices (0, 1, 2, ...) for the
micro MoE kernel, which expects pre-compacted routing.

Expert parallelism: when a shard is given (``local_expert_offset`` /
``num_local_experts``), ids are first shifted into the shard's index space
and pairs outside ``[0, num_local_experts)`` get compact id ``-1`` (the
micro kernel drops them); ``weight_expert_ids`` then maps compact slots to
SHARD-RELATIVE weight indices, matching the rank's sliced weight tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_topk_ids_kernel(
    topk_ids_ptr,
    compact_topk_ids_ptr,
    weight_expert_ids_ptr,
    active_expert_count_ptr,
    total_pairs,
    local_expert_offset,
    num_local_experts,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(topk_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)
    rel = ids - local_expert_offset
    # Pairs routed outside this rank's shard are dropped (-1 sentinel).
    local = valid & (rel >= 0) & (rel < num_local_experts)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_local = local[:, None]
    col_local = local[None, :]

    same_id = rel[:, None] == rel[None, :]
    prior_same = row_local & col_local & same_id & (col_slots < row_slots)

    first_flags = local & (tl.sum(prior_same.to(tl.int32), axis=1) == 0)
    first_prefix = tl.cumsum(first_flags.to(tl.int32), axis=0)

    prior_slots = tl.where(prior_same, col_slots, BLOCK)
    first_match = tl.min(prior_slots, axis=1)
    first_slot = tl.where(first_match < BLOCK, first_match, pair_slots)
    first_slot_mask = col_slots == first_slot[:, None]
    compact_id = tl.sum(tl.where(first_slot_mask, first_prefix[None, :], 0), axis=1) - 1
    compact_id = tl.where(local, compact_id, -1)

    tl.store(compact_topk_ids_ptr + pair_slots, compact_id, mask=valid)
    tl.store(weight_expert_ids_ptr + compact_id, rel, mask=local & first_flags)

    active_expert_count = tl.sum(first_flags.to(tl.int32), axis=0)
    tl.store(active_expert_count_ptr, active_expert_count)


def compact_topk_ids(
    topk_ids: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    weight_expert_ids: torch.Tensor,
    active_expert_count: torch.Tensor,
    *,
    local_expert_offset: int = 0,
    num_local_experts: int | None = None,
) -> None:
    """Remap global expert IDs to dense contiguous local indices.

    Args:
        topk_ids: [total_pairs] int32 — flattened global expert IDs.
        compact_topk_ids: [total_pairs] int32 — output: dense local indices,
            ``-1`` for pairs outside this rank's expert shard.
        weight_expert_ids: [>=total_pairs] int32 — output: compact slot ->
            shard-relative weight expert index (== global id when no shard).
        active_expert_count: [1] int32 — output: number of unique local experts.
        local_expert_offset: start of this rank's contiguous expert shard.
        num_local_experts: shard width; ``None`` disables shard filtering.
    """
    total_pairs = topk_ids.numel()
    if total_pairs == 0:
        active_expert_count.zero_()
        return
    if compact_topk_ids.numel() < total_pairs:
        raise ValueError("compact_topk_ids must have at least total_pairs elements")
    # weight_expert_ids writes at indices 0..active_expert_count-1 (bounded by
    # the number of local experts, not total_pairs), so no size check is needed here.
    if active_expert_count.numel() != 1:
        raise ValueError("active_expert_count must have shape [1]")
    if num_local_experts is None:
        # No shard: every id is local (offset must be 0 for that to hold).
        num_local_experts = 2**30

    block = triton.next_power_of_2(total_pairs)
    num_warps = 1 if block <= 16 else 2
    _compact_topk_ids_kernel[(1,)](
        topk_ids,
        compact_topk_ids,
        weight_expert_ids,
        active_expert_count,
        total_pairs,
        int(local_expert_offset),
        int(num_local_experts),
        BLOCK=block,
        num_warps=num_warps,
    )
