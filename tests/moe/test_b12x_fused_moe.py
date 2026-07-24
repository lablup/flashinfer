"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Numerical accuracy tests for b12x Fused MoE on SM120/SM121 GPUs.

These are SM120-only APIs that take bf16 input directly (no x_sf needed).
The kernel fuses quantization + routing + FC1 + activation + FC2 + scatter.

This test file covers both APIs:
1. Functional API: `b12x_fused_moe`
2. Wrapper API: `B12xMoEWrapper`

Tests include:
- Numerical accuracy against reference implementation (SiLU and ReLU2)
- CUDA graph capture and replay
- API consistency between functional and wrapper APIs
- Micro kernel path for small decode batches
- ReLU2 (non-gated) activation for Nemotron-Super
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from .utils import (
    check_accuracy,
    compute_reference_moe_fp4,
    compute_reference_moe_relu2,
    create_b12x_ct_moe_tensors,
    create_b12x_moe_tensors as create_moe_tensors,
    create_relu2_moe_tensors,
    slice_b12x_moe_tensors_for_ep,
)


def is_sm120_family():
    """Check for SM120 family (SM120, SM121)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 12


def _is_sm12x_supported():
    """Check SM120/SM121 support using repo-standard utility checks."""
    from flashinfer.utils import is_sm120a_supported, is_sm121a_supported

    device = torch.device("cuda")
    return is_sm120a_supported(device) or is_sm121a_supported(device)


def _cuda_13_or_newer():
    """b12x fused MoE kernels require the CUDA 13 toolkit."""
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return get_cuda_version().major >= 13
    except Exception:
        return False


# Skip decorators
cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)
sm120_required = pytest.mark.skipif(
    not _is_sm12x_supported(),
    reason="Requires SM120/SM121 GPU with CUDA 12.8+",
)
cuda_13_required = pytest.mark.skipif(
    not _cuda_13_or_newer(),
    reason="b12x fused MoE requires CUDA 13 or later",
)

_STATIC_CUTOVER_ENV_VARS = (
    "FLASHINFER_B12X_W4A16_STATIC_COMPACT_CUTOVER_PAIRS",
    "B12X_W4A16_STATIC_COMPACT_CUTOVER_PAIRS",
    "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
    "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
)


def _clear_static_cutover_env(monkeypatch):
    for name in _STATIC_CUTOVER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# =============================================================================
# Unit regressions for SM120 dispatch decisions
# =============================================================================


@cute_dsl_available
def test_w4a16_static_tiler_uses_64_when_intermediate_not_128_aligned():
    """The W4A16 backend must not use a 128-wide N tile for n=64 mod 128."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
        _select_tile_config,
    )

    _, tile_n, _, _ = _select_tile_config(
        problem_m=32,
        problem_n=192,
        problem_k=256,
        top_k=2,
        moe_block_size=64,
        sms=1,
        max_shared_mem=101_376,
    )

    assert tile_n == 64


@cute_dsl_available
def test_w4a16_quant_mode_selects_internal_workspace(monkeypatch):
    """Callers provide quant_mode; dispatch owns the concrete workspace type."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    captured = {}

    def fake_allocate(**kwargs):
        captured.update(kwargs)
        return "w4a16-workspace"

    monkeypatch.setattr(moe_dispatch, "_allocate_sm120_w4a16_workspace", fake_allocate)

    workspace = moe_dispatch.allocate_sm120_moe_workspace(
        state_E=1,
        weight_E=1,
        routed_rows=64,
        k=256,
        n=192,
        num_topk=2,
        device=torch.device("cuda"),
        quant_mode="w4a16",
    )

    assert workspace == "w4a16-workspace"
    assert captured["routed_rows"] == 64
    assert captured["k"] == 256
    assert captured["n"] == 192


@cute_dsl_available
def test_sm120_backend_cutovers_are_precision_specific(monkeypatch):
    """W4A16 bypasses NVFP4 static/dynamic cutovers via quant_mode."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    _clear_static_cutover_env(monkeypatch)
    moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    try:
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=80,
                num_topk=8,
                activation_precision="fp4",
            )
            == "static"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=81,
                num_topk=8,
                activation_precision="fp4",
            )
            == "dynamic"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=16,
                num_topk=8,
                quant_mode="w4a16",
            )
            == "w4a16"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=1024,
                num_topk=8,
                quant_mode="w4a16",
            )
            == "w4a16"
        )
    finally:
        moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()


@cute_dsl_available
def test_w4a16_static_cutover_env_override_is_precision_scoped(monkeypatch):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    _clear_static_cutover_env(monkeypatch)
    moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    monkeypatch.setenv("FLASHINFER_B12X_W4A16_STATIC_COMPACT_CUTOVER_PAIRS", "256")
    try:
        assert moe_dispatch._get_static_compact_cutover_pairs("fp4") == 640
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=32,
                num_topk=8,
                quant_mode="w4a16",
            )
            == "w4a16"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=33,
                num_topk=8,
                quant_mode="w4a16",
            )
            == "w4a16"
        )
    finally:
        moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()

    moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    monkeypatch.setenv("FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS", "256")
    try:
        assert moe_dispatch._get_static_compact_cutover_pairs("fp4") == 256
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=32,
                num_topk=8,
                quant_mode="nvfp4",
            )
            == "static"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=33,
                num_topk=8,
                quant_mode="nvfp4",
            )
            == "dynamic"
        )
    finally:
        moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()


@cute_dsl_available
def test_w4a16_direct_micro_rejects_cutlass45_wide_multi_token_shape(monkeypatch):
    """The former direct-micro rejected shape now uses the W4A16 backend."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    captured = {}

    def fake_allocate(**kwargs):
        captured.update(kwargs)
        return "w4a16-workspace"

    monkeypatch.setattr(moe_dispatch, "_allocate_sm120_w4a16_workspace", fake_allocate)

    workspace = moe_dispatch.allocate_sm120_moe_workspace(
        state_E=32,
        weight_E=32,
        routed_rows=4 * 8,
        k=4096,
        n=4096,
        num_topk=8,
        device=torch.device("cuda"),
        quant_mode="w4a16",
    )

    assert workspace == "w4a16-workspace"
    assert captured["k"] == 4096
    assert captured["n"] == 4096
    assert captured["routed_rows"] == 32


@cute_dsl_available
def test_w4a16_direct_micro_shape_guard_rejects_cached_wide_shape(monkeypatch):
    """Wide-shape graph coverage is now expressed as W4A16 workspace capacity."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        max_w4a16_route_capacity,
    )

    routed_rows = 4 * 8
    route_slots, route_blocks = max_w4a16_route_capacity(routed_rows, 32)
    monkeypatch.setattr(moe_dispatch, "get_num_sm", lambda device: 120)
    (
        workspace_slots,
        workspace_blocks,
        fc1_scratch,
        fc2_scratch,
        fc1_cols,
    ) = moe_dispatch._w4a16_workspace_geometry(
        routed_rows=routed_rows,
        route_num_experts=32,
        k=4096,
        n=4096,
        is_gated=True,
        device=torch.device("cuda"),
    )

    assert workspace_slots >= route_slots
    assert workspace_blocks >= route_blocks
    assert fc1_cols == 8192
    assert fc1_scratch > 0
    assert fc2_scratch > 0


@cute_dsl_available
def test_legacy_static_dynamic_allocators_reject_w4a16():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    kwargs = dict(
        state_E=1,
        weight_E=1,
        k=256,
        n=512,
        num_topk=2,
        device=torch.device("cuda"),
        activation_precision="bf16",
    )
    with pytest.raises(ValueError, match="allocate_sm120_moe_workspace"):
        moe_dispatch.allocate_sm120_static_workspace(max_rows=64, **kwargs)
    with pytest.raises(ValueError, match="allocate_sm120_moe_workspace"):
        moe_dispatch.allocate_sm120_dynamic_workspace(routed_rows=64, **kwargs)


def _fake_cuda_13_version():
    class CudaVersion:
        major = 13

        def __str__(self):
            return "13.0"

    return CudaVersion()


def test_functional_cuda_graph_capture_requires_output(monkeypatch):
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    monkeypatch.setattr(b12x_moe_mod, "_is_cuda_graph_capturing", lambda: True)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 1, 1), dtype=torch.uint8)
    scale = torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="pre-allocated output"):
        b12x_moe_mod.b12x_fused_moe(
            x=x,
            w1_weight=weight,
            w1_weight_sf=scale,
            w1_alpha=alpha,
            fc2_input_scale=alpha,
            w2_weight=weight,
            w2_weight_sf=scale,
            w2_alpha=alpha,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            num_experts=1,
            top_k=1,
            quant_mode="w4a16",
        )


@cute_dsl_available
def test_functional_api_passes_source_format_to_dispatch(monkeypatch):
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return kwargs["scatter_output"]

    monkeypatch.setattr(moe_dispatch, "launch_sm120_moe", fake_launch)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 32, 8), dtype=torch.uint8)
    scale = torch.empty((1, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    output = b12x_moe_mod.b12x_fused_moe(
        x=x,
        w1_weight=weight,
        w1_weight_sf=scale,
        w1_alpha=alpha,
        w2_weight=weight,
        w2_weight_sf=scale,
        w2_alpha=alpha,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=1,
        top_k=1,
        quant_mode="w4a16",
        source_format="compressed_tensors",
    )

    assert output is captured["scatter_output"]
    assert captured["source_format"] == "compressed_tensors"


@cute_dsl_available
def test_wrapper_stores_and_passes_source_format(monkeypatch):
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return kwargs["scatter_output"]

    monkeypatch.setattr(moe_dispatch, "launch_sm120_moe", fake_launch)
    moe = b12x_moe_mod.B12xMoEWrapper(
        num_experts=1,
        top_k=1,
        hidden_size=16,
        intermediate_size=16,
        use_cuda_graph=False,
        quant_mode="w4a16",
        source_format="compressed_tensors",
    )

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 32, 8), dtype=torch.uint8)
    scale = torch.empty((1, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    output = moe.run(
        x=x,
        w1_weight=weight,
        w1_weight_sf=scale,
        w1_alpha=alpha,
        w2_weight=weight,
        w2_weight_sf=scale,
        w2_alpha=alpha,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
    )

    assert moe.source_format == "compressed_tensors"
    assert output is captured["scatter_output"]
    assert captured["source_format"] == "compressed_tensors"


@cute_dsl_available
def test_normalize_source_format_accepts_nvfp4_compressed_tensors():
    """nvfp4 + compressed_tensors is accepted pass-through provenance metadata."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        _normalize_source_format_for_quant_mode,
    )

    for alias in ("compressed_tensors", "compressed-tensors", "ct"):
        assert (
            _normalize_source_format_for_quant_mode(alias, "nvfp4")
            == "compressed_tensors"
        )
        assert (
            _normalize_source_format_for_quant_mode(alias, "w4a16")
            == "compressed_tensors"
        )
    assert _normalize_source_format_for_quant_mode("modelopt", "nvfp4") == "modelopt"
    with pytest.raises(ValueError, match=r"source_format"):
        _normalize_source_format_for_quant_mode("bogus", "nvfp4")


@cute_dsl_available
def test_functional_api_passes_nvfp4_ct_source_format_to_dispatch(monkeypatch):
    """nvfp4 + compressed_tensors reaches dispatch instead of raising."""
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return kwargs["scatter_output"]

    monkeypatch.setattr(moe_dispatch, "launch_sm120_moe", fake_launch)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 32, 8), dtype=torch.uint8)
    scale = torch.empty((1, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    output = b12x_moe_mod.b12x_fused_moe(
        x=x,
        w1_weight=weight,
        w1_weight_sf=scale,
        w1_alpha=alpha,
        fc2_input_scale=alpha,
        w2_weight=weight,
        w2_weight_sf=scale,
        w2_alpha=alpha,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=1,
        top_k=1,
        quant_mode="nvfp4",
        source_format="ct",  # alias normalizes on the way in
    )

    assert output is captured["scatter_output"]
    assert captured["source_format"] == "compressed_tensors"
    assert captured["quant_mode"] == "nvfp4"


@cute_dsl_available
def test_dispatch_rejects_nvfp4_ct_reciprocal_input_scales():
    """The reciprocal flag inverts only quant-side global scales in-kernel, so
    combining it with compressed_tensors-tagged nvfp4 tensors is refused."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 32, 8), dtype=torch.uint8)
    scale = torch.empty((1, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    output = torch.empty((1, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"input_scales_are_reciprocal"):
        moe_dispatch.launch_sm120_moe(
            a=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w1_weight=weight,
            w1_weight_sf=scale,
            w1_alpha=alpha,
            fc2_input_scale=alpha,
            w2_weight=weight,
            w2_weight_sf=scale,
            w2_alpha=alpha,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            scatter_output=output,
            input_scales_are_reciprocal=True,
            quant_mode="nvfp4",
            source_format="compressed_tensors",
        )


@cute_dsl_available
def test_wrapper_init_validates_source_format(monkeypatch):
    """Invalid source_format fails at wrapper construction; aliases normalize."""
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)

    with pytest.raises(ValueError, match=r"source_format"):
        b12x_moe_mod.B12xMoEWrapper(
            num_experts=1,
            top_k=1,
            hidden_size=16,
            intermediate_size=16,
            quant_mode="nvfp4",
            source_format="bogus",
        )

    moe = b12x_moe_mod.B12xMoEWrapper(
        num_experts=1,
        top_k=1,
        hidden_size=16,
        intermediate_size=16,
        quant_mode="nvfp4",
        source_format="ct",
    )
    assert moe.source_format == "compressed_tensors"


@cute_dsl_available
def test_w4a16_workspace_validation_rejects_activation_mismatch():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    workspace = moe_dispatch.Sm120W4A16MoEWorkspace(
        state_E=1,
        weight_E=1,
        max_rows=1,
        k=16,
        n=16,
        num_topk=1,
        device=torch.device("cpu"),
        activation="relu2",
        activation_precision="bf16",
        quant_mode="w4a16",
        routed_rows_capacity=1,
        route_num_experts=1,
        intermediate_cache13=torch.empty((16,), dtype=torch.bfloat16),
        intermediate_cache2=torch.empty((1, 16), dtype=torch.bfloat16),
        fc1_c_tmp=torch.empty((1,), dtype=torch.float32),
        fc2_c_tmp=torch.empty((1,), dtype=torch.float32),
        packed_route_indices=torch.empty((1,), dtype=torch.int32),
        block_expert_ids=torch.empty((1,), dtype=torch.int32),
        packed_route_count=torch.empty((1,), dtype=torch.int32),
        expert_offsets=torch.empty((2,), dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="activation mismatch"):
        moe_dispatch._validate_w4a16_workspace(
            workspace,
            state_E=1,
            weight_E=1,
            routed_rows=1,
            k=16,
            n=16,
            num_topk=1,
            device=torch.device("cpu"),
            activation="silu",
        )


@cute_dsl_available
def test_wrapper_cuda_graph_capture_requires_preallocated_buffers(monkeypatch):
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    moe = b12x_moe_mod.B12xMoEWrapper(
        num_experts=1,
        top_k=1,
        hidden_size=16,
        intermediate_size=16,
        use_cuda_graph=False,
    )
    monkeypatch.setattr(b12x_moe_mod, "_is_cuda_graph_capturing", lambda: True)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 1, 1), dtype=torch.uint8)
    scale = torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"use_cuda_graph=True"):
        moe.run(
            x=x,
            w1_weight=weight,
            w1_weight_sf=scale,
            w1_alpha=alpha,
            fc2_input_scale=alpha,
            w2_weight=weight,
            w2_weight_sf=scale,
            w2_alpha=alpha,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
        )


@cute_dsl_available
def test_preallocated_dynamic_workspace_rejects_weight_e_mismatch():
    """Dynamic expert buffers are compiled at weight_E width — a workspace
    allocated at the global expert count must be rejected for an EP shard."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    workspace = object.__new__(moe_dispatch.Sm120DynamicMoEWorkspace)
    workspace.activation_precision = "fp4"
    workspace.weight_E = 4  # global width; num_local_experts below is 2

    x = torch.empty((1, 256), dtype=torch.bfloat16)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    w1 = torch.empty((2, 384, 128), dtype=torch.uint8)
    w1_sf = torch.empty((2, 384, 16), dtype=torch.float8_e4m3fn)
    w2 = torch.empty((2, 256, 96), dtype=torch.uint8)
    w2_sf = torch.empty((2, 256, 12), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((2,), dtype=torch.float32)
    output = torch.empty((1, 256), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"dynamic.*weight_E.*num_local_experts"):
        moe_dispatch.launch_sm120_moe(
            a=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w1_weight=w1,
            w1_weight_sf=w1_sf,
            w1_alpha=alpha,
            fc2_input_scale=torch.ones((1,), dtype=torch.float32),
            w2_weight=w2,
            w2_weight_sf=w2_sf,
            w2_alpha=alpha,
            num_experts=4,
            top_k=1,
            num_local_experts=2,
            scatter_output=output,
            activation_precision="fp4",
            _workspace=workspace,
            _weight_views=object(),
        )


# =============================================================================
# Test Class: Functional API (b12x_fused_moe)
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestB12xFunctional:
    """Tests for the functional API: b12x_fused_moe."""

    @pytest.mark.parametrize(
        "hidden_size,intermediate_size", [(256, 512), (1024, 2048)]
    )
    @pytest.mark.parametrize("top_k", [1, 2, 8])
    @pytest.mark.parametrize("num_tokens", [128, 515, 1024])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_numerical_accuracy(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        """Accuracy test for b12x functional API across configurations."""
        from flashinfer import b12x_fused_moe

        num_local_experts = num_experts

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_local_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    @pytest.mark.parametrize(
        "activation", ["silu", "gelu_tanh", "swigluoai_uninterleave"]
    )
    @pytest.mark.parametrize("num_tokens", [8, 128, 515])
    def test_activation_accuracy(self, activation: str, num_tokens: int):
        """Accuracy of each gated activation: SwiGLU, GeGLU and SwiGLU-OAI.

        Num tokens chosen to trigger the micro, static and dynamic backends to ensure
        that all three backends are tested.
        """
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 1536, 768
        num_experts, top_k = 8, 2
        swiglu_limit = (
            7.0 if activation == "swigluoai_uninterleave" else None
        )  # Minimax-M3 clamp limit
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        assert not torch.isnan(result).any() and not torch.isinf(result).any()
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, f"Only {percent_within * 100:.2f}% within tol (atol={atol:.4f})"

    @pytest.mark.parametrize(
        "activation", ["silu", "gelu_tanh", "swigluoai_uninterleave"]
    )
    @pytest.mark.parametrize("num_tokens", [8, 128])
    def test_intermediate_not_128_aligned(self, activation: str, num_tokens: int):
        """NVFP4 transparently pads non-128-aligned intermediate sizes (e.g.
        Gemma-4's 704) up to a tile multiple; result matches the unpadded ref."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 512, 704  # 704 = 128 * 5.5
        num_experts, top_k = 8, 2
        swiglu_limit = 7.0 if activation == "swigluoai_uninterleave" else None
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any() and not torch.isinf(result).any()
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, f"Only {percent_within * 100:.2f}% within tol (atol={atol:.4f})"

    def test_activation_precision_api_validation(self):
        """W4A4 requires fc2_input_scale; W4A16 tolerates it."""
        from flashinfer import b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 4, 256, 512
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        kwargs = dict(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        with pytest.raises(ValueError, match="fc2_input_scale is required"):
            b12x_fused_moe(**kwargs)

        result = b12x_fused_moe(**kwargs, activation_precision="bf16")
        assert result.shape == (num_tokens, hidden_size)

        result = b12x_fused_moe(
            **kwargs,
            fc2_input_scale=tensors["fc2_input_scale"],
            quant_mode="w4a16",
        )
        assert result.shape == (num_tokens, hidden_size)

    @pytest.mark.parametrize("num_tokens", [64, 384])
    def test_w4a16_functional_accuracy(self, num_tokens: int):
        """Accuracy test for the W4A16 activation path."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens})"
        )

    def test_w4a16_static_accuracy_intermediate_not_128_aligned(self):
        """W4A16 must handle n=64 mod 128 without crossing gate/up tiles."""
        from flashinfer import b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 32, 256, 192
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=789,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 n=192: {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )

    def test_w4a16_dynamic_accuracy_with_wide_scale_tile(self):
        """Exercise W4A16 scale loads when a row spans two scale words."""
        from flashinfer import b12x_fused_moe
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        moe_dispatch._DYNAMIC_KERNEL_CACHE.clear()
        moe_dispatch._WORKSPACE_CACHE.clear()

        num_tokens, hidden_size, intermediate_size = 384, 256, 512
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )

        try:
            result = b12x_fused_moe(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
                num_experts=num_experts,
                top_k=top_k,
                quant_mode="w4a16",
            )
        finally:
            moe_dispatch._DYNAMIC_KERNEL_CACHE.clear()
            moe_dispatch._WORKSPACE_CACHE.clear()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 wide scale tile: {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )

    @pytest.mark.parametrize("num_tokens", [64, 384])
    def test_w4a16_is_more_accurate_than_w4a4(self, num_tokens: int):
        """W4A16 should be closer than W4A4 to the BF16-activation reference."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )
        kwargs = dict(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        a4_result = b12x_fused_moe(**kwargs, quant_mode="nvfp4")
        a16_result = b12x_fused_moe(**kwargs, quant_mode="w4a16")
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        a4_error = (a4_result.float() - ref_output).abs()
        a16_error = (a16_result.float() - ref_output).abs()
        a4_mae = a4_error.mean()
        a16_mae = a16_error.mean()
        a4_mse = a4_error.square().mean()
        a16_mse = a16_error.square().mean()

        assert a16_mae < 0.75 * a4_mae, (
            f"W4A16 MAE should be lower than W4A4 for tokens={num_tokens}: "
            f"a16={a16_mae.item():.6f}, a4={a4_mae.item():.6f}"
        )
        assert a16_mse < 0.5 * a4_mse, (
            f"W4A16 MSE should be lower than W4A4 for tokens={num_tokens}: "
            f"a16={a16_mse.item():.6f}, a4={a4_mse.item():.6f}"
        )


# =============================================================================
# Test Class: Wrapper API (B12xMoEWrapper)
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestB12xWrapper:
    """Tests for the wrapper API: B12xMoEWrapper."""

    @pytest.mark.parametrize(
        "activation", ["silu", "gelu_tanh", "swigluoai_uninterleave"]
    )
    def test_wrapper_intermediate_not_128_aligned(self, activation: str):
        """Wrapper transparently pads a non-128-aligned intermediate size
        (Gemma-4's 704), caching the padded weights across calls."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 128, 512, 704
        num_experts, top_k = 8, 2
        swiglu_limit = 7.0 if activation == "swigluoai_uninterleave" else None
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )
        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any() and not torch.isinf(result).any()
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, f"Only {percent_within * 100:.2f}% within tol (atol={atol:.4f})"

    @pytest.mark.parametrize("num_tokens", [128, 256, 512])
    @pytest.mark.parametrize("top_k", [2, 8])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_wrapper_accuracy(self, num_tokens: int, top_k: int, num_experts: int):
        """Accuracy test for B12xMoEWrapper."""
        from flashinfer import B12xMoEWrapper

        hidden_size, intermediate_size = 256, 512

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Create wrapper WITHOUT CUDA graph
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    def test_w4a16_wrapper_accuracy(self):
        """Accuracy test for B12xMoEWrapper with BF16 intermediates."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 64, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=321,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_mode="w4a16",
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 wrapper: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    @pytest.mark.parametrize("num_tokens", [64, 128, 256])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_wrapper_cuda_graph(self, num_tokens: int, num_experts: int):
        """Test B12xMoEWrapper with CUDA graph capture and replay."""
        from flashinfer import B12xMoEWrapper

        hidden_size, intermediate_size = 256, 512
        top_k = 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Create wrapper WITH CUDA graph
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
        )

        # Warmup
        for _ in range(3):
            moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        # Note: CUDA graph capture doesn't execute - output may be zeros here
        # Actual execution happens during replay
        assert output.shape == (num_tokens, hidden_size)

        # First replay to get actual output
        g.replay()
        torch.cuda.synchronize()

        # Verify output is valid after first replay
        assert not torch.isnan(output).any(), "NaN after first replay"
        assert not (output == 0).all(), "All zeros after first replay"

        # Test replay consistency (allow small numerical differences due to FP4 atomics)
        results = []
        for _ in range(3):
            g.replay()
            torch.cuda.synchronize()
            results.append(output.clone())

        # All replays should produce very similar results (small FP4 tolerance)
        for i in range(1, len(results)):
            max_diff = (results[0] - results[i]).abs().max().item()
            # FP4 atomics can have small non-determinism
            assert max_diff < 0.5, f"Replay {i} differs too much: max_diff={max_diff}"

        # Verify accuracy
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(results[0], ref_output)
        assert passed, (
            f"CUDA graph accuracy: {percent_within * 100:.2f}% (atol={atol:.4f})"
        )


# =============================================================================
# Test Class: API Consistency
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestB12xApiConsistency:
    """Tests verifying consistency between b12x functional and wrapper APIs."""

    def test_functional_vs_wrapper_output(self):
        """Verify b12x_fused_moe and B12xMoEWrapper produce the same output."""
        from flashinfer import B12xMoEWrapper, b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 128, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Functional API
        result_functional = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        # Wrapper API
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result_wrapper = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        # Both should produce valid outputs
        assert result_functional.shape == result_wrapper.shape
        assert not torch.isnan(result_functional).any()
        assert not torch.isnan(result_wrapper).any()

        # Outputs should be very close (may not be exactly equal due to different
        # code paths, but should be within FP4 tolerance)
        diff = (result_functional - result_wrapper).abs()
        max_diff = diff.max().item()
        # Allow small differences from code path differences
        assert max_diff < 1e-3, f"Max diff between APIs: {max_diff}"


# =============================================================================
# Test Class: Micro Kernel (SM120-only, small decode batches)
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestMicroKernel:
    """Tests for the micro kernel path (routed_rows <= 20-40).

    The micro kernel is selected automatically when routed_rows is small.
    These tests use num_tokens=1-4 to exercise the micro dispatch path,
    including the all_rows_unique fast path (num_tokens=1).
    """

    @pytest.mark.parametrize("num_tokens", [1, 2, 4])
    @pytest.mark.parametrize("top_k", [2, 8])
    @pytest.mark.parametrize("num_experts", [256])
    def test_micro_functional_accuracy(
        self, num_tokens: int, top_k: int, num_experts: int
    ):
        """Accuracy test for micro kernel via b12x functional API."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Micro kernel: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens}, top_k={top_k})"
        )

    def test_micro_wrapper_accuracy(self):
        """Accuracy test for micro kernel via B12xMoEWrapper."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 2, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Micro wrapper: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    @pytest.mark.parametrize("num_tokens", [1, 2, 4])
    def test_w4a16_direct_micro_functional_accuracy(self, num_tokens: int):
        """Accuracy test for the W4A16 small-batch route-packing path."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 512, 256
        num_experts, top_k = 64, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=777,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 direct micro: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens})"
        )

    def test_w4a16_direct_micro_wrapper_accuracy(self):
        """Accuracy test for W4A16 small-batch route-packing via B12xMoEWrapper."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 2, 512, 256
        num_experts, top_k = 64, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=778,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_mode="w4a16",
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 direct micro wrapper: {percent_within * 100:.2f}% "
            f"within tolerance (atol={atol:.4f})"
        )

    def test_w4a16_direct_micro_cuda_graph(self):
        """CUDA graph replay test for the W4A16 route-packing wrapper path."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 2, 512, 256
        num_experts, top_k = 64, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=780,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_mode="w4a16",
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
        )

        for _ in range(3):
            moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        graph.replay()
        torch.cuda.synchronize()

        assert output.shape == (num_tokens, hidden_size)
        assert torch.isfinite(output).all()
        assert not (output == 0).all()

    def test_w4a16_direct_micro_workspace_capacity(self):
        """The unified allocator reserves W4A16 route and GEMM scratch space."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            allocate_sm120_moe_workspace,
        )

        num_tokens, hidden_size, intermediate_size = 2, 512, 256
        num_experts, top_k = 64, 2
        routed_rows = num_tokens * top_k
        workspace = allocate_sm120_moe_workspace(
            state_E=num_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=hidden_size,
            n=intermediate_size,
            num_topk=top_k,
            device=torch.device("cuda"),
            quant_mode="w4a16",
        )

        assert workspace.quant_mode == "w4a16"
        assert workspace.routed_rows_capacity >= routed_rows
        assert workspace.intermediate_cache13.numel() >= routed_rows * max(
            hidden_size, 2 * intermediate_size
        )
        assert workspace.intermediate_cache2.shape == (
            workspace.routed_rows_capacity,
            intermediate_size,
        )
        assert workspace.fc1_c_tmp.numel() > 0
        assert workspace.fc2_c_tmp.numel() > 0
        assert workspace.packed_route_indices.numel() >= routed_rows
        assert workspace.block_expert_ids.numel() > 0
        assert workspace.expert_offsets.numel() == workspace.route_num_experts + 1

    def test_micro_single_token_unique_path(self):
        """Test the all_rows_unique fast path (num_tokens=1, top_k=8).

        With 1 token and 8 experts, every expert has exactly 1 row.
        The micro kernel detects this and uses O(1) work tile assignment.
        """
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 1, 256, 512
        num_experts, top_k = 256, 8

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        assert result.shape == (1, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Micro unique path: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    @pytest.mark.parametrize(
        "num_tokens,top_k,num_experts",
        [
            (2, 8, 8),  # total_pairs=16 > num_local_experts=8
            (4, 8, 16),  # total_pairs=32 > num_local_experts=16
            (4, 4, 8),  # total_pairs=16 > num_local_experts=8
        ],
    )
    def test_micro_pairs_exceed_local_experts(
        self, num_tokens: int, top_k: int, num_experts: int
    ):
        """Regression test: micro kernel when num_tokens * top_k > num_local_experts.

        The workspace compact_topk_ids buffer was previously sized state_E
        (num_local_experts), but the micro kernel fills it with total_pairs =
        num_tokens * top_k.  When total_pairs > num_local_experts the assertion
        'flat_ids.numel() <= workspace.compact_topk_ids.numel()' fired.

        Fixed by sizing compact_topk_ids as max(state_E, max_rows).
        """
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Micro pairs>experts: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens}, top_k={top_k}, experts={num_experts})"
        )


# =============================================================================
# Test Class: ReLU2 Activation (SM120-only, non-gated)
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestRelu2Activation:
    """Tests for ReLU2 activation (non-gated, Nemotron-Super)."""

    @pytest.mark.parametrize(
        "hidden_size,intermediate_size", [(256, 512), (1024, 2048)]
    )
    @pytest.mark.parametrize("top_k", [1, 2, 8])
    @pytest.mark.parametrize("num_tokens", [1, 2, 128, 512])
    @pytest.mark.parametrize("num_experts", [256])
    def test_relu2_functional_accuracy(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        """Accuracy test for ReLU2 via b12x functional API."""
        from flashinfer import b12x_fused_moe

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            activation="relu2",
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens})"
        )

    @pytest.mark.parametrize("num_tokens", [64, 384])
    def test_relu2_w4a16_functional_accuracy(self, num_tokens: int):
        """Accuracy test for ReLU2 with the W4A16 activation path."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 256, 2
        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            activation="relu2",
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2 W4A16: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens})"
        )

    def test_relu2_wrapper_accuracy(self):
        """Accuracy test for ReLU2 via B12xMoEWrapper."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 128, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
            activation="relu2",
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2 wrapper: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    def test_relu2_micro_accuracy(self):
        """Accuracy test for ReLU2 with micro kernel (small decode batch)."""
        from flashinfer import b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 2, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            activation="relu2",
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2 micro: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    def test_relu2_w4a16_direct_micro_accuracy(self):
        """Accuracy test for ReLU2 with the W4A16 route-packing small-batch path."""
        from flashinfer import b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 2, 512, 256
        num_experts, top_k = 64, 2

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=779,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            activation="relu2",
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2 W4A16 direct micro: {percent_within * 100:.2f}% "
            f"within tolerance (atol={atol:.4f})"
        )

    def test_relu2_cuda_graph(self):
        """Test ReLU2 with CUDA graph capture and replay."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 128, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
            activation="relu2",
        )

        # Warmup
        for _ in range(3):
            moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        assert output.shape == (num_tokens, hidden_size)

        # Replay and verify
        g.replay()
        torch.cuda.synchronize()
        assert not torch.isnan(output).any(), "NaN after ReLU2 CUDA graph replay"
        assert not (output == 0).all(), "All zeros after ReLU2 CUDA graph replay"


# =============================================================================
# Test Class: compressed-tensors W4A4 (nvfp4 + source_format=compressed_tensors)
# =============================================================================


def _assert_packs_bitwise_equal(pack_a: dict, pack_b: dict):
    """Bitwise pack comparison (fp8/uint8 compared through a uint8 view)."""
    assert pack_a.keys() == pack_b.keys()
    for key in pack_a:
        a, b = pack_a[key], pack_b[key]
        assert a.shape == b.shape, f"{key}: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"{key}: {a.dtype} vs {b.dtype}"
        if a.dtype in (torch.uint8, torch.float8_e4m3fn):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                f"{key} differs bitwise"
            )
        else:
            assert torch.equal(a, b), f"{key} differs"


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestB12xCompressedTensorsW4A4:
    """W4A4 with compressed-tensors checkpoints on the b12x nvfp4 path.

    The decisive oracle is bitwise/exact equivalence against independently
    constructed kernel-convention packs: the loose ``check_accuracy``
    tolerance (rtol=0.5) cannot detect per-expert scale-convention errors of
    magnitude ``weight_global_scale``.
    """

    @pytest.mark.parametrize(
        "hidden_size,intermediate_size",
        [
            (256, 512),
            # Non-swizzle-aligned geometry: 2*I = 320 (% 128 != 0) and
            # H/16 = 13 blocks (% 4 != 0) exercise the swizzle padding.
            (208, 160),
        ],
    )
    def test_prepare_ct_matches_modelopt_twin(
        self, hidden_size: int, intermediate_size: int
    ):
        """prepare(ct) == prepare(modelopt twin) == independently-built pack."""
        from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_packed_weights

        tensors = create_b12x_ct_moe_tensors(
            num_tokens=8,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=8,
            num_local_experts=8,
            top_k=2,
        )

        pack_ct = prepare_b12x_nvfp4_packed_weights(
            tensors["ct"]["w1_fp4"],
            tensors["ct"]["w1_blockscale"],
            tensors["ct"]["w1_global_scale"],
            tensors["ct"]["w2_fp4"],
            tensors["ct"]["w2_blockscale"],
            tensors["ct"]["w2_global_scale"],
            activation="silu",
            source_format="compressed_tensors",
        )
        pack_modelopt = prepare_b12x_nvfp4_packed_weights(
            tensors["modelopt"]["w1_fp4"],
            tensors["modelopt"]["w1_blockscale"],
            tensors["modelopt"]["w1_global_scale"],
            tensors["modelopt"]["w2_fp4"],
            tensors["modelopt"]["w2_blockscale"],
            tensors["modelopt"]["w2_global_scale"],
            activation="silu",
            source_format="modelopt",
        )

        _assert_packs_bitwise_equal(pack_ct, pack_modelopt)
        # The layout oracle: prepared scales must match the pack built through
        # the already-tested bake-on-swizzled + MMA-convert path. This fails
        # loudly if linear scales were fed to convert_sf_to_mma_layout
        # unswizzled (silent garbling) or the reciprocal went the wrong way.
        _assert_packs_bitwise_equal(pack_ct, tensors["native"])
        assert torch.equal(pack_ct["w2_alpha"], pack_ct["fc2_input_scale"])

    def test_fixture_linear_scales_match_quantizer_linear_layout(self):
        """Oracle-independence check: the fixture's linear scales (derived by
        unswizzling fp4_quantize's swizzled output) must equal what the CUDA
        quantizer itself emits in linear layout. This pins the fixture's
        layout assumptions to the quantizer rather than to the same
        swizzle helpers prepare uses."""
        from flashinfer.fp4_quantization import fp4_quantize

        tensors = create_b12x_ct_moe_tensors(
            num_tokens=8,
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
            num_local_experts=4,
            top_k=2,
        )
        w1 = tensors["w1_weight_bf16"]
        gs = tensors["ct"]["w1_global_scale"]
        for e in range(w1.size(0)):
            _, sf_linear = fp4_quantize(
                w1[e],
                global_scale=gs[e : e + 1],
                sf_vec_size=16,
                is_sf_swizzled_layout=False,
            )
            expected = tensors["ct"]["w1_blockscale"][e]
            got = sf_linear.view(torch.float8_e4m3fn).reshape(expected.shape)
            assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8)), (
                f"expert {e}: fixture linear sf differs from quantizer linear sf"
            )

    def test_prepare_gate_up_checkpoint_order(self):
        """A [gate, up]-ordered checkpoint (the vLLM fused-w13 convention)
        prepared with w13_checkpoint_order='gate_up' must produce the same
        pack as the kernel-native [up, gate] ordering."""
        from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_packed_weights

        tensors = create_b12x_ct_moe_tensors(
            num_tokens=8,
            hidden_size=256,
            intermediate_size=512,
            num_experts=8,
            num_local_experts=8,
            top_k=2,
        )
        ct = tensors["ct"]
        inter = 512

        def swapped(t):
            return torch.cat([t[:, inter:], t[:, :inter]], dim=1).contiguous()

        pack_up_gate = prepare_b12x_nvfp4_packed_weights(
            ct["w1_fp4"],
            ct["w1_blockscale"],
            ct["w1_global_scale"],
            ct["w2_fp4"],
            ct["w2_blockscale"],
            ct["w2_global_scale"],
            activation="silu",
            source_format="compressed_tensors",
        )
        pack_gate_up = prepare_b12x_nvfp4_packed_weights(
            swapped(ct["w1_fp4"]),
            swapped(ct["w1_blockscale"]),
            ct["w1_global_scale"],
            ct["w2_fp4"],
            ct["w2_blockscale"],
            ct["w2_global_scale"],
            activation="silu",
            source_format="compressed_tensors",
            w13_checkpoint_order="gate_up",
        )
        _assert_packs_bitwise_equal(pack_gate_up, pack_up_gate)

    def test_prepare_per_shard_w1_global_scales_exact(self):
        """[E, 2] per-shard global scales are baked exactly per row half.

        Constructs a mathematically identical checkpoint where the gate
        half's block scales and global scale are both doubled (a power of
        two, so fp8/fp32 arithmetic is exact): the prepared pack must be
        bitwise identical to the single-scale version. A shard-0 collapse
        (the old lossy behavior) would leave the gate half off by 2x.
        """
        from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_packed_weights

        torch.manual_seed(7)
        num_experts, hidden, inter = 8, 256, 512
        device = "cuda"

        def rand_fp4(rows):
            return torch.randint(
                0,
                256,
                (num_experts, rows, hidden // 2),
                dtype=torch.uint8,
                device=device,
            )

        def rand_sf(rows, cols_blocks):
            # fp8-exact values in [0.25, 64]: doubling stays well below the
            # e4m3 max (448), so the power-of-two shard construction below is
            # exact (no rounding, no saturation).
            raw = torch.empty(num_experts, rows, cols_blocks, device=device).uniform_(
                0.25, 64.0
            )
            return raw.to(torch.float8_e4m3fn)

        w1_fp4 = rand_fp4(2 * inter)
        w1_sf = rand_sf(2 * inter, hidden // 16)
        w2_fp4 = torch.randint(
            0,
            256,
            (num_experts, hidden, inter // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2_sf = rand_sf(hidden, inter // 16)
        gs = torch.empty(num_experts, device=device).uniform_(100.0, 2000.0)
        w2_gs = torch.empty(num_experts, device=device).uniform_(100.0, 2000.0)

        # Mathematically identical per-shard checkpoint: the gate half's
        # block scales and global scale are both doubled (exact in fp8/fp32),
        # so dequant sf/gs is unchanged.
        sf_shard = w1_sf.clone()
        sf_shard[:, inter:] = (sf_shard[:, inter:].float() * 2.0).to(
            torch.float8_e4m3fn
        )
        gs_shard = torch.stack([gs, gs * 2.0], dim=1)
        assert gs_shard.shape == (num_experts, 2)

        pack_single = prepare_b12x_nvfp4_packed_weights(
            w1_fp4,
            w1_sf,
            gs,
            w2_fp4,
            w2_sf,
            w2_gs,
            activation="silu",
            source_format="compressed_tensors",
        )
        pack_shard = prepare_b12x_nvfp4_packed_weights(
            w1_fp4,
            sf_shard,
            gs_shard,
            w2_fp4,
            w2_sf,
            w2_gs,
            activation="silu",
            source_format="compressed_tensors",
        )
        _assert_packs_bitwise_equal(pack_shard, pack_single)

        # Scalar broadcast also accepted.
        pack_scalar_a2 = prepare_b12x_nvfp4_packed_weights(
            w1_fp4,
            w1_sf,
            gs,
            w2_fp4,
            w2_sf,
            w2_gs,
            activation="silu",
            source_format="compressed_tensors",
            a2_input_global_scale=torch.tensor([100.0], device="cuda"),
        )
        assert pack_scalar_a2["fc2_input_scale"].shape == (num_experts,)
        assert torch.equal(
            pack_scalar_a2["w2_alpha"], pack_scalar_a2["fc2_input_scale"]
        )

    def test_prepare_rejects_bad_global_scales(self):
        from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_packed_weights

        tensors = create_b12x_ct_moe_tensors(
            num_tokens=8,
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
            num_local_experts=4,
            top_k=2,
        )
        bad_gs = tensors["ct"]["w1_global_scale"].clone()
        bad_gs[0] = 0.0
        with pytest.raises(ValueError, match=r"finite and positive"):
            prepare_b12x_nvfp4_packed_weights(
                tensors["ct"]["w1_fp4"],
                tensors["ct"]["w1_blockscale"],
                bad_gs,
                tensors["ct"]["w2_fp4"],
                tensors["ct"]["w2_blockscale"],
                tensors["ct"]["w2_global_scale"],
                activation="silu",
                source_format="compressed_tensors",
            )

    @pytest.mark.parametrize(
        "hidden_size,intermediate_size,num_tokens,top_k",
        [
            # Routed pairs = num_tokens * top_k select the backend: micro up
            # to 20 pairs (40 for multi-topk), static up to the ~640-pair
            # compact cutover, dynamic above.
            (256, 512, 8, 1),  # 8 pairs: micro (single-topk cutover 20)
            (256, 512, 2, 8),  # 16 pairs: micro (multi-topk cutover 40)
            (256, 512, 8, 8),  # 64 pairs: static
            (256, 512, 128, 1),  # 128 pairs: static
            (256, 512, 1024, 8),  # 8192 pairs: dynamic
            (256, 320, 128, 8),  # non-128-aligned I: pad-after-conversion path
        ],
    )
    def test_ct_prepared_matches_native_pack_kernel_output(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_tokens: int,
        top_k: int,
    ):
        """Kernel output on ct-prepared weights must exactly match the output
        on the independently-built kernel-convention pack."""
        from flashinfer import b12x_fused_moe
        from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_packed_weights

        num_experts = 32
        tensors = create_b12x_ct_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        pack_ct = prepare_b12x_nvfp4_packed_weights(
            tensors["ct"]["w1_fp4"],
            tensors["ct"]["w1_blockscale"],
            tensors["ct"]["w1_global_scale"],
            tensors["ct"]["w2_fp4"],
            tensors["ct"]["w2_blockscale"],
            tensors["ct"]["w2_global_scale"],
            activation="silu",
            source_format="compressed_tensors",
        )

        def _run(pack: dict, source_format: str) -> torch.Tensor:
            return b12x_fused_moe(
                x=tensors["x_bf16"],
                w1_weight=pack["w1_weight"],
                w1_weight_sf=pack["w1_weight_sf"],
                w1_alpha=pack["w1_alpha"],
                fc2_input_scale=pack["fc2_input_scale"],
                w2_weight=pack["w2_weight"],
                w2_weight_sf=pack["w2_weight_sf"],
                w2_alpha=pack["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
                num_experts=num_experts,
                top_k=top_k,
                quant_mode="nvfp4",
                source_format=source_format,
            )

        result_ct = _run(pack_ct, "compressed_tensors")
        result_native = _run(tensors["native"], "modelopt")

        assert not torch.isnan(result_ct).any()
        assert not (result_ct == 0).all()
        # The packs are bitwise identical (see test_prepare_ct_matches_modelopt
        # _twin), so any kernel-output difference comes from the scatter's
        # atomic accumulation order (~2e-3 typical run-to-run on identical
        # inputs, with rare larger tails on top-8 accumulation). The tolerance
        # stays far below the O(output)-magnitude error a per-expert
        # scale-convention bug would produce.
        torch.testing.assert_close(result_ct, result_native, atol=3e-2, rtol=3e-2)

        # Absolute sanity vs the fp32 reference on the logical bf16 weights.
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float(),
            gemm1_weights=tensors["w1_weight_bf16"].float(),
            gemm2_weights=tensors["w2_weight_bf16"].float(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=torch.ones(1, device="cuda", dtype=torch.float32),
        )
        passed, percent_within, atol = check_accuracy(result_ct, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    def test_ct_solar_shape_smoke(self):
        """Solar-Open2 geometry smoke (subset of experts for memory)."""
        from flashinfer import b12x_fused_moe
        from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_packed_weights

        num_experts, top_k, num_tokens = 32, 8, 64
        hidden_size, intermediate_size = 4096, 1280
        tensors = create_b12x_ct_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        pack_ct = prepare_b12x_nvfp4_packed_weights(
            tensors["ct"]["w1_fp4"],
            tensors["ct"]["w1_blockscale"],
            tensors["ct"]["w1_global_scale"],
            tensors["ct"]["w2_fp4"],
            tensors["ct"]["w2_blockscale"],
            tensors["ct"]["w2_global_scale"],
            activation="silu",
            source_format="compressed_tensors",
        )
        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=pack_ct["w1_weight"],
            w1_weight_sf=pack_ct["w1_weight_sf"],
            w1_alpha=pack_ct["w1_alpha"],
            fc2_input_scale=pack_ct["fc2_input_scale"],
            w2_weight=pack_ct["w2_weight"],
            w2_weight_sf=pack_ct["w2_weight_sf"],
            w2_alpha=pack_ct["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="nvfp4",
            source_format="compressed_tensors",
        )
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float(),
            gemm1_weights=tensors["w1_weight_bf16"].float(),
            gemm2_weights=tensors["w2_weight_bf16"].float(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=torch.ones(1, device="cuda", dtype=torch.float32),
        )
        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    def test_ct_calibrated_activation_scales(self):
        """Calibrated a1/a2 global scales stay numerically consistent (the FC1
        scale cancels; FC2 keeps w2_alpha == fc2_input_scale)."""
        from flashinfer import b12x_fused_moe
        from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_packed_weights

        num_experts, top_k, num_tokens = 16, 4, 128
        hidden_size, intermediate_size = 256, 512
        tensors = create_b12x_ct_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        # ct-convention (BIG) activation global scales at plausible magnitudes.
        amax_x = tensors["x_bf16"].abs().amax().float().clamp(min=1e-6)
        a1_gs_ct = ((448.0 * 6.0) / amax_x).repeat(num_experts)
        a2_gs_ct = ((448.0 * 6.0) / (2.0 * amax_x)).repeat(num_experts)

        pack = prepare_b12x_nvfp4_packed_weights(
            tensors["ct"]["w1_fp4"],
            tensors["ct"]["w1_blockscale"],
            tensors["ct"]["w1_global_scale"],
            tensors["ct"]["w2_fp4"],
            tensors["ct"]["w2_blockscale"],
            tensors["ct"]["w2_global_scale"],
            activation="silu",
            source_format="compressed_tensors",
            a1_input_global_scale=a1_gs_ct,
            a2_input_global_scale=a2_gs_ct,
        )
        assert torch.equal(pack["w2_alpha"], pack["fc2_input_scale"])
        assert torch.allclose(pack["w1_alpha"], 1.0 / a1_gs_ct)

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=pack["w1_weight"],
            w1_weight_sf=pack["w1_weight_sf"],
            w1_alpha=pack["w1_alpha"],
            fc2_input_scale=pack["fc2_input_scale"],
            w2_weight=pack["w2_weight"],
            w2_weight_sf=pack["w2_weight_sf"],
            w2_alpha=pack["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="nvfp4",
            source_format="compressed_tensors",
        )
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float(),
            gemm1_weights=tensors["w1_weight_bf16"].float(),
            gemm2_weights=tensors["w2_weight_bf16"].float(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=torch.ones(1, device="cuda", dtype=torch.float32),
        )
        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    def test_ct_wrapper_cuda_graph(self):
        """Wrapper path with nvfp4 + compressed_tensors: capture and replay."""
        from flashinfer import B12xMoEWrapper, b12x_fused_moe
        from flashinfer.fused_moe.prepare import prepare_b12x_nvfp4_packed_weights

        num_experts, top_k, num_tokens = 32, 2, 64
        hidden_size, intermediate_size = 256, 512
        tensors = create_b12x_ct_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        pack = prepare_b12x_nvfp4_packed_weights(
            tensors["ct"]["w1_fp4"],
            tensors["ct"]["w1_blockscale"],
            tensors["ct"]["w1_global_scale"],
            tensors["ct"]["w2_fp4"],
            tensors["ct"]["w2_blockscale"],
            tensors["ct"]["w2_global_scale"],
            activation="silu",
            source_format="compressed_tensors",
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
            quant_mode="nvfp4",
            source_format="compressed_tensors",
        )

        run_kwargs = dict(
            x=tensors["x_bf16"],
            w1_weight=pack["w1_weight"],
            w1_weight_sf=pack["w1_weight_sf"],
            w1_alpha=pack["w1_alpha"],
            fc2_input_scale=pack["fc2_input_scale"],
            w2_weight=pack["w2_weight"],
            w2_weight_sf=pack["w2_weight_sf"],
            w2_alpha=pack["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        for _ in range(3):
            moe.run(**run_kwargs)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = moe.run(**run_kwargs)
        torch.cuda.synchronize()

        g.replay()
        torch.cuda.synchronize()
        assert not torch.isnan(output).any(), "NaN after ct-W4A4 graph replay"
        assert not (output == 0).all(), "All zeros after ct-W4A4 graph replay"

        # Wrapper output must match the functional path on the same inputs.
        functional = b12x_fused_moe(
            **run_kwargs,
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="nvfp4",
            source_format="compressed_tensors",
        )
        torch.testing.assert_close(output, functional, atol=1e-2, rtol=1e-2)


# =============================================================================
# Expert Parallelism (EP)
# =============================================================================


@cute_dsl_available
def test_ep_shard_validation(monkeypatch):
    """Invalid EP shard descriptions fail fast at both API entry points."""
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((2, 32, 8), dtype=torch.uint8)
    scale = torch.empty((2, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((2,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    common = dict(
        x=x,
        w1_weight=weight,
        w1_weight_sf=scale,
        w1_alpha=alpha,
        fc2_input_scale=alpha,
        w2_weight=weight,
        w2_weight_sf=scale,
        w2_alpha=alpha,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=8,
        top_k=1,
    )
    with pytest.raises(ValueError, match=r"local_expert_offset must be >= 0"):
        b12x_moe_mod.b12x_fused_moe(
            **common, num_local_experts=2, local_expert_offset=-1
        )
    with pytest.raises(ValueError, match=r"must not exceed"):
        b12x_moe_mod.b12x_fused_moe(
            **common, num_local_experts=2, local_expert_offset=7
        )
    with pytest.raises(ValueError, match=r"num_local_experts must be >= 1"):
        b12x_moe_mod.b12x_fused_moe(**common, num_local_experts=0)

    with pytest.raises(ValueError, match=r"must not exceed"):
        b12x_moe_mod.B12xMoEWrapper(
            num_experts=8,
            top_k=1,
            hidden_size=16,
            intermediate_size=32,
            num_local_experts=4,
            local_expert_offset=6,
        )
    with pytest.raises(ValueError, match=r"local_expert_offset must be >= 0"):
        b12x_moe_mod.B12xMoEWrapper(
            num_experts=8,
            top_k=1,
            hidden_size=16,
            intermediate_size=32,
            local_expert_offset=-2,
        )


@cute_dsl_available
def test_functional_api_passes_ep_shard_to_dispatch(monkeypatch):
    """An EP shard reaches dispatch instead of raising NotImplementedError."""
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return kwargs["scatter_output"]

    monkeypatch.setattr(moe_dispatch, "launch_sm120_moe", fake_launch)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((2, 32, 8), dtype=torch.uint8)
    scale = torch.empty((2, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((2,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    output = b12x_moe_mod.b12x_fused_moe(
        x=x,
        w1_weight=weight,
        w1_weight_sf=scale,
        w1_alpha=alpha,
        fc2_input_scale=alpha,
        w2_weight=weight,
        w2_weight_sf=scale,
        w2_alpha=alpha,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=8,
        top_k=1,
        num_local_experts=2,
        local_expert_offset=4,
    )

    assert output is captured["scatter_output"]
    assert captured["num_local_experts"] == 2
    assert captured["local_expert_offset"] == 4


@cute_dsl_available
def test_w4a16_expert_map_offset():
    """The W4A16 expert map places this rank's shard at the offset."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        _make_w4a16_expert_map,
    )

    device = torch.device("cpu")
    expert_map = _make_w4a16_expert_map(
        state_E=2, weight_E=8, device=device, local_expert_offset=3
    )
    assert expert_map.tolist() == [-1, -1, -1, 0, 1, -1, -1, -1]

    # Full-width shard needs no filtering (offset is necessarily 0).
    assert _make_w4a16_expert_map(state_E=8, weight_E=8, device=device) is None

    with pytest.raises(ValueError, match=r"cannot exceed num_experts"):
        _make_w4a16_expert_map(
            state_E=4, weight_E=8, device=device, local_expert_offset=5
        )
    with pytest.raises(ValueError, match=r"local_expert_offset must be >= 0"):
        _make_w4a16_expert_map(
            state_E=4, weight_E=8, device=device, local_expert_offset=-1
        )


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestB12xExpertParallel:
    """EP semantics: per-rank partial sums must reproduce the full model.

    Each simulated rank runs on its contiguous weight shard with global
    routing ids; summing the partial outputs across ranks must match both
    the full-model kernel run and the PyTorch reference. Comparisons use
    check_accuracy tolerances (kernel outputs are nondeterministic at the
    ~2e-3 level from bf16 atomic scatter ordering — never compare bitwise).
    """

    @staticmethod
    def _assert_partials_match_full(
        ep_sum,
        full,
        *,
        atol=3e-2,
        rtol=3e-2,
        max_mismatch_frac=0.005,
        max_abs=0.15,
    ):
        """Tight EP-sum vs full-kernel comparison on identical quantized inputs.

        The shard packs are bitwise slices of the full-model quantization, so
        the runs differ only by scheduler and summation order. Different
        accumulation orders occasionally flip an FP4 rounding boundary before
        FC2 (observed ~0.02% of elements between the dynamic and static
        schedulers), so allow a tiny out-of-tolerance fraction — while an
        inert filter, dropped expert, or all-zero output shifts far more
        elements than the allowance and still fails. check_accuracy's
        scale-aware atol floor is too loose here (an all-zero output passes
        it at this geometry).
        """
        full = full.float()
        diff = (ep_sum.float() - full).abs()
        thresh = atol + rtol * full.abs()
        mismatch_frac = (diff > thresh).float().mean().item()
        assert mismatch_frac <= max_mismatch_frac, (
            f"{mismatch_frac * 100:.3f}% of elements outside "
            f"atol={atol}/rtol={rtol} (allowed {max_mismatch_frac * 100:.2f}%)"
        )
        max_diff = diff.max().item()
        assert max_diff < max_abs, f"max abs diff {max_diff:.4f} >= {max_abs}"

    def _run_functional(self, tensors, num_experts, top_k, **kwargs):
        from flashinfer import b12x_fused_moe

        return b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            **kwargs,
        )

    def _ep_sum(self, tensors, num_experts, top_k, ep_size, **kwargs):
        """Run every rank's shard and sum the partial outputs in fp32."""
        assert num_experts % ep_size == 0
        n_local = num_experts // ep_size
        ep_sum = None
        for rank in range(ep_size):
            shard = slice_b12x_moe_tensors_for_ep(
                tensors,
                local_expert_offset=rank * n_local,
                num_local_experts=n_local,
            )
            partial = self._run_functional(
                shard,
                num_experts,
                top_k,
                num_local_experts=n_local,
                local_expert_offset=rank * n_local,
                **kwargs,
            )
            ep_sum = partial.float() if ep_sum is None else ep_sum + partial.float()
        return ep_sum

    def _reference(self, tensors, num_tokens, num_experts, top_k, h, i):
        return compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=h,
            intermediate_size=i,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

    @pytest.mark.parametrize(
        "num_tokens,top_k,ep_size",
        [
            (4, 2, 2),  # micro-range: EP runs native micro via the filtered pre-pass
            (128, 2, 2),
            (128, 2, 4),
        ],
    )
    def test_nvfp4_partial_sums_match_full(self, num_tokens, top_k, ep_size):
        num_experts = 32
        hidden_size, intermediate_size = 256, 512
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        full = self._run_functional(tensors, num_experts, top_k)
        ep_sum = self._ep_sum(tensors, num_experts, top_k, ep_size)

        self._assert_partials_match_full(ep_sum, full)
        ref = self._reference(
            tensors, num_tokens, num_experts, top_k, hidden_size, intermediate_size
        )
        passed, percent_within, atol = check_accuracy(ep_sum, ref)
        assert passed, (
            f"EP sum vs reference: {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )

    def test_triton_compact_ep_filter(self):
        """The compact pre-pass shifts ids and emits -1 for non-local pairs."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.triton_compact import (
            compact_topk_ids,
        )

        # Shard [2, 8): rel = [3, 0, 7, 0, 5, 98] -> local mask [T,T,F,T,T,F];
        # unique local rel ids in first-occurrence order: 3, 0, 5.
        ids = torch.tensor([5, 2, 9, 2, 7, 100], dtype=torch.int32, device="cuda")
        compact = torch.empty(6, dtype=torch.int32, device="cuda")
        weight_ids = torch.full((6,), -7, dtype=torch.int32, device="cuda")
        count = torch.zeros(1, dtype=torch.int32, device="cuda")
        compact_topk_ids(
            ids,
            compact,
            weight_ids,
            count,
            local_expert_offset=2,
            num_local_experts=6,
        )
        assert compact.tolist() == [0, 1, -1, 1, 2, -1]
        assert weight_ids[:3].tolist() == [3, 0, 5]
        assert count.item() == 3

        # No shard: original behavior (every id gets a slot, global ids kept).
        compact_topk_ids(ids[:5], compact[:5], weight_ids, count)
        assert count.item() == 4  # uniques: 5, 2, 9, 7
        assert compact[:5].tolist() == [0, 1, 2, 1, 3]
        assert weight_ids[:4].tolist() == [5, 2, 9, 7]

    def test_nvfp4_micro_ep_takes_micro_kernel(self, monkeypatch):
        """Micro-range EP shards run the micro kernel at shard width."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        num_tokens, top_k, num_experts = 4, 2, 32
        hidden_size, intermediate_size = 256, 512
        n_local = num_experts // 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        shard = slice_b12x_moe_tensors_for_ep(
            tensors, local_expert_offset=n_local, num_local_experts=n_local
        )

        micro_weight_e = []
        real = moe_dispatch._get_micro_kernel

        def spy(*args, **kwargs):
            micro_weight_e.append(args[1])
            return real(*args, **kwargs)

        monkeypatch.setattr(moe_dispatch, "_get_micro_kernel", spy)
        partial = self._run_functional(
            shard,
            num_experts,
            top_k,
            num_local_experts=n_local,
            local_expert_offset=n_local,
        )
        assert micro_weight_e == [n_local], (
            f"EP micro-range call must compile the micro kernel at shard "
            f"width, got {micro_weight_e}"
        )
        assert not torch.isnan(partial).any()

    def test_nvfp4_rank_without_routed_tokens_returns_zeros(self):
        """A rank whose experts receive no tokens must output exact zeros."""
        num_tokens, top_k, num_experts = 8, 2, 32
        hidden_size, intermediate_size = 256, 512
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        # Route every token to the upper half of the expert space only.
        upper_ids = torch.randint(
            num_experts // 2,
            num_experts,
            (num_tokens, top_k),
            device="cuda",
            dtype=torch.int32,
        )
        tensors = dict(tensors)
        tensors["token_selected_experts"] = upper_ids

        n_local = num_experts // 2
        rank0 = slice_b12x_moe_tensors_for_ep(
            tensors, local_expert_offset=0, num_local_experts=n_local
        )
        partial0 = self._run_functional(
            rank0,
            num_experts,
            top_k,
            num_local_experts=n_local,
            local_expert_offset=0,
        )
        # Deterministic: every pair is dropped, Phase 0 zeroes the output.
        assert (partial0 == 0).all(), "rank without routed tokens must be all-zero"

        rank1 = slice_b12x_moe_tensors_for_ep(
            tensors, local_expert_offset=n_local, num_local_experts=n_local
        )
        partial1 = self._run_functional(
            rank1,
            num_experts,
            top_k,
            num_local_experts=n_local,
            local_expert_offset=n_local,
        )
        # All routing lands on rank 1's shard, so its partial must equal the
        # full-model run (tight bound — catches an over-dropping filter that
        # would otherwise hide under loose accuracy checks).
        full = self._run_functional(tensors, num_experts, top_k)
        self._assert_partials_match_full(partial1, full)
        ref = self._reference(
            tensors, num_tokens, num_experts, top_k, hidden_size, intermediate_size
        )
        passed, percent_within, atol = check_accuracy(
            partial0.float() + partial1.float(), ref
        )
        assert passed, (
            f"EP sum vs reference: {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )

    def test_w4a16_partial_sums_match_full(self):
        num_tokens, top_k, num_experts, ep_size = 32, 2, 32, 2
        hidden_size, intermediate_size = 256, 512
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )

        full = self._run_functional(tensors, num_experts, top_k, quant_mode="w4a16")
        ep_sum = self._ep_sum(tensors, num_experts, top_k, ep_size, quant_mode="w4a16")

        self._assert_partials_match_full(ep_sum, full)
        ref = self._reference(
            tensors, num_tokens, num_experts, top_k, hidden_size, intermediate_size
        )
        passed, percent_within, atol = check_accuracy(ep_sum, ref)
        assert passed, (
            f"W4A16 EP sum vs reference: {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )

    def test_nvfp4_unaligned_offset_shard(self):
        """Shards whose offset is not a multiple of 4 must launch and be correct.

        fp32 per-expert scale slices at offset % 4 != 0 are only 4-byte
        aligned; the compiled kernel must accept them (regression test for
        the assumed_align=16 rejection on sliced w1_alpha/w2_alpha). Under
        the shard-scaled cutovers this geometry runs the MICRO kernel; the
        static-regime twin below covers the static fakes.
        """
        num_tokens, top_k, num_experts = 16, 2, 32
        hidden_size, intermediate_size = 256, 512
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        # Restrict routing to experts [2, 6) so two tiny shards at offsets
        # 2 and 4 (both width 2) cover every routed expert and their partial
        # sums must reproduce the full-model run.
        tensors = dict(tensors)
        tensors["token_selected_experts"] = torch.randint(
            2, 6, (num_tokens, top_k), device="cuda", dtype=torch.int32
        )

        full = self._run_functional(tensors, num_experts, top_k)
        ep_sum = None
        for offset in (2, 4):
            shard = slice_b12x_moe_tensors_for_ep(
                tensors, local_expert_offset=offset, num_local_experts=2
            )
            # The point of this test: the offset-2 shard's sliced scale
            # tensors really are misaligned relative to 16 bytes (offset 4
            # lands back on a 16-byte boundary: 4 experts * 4 bytes).
            if offset % 4 != 0:
                assert shard["w1_alpha"].data_ptr() % 16 != 0
            partial = self._run_functional(
                shard,
                num_experts,
                top_k,
                num_local_experts=2,
                local_expert_offset=offset,
            )
            ep_sum = partial.float() if ep_sum is None else ep_sum + partial.float()

        self._assert_partials_match_full(ep_sum, full)

    def test_nvfp4_unaligned_offset_shard_static_regime(self, monkeypatch):
        """Unaligned-offset shards in the STATIC regime (micro cutover exceeded).

        Regression for the static kernel's [E] fp32 scale-fake alignment:
        expected local pairs = 1024 * 2/32 = 64 > 40 keeps this off the
        micro path, so the offset-2 shard exercises the static compile.
        The full-model baseline is pinned to the static scheduler too —
        static-vs-dynamic accumulation-order divergence at this
        expert-concentrated geometry exceeds the tight mismatch allowance
        (~1.8% observed) while static-vs-static sits at ~0.3%.
        """
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        monkeypatch.setitem(
            moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE, "fp4", 4096
        )
        num_tokens, top_k, num_experts = 512, 2, 32
        hidden_size, intermediate_size = 256, 512
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        tensors = dict(tensors)
        tensors["token_selected_experts"] = torch.randint(
            2, 6, (num_tokens, top_k), device="cuda", dtype=torch.int32
        )

        full = self._run_functional(tensors, num_experts, top_k)
        ep_sum = None
        for offset in (2, 4):
            shard = slice_b12x_moe_tensors_for_ep(
                tensors, local_expert_offset=offset, num_local_experts=2
            )
            if offset % 4 != 0:
                assert shard["w1_alpha"].data_ptr() % 16 != 0
            partial = self._run_functional(
                shard,
                num_experts,
                top_k,
                num_local_experts=2,
                local_expert_offset=offset,
            )
            ep_sum = partial.float() if ep_sum is None else ep_sum + partial.float()
        self._assert_partials_match_full(ep_sum, full)

    def test_nvfp4_micro_admission_bounded_by_global_pairs(self):
        """High-ratio EP in the micro window must not crash the Triton pre-pass.

        E=64 with width-2 shards at 150 tokens * top_k 8 gives 1200 GLOBAL
        pairs but only ~37 expected local pairs — inside the micro cutover.
        The pre-pass tops out at 1024 pairs (BLOCK^2 Triton tensor limit), so
        admission must also bound the global count and route this to static.
        """
        num_tokens, top_k, num_experts = 150, 8, 64
        hidden_size, intermediate_size = 256, 512
        n_local, offset = 2, 4
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        shard = slice_b12x_moe_tensors_for_ep(
            tensors, local_expert_offset=offset, num_local_experts=n_local
        )
        partial = self._run_functional(
            shard,
            num_experts,
            top_k,
            num_local_experts=n_local,
            local_expert_offset=offset,
        )
        assert not torch.isnan(partial).any()
        # The shard's contribution must match the reference restricted to
        # its experts: compare via full-model minus other-experts is noisy;
        # instead check against the full kernel run masked to shard experts.
        ids = tensors["token_selected_experts"]
        has_local = ((ids >= offset) & (ids < offset + n_local)).any(dim=1).unsqueeze(1)
        # Tokens with no local expert must be exactly zero.
        assert (partial[~has_local.squeeze(1)] == 0).all()

    def test_nvfp4_prefill_ep_runs_native_dynamic(self, monkeypatch):
        """Prefill-sized EP runs natively on the dynamic backend and matches."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        num_tokens, top_k, num_experts, ep_size = 512, 8, 32, 2
        hidden_size, intermediate_size = 256, 512
        # Precondition: this shape really is in the dynamic regime.
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=num_tokens, num_topk=top_k, activation_precision="fp4"
            )
            == "dynamic"
        )
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Spy on the dynamic launcher: EP shards must take it (no static
        # fallback — that path overflows Int32 packed-A offsets at scale).
        dynamic_calls = []
        real_dynamic = moe_dispatch.launch_sm120_dynamic_moe

        def spy_dynamic(**kwargs):
            dynamic_calls.append(kwargs["num_local_experts"])
            return real_dynamic(**kwargs)

        monkeypatch.setattr(moe_dispatch, "launch_sm120_dynamic_moe", spy_dynamic)

        full = self._run_functional(tensors, num_experts, top_k)
        ep_sum = self._ep_sum(tensors, num_experts, top_k, ep_size)
        assert dynamic_calls == [num_experts] + [num_experts // ep_size] * ep_size, (
            f"expected native dynamic launches, got {dynamic_calls}"
        )

        self._assert_partials_match_full(ep_sum, full)
        ref = self._reference(
            tensors, num_tokens, num_experts, top_k, hidden_size, intermediate_size
        )
        passed, percent_within, atol = check_accuracy(ep_sum, ref)
        assert passed, (
            f"EP prefill sum vs reference: {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )

    def test_nvfp4_prefill_ep_unaligned_and_empty_shards(self):
        """Dynamic-path EP: unaligned offsets launch; an empty shard is zero.

        Covers the dynamic kernel's all-drop path (no tasks published, no
        deadlock, exact-zero output) and the fp32 scale-slice alignment at
        offset % 4 != 0 (the dynamic fakes are compiled with align=4).
        """
        # num_experts=8 keeps width-2 shards in the dynamic regime under the
        # shard-scaled cutover (expected local pairs = 4096 * 2/8 = 1024 > 640).
        num_tokens, top_k, num_experts = 512, 8, 8
        hidden_size, intermediate_size = 256, 512
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        # Restrict routing to experts [2, 6): shards at offsets 2 and 4
        # (width 2, both prefill-sized -> dynamic backend) cover all routed
        # experts; the shard at offset 6 receives nothing.
        tensors = dict(tensors)
        tensors["token_selected_experts"] = torch.randint(
            2, 6, (num_tokens, top_k), device="cuda", dtype=torch.int32
        )

        full = self._run_functional(tensors, num_experts, top_k)
        ep_sum = None
        for offset in (2, 4):
            shard = slice_b12x_moe_tensors_for_ep(
                tensors, local_expert_offset=offset, num_local_experts=2
            )
            if offset % 4 != 0:
                assert shard["w1_alpha"].data_ptr() % 16 != 0
            partial = self._run_functional(
                shard,
                num_experts,
                top_k,
                num_local_experts=2,
                local_expert_offset=offset,
            )
            ep_sum = partial.float() if ep_sum is None else ep_sum + partial.float()
        self._assert_partials_match_full(ep_sum, full)

        empty = slice_b12x_moe_tensors_for_ep(
            tensors, local_expert_offset=6, num_local_experts=2
        )
        partial_empty = self._run_functional(
            empty,
            num_experts,
            top_k,
            num_local_experts=2,
            local_expert_offset=6,
        )
        assert (partial_empty == 0).all(), (
            "an EP shard with no routed pairs must return exact zeros"
        )

    def test_wrapper_ep_dynamic_cuda_graph_matches_functional(self):
        """EP wrapper on the DYNAMIC path (prefill-sized) with CUDA graph.

        Exercises needs_dynamic under EP: the wrapper allocates a dynamic
        workspace at weight_E=num_local_experts, run() picks it for a
        prefill-sized call, and graph capture/replay must match the
        functional shard run.
        """
        from flashinfer import B12xMoEWrapper
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        num_tokens, top_k, num_experts = 512, 8, 32
        hidden_size, intermediate_size = 256, 512
        n_local = num_experts // 2
        offset = n_local  # rank 1
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=num_tokens, num_topk=top_k, activation_precision="fp4"
            )
            == "dynamic"
        )
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        shard = slice_b12x_moe_tensors_for_ep(
            tensors, local_expert_offset=offset, num_local_experts=n_local
        )
        functional = self._run_functional(
            shard,
            num_experts,
            top_k,
            num_local_experts=n_local,
            local_expert_offset=offset,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
            num_local_experts=n_local,
            local_expert_offset=offset,
        )
        assert moe._dynamic_workspace is not None, (
            "prefill-sized EP wrapper must pre-allocate a dynamic workspace"
        )
        assert moe._dynamic_workspace.weight_E == n_local

        run_kwargs = dict(
            x=shard["x_bf16"],
            w1_weight=shard["w1_weight"],
            w1_weight_sf=shard["w1_weight_sf"],
            w1_alpha=shard["w1_alpha"],
            fc2_input_scale=shard["fc2_input_scale"],
            w2_weight=shard["w2_weight"],
            w2_weight_sf=shard["w2_weight_sf"],
            w2_alpha=shard["w2_alpha"],
            token_selected_experts=shard["token_selected_experts"],
            token_final_scales=shard["token_final_scales"],
        )
        for _ in range(3):
            moe.run(**run_kwargs)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = moe.run(**run_kwargs)
        g.replay()
        torch.cuda.synchronize()

        self._assert_partials_match_full(output, functional)

    def test_wrapper_ep_cuda_graph_matches_functional(self):
        """EP wrapper with CUDA graph capture matches the functional shard."""
        from flashinfer import B12xMoEWrapper

        num_tokens, top_k, num_experts = 64, 2, 32
        hidden_size, intermediate_size = 256, 512
        n_local = num_experts // 2
        offset = n_local  # rank 1
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        shard = slice_b12x_moe_tensors_for_ep(
            tensors, local_expert_offset=offset, num_local_experts=n_local
        )

        functional = self._run_functional(
            shard,
            num_experts,
            top_k,
            num_local_experts=n_local,
            local_expert_offset=offset,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
            num_local_experts=n_local,
            local_expert_offset=offset,
        )
        run_kwargs = dict(
            x=shard["x_bf16"],
            w1_weight=shard["w1_weight"],
            w1_weight_sf=shard["w1_weight_sf"],
            w1_alpha=shard["w1_alpha"],
            fc2_input_scale=shard["fc2_input_scale"],
            w2_weight=shard["w2_weight"],
            w2_weight_sf=shard["w2_weight_sf"],
            w2_alpha=shard["w2_alpha"],
            token_selected_experts=shard["token_selected_experts"],
            token_final_scales=shard["token_final_scales"],
        )

        for _ in range(3):
            moe.run(**run_kwargs)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = moe.run(**run_kwargs)
        g.replay()
        torch.cuda.synchronize()

        # Same backend + identical shard tensors: only bf16 atomic-scatter
        # ordering differs (~2e-3), so a tight bound applies.
        torch.testing.assert_close(
            output.float(), functional.float(), atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
