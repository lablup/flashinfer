"""
B12x fused MoE API for SM120/SM121.

Provides high-level APIs for running Mixture of Experts (MoE)
computations using b12x CuTe DSL kernels on Blackwell GeForce GPUs.

The b12x kernels take bf16 input and run the SM12x MoE route, FC1,
activation, FC2, and scatter pipeline through the selected backend.
Supports SiLU (gated, SwiGLU) and ReLU2 (non-gated, Nemotron-Super)
activations.

Two APIs are provided:

1. **Functional API** (`b12x_fused_moe`):
   Simple function call. Workspace is cached internally.

2. **Wrapper API** (`B12xMoEWrapper`):
   Class-based API with pre-allocated buffers for CUDA graph compatibility.

Example (Functional API):
    >>> from flashinfer import b12x_fused_moe
    >>> output = b12x_fused_moe(
    ...     x=hidden_states_bf16,
    ...     w1_weight=w1_fp4, w1_weight_sf=w1_sf, w1_alpha=w1_alpha,
    ...     fc2_input_scale=fc2_scale,
    ...     w2_weight=w2_fp4, w2_weight_sf=w2_sf, w2_alpha=w2_alpha,
    ...     token_selected_experts=topk_ids, token_final_scales=topk_weights,
    ...     num_experts=256, top_k=8,
    ... )

Example (Wrapper API with CUDA Graph):
    >>> from flashinfer import B12xMoEWrapper
    >>> moe = B12xMoEWrapper(
    ...     num_experts=256, top_k=8, hidden_size=4096,
    ...     intermediate_size=14336, use_cuda_graph=True,
    ... )
    >>> output = moe.run(x=hidden_states_bf16, ...)
"""

import logging
import os
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

from ...api_logging import flashinfer_api
from ...trace.templates.moe import b12x_fused_moe_trace, b12x_moe_wrapper_run_trace
from ...utils import supported_compute_capability

# "0" disables the wrapper's in-place weight-padding rebind (see
# B12xMoEWrapper.run): padded copies are then only cached on the wrapper and
# the caller's original tensors stay alive, roughly doubling MoE weight
# memory when the per-rank intermediate size is not a multiple of 128.
_PAD_IN_PLACE_ENV = "FLASHINFER_B12X_PAD_IN_PLACE"


def _pad_in_place_enabled() -> bool:
    return os.environ.get(_PAD_IN_PLACE_ENV, "1") != "0"


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _validate_expert_shard(
    *, num_experts: int, num_local_experts: int, local_expert_offset: int
) -> None:
    """Validate an expert-parallel shard: a contiguous slice of experts."""
    if num_local_experts < 1:
        raise ValueError(f"num_local_experts must be >= 1, got {num_local_experts}")
    if local_expert_offset < 0:
        raise ValueError(f"local_expert_offset must be >= 0, got {local_expert_offset}")
    if local_expert_offset + num_local_experts > num_experts:
        raise ValueError(
            "local_expert_offset + num_local_experts must not exceed "
            f"num_experts ({local_expert_offset} + {num_local_experts} > "
            f"{num_experts})."
        )


@supported_compute_capability([120, 121])
@flashinfer_api(trace=b12x_fused_moe_trace)
def b12x_fused_moe(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    *,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    fc2_input_scale: Optional[torch.Tensor] = None,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    output: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: Optional[float] = None,
    activation_precision: str = "fp4",
    quant_mode: Optional[str] = None,
    source_format: str = "modelopt",
) -> torch.Tensor:
    r"""Run fused MoE on SM120/SM121 using b12x CuTe-DSL kernels.

    The kernel takes bf16 input and runs routing, FC1, activation, FC2, and
    scatter through the selected backend.  Automatically selects the micro
    (decode), static, or dynamic backend based on the routed row count.

    Parameters
    ----------
    x : torch.Tensor
        Input activations of shape ``[num_tokens, hidden_size]``, ``bfloat16``.
    w1_weight : torch.Tensor
        FC1 weights, FP4 packed.  Gated (SiLU) layout
        ``[E, 2 * intermediate_size, hidden_size // 2]``; non-gated (ReLU2)
        layout ``[E, intermediate_size, hidden_size // 2]``.
    w1_weight_sf : torch.Tensor
        Scale factors for ``w1_weight``.
    w2_weight : torch.Tensor
        FC2 weights of shape ``[E, hidden_size, intermediate_size // 2]``,
        FP4.
    w2_weight_sf : torch.Tensor
        Scale factors for ``w2_weight``.
    token_selected_experts : torch.Tensor
        Expert assignments of shape ``[num_tokens, top_k]``.
    token_final_scales : torch.Tensor
        Routing weights of shape ``[num_tokens, top_k]``.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts routed to per token.
    w1_alpha : torch.Tensor
        Per-expert global scale for FC1.
    w2_alpha : torch.Tensor
        Per-expert global scale for FC2.
    fc2_input_scale : Optional[torch.Tensor]
        Global scale for FC2 input quantization.  Required for
        ``quant_mode="nvfp4"``; accepted but ignored for
        ``quant_mode="w4a16"``.
    num_local_experts : Optional[int]
        Number of experts owned by this rank under expert parallelism.
        Defaults to ``num_experts`` (no EP).
    local_expert_offset : int
        Start of this rank's contiguous expert shard in the global expert
        space.  ``token_selected_experts`` always carries GLOBAL expert ids;
        the kernel drops ids outside
        ``[local_expert_offset, local_expert_offset + num_local_experts)``.
        All weight and per-expert scale tensors (``w1_weight``,
        ``w1_weight_sf``, ``w1_alpha``, ``fc2_input_scale`` when per-expert,
        ``w2_*``) must be this rank's ``[num_local_experts, ...]`` slices.
        The returned tensor is the PARTIAL sum over local experts — zero
        rows for tokens with no local expert — and the caller must sum
        (all-reduce) partial outputs across EP ranks.
        ``token_final_scales`` must be normalized over the global top-k;
        do not renormalize per rank.  Defaults to ``0``.
    output : Optional[torch.Tensor]
        Pre-allocated output buffer of shape ``[num_tokens, hidden_size]``,
        ``bfloat16``.
    output_dtype : torch.dtype
        Output data type.  Only ``torch.bfloat16`` is currently supported.
    activation : str
        Activation function — ``"silu"`` (gated SwiGLU), ``"gelu_tanh"`` (gated
        tanh-approx GeGLU), ``"swigluoai_uninterleave"`` (gated SwiGLU-OAI)
        or ``"relu2"`. Defaults to ``"silu"``.
    swiglu_alpha, swiglu_beta, swiglu_limit : float
        SwiGLU-OAI parameters used only when
        ``activation="swigluoai_uninterleave"``: ``gate*sigmoid(alpha*gate)*
        (up+beta)`` with optional clamp to ``swiglu_limit`` (``None`` disables).
        Defaults to 1.702 / 1.0 / None as standard parameters for approximating GELU.
    activation_precision : str
        Backward-compatible alias for ``quant_mode``.  ``"fp4"`` selects
        ``quant_mode="nvfp4"``; ``"bf16"`` selects ``quant_mode="w4a16"``.
    quant_mode : Optional[str]
        Quantization mode, ``"nvfp4"`` / ``"w4a4"`` or ``"w4a16"``.  When set,
        selects the backend and internal workspace family.
    source_format : str
        Checkpoint source format — ``"modelopt"`` or ``"compressed_tensors"``.
        Defaults to ``"modelopt"``.  For ``quant_mode="w4a16"`` this selects
        the scale translation applied at weight-preparation time.  For
        ``quant_mode="nvfp4"`` it is accepted as pass-through provenance
        metadata: tensors must already be in kernel convention (full dequant
        block scales; ``w2_alpha == fc2_input_scale``) — use
        :func:`flashinfer.fused_moe.prepare.prepare_b12x_nvfp4_packed_weights`
        to translate raw checkpoint tensors.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``[num_tokens, hidden_size]``.
    """
    from ...jit.cpp_ext import get_cuda_version

    if get_cuda_version().major < 13:
        raise ValueError(
            "b12x fused MoE requires CUDA 13 or later. "
            f"Current CUDA version: {get_cuda_version()}."
        )

    # SM12x kernels hardcode BF16 for scatter_output (see Phase 0 zero-init
    # in moe_{static,micro,dynamic}_kernel.py). Other dtypes will fail the
    # kernel tensor binding.
    if output_dtype != torch.bfloat16:
        raise ValueError(
            f"b12x fused MoE only supports output_dtype=torch.bfloat16, "
            f"got {output_dtype}."
        )
    if output is not None and output.dtype != torch.bfloat16:
        raise ValueError(
            f"b12x fused MoE only supports bf16 output buffers, "
            f"got output.dtype={output.dtype}."
        )

    if num_local_experts is None:
        num_local_experts = num_experts

    _validate_expert_shard(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
    )

    from .blackwell_sm12x.moe_dispatch import (
        _normalize_quant_mode,
        _normalize_source_format_for_quant_mode,
    )

    # Validate/normalize eagerly so bad values fail before buffer allocation.
    source_format = _normalize_source_format_for_quant_mode(
        source_format, _normalize_quant_mode(quant_mode, activation_precision)
    )

    num_tokens = token_selected_experts.size(0)
    hidden_size = x.size(1)

    if output is None:
        if _is_cuda_graph_capturing():
            raise RuntimeError(
                "b12x_fused_moe requires a pre-allocated output buffer during "
                "CUDA graph capture."
            )
        output = torch.empty(
            (num_tokens, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )

    from .blackwell_sm12x.moe_dispatch import launch_sm120_moe

    return launch_sm120_moe(
        a=x,
        topk_ids=token_selected_experts,
        topk_weights=token_final_scales,
        w1_weight=w1_weight,
        w1_weight_sf=w1_weight_sf,
        w1_alpha=w1_alpha,
        fc2_input_scale=fc2_input_scale,
        w2_weight=w2_weight,
        w2_weight_sf=w2_weight_sf,
        w2_alpha=w2_alpha,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        scatter_output=output,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        source_format=source_format,
    )


class B12xMoEWrapper:
    """B12x fused MoE wrapper for SM120/SM121 with CUDA graph support.

    Pre-allocates workspace buffers for CUDA graph compatibility.
    Automatically selects micro/static/dynamic backend per call.

    Args:
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        hidden_size: Hidden dimension size.
        intermediate_size: Intermediate size.
        use_cuda_graph: Pre-allocate buffers for CUDA graph compatibility.
        max_num_tokens: Maximum tokens (only for use_cuda_graph=True).
        num_local_experts: Local experts for EP. Default: num_experts.
        output_dtype: Output data type. Only torch.bfloat16 is currently
            supported. Default: torch.bfloat16.
        device: Device for buffer allocation. Default: "cuda".
        activation: Activation — "silu", "gelu_tanh", "swigluoai_uninterleave", or
            "relu2". Default: "silu". swiglu_alpha/beta/limit apply to swigluoai.
        activation_precision: Backward-compatible alias for quant_mode.
            "fp4" selects quant_mode="nvfp4"; "bf16" selects quant_mode="w4a16".
        quant_mode: Quantization mode, "nvfp4"/"w4a4" or "w4a16". When set,
            this selects the backend and internal workspace family.
        source_format: Checkpoint source format, "modelopt" or
            "compressed_tensors". Default: "modelopt". Selects the w4a16
            scale translation; pass-through provenance metadata for nvfp4
            (tensors must already be in kernel convention — see
            prepare_b12x_nvfp4_packed_weights).

    Example:
        >>> moe = B12xMoEWrapper(num_experts=256, top_k=8, ...)
        >>> output = moe.run(x=hidden_states_bf16, ...)
    """

    @supported_compute_capability([120, 121])
    @flashinfer_api
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        use_cuda_graph: bool = False,
        max_num_tokens: int = 4096,
        num_local_experts: Optional[int] = None,
        local_expert_offset: int = 0,
        output_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        activation: str = "silu",
        swiglu_alpha: float = 1.702,
        swiglu_beta: float = 1.0,
        swiglu_limit: Optional[float] = None,
        activation_precision: str = "fp4",
        quant_mode: Optional[str] = None,
        source_format: str = "modelopt",
    ):
        r"""Configure the b12x fused-MoE wrapper.

        Parameters
        ----------
        num_experts : int
            Total number of experts.
        top_k : int
            Number of experts routed to per token.
        hidden_size : int
            Hidden dimension size.
        intermediate_size : int
            Intermediate dimension size.
        use_cuda_graph : bool
            If ``True``, pre-allocate workspace buffers sized for
            ``max_num_tokens`` so the wrapper can be captured into a CUDA
            graph.  Defaults to ``False``.
        max_num_tokens : int
            Maximum batch size, only used when ``use_cuda_graph=True``.
            Defaults to ``4096``.
        num_local_experts : Optional[int]
            Number of experts owned by this rank under expert parallelism.
            Defaults to ``num_experts`` (no EP).
        local_expert_offset : int
            Start of this rank's contiguous expert shard in the global
            expert space (see :func:`b12x_fused_moe` for the full EP
            contract: global routing ids, local weight/scale slices,
            partial-sum output reduced by the caller).  Defaults to ``0``.
            EP runs natively on all backends (micro / static / dynamic /
            w4a16); backend selection scales its cutovers by the shard
            fraction so shards keep low-latency kernels longer.
        output_dtype : torch.dtype
            Output dtype.  Only ``torch.bfloat16`` is currently supported.
        device : str
            Device on which to allocate workspace buffers.  Defaults to
            ``"cuda"``.
        activation : str
            Activation function — ``"silu"`` (gated SwiGLU), ``"gelu_tanh"``
            (gated GeGLU, tanh-approx GELU), ``"swigluoai_uninterleave"`` (gated
            SwiGLU-OAI) or ``"relu2"`` (non-gated). Defaults to ``"silu"``.
        swiglu_alpha, swiglu_beta, swiglu_limit : float
            SwiGLU-OAI parameters (only for ``"swigluoai_uninterleave"``):
            ``gate*sigmoid(alpha*gate)*(up+beta)`` with optional clamp to
            ``swiglu_limit`` (``None`` disables). Defaults 1.702 / 1.0 / None.
        activation_precision : str
            Backward-compatible alias for ``quant_mode``.  ``"fp4"`` selects
            ``quant_mode="nvfp4"``; ``"bf16"`` selects ``quant_mode="w4a16"``.
        quant_mode : Optional[str]
            Quantization mode, ``"nvfp4"`` / ``"w4a4"`` or ``"w4a16"``.
        source_format : str
            Checkpoint source format — ``"modelopt"`` (default) or
            ``"compressed_tensors"``.  Selects the w4a16 scale translation;
            pass-through provenance metadata for ``quant_mode="nvfp4"``
            (tensors must already be in kernel convention — see
            ``prepare_b12x_nvfp4_packed_weights``).
        """
        from ...jit.cpp_ext import get_cuda_version
        from .blackwell_sm12x.moe_dispatch import (
            _activation_precision_from_quant_mode,
            _normalize_quant_mode,
            _normalize_source_format_for_quant_mode,
        )

        if get_cuda_version().major < 13:
            raise ValueError(
                "b12x fused MoE requires CUDA 13 or later. "
                f"Current CUDA version: {get_cuda_version()}."
            )

        # SM12x kernels hardcode BF16 for scatter_output.
        if output_dtype != torch.bfloat16:
            raise ValueError(
                f"b12x fused MoE only supports output_dtype=torch.bfloat16, "
                f"got {output_dtype}."
            )

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_cuda_graph = use_cuda_graph
        self.max_num_tokens = max_num_tokens
        # Only None means "unset" — 0 must reach validation and fail loudly,
        # matching b12x_fused_moe (an `or` would silently coerce 0 to
        # num_experts and full-width weights would double-count after the
        # caller's EP all-reduce).
        self.num_local_experts = (
            num_local_experts if num_local_experts is not None else num_experts
        )
        self.local_expert_offset = local_expert_offset

        _validate_expert_shard(
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
        )
        self.output_dtype = output_dtype
        self.device = device
        self.activation = activation
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
        self.activation_precision = _activation_precision_from_quant_mode(
            self.quant_mode
        )
        # Validate/normalize eagerly so invalid values fail at construction
        # rather than on the first run() call.
        self.source_format = _normalize_source_format_for_quant_mode(
            source_format, self.quant_mode
        )

        # Pre-allocated objects. Both workspace slots may be populated so
        # run() can pick per-call; without this, the backend would be locked
        # to whichever workspace was allocated at init time.
        self._static_workspace: object = None
        self._dynamic_workspace: object = None
        self._weight_views: object = None
        self._weight_key: Optional[Tuple] = None
        self._padded_weights: Any = None
        self._padded_weight_key: Optional[Tuple] = None
        self._moe_output: Optional[torch.Tensor] = None

        if use_cuda_graph:
            self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Pre-allocate buffers for CUDA graph compatibility."""
        from .blackwell_sm12x.moe_dispatch import (
            _get_cached_workspace,
            select_sm120_moe_backend,
            _get_static_compact_cutover_pairs,
        )

        # Workspaces come from the module-level pool so wrappers with the
        # same geometry SHARE scratch buffers. Serving stacks build one
        # wrapper per MoE layer (e.g. 48 for Solar-Open2); private buffers
        # would multiply ~0.5 GB of scratch by the layer count. Sharing is
        # stream-order-safe: every kernel fully rewrites its scratch in
        # Phase 0 and layers execute sequentially (callers copy the output
        # before the next layer runs), including inside one CUDA graph.

        max_routed_rows = self.max_num_tokens * self.top_k
        if self.quant_mode == "w4a16":
            self._static_workspace = _get_cached_workspace(
                backend="w4a16",
                state_E=self.num_local_experts,
                weight_E=self.num_experts,
                routed_rows=max_routed_rows,
                k=self.hidden_size,
                n=self.intermediate_size,
                num_topk=self.top_k,
                device=torch.device(self.device),
                quant_mode=self.quant_mode,
                activation=self.activation,
                local_expert_offset=self.local_expert_offset,
            )
            self._moe_output = torch.empty(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.output_dtype,
                device=self.device,
            )
            return

        # Allocate a dynamic workspace alongside the static one when
        # max_num_tokens is large enough to cross the cutover. This lets
        # run() adapt backend per call rather than locking at init.
        needs_dynamic = (
            select_sm120_moe_backend(
                num_tokens=self.max_num_tokens,
                num_topk=self.top_k,
                activation_precision=self.activation_precision,
                num_local_experts=self.num_local_experts,
                num_experts=self.num_experts,
            )
            == "dynamic"
        )

        # When both workspaces exist, static only serves calls whose
        # EXPECTED-local pair count is <= cutover; size it accordingly to
        # avoid paying for the full capacity twice. Under EP the floor-based
        # selection admits up to ((cutover+1) * num_experts - 1) //
        # num_local GLOBAL pairs, and skewed routing could land every one of
        # them on this shard — capacity must cover that exact admission
        # bound, not the expectation (launch_sm120_static_moe re-validates).
        static_cutover = _get_static_compact_cutover_pairs(self.activation_precision)
        if self.num_local_experts < self.num_experts:
            static_cutover = (
                (static_cutover + 1) * self.num_experts - 1
            ) // self.num_local_experts
        static_max_rows = (
            min(max_routed_rows, static_cutover) if needs_dynamic else max_routed_rows
        )
        # weight_E is the weight tensors' expert count — this rank's shard
        # width under EP (the static kernel binds per-expert tensors at that
        # width; the shard offset lives in the compiled kernel, not here).
        self._static_workspace = _get_cached_workspace(
            backend="static",
            state_E=self.num_local_experts,
            weight_E=self.num_local_experts,
            routed_rows=max(1, static_max_rows),
            k=self.hidden_size,
            n=self.intermediate_size,
            num_topk=self.top_k,
            device=torch.device(self.device),
            quant_mode=self.quant_mode,
            activation=self.activation,
            local_expert_offset=self.local_expert_offset,
        )

        if needs_dynamic:
            self._dynamic_workspace = _get_cached_workspace(
                backend="dynamic",
                state_E=self.num_local_experts,
                weight_E=self.num_local_experts,
                routed_rows=max_routed_rows,
                k=self.hidden_size,
                n=self.intermediate_size,
                num_topk=self.top_k,
                device=torch.device(self.device),
                quant_mode=self.quant_mode,
                activation=self.activation,
                local_expert_offset=self.local_expert_offset,
            )

        # Allocated after arch-specific buffers to preserve memory layout
        # that the autotuner's CUDA graph profiling is sensitive to.
        self._moe_output = torch.empty(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.output_dtype,
            device=self.device,
        )

    @flashinfer_api(trace=b12x_moe_wrapper_run_trace)
    def run(
        self,
        x: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        *,
        w1_alpha: torch.Tensor,
        w2_alpha: torch.Tensor,
        fc2_input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run the b12x fused-MoE forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input activations of shape ``[num_tokens, hidden_size]``,
            ``bfloat16``.
        w1_weight : torch.Tensor
            FC1 weights, FP4-packed.
        w1_weight_sf : torch.Tensor
            Scale factors for ``w1_weight``.
        w2_weight : torch.Tensor
            FC2 weights, FP4-packed.
        w2_weight_sf : torch.Tensor
            Scale factors for ``w2_weight``.
        token_selected_experts : torch.Tensor
            Expert assignments of shape ``[num_tokens, top_k]``.
        token_final_scales : torch.Tensor
            Routing weights of shape ``[num_tokens, top_k]``.
        w1_alpha : torch.Tensor
            Per-expert global scale for FC1.
        w2_alpha : torch.Tensor
            Per-expert global scale for FC2.
        fc2_input_scale : Optional[torch.Tensor]
            Global scale for FC2 input quantization.  Required for
            ``quant_mode="nvfp4"``; accepted but ignored for ``"w4a16"``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[num_tokens, hidden_size]``.

        Notes
        -----
        When ``quant_mode="nvfp4"`` and the per-rank intermediate size is not
        a multiple of 128, the weights must be zero-padded to the tile
        boundary. On the first such call the wrapper builds padded copies and
        **repoints the caller's weight tensors** (``w1_weight``,
        ``w1_weight_sf``, ``w2_weight``, ``w2_weight_sf``) at the padded
        storage via ``.data`` rebinding, so the unpadded originals can be
        freed instead of staying resident next to the padded copies (which
        would roughly double MoE weight memory). The caller-visible tensors
        change shape along the intermediate dimension; the extra channels are
        zero and the results are numerically identical. Set
        ``FLASHINFER_B12X_PAD_IN_PLACE=0`` to disable the rebind and keep the
        old keep-both-copies behavior.
        """
        num_tokens = token_selected_experts.size(0)

        if self.use_cuda_graph and num_tokens > self.max_num_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds max_num_tokens "
                f"({self.max_num_tokens})"
            )

        if self.use_cuda_graph:
            moe_output = self._moe_output[:num_tokens]
        else:
            if _is_cuda_graph_capturing():
                raise RuntimeError(
                    "B12xMoEWrapper must be constructed with use_cuda_graph=True "
                    "to run during CUDA graph capture."
                )
            moe_output = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=self.output_dtype,
                device=x.device,
            )

        from .blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
            select_sm120_moe_backend,
            _get_weight_views as _get_sm120_weight_views,
            _pad_intermediate_to_tile,
            _LEVEL_TILE_N,
            is_gated_activation,
        )

        # Pick the right pre-allocated workspace for this call's token
        # count. launch_sm120_moe otherwise infers backend from workspace
        # type, which would lock us to whichever one was allocated at init.
        workspace = None
        if self.use_cuda_graph:
            if self.quant_mode == "w4a16":
                workspace = self._static_workspace
            elif (
                self._dynamic_workspace is not None
                and select_sm120_moe_backend(
                    num_tokens=num_tokens,
                    num_topk=self.top_k,
                    activation_precision=self.activation_precision,
                    num_local_experts=self.num_local_experts,
                    num_experts=self.num_experts,
                )
                == "dynamic"
            ):
                workspace = self._dynamic_workspace
            else:
                workspace = self._static_workspace

        if self.quant_mode == "nvfp4":
            is_gated = is_gated_activation(self.activation)
            # The effective intermediate size comes from the weight tensor,
            # not the configured value: after an earlier in-place pad the
            # caller's tensors are already tile-aligned while
            # self.intermediate_size still holds the unpadded size (matching
            # how launch_sm120_moe infers it).
            n_eff = w1_weight.size(1) // 2 if is_gated else w1_weight.size(1)
            # Pad non-128-aligned intermediate sizes once and cache.
            if n_eff % _LEVEL_TILE_N != 0:
                padded_weight_key = (
                    self.quant_mode,
                    w1_weight.data_ptr(),
                    w1_weight_sf.data_ptr(),
                    w1_alpha.data_ptr(),
                    w2_weight.data_ptr(),
                    w2_weight_sf.data_ptr(),
                    w2_alpha.data_ptr(),
                    fc2_input_scale.data_ptr() if fc2_input_scale is not None else 0,
                )
                if (
                    self._padded_weights is None
                    or self._padded_weight_key != padded_weight_key
                ):
                    # Build the padded copies as normal tensors even under
                    # torch.inference_mode() (serving forward passes): they
                    # replace the caller's load-time weights below, and
                    # inference tensors would break any later
                    # out-of-inference-mode weight mutation by the caller.
                    with torch.inference_mode(False):
                        self._padded_weights = _pad_intermediate_to_tile(
                            w1_weight,
                            w1_weight_sf,
                            w2_weight,
                            w2_weight_sf,
                            fc2_input_scale,
                            n_eff,
                            _LEVEL_TILE_N,
                            self.hidden_size,
                            w1_weight.size(0),
                            is_gated,
                            cache=False,
                        )
                    self._padded_weight_key = padded_weight_key
                    if _pad_in_place_enabled():
                        # Repoint the caller's tensors at the padded storage
                        # so the unpadded originals can be freed. Callers
                        # (e.g. vLLM) hold the weights for the process
                        # lifetime and pass the same tensor objects every
                        # call; without this, original + padded copy stay
                        # resident together (~2.2x MoE weight memory —
                        # Solar-Open2 TP4 lost ~41 GiB of KV cache to it).
                        # Rebinding via .data mirrors how serving stacks
                        # rewrite weights in process_weights_after_loading.
                        # Later calls then arrive with tile-aligned tensors
                        # and skip this branch; if the caller passes fresh
                        # objects instead, the key above still matches and
                        # the wrapper-held copy is reused as before.
                        w1p, w1_sf_p, w2p, w2_sf_p, _, _ = self._padded_weights
                        try:
                            w1_weight.data = w1p
                            w1_weight_sf.data = w1_sf_p
                            w2_weight.data = w2p
                            w2_weight_sf.data = w2_sf_p
                        except RuntimeError as exc:
                            # Keep serving off the wrapper-held padded copy —
                            # correct but keeps the unpadded originals
                            # resident too (~2x MoE weight memory), so warn.
                            logger.warning(
                                "B12xMoEWrapper: in-place weight-padding "
                                "rebind failed (%s); falling back to cached "
                                "padded copies. Original + padded weights "
                                "will both stay resident.",
                                exc,
                            )
                (
                    w1_weight,
                    w1_weight_sf,
                    w2_weight,
                    w2_weight_sf,
                    fc2_input_scale,
                    n_eff,
                ) = self._padded_weights

            # Cache weight views; invalidate if weight pointers change. Keyed
            # AFTER padding so the key is stable across calls on both the
            # rebind path (callers now pass the padded tensors) and the
            # fallback path (wrapper-held padded tensors).
            weight_key = (
                self.quant_mode,
                w1_weight.data_ptr(),
                w1_weight_sf.data_ptr(),
                w1_alpha.data_ptr(),
                w2_weight.data_ptr(),
                w2_weight_sf.data_ptr(),
                w2_alpha.data_ptr(),
            )
            if self._weight_views is None or self._weight_key != weight_key:
                self._weight_views = _get_sm120_weight_views(
                    w1_fp4=w1_weight,
                    w1_blockscale=w1_weight_sf,
                    w2_fp4=w2_weight,
                    w2_blockscale=w2_weight_sf,
                    w1_alphas=w1_alpha,
                    w2_alphas=w2_alpha,
                    n=n_eff,
                    k=self.hidden_size,
                    activation_precision=self.activation_precision,
                )
                self._weight_key = weight_key
        else:
            self._weight_views = None
            self._weight_key = None

        return launch_sm120_moe(
            a=x,
            topk_ids=token_selected_experts,
            topk_weights=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            scatter_output=moe_output,
            activation=self.activation,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            activation_precision=self.activation_precision,
            quant_mode=self.quant_mode,
            source_format=self.source_format,
            _workspace=workspace,
            _weight_views=self._weight_views,
        )
