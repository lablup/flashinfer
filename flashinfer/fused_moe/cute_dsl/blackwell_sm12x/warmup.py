"""Pre-compile the SM12x B12x MoE kernel set for a model deployment.

Populates the persistent CuTe-DSL kernel cache
(``cached_ops/b12x_moe_<arch>_cute_dsl/``) with every kernel a vLLM
deployment of the model will need, so engine startup (CUDA-graph capture in
particular) loads object files instead of running ``cute.compile``.

The tool drives the real ``B12xMoEWrapper`` dispatch path with dummy weights,
constructed with the same arguments vLLM's ``FlashInferB12xExperts`` uses —
cache keys therefore match the deployment by construction. Kernels are
device-specific (SM count participates in the specialization), so run this on
the same GPU model that will serve.

Usage::

    python -m flashinfer.fused_moe.cute_dsl.blackwell_sm12x.warmup \
        --config /path/to/model --tp-size 2 [--workers 4]

Models with per-layer expert counts (``n_routed_experts_per_layer`` in
config.json, e.g. per-layer-pruned checkpoints) warm every distinct count.
The bake workflow: run once per (GPU model, TP degree), then snapshot the
FlashInfer workspace (``~/.cache/flashinfer``) into the serving image or a
persistent volume.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _vllm_default_capture_sizes(max_size: int) -> list[int]:
    """vLLM's default cudagraph capture sizes: 1, 2, 4, then multiples of 8."""
    sizes = [s for s in (1, 2, 4) if s <= max_size]
    sizes += list(range(8, max_size + 1, 8))
    return sizes


def _load_model_geometry(config_path: Path) -> dict:
    if config_path.is_dir():
        config_path = config_path / "config.json"
    cfg = json.loads(config_path.read_text())
    experts = cfg.get("n_routed_experts_per_layer") or [cfg["n_routed_experts"]]
    return {
        "hidden_size": cfg["hidden_size"],
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "top_k": cfg["num_experts_per_tok"],
        "activation": cfg.get("hidden_act", "silu"),
        "expert_counts": sorted(set(experts)),
    }


def _make_dummy_weights(E: int, n: int, k: int, device: str):
    """Random tensors in the exact layouts the wrapper consumes.

    Values are irrelevant (warmup only needs the dispatch path to run);
    shapes and the MMA scale-factor layout must match vLLM's
    ``process_weights_after_loading`` output.
    """
    import torch

    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from .moe_w4a16_fp4_helpers import swizzle_block_scale

    sf_vec = 16

    def sf_mma(rows: int, cols: int) -> torch.Tensor:
        lin = torch.full(
            (E, rows, cols // sf_vec), 0.01, dtype=torch.float32, device=device
        )
        sw = torch.stack([swizzle_block_scale(lin[e]) for e in range(E)], 0)
        sw2d = sw.reshape(E * sw.shape[1], sw.shape[2]).to(torch.float8_e4m3fn)
        return convert_sf_to_mma_layout(sw2d, m=rows, k=cols, num_groups=E)

    return {
        "w1_weight": torch.randint(
            0, 255, (E, 2 * n, k // 2), dtype=torch.uint8, device=device
        ),
        "w1_weight_sf": sf_mma(2 * n, k),
        "w2_weight": torch.randint(
            0, 255, (E, k, n // 2), dtype=torch.uint8, device=device
        ),
        "w2_weight_sf": sf_mma(k, n),
        "w1_alpha": torch.ones(E, dtype=torch.float32, device=device),
        "w2_alpha": torch.ones(E, dtype=torch.float32, device=device),
        "fc2_input_scale": torch.ones(E, dtype=torch.float32, device=device),
    }


def _warm_one_expert_count(
    E: int,
    *,
    hidden_size: int,
    n: int,
    top_k: int,
    activation: str,
    token_sizes: list[int],
    max_num_tokens: int,
    source_format: str,
) -> None:
    import torch

    from flashinfer.fused_moe import B12xMoEWrapper

    device = "cuda"
    weights = _make_dummy_weights(E, n, hidden_size, device)
    wrapper = B12xMoEWrapper(
        num_experts=E,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=n,
        use_cuda_graph=True,
        max_num_tokens=max_num_tokens,
        num_local_experts=E,
        local_expert_offset=0,
        activation=activation,
        activation_precision="fp4",
        source_format=source_format,
    )
    for m in token_sizes:
        x = torch.randn(m, hidden_size, dtype=torch.bfloat16, device=device)
        topk_ids = torch.randint(0, E, (m, top_k), dtype=torch.int32, device=device)
        topk_w = torch.softmax(torch.rand(m, top_k, device=device), dim=-1)
        wrapper.run(
            x,
            weights["w1_weight"],
            weights["w1_weight_sf"],
            weights["w2_weight"],
            weights["w2_weight_sf"],
            topk_ids,
            topk_w,
            w1_alpha=weights["w1_alpha"],
            w2_alpha=weights["w2_alpha"],
            fc2_input_scale=weights["fc2_input_scale"],
        )
    torch.cuda.synchronize()


def _worker(worker_args: tuple) -> str:
    E, kwargs = worker_args
    t0 = time.perf_counter()
    _warm_one_expert_count(E, **kwargs)
    return f"E={E}: done in {time.perf_counter() - t0:.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Model directory or config.json path",
    )
    parser.add_argument("--tp-size", required=True, type=int)
    parser.add_argument(
        "--max-capture-size",
        type=int,
        default=512,
        help="vLLM max_cudagraph_capture_size (default 512)",
    )
    parser.add_argument(
        "--capture-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit capture sizes; default is vLLM's 1,2,4,8,16,...,max",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=8192,
        help="vLLM max_num_batched_tokens; sizes the wrapper workspaces "
        "(must match the deployment for cache keys to hit) (default 8192)",
    )
    parser.add_argument(
        "--source-format",
        default="compressed_tensors",
        choices=["compressed_tensors", "modelopt"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel compile processes, partitioned by expert count",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs="*",
        default=None,
        help="Warm only these expert counts (default: all distinct counts "
        "from the model config; useful for smoke tests or splitting a bake)",
    )
    args = parser.parse_args()

    geo = _load_model_geometry(args.config)
    if geo["moe_intermediate_size"] % args.tp_size:
        raise SystemExit(
            f"moe_intermediate_size {geo['moe_intermediate_size']} is not "
            f"divisible by tp_size {args.tp_size}"
        )
    n = geo["moe_intermediate_size"] // args.tp_size
    token_sizes = args.capture_sizes or _vllm_default_capture_sizes(
        args.max_capture_size
    )
    # The profile run (max_num_tokens) exercises the same dynamic kernel the
    # large capture sizes compile, but include it for completeness.
    token_sizes = sorted(set(token_sizes) | {args.max_num_tokens})

    kwargs = dict(
        hidden_size=geo["hidden_size"],
        n=n,
        top_k=geo["top_k"],
        activation=geo["activation"],
        token_sizes=token_sizes,
        max_num_tokens=args.max_num_tokens,
        source_format=args.source_format,
    )
    counts = args.experts if args.experts else geo["expert_counts"]
    print(
        f"Warming B12x MoE kernels: {len(counts)} expert count(s) "
        f"{counts}, n={n} (tp={args.tp_size}), top_k={geo['top_k']}, "
        f"activation={geo['activation']}, {len(token_sizes)} token sizes"
    )
    t0 = time.perf_counter()
    work = [(E, kwargs) for E in counts]
    if args.workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for line in pool.imap_unordered(_worker, work):
                print(line, flush=True)
    else:
        for item in work:
            print(_worker(item), flush=True)
    from flashinfer.jit import env as jit_env

    print(
        f"All kernels warm in {time.perf_counter() - t0:.1f}s. Cache: "
        f"{jit_env.FLASHINFER_JIT_DIR / 'b12x_moe*'}"
    )
    if os.environ.get("FLASHINFER_CUTE_DSL_DISABLE_CACHE") == "1":
        print(
            "WARNING: FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 — nothing was "
            "persisted to disk."
        )


if __name__ == "__main__":
    main()
