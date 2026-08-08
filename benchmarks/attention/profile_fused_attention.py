# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Profile C++ and Python cuDNN frontend fused attention implementations.

Example:

.. code-block:: bash

   nsys profile \
       --capture-range=cudaProfilerApi \
       --capture-range-end=stop \
       --trace=cuda,nvtx,osrt \
       --output=cpp_fused_attention \
       --force-overwrite=true \
       python benchmarks/attention/profile_fused_attention.py \
           --impl cpp --profile --cpu-overhead --iterations 50

   nsys profile \
       --capture-range=cudaProfilerApi \
       --capture-range-end=stop \
       --trace=cuda,nvtx,osrt \
       --output=python_fused_attention \
       --force-overwrite=true \
       python benchmarks/attention/profile_fused_attention.py \
           --impl python --profile --cpu-overhead --iterations 50

Use the small shape preset to emphasize host submission overhead with short-running
GPU kernels:

.. code-block:: bash

   python benchmarks/attention/profile_fused_attention.py \
       --impl cpp --shape small --iterations 100

Measure warmed-process, cold-cache cuDNN graph construction:

.. code-block:: bash

   nsys profile \
       --capture-range=cudaProfilerApi \
       --capture-range-end=stop \
       --trace=cuda,nvtx \
       --sample=none \
       --output=cpp_graph_creation \
       --force-overwrite=true \
       python benchmarks/attention/profile_fused_attention.py \
           --impl cpp --measure graph_creation --mode fwd_bwd \
           --warmup 2 --iterations 10 --profile

   nsys stats --report nvtx_sum cpp_graph_creation.nsys-rep

Run the same command with ``--impl python --match-cpp-graph-build`` and a different
output name to compare the Python frontend with the C++ path using the same cuDNN
heuristic mode. Graph-build durations are reported by the inclusive
``cudnn_graph_build_{cpp,python}_{fwd,bwd}`` ranges. Both implementations also emit
``materialization_and_validation``, ``build_operation_graph``,
``create_execution_plans``, ``check_support``, and ``build_plans`` stages. Python
adds ``ir_definition`` and ``lowering_and_native_validation`` detail;
C++ adds ``native_definition`` and ``native_validation`` detail. Forward-only mode
builds an inference graph, while forward-plus-backward mode builds a training
forward graph with softmax statistics.
"""

import argparse
import os
import statistics
import time
from typing import Literal

import torch

SHAPE_PRESETS = {
    "large": {
        "batch_size": 2,
        "sequence_length": 4096,
        "num_heads": 128,
        "head_dim": 128,
    },
    "small": {
        "batch_size": 1,
        "sequence_length": 128,
        "num_heads": 1,
        "head_dim": 64,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("cpp", "python"), required=True)
    parser.add_argument(
        "--measure",
        choices=("runtime", "graph_creation"),
        default="runtime",
        help=(
            "Measure steady-state runtime or warmed-process, cold-cache cuDNN "
            "graph construction."
        ),
    )
    parser.add_argument(
        "--shape",
        choices=tuple(SHAPE_PRESETS),
        default="large",
        help="Named shape preset. Explicit dimension options override the preset.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--mask", choices=("causal", "no_mask"), default="causal")
    parser.add_argument("--mode", choices=("fwd", "fwd_bwd"), default="fwd_bwd")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--cpu-overhead",
        action="store_true",
        help=(
            "Synchronize outside each repeated cpu_submit NVTX range so nvtx_sum "
            "measures host submission overhead without GPU queue backpressure."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop for nsys capture-range.",
    )
    parser.add_argument(
        "--match-cpp-graph-build",
        action="store_true",
        help=(
            "For Python graph-creation profiling, use the C++ path's heur_mode.A-only "
            "policy instead of the production Python A-plus-FALLBACK policy."
        ),
    )
    args = parser.parse_args()
    for name, value in SHAPE_PRESETS[args.shape].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def _run_step(
    attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    d_out: torch.Tensor | None,
    mask: Literal["causal", "no_mask"],
    run_backward: bool,
):
    if run_backward:
        output = attention(
            query,
            key,
            value,
            qkv_format="bshd",
            attn_mask_type=mask,
        )
        grads = torch.autograd.grad(
            output,
            (query, key, value),
            grad_outputs=d_out,
        )
        return output, grads

    with torch.no_grad():
        return (
            attention(
                query,
                key,
                value,
                qkv_format="bshd",
                attn_mask_type=mask,
            ),
            None,
        )


def _set_graph_build_profiling(impl: str, enabled: bool, tex, cudnn_attention) -> None:
    """Toggle graph-build instrumentation for one fused-attention implementation."""
    if impl == "cpp":
        tex._set_fused_attn_graph_build_profiling(enabled)
    else:
        cudnn_attention._set_cudnn_attention_graph_build_profiling(enabled)


def _reset_graph_cache(impl: str, tex, cudnn_attention) -> None:
    """Invalidate the selected implementation's graph cache."""
    if impl == "cpp":
        tex._reset_fused_attn_graph_caches()
    else:
        cudnn_attention._reset_cudnn_attention_graph_cache()


def _benchmark_graph_creation(
    args,
    shape,
    attention,
    query,
    key,
    value,
    d_out,
    run_backward: bool,
    device: torch.device,
    tex,
    cudnn_attention,
) -> None:
    """Emit NVTX ranges for warmed-process, cold-cache graph builds."""
    if args.cpu_overhead:
        raise ValueError("--cpu-overhead applies only to --measure runtime.")

    range_prefix = (
        f"fused_attention_{args.impl}_graph_creation_{args.mode}_"
        f"b{args.batch_size}_s{args.sequence_length}_"
        f"h{args.num_heads}_d{args.head_dim}_{args.dtype}_{args.mask}"
    )

    match_cpp_graph_build = args.impl == "python" and args.match_cpp_graph_build
    if match_cpp_graph_build:
        cudnn_attention._set_cudnn_attention_cpp_comparable_graph_build(True)
    _set_graph_build_profiling(args.impl, True, tex, cudnn_attention)
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for trial in range(args.iterations):
            _reset_graph_cache(args.impl, tex, cudnn_attention)
            with torch.cuda.nvtx.range(f"{range_prefix}_trial_{trial}"):
                _run_step(
                    attention,
                    query,
                    key,
                    value,
                    d_out,
                    args.mask,
                    run_backward,
                )
            torch.cuda.synchronize(device)
    finally:
        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()
        _set_graph_build_profiling(args.impl, False, tex, cudnn_attention)
        if match_cpp_graph_build:
            cudnn_attention._set_cudnn_attention_cpp_comparable_graph_build(False)

    print(
        f"impl={args.impl} measure=graph_creation mode={args.mode} "
        f"dtype={args.dtype} mask={args.mask} shape={shape} "
        f"trials={args.iterations} timing_source=nsys_nvtx "
        f"heuristic_policy={'A' if args.impl == 'cpp' or match_cpp_graph_build else 'A+FALLBACK'}"
    )


def main() -> None:
    args = _parse_args()
    if args.warmup < 1:
        raise ValueError(
            "--warmup must be at least 1 so graph construction is outside capture."
        )
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if args.match_cpp_graph_build and not (
        args.impl == "python" and args.measure == "graph_creation"
    ):
        raise ValueError(
            "--match-cpp-graph-build requires --impl python --measure graph_creation."
        )

    # The implementation selector only applies after FusedAttention is chosen.
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"

    from transformer_engine.pytorch import DotProductAttention
    from transformer_engine.pytorch.attention.dot_product_attention import (
        _attention_backends,
    )
    from transformer_engine.pytorch.attention.dot_product_attention import (
        cudnn_attention,
    )
    import transformer_engine_torch as tex

    if args.match_cpp_graph_build:
        # Warm the same heuristic path that will be measured after each cache reset.
        cudnn_attention._set_cudnn_attention_cpp_comparable_graph_build(True)

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    run_backward = args.mode == "fwd_bwd"
    shape = (
        args.batch_size,
        args.sequence_length,
        args.num_heads,
        args.head_dim,
    )

    torch.manual_seed(1234)
    query, key, value = [
        (0.01 * torch.randn(shape, dtype=dtype, device=device)).requires_grad_(
            run_backward
        )
        for _ in range(3)
    ]
    output_shape = (
        args.batch_size,
        args.sequence_length,
        args.num_heads * args.head_dim,
    )
    d_out = (
        torch.randn(output_shape, dtype=dtype, device=device) if run_backward else None
    )

    attention = DotProductAttention(
        num_attention_heads=args.num_heads,
        kv_channels=args.head_dim,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type=args.mask,
        fused_attention_impl=args.impl,
    ).to(device)
    attention.train(run_backward)

    last_output = None
    last_grads = None
    for _ in range(args.warmup):
        last_output, last_grads = _run_step(
            attention,
            query,
            key,
            value,
            d_out,
            args.mask,
            run_backward,
        )
    torch.cuda.synchronize(device)

    if not _attention_backends["use_fused_attention"]:
        raise RuntimeError(
            "FusedAttention was not selected despite the forced backend settings."
        )

    if args.measure == "graph_creation":
        _benchmark_graph_creation(
            args,
            shape,
            attention,
            query,
            key,
            value,
            d_out,
            run_backward,
            device,
            tex,
            cudnn_attention,
        )
        return

    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    range_name = (
        f"fused_attention_{args.impl}_{args.mode}_"
        f"b{args.batch_size}_s{args.sequence_length}_"
        f"h{args.num_heads}_d{args.head_dim}_{args.dtype}_{args.mask}"
    )

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    start.record()
    cpu_submit_times_ns = []
    cpu_range_name = f"{range_name}_cpu_submit"
    torch.cuda.nvtx.range_push(range_name)
    try:
        for iteration in range(args.iterations):
            if args.cpu_overhead:
                # Drain the previous iteration outside the measured CPU range. This
                # prevents a deep GPU queue from moving backpressure into submission.
                torch.cuda.synchronize(device)
                iteration_range_name = cpu_range_name
            else:
                iteration_range_name = f"{range_name}_iteration_{iteration}"
            torch.cuda.nvtx.range_push(iteration_range_name)
            cpu_start_ns = time.perf_counter_ns()
            try:
                last_output, last_grads = _run_step(
                    attention,
                    query,
                    key,
                    value,
                    d_out,
                    args.mask,
                    run_backward,
                )
            finally:
                cpu_submit_times_ns.append(time.perf_counter_ns() - cpu_start_ns)
                torch.cuda.nvtx.range_pop()
        if args.cpu_overhead:
            torch.cuda.synchronize(device)
        end.record()
        torch.cuda.synchronize(device)
    finally:
        torch.cuda.nvtx.range_pop()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    # Keep the final results alive until all work has completed.
    assert last_output is not None
    if run_backward:
        assert last_grads is not None

    average_ms = start.elapsed_time(end) / args.iterations
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    summary = (
        f"impl={args.impl} mode={args.mode} dtype={args.dtype} mask={args.mask} "
        f"shape={shape} average={average_ms:.3f} ms peak_allocated={peak_gib:.3f} GiB"
    )
    if args.cpu_overhead:
        cpu_submit_times_us = [duration / 1e3 for duration in cpu_submit_times_ns]
        summary += (
            f" cpu_submit_average={statistics.mean(cpu_submit_times_us):.3f} us"
            f" cpu_submit_median={statistics.median(cpu_submit_times_us):.3f} us"
        )
    print(summary)


if __name__ == "__main__":
    main()
