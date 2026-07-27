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
"""

import argparse
import os
import statistics
import time
from typing import Literal

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("cpp", "python"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
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
    return parser.parse_args()


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


def main() -> None:
    args = _parse_args()
    if args.warmup < 1:
        raise ValueError(
            "--warmup must be at least 1 so graph construction is outside capture."
        )
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")

    # The implementation selector only applies after FusedAttention is chosen.
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"

    from transformer_engine.pytorch import DotProductAttention
    from transformer_engine.pytorch.attention.dot_product_attention import (
        _attention_backends,
    )

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
