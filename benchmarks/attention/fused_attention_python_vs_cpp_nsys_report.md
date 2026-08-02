# C++ vs. Python cuDNN Fused Attention: Post-Fix Nsight Systems Report

Date: 2026-08-02  
Transformer Engine source commit: `da8b11b2`

## Executive summary

This report compares Transformer Engine's C++ and Python cuDNN frontend fused-attention paths after the C++ zero-dropout RNG fix. The profiles now contain identical kernel counts: one kernel per forward step and four kernels per forward-plus-backward step. The former C++-only `extract_seed_and_offset` kernel is absent.

For the profiled BF16 causal-attention workload:

- Forward: Python is 2.09% faster end-to-end, with 13.12% lower average and 15.67% lower median CPU submission time.
- Forward + backward: performance is effectively tied. Python is 0.27% slower end-to-end, with 0.32% lower average and 1.96% lower median CPU submission time.
- GPU kernel time differs by -0.52% in forward and +0.28% in forward + backward. These differences are smaller than the observed per-kernel variation and indicate GPU parity.
- Peak PyTorch allocation is identical for forward + backward and approximately 4 MiB lower for Python in forward-only execution.

Removing zero-dropout RNG preparation makes the training comparison essentially neutral. A forward-only host-side difference of about 39 us/step remains, suggesting overhead elsewhere in the C++ wrapper rather than in cuDNN execution.

## Fix verification

The rebuilt extension passed all 23 tests in `tests/pytorch/attention/test_cudnn_frontend_attention.py`, including train and eval checks that zero-dropout C++ attention does not advance CUDA RNG state.

Nsight confirms the intended runtime change:

| Mode | C++ kernels/step | Python kernels/step | `extract_seed_and_offset` instances |
|---|---:|---:|---:|
| Forward | 1 | 1 | 0 |
| Forward + backward | 4 | 4 | 0 |

## Test environment

| Component | Value |
|---|---|
| GPU | NVIDIA GB200, device 0, 189,471 MiB |
| Driver | 595.84.01 |
| CUDA | 13.3 |
| cuDNN runtime | 9.23.0 (`torch.backends.cudnn.version() == 92300`) |
| cuDNN frontend Python package | 1.26.0 |
| PyTorch | 2.13.0a0+8145d630e8.nv26.06 |
| Transformer Engine source | `da8b11b2` |
| Package-reported TE version | 2.19.0.dev0+4d4088b3 |
| Nsight Systems | 2026.3.1.117-263137992252v0 |

The package version retains its earlier editable-install metadata, while the extension was rebuilt from the source at `da8b11b2`. No other GPU processes were present before the benchmark. GPU clocks were not locked.

## Workload and methodology

All four runs used `benchmarks/attention/profile_fused_attention.py` with:

| Parameter | Value |
|---|---|
| Batch size | 2 |
| Sequence length | 4096 |
| Attention heads | 128 |
| Head dimension | 128 |
| QKV layout | BSHD |
| Data type | BF16 |
| Mask | Top-left causal |
| Dropout | 0 |
| Warmup | 5 iterations |
| Measured iterations | 50 |
| Modes | Forward; forward + backward |

The benchmark disables FlashAttention and unfused attention, forcing Transformer Engine's fused-attention backend. `fused_attention_impl` then selects the C++ extension or the cuDNN frontend Python implementation.

The command template was:

```bash
TE_PATH=/workspace/TransformerEngine/ \
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,nvtx,osrt \
    --output=/tmp/te_attention_profiles/<implementation>_<mode> \
    --force-overwrite=true \
    python benchmarks/attention/profile_fused_attention.py \
        --impl <cpp|python> \
        --mode <fwd|fwd_bwd> \
        --warmup 5 \
        --iterations 50 \
        --profile \
        --cpu-overhead
```

`--cpu-overhead` synchronizes before each measured submission, with synchronization outside the `cpu_submit` NVTX range. This prevents GPU queue backpressure from being charged to CPU submission and makes the run a serialized latency measurement. Warmup constructs and caches cuDNN graphs before capture.

Metric definitions:

- Step latency is the benchmark's CUDA-event elapsed time divided by 50.
- CPU submit is `perf_counter_ns` around the Python forward or forward-plus-autograd call; it excludes pre-step synchronization.
- CUDA launch API time is `cuLaunchKernelEx` time from `cuda_api_sum`, divided by 50.
- GPU kernel time is the sum of `cuda_gpu_kern_sum`, divided by 50.
- Peak memory is reported by PyTorch's CUDA caching allocator.

## Results

Lower is better for every timing column.

| Mode | Implementation | Step latency (ms) | CPU submit avg / median (us) | CUDA launch API (us/step) | GPU kernels (ms/step) | Kernels/step | Peak allocation (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Forward | C++ | 1.054 | 298.560 / 283.394 | 12.888 | 0.789130 | 1 | 1.254 |
| Forward | Python | 1.032 | 259.399 / 238.994 | 12.582 | 0.785040 | 1 | 1.250 |
| Forward | Python vs. C++ | **-2.09%** | **-13.12% / -15.67%** | -2.37% | -0.52% | 0 | -0.32% |
| Forward + backward | C++ | 3.690 | 634.679 / 618.596 | 35.740 | 3.377366 | 4 | 3.508 |
| Forward + backward | Python | 3.700 | 632.618 / 606.485 | 39.065 | 3.386984 | 4 | 3.508 |
| Forward + backward | Python vs. C++ | +0.27% | **-0.32% / -1.96%** | +9.30% | +0.28% | 0 | 0.00% |

Nsight's outer NVTX range independently agrees with the benchmark latency:

| Mode | C++ (ms/step) | Python (ms/step) | Python vs. C++ |
|---|---:|---:|---:|
| Forward | 1.0534 | 1.0310 | -2.13% |
| Forward + backward | 3.6892 | 3.6987 | +0.26% |

The NVTX `cpu_submit` ranges report 301.607 us versus 262.664 us for forward and 638.561 us versus 636.222 us for forward + backward, consistent with the benchmark's direct host timing.

## Trace analysis

### Forward

Both implementations execute exactly one instance per step of the same generated cuDNN kernel:

```text
cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_1_128x128x128_4x1x1_cga1x1x1_kernel0_0
```

| Metric | C++ | Python |
|---|---:|---:|
| Mean kernel duration | 789.130 us | 785.040 us |
| Median kernel duration | 790.070 us | 781.877 us |
| Kernel standard deviation | 12.644 us | 12.499 us |
| Mean `cuLaunchKernelEx` duration | 12.888 us | 12.582 us |

The launch API difference is only 0.306 us/step, and GPU kernel time is within normal variation. Neither explains the 39.161 us difference in direct CPU submission time.

The C++ wrapper invokes the native fused-attention API twice per step: once to query auxiliary shapes and workspace requirements and once to execute. Nsight records 100 `nvte_flash_attn_fwd` ranges for 50 steps. The Python path performs a cached graph lookup, allocates its workspace, and calls `graph.execute` once. More granular sibling NVTX ranges around tensor preparation, workspace query/allocation, and graph execution would be needed to assign the remaining host difference precisely.

### Forward + backward

Both implementations execute the same four cuDNN kernels per step. Mean durations remain closely matched:

| Kernel role | C++ (us) | Python (us) | Python vs. C++ |
|---|---:|---:|---:|
| SDPA forward | 843.837 | 846.341 | +0.30% |
| SDPA backward main | 2,251.467 | 2,258.779 | +0.32% |
| `compute_dot_do_o_specialized` | 163.289 | 162.651 | -0.39% |
| `convert_dq_to_16bits` | 118.774 | 119.214 | +0.37% |

Average CPU submission differs by only 2.061 us in Python's favor. Python's median is 12.111 us lower, but its submission distribution has a larger standard deviation and maximum, so the average is the more conservative summary. The 10 us end-to-end difference follows the similarly sized GPU kernel-time difference and is not evidence of a systematic codepath regression.

The C++ trace labels internal ranges `nvte_flash_attn_fwd` and `nvte_flash_attn_bwd`. These labels do not indicate selection of the external FlashAttention backend: actual kernels in both traces are generated cuDNN SDPA kernels, and the benchmark verifies that Transformer Engine selected FusedAttention.

### OS runtime

OS-runtime behavior is effectively identical in forward + backward. Nsight observes 99 `pthread_cond_wait` calls in both paths, totaling 181.268 ms for C++ and 181.612 ms for Python across profiler-observed threads. Background `poll` time is 361.088 ms and 361.115 ms, respectively. These cumulative multi-threaded waits are not step latency and should not be added to the timing results.

## Interpretation

After removing zero-dropout RNG preparation, the two implementations submit the same GPU work and reach the same cuDNN execution plans. GPU performance and memory usage are at parity.

Training CPU cost is also at parity: the average submission difference is 0.32%, well below the variability visible in the trace. The prior raw Python training advantage was largely attributable to the unnecessary C++ RNG path.

Forward-only Python execution retains a measurable host advantage of approximately 39 us/step. Since kernel count, launch API time, and GPU time now match, this difference lies in wrapper-level work before and around cuDNN execution. The repeated native workspace/auxiliary query in the C++ wrapper is a likely contributor, but the current ranges do not provide an exclusive breakdown.

## Limitations and follow-up

- This is one GB200, one shape, BF16, BSHD, and causal masking. It does not establish performance across the Python path's full supported matrix.
- Each configuration was profiled once for 50 iterations. Per-kernel distributions are available, but runs were not repeated or counterbalanced.
- GPU clocks were not fixed, so sub-1% differences are not strong evidence of a speedup or regression.
- Graph construction and cuDNN plan selection are intentionally excluded; results describe warm-cache steady state.
- Per-step synchronization isolates host overhead but differs from a deeply queued throughput workload.
- Nsight tracing adds overhead. Comparisons are matched, but absolute CPU times should not be treated as uninstrumented production latency.

Useful follow-up work would add NVTX ranges for C++ and Python wrapper phases, repeat profiles in counterbalanced order, and cover unmasked attention, FP16, smaller sequence lengths, and GQA shapes.

## Artifacts

The recreated reports and SQLite exports are under `/tmp/te_attention_profiles`:

```text
cpp_fwd.nsys-rep
python_fwd.nsys-rep
cpp_fwd_bwd.nsys-rep
python_fwd_bwd.nsys-rep
```

The fresh SQLite exports and summary tables were generated with:

```bash
nsys stats \
    --force-export=true \
    --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,osrt_sum \
    /tmp/te_attention_profiles/<report>.nsys-rep
```
