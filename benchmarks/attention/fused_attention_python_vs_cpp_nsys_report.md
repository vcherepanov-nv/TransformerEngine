# C++ vs. Python cuDNN Fused Attention: Nsight Systems Report

Date: 2026-08-02  
Transformer Engine commit: `4d4088b3`

## Executive summary

For the profiled BF16 causal-attention workload, the Python cuDNN frontend path and the existing C++ path select the same generated cuDNN SDPA kernels and have effectively equivalent GPU compute time. The Python path reduces measured host submission overhead, most noticeably in forward-only execution:

- Forward: Python is 3.08% faster end-to-end and uses 18.42% less average CPU submission time.
- Forward + backward: Python is 0.86% faster end-to-end and uses 1.81% less average CPU submission time.
- GPU kernel time differs by only +0.25% in forward and -0.60% in forward + backward. These differences are smaller than the observed per-kernel variation and should be treated as performance parity, not a GPU speedup.
- Peak PyTorch allocation is unchanged for forward + backward and approximately 4 MiB lower for Python in forward-only execution.

The clearest trace-level difference is that the C++ path launches a small `extract_seed_and_offset` kernel even with zero dropout. The Python path does not: it launches one kernel instead of two for forward and four instead of five for forward + backward. This removes some CUDA launch work, although the full host-time difference also includes wrapper and dispatch overhead not separately attributable by Nsight Systems.

## Test environment

| Component | Value |
|---|---|
| GPU | NVIDIA GB200, device 0, 189,471 MiB |
| Driver | 595.84.01 |
| CUDA | 13.3 |
| cuDNN runtime | 9.23.0 (`torch.backends.cudnn.version() == 92300`) |
| cuDNN frontend Python package | 1.26.0 |
| PyTorch | 2.13.0a0+8145d630e8.nv26.06 |
| Transformer Engine | 2.19.0.dev0+4d4088b3 |
| Nsight Systems | 2026.3.1.117-263137992252v0 |

No other GPU processes were present immediately before the run. GPU clocks were not locked.

## Workload and methodology

All four runs used the benchmark in `benchmarks/attention/profile_fused_attention.py` with:

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

The benchmark disables FlashAttention and unfused attention, forcing Transformer Engine's fused-attention backend. `fused_attention_impl` then selects either the existing C++ extension or the new cuDNN frontend Python implementation.

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

`--cpu-overhead` synchronizes before each measured submission, with the synchronization outside the `cpu_submit` NVTX range. This prevents GPU queue backpressure from being charged to submission time and makes the run a serialized latency measurement rather than a maximum-throughput measurement. Warmup constructs and caches cuDNN graphs before profiling starts.

Metric definitions:

- Step latency is the benchmark's CUDA-event elapsed time divided by 50.
- CPU submit is `perf_counter_ns` around the Python forward or forward-plus-autograd call; it excludes the pre-step device synchronization.
- CUDA launch API time is the sum of `cudaLaunchKernel` and `cuLaunchKernelEx` durations from `cuda_api_sum`, divided by 50.
- GPU kernel time is the sum of all entries in `cuda_gpu_kern_sum`, divided by 50.
- Peak memory is reported by PyTorch's CUDA caching allocator.

## Results

Lower is better for every timing column.

| Mode | Implementation | Step latency (ms) | CPU submit avg / median (us) | CUDA launch API (us/step) | GPU kernels (ms/step) | Kernels/step | Peak allocation (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Forward | C++ | 1.070 | 320.714 / 306.610 | 22.577 | 0.786811 | 2 | 1.254 |
| Forward | Python | 1.037 | 261.628 / 244.274 | 13.285 | 0.788792 | 1 | 1.250 |
| Forward | Python vs. C++ | **-3.08%** | **-18.42% / -20.33%** | **-41.16%** | +0.25% | -1 | -0.32% |
| Forward + backward | C++ | 3.717 | 645.284 / 628.405 | 45.929 | 3.398979 | 5 | 3.508 |
| Forward + backward | Python | 3.685 | 633.574 / 615.365 | 39.891 | 3.378625 | 4 | 3.508 |
| Forward + backward | Python vs. C++ | **-0.86%** | **-1.81% / -2.08%** | **-13.15%** | -0.60% | -1 | 0.00% |

Nsight's outer NVTX range independently agrees with the benchmark latency: 1.0692 ms/step for C++ versus 1.0356 ms/step for Python in forward, and 3.7168 ms/step versus 3.6835 ms/step in forward + backward.

## Trace analysis

### Forward

Both implementations execute the same main kernel:

```text
cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_1_128x128x128_4x1x1_cga1x1x1_kernel0_0
```

Its mean duration is 785.747 us in the C++ trace and 788.792 us in the Python trace. The 3.045 us difference is small relative to the respective 12.303 us and 13.661 us standard deviations.

The C++ trace additionally contains one `transformer_engine::fused_attn::extract_seed_and_offset` kernel per step. It averages 1.064 us of GPU time, while its `cudaLaunchKernel` call averages 15.763 us of host CUDA API time. The Python path makes only the main kernel's `cuLaunchKernelEx` call.

Consequently, Python reduces launch API time by 9.292 us/step and complete CPU submission time by 59.086 us/step. The trace directly explains part, but not all, of that reduction; the remainder is host-side wrapper, validation, allocation, and dispatch work outside CUDA API calls.

### Forward + backward

The implementations again select the same four cuDNN kernels. Mean durations are closely matched:

| Kernel role | C++ (us) | Python (us) | Python vs. C++ |
|---|---:|---:|---:|
| SDPA forward | 848.642 | 844.027 | -0.54% |
| SDPA backward main | 2,266.892 | 2,252.829 | -0.62% |
| `compute_dot_do_o_specialized` | 163.311 | 162.504 | -0.49% |
| `convert_dq_to_16bits` | 118.892 | 119.266 | +0.31% |

The C++ path also launches the seed/offset extraction kernel, averaging 1.242 us. Python therefore executes four kernels per step instead of five and spends 6.038 us/step less inside CUDA kernel-launch APIs. End-to-end CPU submission falls by 11.710 us/step; the percentage benefit is smaller than in forward-only mode because PyTorch autograd scheduling and backward submission dominate the host path.

The C++ trace labels its internal ranges `nvte_flash_attn_fwd` and `nvte_flash_attn_bwd`. Those labels do not mean the benchmark selected the external FlashAttention backend: the actual kernels in both traces are generated cuDNN SDPA kernels, and the benchmark verifies that Transformer Engine selected FusedAttention.

### OS runtime

OS-runtime tracing did not expose a new blocking source in the Python implementation. Forward-plus-backward traces are dominated by background `poll` calls and 99 `pthread_cond_wait` calls in both implementations; cumulative condition-wait time is 182.397 ms for C++ and 181.071 ms for Python across profiler-observed threads. These cumulative multi-threaded wait times are not step latency and should not be added to the timing results.

## Interpretation

The Python cuDNN frontend path reaches the same cuDNN execution plans as the C++ implementation for this supported configuration. GPU performance and memory usage are effectively equivalent. Its practical advantage in this profile is lower host overhead, particularly for forward-only use, where eliminating C++-side RNG plumbing and other wrapper work matters relative to a roughly 0.79 ms attention kernel.

For training, the attention kernels are longer and autograd adds substantial host work, so the same reduction produces only a small end-to-end improvement. The measured sub-1% training difference should be interpreted as parity unless it reproduces across repeated runs and additional shapes.

## Limitations and follow-up

- This is one GB200, one shape, BF16, BSHD, and causal masking. It does not establish performance across the Python path's full supported matrix.
- Each configuration was profiled once for 50 iterations. The trace contains per-kernel distributions, but the runs were not repeated or counterbalanced.
- GPU clocks were not fixed, so differences below approximately 1% are not strong evidence of a speedup or regression.
- Graph construction and cuDNN plan selection are intentionally excluded. Results describe warm-cache steady state.
- Per-step synchronization is useful for host-overhead isolation but differs from a deeply queued throughput workload.
- Nsight tracing adds overhead. Comparisons are matched, but the absolute CPU times should not be treated as uninstrumented production latency.

Useful follow-up coverage would include unmasked attention, FP16, smaller sequence lengths where host overhead is a larger fraction, GQA shapes, and repeated non-profiled runs for confidence intervals.

## Artifacts

The generated reports and SQLite exports are available in `/tmp/te_attention_profiles` for the lifetime of this environment:

```text
cpp_fwd.nsys-rep
python_fwd.nsys-rep
cpp_fwd_bwd.nsys-rep
python_fwd_bwd.nsys-rep
```

The summary tables were extracted with:

```bash
nsys stats \
    --report nvtx_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum,cuda_api_sum,osrt_sum \
    /tmp/te_attention_profiles/<report>.nsys-rep
```
