# C++ vs. Python cuDNN Fused Attention: Post-Fix Nsight Systems Report

Date: 2026-08-02  
Transformer Engine extension source commit: `da8b11b2`
Benchmark source commit: `dc25d6b5`

## Executive summary

This report compares Transformer Engine's C++ and Python cuDNN frontend fused-attention paths after the C++ zero-dropout RNG fix. The profiles now contain identical kernel counts: one kernel per forward step and four kernels per forward-plus-backward step. The former C++-only `extract_seed_and_offset` kernel is absent.

For the profiled BF16 causal-attention workload:

- Forward: Python is 2.09% faster end-to-end, with 13.12% lower average and 15.67% lower median CPU submission time.
- Forward + backward: performance is effectively tied. Python is 0.27% slower end-to-end, with 0.32% lower average and 1.96% lower median CPU submission time.
- GPU kernel time differs by -0.52% in forward and +0.28% in forward + backward. These differences are smaller than the observed per-kernel variation and indicate GPU parity.
- Peak PyTorch allocation is identical for forward + backward and approximately 4 MiB lower for Python in forward-only execution.

Removing zero-dropout RNG preparation makes the training comparison essentially neutral. A forward-only host-side difference of about 39 us/step remains, suggesting overhead elsewhere in the C++ wrapper rather than in cuDNN execution.

Follow-up profiles without per-iteration synchronization confirm that the original 225-248 us forward GPU gaps were caused by the serialized CPU-overhead methodology. With the large shape, queued kernels are effectively contiguous. With the new small shape `(B=1, S=128, H=1, D=64)`, the cuDNN forward kernel takes only about 4.5 us and forward-plus-backward GPU work totals about 18.5 us, exposing host orchestration directly. Python is 18.0% faster for small inference forward, while C++ is 4.9% faster for small training forward plus backward.

These two modes do not execute the same forward path. Forward-only runs with the module in evaluation mode under `torch.no_grad()`, while forward plus backward enables training, generates softmax statistics, saves autograd state, and uses a different cuDNN graph. The small-shape crossover is host-driven. The large-shape forward-plus-backward result is instead GPU-dominated and follows sub-1% kernel-time variation.

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
| Transformer Engine extension source | `da8b11b2` |
| Benchmark source | `dc25d6b5` |
| Package-reported TE version | 2.19.0.dev0+4d4088b3 |
| Nsight Systems | 2026.3.1.117-263137992252v0 |

The package version retains its earlier editable-install metadata, while the extension was rebuilt from the source at `da8b11b2`. The small-shape preset was added in benchmark commit `dc25d6b5`. No other GPU processes were present before the benchmark. GPU clocks were not locked.

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

## Metric definitions and calculations

All profile-derived timestamps are in nanoseconds on Nsight Systems' unified timeline unless stated otherwise. `N` is the number of measured iterations: 50 for the large shape and 100 for the small shape. Warmup is outside every reported measurement.

### Step latency and step time

The benchmark records CUDA events immediately before and after the complete measured loop. After synchronizing the device, it calculates:

```text
CUDA-event step latency (ms) = start.elapsed_time(end) / N
```

The `average=... ms` printed by the benchmark and the initial serialized `Step latency` table use this measurement. It is one aggregate makespan divided by `N`, not the mean of `N` independently timed steps, so it has no per-step median. CUDA events measure the device-stream timeline. GPU execution and GPU-idle periods caused by late host submission between the events are included; warmup and work outside the events are excluded.

The benchmark also places one outer NVTX range around the complete measured loop and its final synchronization. For the queued large- and small-shape tables, the six-decimal `Step time` values are calculated from this range:

```text
outer NVTX step time (ms) = (outer_end_ns - outer_start_ns) / 1,000,000 / N
```

This is end-to-end host wall-clock makespan per iteration. It includes host submission, device execution, host-induced GPU gaps, and the final queue drain. It excludes warmup and graph construction. The final synchronization is necessary to include completion of all queued work; its trailing CPU return overhead is also inside the outer range. The report gives a separate outer-NVTX table for the serialized runs to cross-check the CUDA-event measurement. The two methods agree closely, with small differences from their exact start/end envelopes and timestamp domains.

For queued execution, step time is approximately the slower of steady-state host submission cadence and GPU service time, plus startup and final-drain overhead amortized over `N`. CPU and GPU work can overlap, so step time is not calculated by adding CPU iteration time and GPU kernel time.

### Direct CPU submission time

Every call to `_run_step` is bracketed with `time.perf_counter_ns()`:

```text
direct_cpu_i_us = (perf_counter_end_i_ns - perf_counter_start_i_ns) / 1,000
CPU submit average = arithmetic mean(direct_cpu_i_us)
CPU submit median = median(direct_cpu_i_us)
```

The initial serialized `CPU submit avg / median` values come from these direct benchmark timers. With `--cpu-overhead`, `torch.cuda.synchronize()` runs before the timer and before the `cpu_submit` NVTX range, so the metric excludes the deliberate pre-step synchronization. It includes the Python attention call and, in forward-plus-backward mode, `torch.autograd.grad`, along with any CUDA API blocking or OS scheduling encountered inside that interval. It measures host wall time, not CPU cycles or exclusive on-core execution.

The script records the same direct timer in queued mode but does not print it unless `--cpu-overhead` is set, so the queued tables use the persisted NVTX iteration ranges described below.

### NVTX CPU submission and CPU iteration

In serialized mode, all measured calls use the `..._cpu_submit` NVTX name. Nsight's `nvtx_sum` groups those ranges and reports their count, total, mean, median, minimum, maximum, and standard deviation. The report cites the NVTX mean separately as an independent check of the direct CPU timer. The NVTX envelope starts just before `perf_counter_ns()` and ends just after the direct duration is appended, so it is slightly wider than the direct timer.

In queued mode, each call uses a unique `..._iteration_<index>` NVTX range. The `CPU iteration` columns are calculated directly from those ranges:

```text
cpu_iteration_i_us = (iteration_end_i_ns - iteration_start_i_ns) / 1,000
CPU iteration average = arithmetic mean(cpu_iteration_i_us)
CPU iteration median = median(cpu_iteration_i_us)
```

All `N` ranges are included, including the first captured iteration. The range includes forward, optional autograd/backward submission, replacement and release of the previous returned objects, the two `perf_counter_ns()` calls, and appending the direct duration. It excludes iteration-name construction, the NVTX push call itself, final CUDA-event handling, and final synchronization. Because no per-iteration synchronization is present, it normally measures asynchronous submission rather than GPU completion, except where a CUDA API or queue backpressure blocks the host.

The equivalent SQLite selection is:

```sql
SELECT n.start, n.end
FROM NVTX_EVENTS AS n
LEFT JOIN StringIds AS s ON s.id = n.textId
WHERE COALESCE(n.text, s.value) LIKE '%_iteration_%'
ORDER BY n.start;
```

### CUDA launch API time

CUDA runtime/driver calls come from `CUPTI_ACTIVITY_KIND_RUNTIME`, summarized by `cuda_api_sum`. For each `cuLaunchKernelEx` call:

```text
launch_api_duration_us = (api_end_ns - api_start_ns) / 1,000
CUDA launch API us/step = sum(all cuLaunchKernelEx durations) / N / 1,000
```

The result is total launch-API wall time per benchmark step, not average time per launch. These are host API durations; they do not include later kernel execution. Forward has one launch per step after the RNG fix, while forward plus backward has four.

### GPU kernel time and per-kernel statistics

Kernel events come from `CUPTI_ACTIVITY_KIND_KERNEL` and are also summarized by `cuda_gpu_kern_sum`. Each kernel duration is:

```text
kernel_duration_us = (kernel_end_ns - kernel_start_ns) / 1,000
GPU kernel time per step = sum(all kernel durations) / N
```

Kernels execute on one CUDA stream in these profiles, so summing durations does not double-count overlap. Mean, median, minimum, maximum, and standard deviation for a kernel role are calculated over all instances with the same demangled kernel name. The role-specific forward and backward tables use the mean emitted by `cuda_gpu_kern_sum`. `GPU kernels` is device-busy kernel time only: host submission, launch API time, GPU-idle gaps, CUDA events, and synchronization are excluded.

### Kernel counts and removed-kernel verification

`Kernels/step` is:

```text
kernel instances in CUPTI_ACTIVITY_KIND_KERNEL / N
```

The Python-vs.-C++ row shows the arithmetic count difference, not a percentage. Verification of the removed RNG kernel searches captured kernel names for `extract_seed_and_offset` and reports the number of matching instances. A zero count in both paths establishes that the former C++-only kernel is absent from the captured ranges.

### Inter-kernel gaps

Kernel events are ordered by GPU start timestamp. For each adjacent pair on the captured device and stream:

```text
gap_i_us = (next_kernel_start_ns - previous_kernel_end_ns) / 1,000
gap average = arithmetic mean(gap_i_us)
gap median = median(gap_i_us)
```

The interval before the first kernel and after the last kernel is not included. Therefore, 50 forward kernels produce 49 gaps, 200 large forward-plus-backward kernels produce 199 gaps, 100 small forward kernels produce 99 gaps, and 400 small forward-plus-backward kernels produce 399 gaps.

Forward-plus-backward gaps are additionally classified by their position in the repeating four-kernel sequence: forward-to-first-backward, first-to-second backward, second-to-third backward, and last-backward-to-next-forward step boundary. The overall small-shape median is low because slightly more than half of adjacent transitions are short intra-cuDNN-graph gaps; the mean reflects the long host-driven transitions as well.

### Cross CPU/GPU phase intervals

Nsight places CPU NVTX and GPU kernel timestamps on a unified timeline. Each iteration range is paired with its ordered group of one forward kernel or four forward-plus-backward kernels. The phase table uses direct timestamp subtraction:

```text
pre-forward = first_kernel_start - iteration_range_start
forward-to-backward = first_backward_start - forward_kernel_end
backward internal gap = next_backward_start - previous_backward_end
post-backward = iteration_range_end - last_kernel_end
between iteration ranges = next_iteration_start - previous_iteration_end
```

These intervals include all host work, API calls, scheduling, and idle time between their endpoints. They localize overhead but do not exclusively assign it to cache lookup, allocation, Python execution, autograd, or cuDNN without narrower sibling NVTX ranges.

### Peak PyTorch allocation

Immediately before measurement, the benchmark calls `torch.cuda.reset_peak_memory_stats()`. After completion it calculates:

```text
peak allocation GiB = torch.cuda.max_memory_allocated(device) / 1024^3
```

This is PyTorch caching-allocator memory actively allocated at the high-water mark, not reserved capacity, device-wide free memory, or an incremental allocation attributable only to attention. Resetting peak statistics preserves the current baseline allocation, so inputs and other live tensors that already exist at reset remain part of the reported peak.

### Percentage comparisons

Unless explicitly labeled as a count difference, every `Python vs. C++` percentage uses:

```text
percentage delta = (Python value - C++ value) / C++ value * 100
```

A negative timing or memory percentage means Python measured lower; a positive value means Python measured higher. Percentages are calculated from unrounded values where available, then rounded for display.

### OS-runtime totals

OS runtime calls come from `OSRT_API` and `osrt_sum`. Counts are the number of captured calls with the same API name, and total time is the sum of `end - start` over those calls across all captured threads. These multi-thread cumulative totals can exceed benchmark wall time and are reported only to compare runtime behavior. They are not divided into, or added to, step latency.

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

## Queued-execution follow-up

The initial profiles intentionally used `--cpu-overhead` to isolate host submission cost. A second round removed that flag to model normal asynchronous execution more closely. These runs retain one synchronization after the complete measured loop, but do not synchronize between iterations. The CPU can therefore prepare and queue later operations while the GPU executes earlier ones.

Both large-shape implementations used 5 warmups and 50 measured iterations. The small-shape runs used 5 warmups and 100 measured iterations. All runs remained BF16, BSHD, top-left causal, and zero-dropout.

### Large shape with queued execution

| Mode | Implementation | Step time (ms) | CPU iteration (us) | GPU kernels (us/step) | Inter-kernel gap avg / median (us) |
|---|---|---:|---:|---:|---:|
| Forward | C++ | 0.793642 | 305.049 | 778.073 | 0.262 / 0.256 |
| Forward | Python | 0.799175 | 265.803 | 785.328 | 0.263 / 0.256 |
| Forward + backward | C++ | 3.540529 | 621.834 | 3,523.344 | 0.381 / 0.288 |
| Forward + backward | Python | 3.551236 | 611.290 | 3,534.819 | 0.273 / 0.288 |

Removing per-iteration synchronization collapses the earlier forward GPU bubbles from 251.952 us average for C++ and 233.099 us for Python to approximately 0.26 us in both paths. Forward-plus-backward kernels are also effectively contiguous. The CPU queues work tens of milliseconds ahead of the device, so host differences are hidden by GPU execution.

Python's large-shape CPU iteration is 12.9% lower in forward and 1.7% lower in forward plus backward. Nevertheless, Python's measured GPU kernel totals are 7.255 us and 11.475 us higher, respectively. The end-to-end differences closely follow those GPU differences: 5.533 us in forward and 10.707 us in forward plus backward. These sub-1% kernel differences are within the observed run-to-run and per-kernel variation, so the large queued results are not evidence that the C++ backward wrapper is intrinsically faster.

### Small shape with queued execution

The small preset is `(batch=1, sequence=128, heads=1, head_dim=64)`. It minimizes the amount of device work while retaining a conventional cuDNN-supported attention tile. The generated forward kernel takes approximately 4.5 us, making host orchestration the throughput limiter.

| Mode | Implementation | Step time (ms) | CPU iteration avg / median (us) | GPU kernels (us/step) | Python vs. C++ step time |
|---|---|---:|---:|---:|---:|
| Forward | C++ | 0.299259 | 295.777 / 291.922 | 4.629 | - |
| Forward | Python | 0.245501 | 241.908 / 230.754 | 4.540 | **-18.0%** |
| Forward + backward | C++ | 0.635569 | 631.457 / 617.605 | 18.521 | - |
| Forward + backward | Python | 0.666541 | 662.722 / 650.292 | 18.539 | **+4.9%** |

The device work is at parity. Python's forward kernel is 0.089 us shorter, and its complete forward-plus-backward kernel sequence is only 0.018 us longer. Neither difference can explain the tens of microseconds in step time.

Forward-only inter-kernel gaps average 290.445 us for C++ and 234.312 us for Python, with medians of 290.690 us and 229.858 us. These gaps are host submission intervals: unlike the large shape, the CPU cannot enqueue the next step before the short kernel finishes.

The forward-plus-backward gap distribution is bimodal because two of the three backward-kernel transitions are internal to a single cuDNN graph execution. The overall means are 152.353 us for C++ and 160.285 us for Python, while the medians are only 18.976 us and 18.944 us. A phase breakdown is more informative:

| Host/GPU interval | C++ avg (us) | Python avg (us) | Python - C++ (us) |
|---|---:|---:|---:|
| Iteration start to training-forward kernel | 276.497 | 300.937 | +24.440 |
| Forward kernel end to first backward kernel | 250.661 | 265.279 | +14.618 |
| First-to-second backward-kernel gap | 8.364 | 8.785 | +0.421 |
| Second-to-third backward-kernel gap | 0.272 | 0.290 | +0.018 |
| Last backward kernel end to iteration end | 77.142 | 68.892 | -8.250 |
| Between iteration NVTX ranges | 2.959 | 2.933 | -0.026 |

The approximately 31 us net difference from these phases matches the observed CPU-iteration and step-time difference.

### Why inference forward favors Python but training favors C++

The benchmark couples execution mode and autograd behavior:

- `mode=fwd` sets `run_backward=False`, puts `DotProductAttention` in evaluation mode, disables input gradients, and calls attention under `torch.no_grad()`.
- `mode=fwd_bwd` sets `run_backward=True`, puts the module in training mode, enables input gradients, and invokes `torch.autograd.grad` after forward.

Forward plus backward is therefore not simply the forward-only operation followed by a backward operation. Its forward graph generates softmax statistics, returns auxiliary tensors, and saves training state. The two modes can have different host-side rankings even before backward begins.

For inference forward, the cached Python path is relatively lean. It allocates the output, constructs and looks up the graph-cache key, builds a four-entry variant pack, allocates workspace, and calls `graph.execute`. It skips the training statistics tensor. The generic C++ wrapper still constructs Transformer Engine tensor wrappers and an auxiliary tensor pack, creates its ABI-required RNG-state tensor, calls the native fused-attention API once to query auxiliary/workspace requirements, allocates the returned buffers, and calls the native API again to execute. The small trace splits the inference-forward advantage as follows:

| Inference-forward interval | C++ avg (us) | Python avg (us) | Python - C++ (us) |
|---|---:|---:|---:|
| Iteration start to kernel | 244.398 | 215.134 | -29.264 |
| Kernel end to iteration end | 46.750 | 22.234 | -24.516 |
| Total CPU iteration | 295.777 | 241.908 | -53.869 |

In training, the Python implementation allocates the softmax-statistics tensor, includes it in a larger cache key and variant pack, and saves five tensors for autograd. Its backward method makes `d_out` contiguous, allocates `dQ`, `dK`, and `dV`, constructs a backward cache key from the metadata of Q, K, V, O, dO, and statistics, builds a nine-entry variant pack, allocates another workspace, obtains the current-stream cuDNN handle, and executes the cached backward graph.

The C++ path is also generic and still performs workspace queries and buffer allocation, including two native fused-attention calls in backward. However, much of its tensor metadata, layout handling, auxiliary wrapping, gradient allocation, and native argument preparation executes in compiled C++ rather than as per-call Python object and dictionary manipulation. The small trace is consistent with that compiled orchestration offsetting the C++ forward-query cost during training.

The current NVTX ranges establish where the extra time occurs but do not exclusively attribute each microsecond to cache-key creation, allocation, autograd bookkeeping, or graph execution. Phase-specific sibling NVTX ranges would be required for an exclusive breakdown.

## Limitations and follow-up

- This is one GB200, two shapes, BF16, BSHD, and causal masking. It does not establish performance across the Python path's full supported matrix.
- Each configuration was profiled once: 50 iterations for the large shape and 100 for the small shape. Per-kernel distributions are available, but runs were not repeated or counterbalanced.
- GPU clocks were not fixed, so sub-1% differences are not strong evidence of a speedup or regression.
- Graph construction and cuDNN plan selection are intentionally excluded; results describe warm-cache steady state.
- Per-step synchronization isolates host overhead but differs from a deeply queued throughput workload.
- The benchmark currently couples forward-only with evaluation mode and forward-plus-backward with training mode. A training-forward-only mode is needed for a strict additive forward/backward comparison.
- Nsight tracing adds overhead. Comparisons are matched, but absolute CPU times should not be treated as uninstrumented production latency.

Useful follow-up work would decouple training mode from backward execution, add NVTX ranges for C++ and Python cache-key construction, allocation, workspace query, graph execution, and autograd-return phases, repeat profiles in counterbalanced order, and cover unmasked attention, FP16, other sequence lengths, and GQA shapes.

## Artifacts

The recreated reports and SQLite exports are under `/tmp/te_attention_profiles`. The serialized CPU-overhead profiles are:

```text
cpp_fwd.nsys-rep
python_fwd.nsys-rep
cpp_fwd_bwd.nsys-rep
python_fwd_bwd.nsys-rep
```

The queued large-shape profiles are:

```text
cpp_fwd_throughput.nsys-rep
python_fwd_throughput.nsys-rep
cpp_fwd_bwd_throughput.nsys-rep
python_fwd_bwd_throughput.nsys-rep
```

The queued small-shape profiles are:

```text
cpp_fwd_small_throughput.nsys-rep
python_fwd_small_throughput.nsys-rep
cpp_fwd_bwd_small_throughput.nsys-rep
python_fwd_bwd_small_throughput.nsys-rep
```

Each profile has a matching fresh SQLite export in the same directory.

The fresh SQLite exports and summary tables were generated with:

```bash
nsys stats \
    --force-export=true \
    --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,osrt_sum \
    /tmp/te_attention_profiles/<report>.nsys-rep
```

## Graph-build follow-up: aligned C++ and Python paths (2026-08-08)

This follow-up measures warmed-process, cold-TE-cache cuDNN graph construction after
commit `d23ffc756c547591e9dcadb56ef8205e8ebfa7da` aligned the two paths' attention-scale
representation and added equivalent graph-build NVTX stages. It is separate from the
steady-state execution results above.

Three profiles distinguish frontend-language overhead from heuristic-policy overhead:

1. C++ with its normal `heur_mode.A` policy.
2. Python with `--match-cpp-graph-build`, which also uses only `heur_mode.A`.
3. Python with its production `heur_mode.A + heur_mode.FALLBACK` policy.

The match option affects only this benchmark. Production Python retains `A + FALLBACK`
for broader engine coverage. Both TE implementations now describe attention scale as a
rank-4 FP32 pass-by-value tensor backed by a host scalar, so the matched profiles no
longer compare an explicit C++ input with an embedded Python scalar attribute.

### Environment and methodology

| Component | Value |
|---|---|
| GPU | NVIDIA GB200, device 0 (189,471 MiB; four GB200s visible) |
| Driver | 595.84.01 |
| CUDA | 13.3 |
| cuDNN runtime | 9.24.0 (`torch.backends.cudnn.version() == 92400`) |
| cuDNN frontend Python package | 1.27.0 |
| PyTorch | 2.13.0a0+9186a08b2c.nv26.07 |
| Nsight Systems | 2026.3.1.117-263137992252v0 |
| TE source | `d23ffc756c547591e9dcadb56ef8205e8ebfa7da` |

All runs used the small BF16, BSHD, top-left-causal shape `(B=1, S=128, H=1,
D=64)`, training forward plus backward, five warmups, and 30 measured trials. Each
measured trial clears the selected TE graph cache, constructs a training-forward graph
and a backward graph, executes them, and synchronizes afterward. The reported graph
ranges contain host graph construction only; the synchronization and GPU execution are
outside them. The three profiles were captured once, in C++, matched-Python,
production-Python order, without locked GPU clocks.

The command template was:

```bash
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,nvtx \
    --sample=none \
    --output=<report> \
    --force-overwrite=true \
    python benchmarks/attention/profile_fused_attention.py \
        --impl <cpp|python> \
        --measure graph_creation \
        --shape small \
        --mode fwd_bwd \
        --warmup 5 \
        --iterations 30 \
        --profile \
        [--match-cpp-graph-build]
```

### End-to-end graph-build results

Times are Nsight NVTX CPU wall-clock durations in milliseconds. `Fwd` is the
statistics-producing training-forward graph, not the inference graph. Average and
median are both shown because one Python backward IR-definition sample was an outlier:
3.934 ms in the matched run versus a 0.183 ms median.

| Direction | C++ `A` avg / median | Python `A` avg / median | Matched Python - C++ avg / median | Python `A+FALLBACK` avg / median | `FALLBACK` increment avg / median |
|---|---:|---:|---:|---:|---:|
| Fwd | 26.416 / 26.460 | 27.006 / 26.997 | +0.590 / +0.537 | 27.392 / 27.288 | +0.386 / +0.291 |
| Bwd | 30.199 / 30.217 | 31.208 / 31.027 | +1.010 / +0.810 | 33.016 / 32.760 | +1.808 / +1.733 |

With graph representation and heuristic mode matched, Python's median overhead is
0.537 ms (2.0%) for forward and 0.810 ms (2.7%) for backward. The normal Python
`FALLBACK` query raises the median difference from C++ to 0.828 ms (3.1%) and
2.543 ms (8.4%), respectively.

### Matched-policy stage breakdown

The following table uses medians. The stages are sibling ranges inside the inclusive
`cudnn_graph_build_*` range, except that materialization contains the frontend-specific
definition and validation subranges. Per-stage medians do not have to add exactly to
the median of the enclosing range.

| Stage | C++ fwd | Python fwd | Fwd delta | C++ bwd | Python bwd | Bwd delta |
|---|---:|---:|---:|---:|---:|---:|
| Materialization + validation | 0.038 | 0.238 | +0.200 | 0.030 | 0.329 | +0.300 |
| Build operation graph | 0.443 | 0.456 | +0.014 | 1.032 | 1.053 | +0.021 |
| Create execution plans | 1.732 | 1.828 | +0.096 | 2.918 | 3.004 | +0.086 |
| Check support | 0.024 | 0.032 | +0.008 | 0.022 | 0.031 | +0.010 |
| Build plans | 24.208 | 24.394 | +0.186 | 26.212 | 26.550 | +0.338 |
| **Inclusive graph build** | **26.460** | **26.997** | **+0.537** | **30.217** | **31.027** | **+0.810** |

Plan construction dominates both paths: `build_plans` accounts for about 24.2 ms of
forward build time and 26.2-26.6 ms of backward build time. Once the graph has been
lowered, `build_operation_graph` is within 0.014-0.021 ms and the matched heuristic
query is within 0.086-0.096 ms. These close native-stage results are evidence that the
large original difference was not primarily a different cuDNN backend implementation.

The materialization subranges expose work that cannot be named identically because the
public APIs have different architectures:

| Direction | C++ native definition | C++ native validation | Python IR definition | Python lowering + native validation |
|---|---:|---:|---:|---:|
| Fwd | 0.020 | 0.017 | 0.116 | 0.113 |
| Bwd | 0.013 | 0.016 | 0.183 | 0.132 |

### Cost of Python's production heuristic policy

Adding `FALLBACK` has little effect before planning and is therefore best isolated in
the planning stages:

| Direction and stage | Python `A` avg / median | Python `A+FALLBACK` avg / median | Increment avg / median |
|---|---:|---:|---:|
| Fwd total | 27.006 / 26.997 | 27.392 / 27.288 | +0.386 / +0.291 |
| Fwd create execution plans | 1.846 / 1.828 | 1.937 / 1.898 | +0.091 / +0.071 |
| Fwd build plans | 24.377 / 24.394 | 24.642 / 24.546 | +0.265 / +0.152 |
| Bwd total | 31.208 / 31.027 | 33.016 / 32.760 | +1.808 / +1.733 |
| Bwd create execution plans | 3.017 / 3.004 | 4.643 / 4.614 | +1.626 / +1.610 |
| Bwd build plans | 26.581 / 26.550 | 26.717 / 26.677 | +0.135 / +0.127 |

For backward, 1.610 ms of the 1.733 ms median policy increment occurs directly in
`create_execution_plans`. `FALLBACK` is therefore the main reason the production
Python backward graph build remains visibly farther from C++ than the matched run.
This is a coverage-versus-build-latency policy difference in TE, not Python binding
overhead.

### Why the matched Python path is still slower

There are differences on both the TE side and in the public cuDNN frontend Python API:

- TE's C++ path constructs `cudnn_frontend::graph::Graph`, tensor descriptors, and the
  SDPA node directly, then calls the five native validation/planning methods.
- TE's Python path first constructs the public `cudnn.pygraph` IR. In cuDNN frontend
  1.27.0, `python/cudnn/_pygraph.py::validate()` performs Python property inference and
  validation, calls `_lower_to_cpp()`, recreates tensors and the SDPA node in the
  internal pybind `backend_graph`, and finally invokes native validation. This explains
  the 0.200-0.300 ms median materialization difference.
- The pybind methods themselves are thin wrappers around the same C++
  `cudnn_frontend::graph::Graph` methods. The nearly equal operation-graph and support
  stages are consistent with that shared native implementation.
- Public Python `create_execution_plans()` additionally discovers candidate engines,
  invokes its router, and constructs a unified Python plan list around the backend
  results. Its `build_plans()` walks that list and may dispatch a backend entry through
  `build_plan_at_index`; C++ TE calls `Graph::build_plans()` directly. That public-API
  orchestration accounts for the remaining planning delta, especially the 0.338 ms
  backward `build_plans` median difference.
- Without `--match-cpp-graph-build`, TE also deliberately asks Python cuDNN frontend for
  both `A` and `FALLBACK`, while C++ TE asks only for `A`. That separate policy choice
  accounts for most of the additional production backward gap.

The result is now an apples-to-apples comparison at two useful levels: the inclusive
range measures the real TE C++ and public-Python codepaths, while the common planning
ranges show that their lowered native cuDNN work is much closer. Making the inclusive
ranges identical would require bypassing the public Python IR/router and benchmarking
the private `backend_graph` binding, which would no longer represent TE's Python
codepath.

### Graph-build artifacts

The reports and fresh SQLite exports are retained under:

```text
benchmarks/attention/artifacts/cudnn_graph_build_20260808/
```

| Configuration | Nsight report | SQLite export |
|---|---|---|
| C++ `A` | `cpp_a_fwd_bwd_30.nsys-rep` | `cpp_a_fwd_bwd_30.sqlite` |
| Python `A` | `python_a_fwd_bwd_30.nsys-rep` | `python_a_fwd_bwd_30.sqlite` |
| Python `A+FALLBACK` | `python_a_fallback_fwd_bwd_30.nsys-rep` | `python_a_fallback_fwd_bwd_30.sqlite` |

The SQLite files and tables above were generated with:

```bash
nsys stats \
    --force-export=true \
    --report nvtx_sum \
    --format csv \
    benchmarks/attention/artifacts/cudnn_graph_build_20260808/<report>.nsys-rep
```

This round is still one ordered profile per configuration on one shape. Nsight adds
instrumentation overhead, clocks were not locked, and occasional Python/OS scheduling
outliers affect averages. The medians are the more stable comparison; broader claims
would require counterbalanced repetitions across shapes, dtypes, masks, and GQA.
