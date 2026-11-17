# cuDNN attention CI comparison

This report compares the affected PyTorch and JAX attention tests across 17 jobs with the same suite, GPU model, and GPU count. [Main nightly 82552941](https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82552941) · [Feature pipeline 82782590](https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82782590)

## Matched jobs

| Pipeline | Test cases run | Skipped | Failed | Total job runtime |
|---|---:|---:|---:|---:|
| Main nightly | 21,988 | 120,667 | 0 | 21:23:06 |
| Feature | 24,936 | 120,768 | 0 | 18:51:49 |
| Feature minus main | +2,948 | +101 | 0 | −02:31:17 (-11.8%) |

**Zero test cases went from running on main to skipped on the feature branch.** All main test IDs were present in the matching feature job and invocation. The net increase of 101 skips came from 231 additional skipped cases and 130 shared cases that went from skipped on main to executed on the branch. [Module comparison](module_comparison.csv)

PyTorch’s `test_attention.py` accounted for 2,274 additional runs. The largest runtime drop was the single-GPU B200 JAX job, from 03:29:46 to 01:29:55. [Job comparison](job_comparison.csv)

## Per-job results

Counts cover the affected attention cases. Runtimes cover the full job, including setup, other tests, and cleanup. All deltas are feature minus main.

| Job | Main run | Feature run | Δ run | Main skipped | Feature skipped | Δ skipped | Main runtime | Feature runtime | Δ runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0_jax_distributed_unittest--H100_4GPU | 3 | 3 | 0 | 0 | 0 | 0 | 00:15:53 | 00:22:21 | +00:06:29 |
| L0_jax_distributed_unittest--B200_8GPU | 3 | 3 | 0 | 0 | 0 | 0 | 00:36:23 | 00:55:45 | +00:19:22 |
| L0_jax_distributed_unittest--GB200_4GPU | 3 | 3 | 0 | 0 | 0 | 0 | 00:17:43 | 00:22:39 | +00:04:56 |
| L0_jax_unittest--A100_1GPU | 1,333 | 1,389 | +56 | 4,113 | 4,174 | +61 | 00:49:41 | 00:47:46 | −00:01:55 |
| L0_jax_unittest--H100_1GPU | 1,751 | 1,813 | +62 | 3,695 | 3,750 | +55 | 00:45:53 | 00:47:30 | +00:01:37 |
| L0_jax_unittest--B200_1GPU | 1,650 | 1,767 | +117 | 3,928 | 3,928 | 0 | 03:29:46 | 01:29:55 | −01:59:52 |
| L0_jax_unittest--GB200_1GPU | 1,650 | 1,767 | +117 | 3,928 | 3,928 | 0 | 00:57:04 | 00:56:36 | −00:00:28 |
| L0_pytorch_unittest--A100_1GPU | 794 | 1,371 | +577 | 9,816 | 9,864 | +48 | 00:54:18 | 00:50:07 | −00:04:11 |
| L0_pytorch_unittest--H100_1GPU | 1,529 | 2,270 | +741 | 9,153 | 9,037 | −116 | 00:46:27 | 00:47:22 | +00:00:55 |
| L0_pytorch_unittest--B200_1GPU | 5,116 | 5,727 | +611 | 5,710 | 5,724 | +14 | 01:33:12 | 01:33:31 | +00:00:19 |
| L0_pytorch_unittest--GB200_1GPU | 5,116 | 5,727 | +611 | 5,710 | 5,724 | +14 | 01:30:30 | 01:28:51 | −00:01:40 |
| L1_jax_distributed_unittest--H100_4GPU | 601 | 603 | +2 | 822 | 847 | +25 | 00:41:07 | 00:39:25 | −00:01:42 |
| L1_jax_distributed_unittest--B200_8GPU | 697 | 724 | +27 | 726 | 726 | 0 | 01:51:01 | 00:56:28 | −00:54:33 |
| L1_jax_distributed_unittest--GB200_4GPU | 552 | 579 | +27 | 522 | 522 | 0 | 00:32:18 | 00:29:01 | −00:03:17 |
| L1_pytorch_distributed_unittest--H100_4GPU | 80 | 80 | 0 | 24,498 | 24,498 | 0 | 01:18:01 | 01:18:18 | +00:00:17 |
| L1_pytorch_distributed_unittest--B200_8GPU | 555 | 555 | 0 | 24,023 | 24,023 | 0 | 02:31:15 | 02:30:29 | −00:00:46 |
| L1_pytorch_distributed_unittest--GB200NVL_4GPU | 555 | 555 | 0 | 24,023 | 24,023 | 0 | 02:32:33 | 02:35:46 | +00:03:13 |

## Scope and timing

- The nightly also ran seven R100 and VR200NVL jobs. Across all 24 nightly jobs in scope, 36,554 attention cases ran and passed, 164,564 were skipped, and job runtimes summed to 27:58:19. [Nightly jobs](jobs.csv) · [Nightly module counts](modules.csv)
- Counts come from filtered JUnit results checked against pytest summaries in the job logs. Default and deterministic invocations count separately. Deselected cases are excluded. Run-to-skip transitions use exact pytest IDs within the same job and invocation. [Counting notes](README.txt)
- GitLab’s duration excludes time queued before the job starts. The reported runtime also subtracts logged Slurm GPU allocation waits within the job. Total runtime sums individual job runtimes, including jobs that ran concurrently. [Timing data](job_comparison.csv)
- C++ job durations are recorded separately. The two new sequence-length tests are absent on main, and the feature CTest artifact reports their operator binary as a single result. [Native job durations](native_jobs.csv)
