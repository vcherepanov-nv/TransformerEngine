L2 JAX attention CI results

Main nightly pipeline 82552941
https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82552941
Tested commit 080397c658ed5e1ca78bce65f96b8f637606ef7e

Feature pipeline 82782590
https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82782590
Tested CI merge commit 0065d88c63a1d5949a597d181e941d1f89b4aa1e

Files and scope
comparison.md summarizes the job and module counts and runtime comparison.
jobs.csv lists all 19 L2 JAX test-job records across both pipelines, including
three main-only manual jobs and two superseded feature attempts.
modules.csv contains 42 module records for job attempts that reached attention.
job_comparison.csv pairs the latest attempts by exact job name.
module_comparison.csv compares module outcomes and records shared pytest IDs and
run-to-skip transitions for pairs with complete JUnit reports.

Counts cover tests/jax/test_fused_attn.py, test_fused_attn_score_mod.py,
test_layer.py, test_mxfp8_attention.py, and test_distributed_fused_attn.py.
The Flax layer counts include configurations that may use unfused attention.
The source columns link to the original GitLab JUnit reports or job traces.

Counting
All 12 available JUnit reports were checked against their corresponding pytest
summaries. Reported includes skipped cases. Executed is reported minus skipped.
Deselected tests are excluded. Shared IDs match the JUnit classname and test
name within each job and module. Deltas are feature minus main.

The GB200 NVL unit runs each stopped during the fused-attention module. For
the feature A100 retry, the trace records all outcomes through the end of the
Flax layer module. The job timed out in mHC before reaching MXFP8 attention.
Partial jobs use completed pytest case lines in the trace. A module marked
not_reached has blank counts. A module marked module_absent contributes zero
cases at that revision. Count deltas and shared-ID statistics are blank for
partial job pairs. The original feature A100 attempt and main-only manual jobs
have no attention counts.

Timing
GitLab job duration excludes the queue before the job starts. The adjusted
running time subtracts the resource wait between the matching Slurm queue and
grant messages. The GB200 NVL wait uses the interval from srun submission to
container start. Full job times include setup, other tests, artifact upload,
and cleanup. Concurrent jobs contribute independently to the totals. Sums and
deltas are calculated before rounding to whole seconds.

The primary comparison uses the latest attempts. The original feature B200
unit attempt is included in the CSVs for reference. Its exact attention test
IDs and outcomes matched the retry.
