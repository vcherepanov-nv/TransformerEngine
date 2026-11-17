Attention CI comparison

Main nightly 82552941 (failed)
https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82552941
Tested CI commit 080397c658ed5e1ca78bce65f96b8f637606ef7e

Feature pipeline 82782590 (success)
https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82782590
Tested CI merge commit 0065d88c63a1d5949a597d181e941d1f89b4aa1e

Nightly, all           24 jobs   36,554 executed  164,564 skipped   36,554 passed    0 failed    0 errors  27:58:19 running
Nightly, matched jobs  17 jobs   21,988 executed  120,667 skipped   21,988 passed    0 failed    0 errors  21:23:06 running
Feature, matched jobs  17 jobs   24,936 executed  120,768 skipped   24,936 passed    0 failed    0 errors  18:51:49 running

Feature minus nightly across the 17 matched job names
Executed +2,948
Skipped +101
Running time -02:31:17 (-11.8%)
GitLab duration before removing runner-side waits
  Nightly 21:50:42
  Feature 19:47:34
  Delta -02:03:08

Largest runtime reduction
L0_jax_unittest--B200_1GPU
  Nightly 03:29:46
  Feature 01:29:55
  Delta -01:59:52
  Its mixed JAX pytest invocation took 03:03:27 on main
  and 01:05:33 on the feature branch, based on the JUnit
  suite time. This invocation includes other tests. The pipeline timing
  comparison does not isolate attention performance.

Counts from exact pytest IDs on the matched jobs
130 shared IDs changed from skipped to executed.
0 shared IDs changed from executed to skipped.
2,818 executed IDs appear only on the feature.
0 executed IDs appear only on main.
Renamed IDs or parameter values may appear as one removal and one addition.

Files and measurement
jobs.csv and modules.csv hold the nightly results. job_comparison.csv pairs
the 17 exact job names with matching suite, GPU model and GPU count. It also
lists the seven nightly-only R100 and VR200NVL jobs with blank comparison
deltas. module_comparison.csv compares selected JUnit test cases by module,
job and invocation. native_jobs.csv records the analogous C++ job durations.
The two new SequenceLengthsTest cases are absent from the nightly revision.
The feature CI CTest artifact reports aggregate results for their executable.

GitLab duration excludes GitLab's pre-start queue and can still include
Slurm queueing inside runner setup. Running time subtracts the timestamped
Slurm resource wait for the same allocation ID. It includes setup, the full
job script, artifacts and cleanup. Sums are job elapsed times, so concurrent
jobs contribute independently. The two pipelines used different revisions,
and totals do not control for runner or test matrix differences.

The same attention selectors as the feature report were applied. Reports
for mixed suites were filtered at the test-case level. The JAX MXFP8 module
is absent on main and contributes zero nightly cases. Reported counts
include skipped tests. Executed equals reported minus skipped. Pytest
deselections do not appear in JUnit. Default and deterministic runs count
separately. Delta columns use feature minus nightly.
JUnit counts were checked against each archived XML suite's totals.
Trace summaries matched for 88 of 88 nightly reports
and 60 of 60 feature reports.
