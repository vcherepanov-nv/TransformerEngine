Attention test results for GitLab pipeline 82782590
https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82782590
Pipeline ref pr_3693_main_82782460_CI
Tested commit 0065d88c63a1d5949a597d181e941d1f89b4aa1e (GitLab CI merge commit)

jobs.csv contains 17 PyTorch and JAX jobs with in-scope pytest results.
modules.csv contains 72 rows, one per job, module, and pytest invocation.
native_jobs.csv records 9 C++ jobs that ran the test_operator binary
containing the two SequenceLengthsTest cases. CTest reports a single result for
the binary and truncates its output before those cases. Individual outcomes are unavailable.

Counts across the PyTorch and JAX jobs
PyTorch   7 jobs  119,178 reported   16,285 executed   16,285 passed  102,893 skipped     0 failed     0 errors     0 xfailed
JAX      10 jobs   26,526 reported    8,651 executed    8,651 passed   17,875 skipped     0 failed     0 errors     0 xfailed
Total    17 jobs  145,704 reported   24,936 executed   24,936 passed  120,768 skipped     0 failed     0 errors     0 xfailed

How the numbers were collected
glab retrieved full GitLab job traces, job metadata, and JUnit artifacts.
All 60 included pytest report totals were checked against
the matching summary in the trace. Counts come from individual JUnit test cases.
Reported equals passed + skipped + failed + errors + xfailed. Executed equals
reported minus skipped. Deselected tests are absent from JUnit and excluded.
Separate default and deterministic invocations count as separate runs.
GitLab duration is elapsed time for the entire job, excluding queue time.
The failed A100 JAX attempt (553688735, 41.253 seconds)
stopped during GPU allocation before pytest. Its successful retry is included.

Attention scope
PyTorch test_attention.py includes fused and mixed-backend cases. Dedicated
FlashAttention version, FlashAttention 4, FlashAttention THD, num_splits,
and standalone softcap tests were excluded. The zero-softcap backend-selection
guard is included. In test_attention_backend_selection.py, FA4-only tests
were excluded. The KV-cache and context-parallel counts are for fused cases.
The torch.compile counts include fused DPA parameters, fused-op fallback, and
backend-selection tracing. CPU offload counts include only explicit FusedAttention
cases for MultiheadAttention and TransformerLayer. The identity quantizer counts
cover only the FP8 DPA rejection check. Flex Attention and MXFP8 MLA modules
are included in full.

JAX counts include the fused-attention, score-modifier, MXFP8, distributed
fused-attention and Flax layer modules. The attention tutorial was included
in both the single-device and multi-GPU jobs, with other tutorials excluded.
Mixed-backend and integration cases are counted as reported by pytest. Their
JUnit outcomes do not establish which backend executed inside a passing case.
