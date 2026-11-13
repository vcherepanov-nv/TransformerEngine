..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX attention cache and execution probe
=======================================

This profiling utility records the cache activity and native FFI executions of
JAX fused attention. It targets the private FFI names and score-mod cache symbols
in the two revisions recorded below. The probe runs outside the TE source and
build trees.

Recorded comparison
-------------------

``results/2026-11-13`` compares ``py-all`` at ``8ad48332`` with ``main`` at
``ab4b9222``. Each revision ran the same ordered list of 5,135 pytest node IDs in
one fresh process on one GB200 (SM100). Both produced 1,234 passes, 3,901 skips,
and no failures. All case identities, outcomes, and skip reasons match.

==================  ======  ======
Metric                main  py-all
==================  ======  ======
Graphs created       1,090     800
Graphs cached          977     687
Graphs compiled        917     625
Graph executions     2,420   2,420
Cache hits          33,176  10,263
Cache misses         1,090     800
==================  ======  ======

All rows combine forward and backward and include ordinary SDPA and score-mod
on both revisions. The separate deterministic run produced 16 passes and 24
skips, with 32 graph executions on each revision. Its counters are kept separate
from the L0 comparison.

The recorded environment used CUDA 13.4.3, cuDNN 9.27.0, cuDNN Frontend 1.31.0,
JAX 0.12.1.dev20261113+289b65056b, jaxlib 0.12.1.dev20261113, Flax 0.12.12,
pytest 9.2.0, and driver 595.84.01. Four GPUs were available; each pytest process
saw one GPU. ``run-metadata.json`` preserves the exact commands, working
directories, software versions, source hashes, and validation results.

Measurement method
------------------

* ``NVTE_FUSED_ATTN_CACHE_DEBUG=1:all`` collects exit summaries from all ranks.
  Level ``2:all`` additionally traces individual events.
* ``ffi_probe_plugin.py`` wraps only the ``execute`` entries of attention FFI
  handler bundles, before TE registers them. ``ffi_probe.cpp`` forwards the
  original handlers and counts successful native EXECUTE-stage calls. Metadata
  queries and initialization are excluded. Counting stays in C++ during runtime
  execution. GPU command-buffer replays that bypass the host handler are not
  counted separately.
* On the old main revision, ``TE_PROBE_MAIN_SCORE_MOD=1`` also instruments its
  Python score-mod cache. This adds six created/cached/compiled graphs, six
  misses, and two hits to its native diagnostic totals. Its eight score-mod
  executions are included by the FFI counter. On ``py-all``, the existing Python
  diagnostics already include score-mod cache activity.
* Execution counts replace the native EXECUTE column, avoiding double counting.
  Cache hit counts reflect different lookup sites in the two implementations.
* Persistent JAX compilation caching is disabled. Each revision and determinism
  policy uses a fresh process and retains normal in-process caches throughout
  the suite. Keep the whole suite in one process to preserve cache reuse.

Calibration used one ordinary max-logit forward case and the three numerical
score-mod cases. Each revision passed all four cases and recorded nine runtime
executions. The full runs preserved every raw cache counter from the preceding
run without the execution probe. Raw cuDNN API logging was rejected as a
measurement method because it includes warm-ups and entry/exit records.

Data layout and verification
----------------------------

The results directory contains the recorded combined JSON/CSV, run metadata,
native FFI counter JSON, stderr summaries, compressed stdout logs and JUnit XML,
and the compressed common test selection. The calibration records are included.
``measured-sources`` preserves the exact temporary source bytes, including the
SHA256-identified probe sources used for the measurements. The adjacent toolkit
sources add license headers, formatting, command-line helpers, and relocatable
paths. The readable CSV uses LF line endings for version control.

Regenerate the table without importing TE or accessing a GPU::

    python benchmarks/attention/jax_cache_probe/aggregate.py \
      benchmarks/attention/jax_cache_probe/results/2026-11-13 \
      --output-dir /tmp/te-attention-cache-report

The aggregator checks all test identities, outcomes, and skip reasons, rejects
FFI errors, and checks main's ordinary-attention FFI counts against its native
EXECUTE counters. The regenerated metric values match the committed records.

Reproduce the measurement
-------------------------

Use separate TE builds for the recorded revisions, with the same JAX and CUDA
dependencies. The recorded editable build configuration was
``NVTE_CUDA_ARCHS=100 NVTE_FRAMEWORK=jax``. See ``docs/installation.rst`` for the
build procedure. Verify that each interpreter loads its intended worktree.

Build the probe with the active JAX installation's FFI headers and unpack the
frozen common selection::

    python benchmarks/attention/jax_cache_probe/prepare.py \
      --output-dir /tmp/te-attention-probe

Run this from each TE checkout, using that checkout's interpreter. Set
``TE_PROBE_MAIN_SCORE_MOD=1`` for the recorded main revision and ``0`` for the
recorded py-all revision. Set ``probe_label`` to ``main-l0`` or ``branch-l0``::

    probe_dir=/tmp/te-attention-probe
    probe_label=branch-l0
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$probe_dir" \
    TE_FFI_PROBE_OUTPUT="$probe_dir/$probe_label.probe.json" \
    TE_PROBE_MAIN_SCORE_MOD=0 \
    NVTE_FUSED_ATTN_CACHE_DEBUG=1:all \
    NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 \
    NVTE_JAX_UNITTEST_LEVEL=L0 \
    NVTE_REPORT_IF_CUTEDSL_BACKEND_NOT_CHOSEN=ERROR \
    JAX_ENABLE_COMPILATION_CACHE=false \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    python -m pytest -p ffi_probe_plugin -c tests/jax/pytest.ini -v -s --tb=short \
      --junitxml="$probe_dir/$probe_label.xml" \
      @"$probe_dir/common-tests.args" \
      > "$probe_dir/$probe_label.stdout.log" \
      2> "$probe_dir/$probe_label.stderr.log"

For the separate deterministic run, use labels ``main-deterministic`` and
``branch-deterministic``, set ``NVTE_ALLOW_NONDETERMINISTIC_ALGO=0``, and replace
the argument-file selection with
``tests/jax/test_fused_attn.py -k TestFusedAttnWithDeterminism``.

Aggregate newly collected runs with ``aggregate.py INPUT_DIR --output-dir
OUTPUT_DIR``. Add ``--modes l0`` when only the L0 pair is available. The exact
historical commands in the committed metadata retain their original absolute
paths; the example above uses a relocatable probe directory.
