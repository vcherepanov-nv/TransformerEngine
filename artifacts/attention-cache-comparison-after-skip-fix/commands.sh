#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Run from the root of the revision being tested, with that revision built/installed.
# Usage: bash /absolute/path/to/commands.sh {collect|run} {main|pr}
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != collect && "$1" != run ) ]]; then
    echo "Usage: bash $0 {collect|run} LABEL" >&2
    exit 2
fi

mode=$1
label=$2
comparison_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pytest_options=(-s --tb=auto)
if [[ "$mode" == collect ]]; then
    pytest_options+=(--collect-only -q)
fi

# Preserve the relevant qa/L0_pytorch_unittest/test.sh settings.
# Level 1 prints graph-cache counters/build timings at process exit.
# Use one pytest process, without xdist or extra selection options.
env \
    NVTE_FUSED_ATTN_CACHE_DEBUG=1 \
    NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 \
    NVTE_FLASH_ATTN_V2=1 \
    NVTE_FLASH_ATTN_V3=0 \
    NVTE_FLASH_ATTN_V4=0 \
    NVTE_REPORT_IF_CUTEDSL_BACKEND_NOT_CHOSEN=ERROR \
    python -m pytest "${pytest_options[@]}" \
        "@$comparison_dir/common-tests.txt" \
        2>&1 | tee "$comparison_dir/$label-$mode.log"
