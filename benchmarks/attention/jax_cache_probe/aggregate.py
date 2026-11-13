# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Combine native/Python cache diagnostics and runtime FFI execution measurements."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

EVENTS = {
    "CREATE_GRAPH": "Graphs created",
    "CACHE_GRAPH": "Graphs cached",
    "BUILD_PLANS": "Graphs compiled",
    "EXECUTE": "Graph executions",
    "HIT": "Cache hits",
    "MISS": "Cache misses",
}
NATIVE_COUNTERS = re.compile(
    r"\[FUSED-ATTN-CACHE\] (?:rank=\d+ \| )?tid=all\s+dev=all\s+\|\s+"
    r"(?:f16|fp8) (?:fwd|bwd)\s+\|\s+(.*)"
)
PYTHON_COUNTER = re.compile(r"  (?:f16|fp8) (?:fwd|bwd) (\w+)=(\d+)")


def cache_events(path):
    """Read process totals, excluding the duplicate native per-thread breakdown."""
    native, python = Counter(), Counter()
    for line in path.read_text().splitlines():
        match = NATIVE_COUNTERS.match(line)
        if match:
            native.update(
                {
                    name.upper(): int(count)
                    for name, count in re.findall(r"(\w+)=\s*(\d+)", match[1])
                }
            )
        match = PYTHON_COUNTER.fullmatch(line)
        if match:
            python[match[1]] += int(match[2])
    return native, python


def test_results(path):
    """Read test outcomes and skip reasons from an original or compressed JUnit file."""
    opener = gzip.open if path.suffix == ".gz" else open
    cases = {}
    with opener(path, "rb") as stream:
        tree = ET.parse(stream)
    for case in tree.iter("testcase"):
        status, reason = "passed", ""
        for tag in ("skipped", "failure", "error"):
            child = case.find(tag)
            if child is not None:
                status, reason = tag, child.get("message", "")
                break
        key = case.get("classname") + "::" + case.get("name")
        if key in cases:
            raise ValueError(f"Duplicate JUnit case: {key}")
        cases[key] = (status, reason)
    return cases


def collect_mode(directory, mode):
    """Aggregate both passes while checking execution counts and matching coverage."""
    data, testcase_results = {}, {}
    for side in ("main", "branch"):
        label = side + "-" + mode
        stderr = directory / (label + ".stderr.log")
        if not stderr.exists():
            stderr = directory / (label + ".stderr.txt")
        native, python = cache_events(stderr)
        probe = json.loads((directory / (label + ".probe.json")).read_text())
        if any(target["errors"] for target in probe["targets"].values()):
            raise ValueError(f"{label}: FFI handler errors were recorded.")
        if probe["total_executions"] != sum(
            target["executions"] for target in probe["targets"].values()
        ):
            raise ValueError(f"{label}: Inconsistent FFI execution total.")

        junit = directory / (label + ".xml")
        if not junit.exists():
            junit = directory / (label + ".xml.gz")
        cases = test_results(junit)
        testcase_results[side] = cases
        totals = native + python + Counter(probe["main_score_mod_cache"])
        if side == "main":
            ordinary_executions = sum(
                item["executions"]
                for name, item in probe["targets"].items()
                if "score_mod" not in name
            )
            if ordinary_executions != native["EXECUTE"]:
                raise ValueError(f"{label}: Native and FFI ordinary-attention counts differ.")
        # Native EXECUTE is already included in the FFI total; replace it rather than add it.
        totals["EXECUTE"] = probe["total_executions"]
        data[side] = {
            "tests": dict(Counter(status for status, _ in cases.values())),
            "metrics": {metric: totals[event] for event, metric in EVENTS.items()},
            "native_cache_events": dict(native),
            "python_cache_events": dict(python),
            "ffi_probe": probe,
        }

    if testcase_results["main"] != testcase_results["branch"]:
        raise ValueError(f"{mode}: Test identities, outcomes, or skip reasons differ.")
    data["common_case_count"] = len(testcase_results["main"])
    data["identical_case_outcomes_and_skip_reasons"] = True
    return data


def main():
    """Regenerate the combined CSV and JSON for recorded or newly collected runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modes", nargs="+", choices=("l0", "deterministic"), default=("l0", "deterministic")
    )
    args = parser.parse_args()
    data = {
        "method": {
            "execution": (
                "Successful EXECUTE-stage attention FFI handler invocations in the parent pytest"
                " process. Metadata queries and initialization stages are excluded."
            ),
            "caches": (
                "NVTE_FUSED_ATTN_CACHE_DEBUG=1:all, summed across passes. Main Python score-mod"
                " cache events are instrumented separately and included."
            ),
            "scope": "One fresh process per revision and mode; deterministic totals are separate.",
            "hit_counts": "Lookup sites differ between native and Python implementations.",
        },
        "results": {},
    }
    rows = []
    for mode in args.modes:
        result = collect_mode(args.input_dir, mode)
        data["results"][mode] = result
        for metric in EVENTS.values():
            rows.append(
                {
                    "mode": mode,
                    "Metric": metric,
                    "main": result["main"]["metrics"][metric],
                    "branch": result["branch"]["metrics"][metric],
                }
            )
        print(mode, result["main"]["tests"], result["branch"]["tests"])
        for side in ("main", "branch"):
            print(side, result[side]["metrics"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "combined-stats.json").write_text(json.dumps(data, indent=2) + "\n")
    with (args.output_dir / "combined-stats.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["mode", "Metric", "main", "branch"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
