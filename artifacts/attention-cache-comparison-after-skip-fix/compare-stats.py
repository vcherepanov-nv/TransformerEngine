# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Render the retained cache-statistics snapshot without the original run logs."""

import json
from pathlib import Path

stats = json.loads(Path(__file__).with_name("cache-comparison.json").read_text())
events = ("CREATE_GRAPH", "CACHE_GRAPH", "BUILD_PLANS", "EXECUTE", "HIT", "MISS")


def table(title, main, pr):
    print("\n" + title)
    print(f"{'Event':16} {'main':>12} {'PR':>12} {'Delta':>12}")
    for event in events:
        a, b = main[event], pr[event]
        print(f"{event:16} {a:12,d} {b:12,d} {b-a:+12,d}")


for label in ("main", "pr"):
    print(f"{label}: {stats['revisions'][label]} — {stats[label]['test_outcomes']}")
print(stats["scope"])
table("Totals", stats["main"]["totals"], stats["pr"]["totals"])
for site in stats["main"]["counters"]:
    table(site, stats["main"]["counters"][site], stats["pr"]["counters"][site])
print("\nBuild stages (CPU wall time; warm-up histories differ)")
print(f"{'Stage':40} {'main calls':>10} {'main ms/call':>13} {'PR calls':>10} {'PR ms/call':>12}")
for stage, main in stats["main"]["build_stage_timings"].items():
    pr = stats["pr"]["build_stage_timings"][stage]
    print(
        f"{stage:40} {main['calls']:10,d} {main['average_ms']:13.3f} "
        f"{pr['calls']:10,d} {pr['average_ms']:12.3f}"
    )
