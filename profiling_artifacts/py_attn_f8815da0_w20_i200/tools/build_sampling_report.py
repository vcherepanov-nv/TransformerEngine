"""Render SVGs and write the human-readable attention sampling report."""

import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET


directory = Path(sys.argv[1]).resolve()
repository = directory.parent.parent
report = repository / 'benchmarks/attention/fused_attention_python_vs_cpp_sampling_report.md'
asset_dir = report.parent / 'fused_attention_python_vs_cpp_sampling_flamegraphs'
asset_dir.mkdir(exist_ok=True)
results = json.loads((directory / 'report_analysis.json').read_text())
manifest = json.loads((directory / 'manifest.json').read_text())
validation = json.loads((directory / 'validation.json').read_text())
names = {result['name']: result for result in results}
renderer = directory / 'tools/flamegraph.pl'
render_commands = []
for result in results:
    name = result['name']
    impl, shape = name.split('_')
    title = ('C++' if impl == 'cpp' else 'Python') + ' / ' + shape
    for kind, color, label in (('cpu', 'hot', 'native CPU'), ('python', 'green', 'Python stacks')):
        for scope in ('submission', 'full'):
            data = result[scope + '_' + kind]
            source = directory / f'{name}.{kind}.{scope}.folded'
            svg = asset_dir / f'{name}.{kind}.{scope}.svg'
            subtitle = (f'{data["samples"]} samples; all captured process threads; '
                        + ('inside 200 iteration NVTX ranges' if scope == 'submission' else 'complete measured NVTX window'))
            argv = ['perl', str(renderer), '--width=1600', '--fontsize=12', '--minwidth=0.1',
                    '--hash', '--colors=' + color, '--countname=samples',
                    '--title=' + title + ' - ' + label + ' (' + scope + ')',
                    '--subtitle=' + subtitle, str(source)]
            with svg.open('w') as output:
                subprocess.run(argv, stdout=output, check=True)
            root = ET.parse(svg).getroot()
            assert root.tag.endswith('svg')
            assert 'Search' in svg.read_text() and 'samples' in svg.read_text()
            render_commands.append(shlex.join(argv) + ' > ' + shlex.quote(str(svg)))
(directory / 'flamegraph_commands.sh').write_text('#!/usr/bin/env bash\nset -eu\n\n' + '\n\n'.join(render_commands) + '\n')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def lookup(data, list_name, function):
    return next((row['samples'] for row in data[list_name] if row['function'] == function), 0)


def percent_cell(count, denominator):
    return f'{count} ({count * 100 / denominator:.1f}%)'


def function_label(value):
    function, _, filename = value.partition(' (')
    filename = filename.removesuffix(')')
    filename = filename.removeprefix('/workspace/TransformerEngine/').removeprefix('/usr/local/lib/python3.12/dist-packages/')
    return '`' + filename + '::' + function + '`'


def pretty_profile(command):
    argv = shlex.split(command)
    python_index = argv.index('python')
    lines = ['nsys profile \\']
    lines.extend('  ' + option + ' \\' for option in argv[2:python_index])
    lines.append('  python ' + argv[python_index + 1] + ' \\')
    options = argv[python_index + 2:]
    lines.append('    ' + shlex.join(options[:8]) + ' \\')
    lines.append('    ' + shlex.join(options[8:14]) + ' \\')
    lines.append('    ' + shlex.join(options[14:]))
    assert shlex.split('\n'.join(lines).replace('\\\n', ' ')) == argv
    return '\n'.join(lines)


small_delta = (names['python_small']['nvtx_wall_step_ms'] / names['cpp_small']['nvtx_wall_step_ms'] - 1) * 100
large_delta = (names['python_large']['nvtx_wall_step_ms'] / names['cpp_large']['nvtx_wall_step_ms'] - 1) * 100
small_fwd_delta = names['python_small']['cpu_nvtx']['forward_call']['mean_us'] - names['cpp_small']['cpu_nvtx']['forward_call']['mean_us']
small_bwd_delta = names['python_small']['cpu_nvtx']['backward_call']['mean_us'] - names['cpp_small']['cpu_nvtx']['backward_call']['mean_us']

sections = [f'''# C++ vs. Python cuDNN frontend: runtime and CPU sampling report

Capture date: 2026-09-30 (UTC)

Branch: `py-attn`

Source commit: `f8815da06f194cbceba689987a176ec62172d710`

## Findings

For this instrumented forward-plus-backward benchmark, Python has **{-small_delta:.2f}% lower wall time per step for the small shape**. The large-shape wall times differ by **{large_delta:.2f}%**, with Python slightly slower. This is one capture per scenario, so the large-shape difference does not establish a performance ranking.

Both paths execute the same four GPU kernels per step for each shape. Small-shape GPU work is about 17.5 us/step and occupies only 2.4–2.5% of measured wall time. Host submission dominates this run. Large-shape GPU work is about 3.72 ms/step and occupies more than 99% of measured wall time. Submission runs ahead of the GPU, followed by one final queue drain.

The small-shape Python advantage is concentrated in the forward call: its mean CPU range is {abs(small_fwd_delta):.1f} us shorter; its backward call is {small_bwd_delta:.1f} us longer. Those are inclusive host wall times, including profiling and any waiting during submission. Sampling identifies the call paths associated with those intervals, but cannot exclusively assign the timing difference to a specific allocation, cache lookup, or binding conversion.

## Branch and comparison scope

This branch adds `fused_attention_impl="python"` alongside TE's existing C++ fused-attention implementation. The Python path builds and caches cuDNN frontend SDPA graphs and supplies a PyTorch autograd implementation. The branch also removes zero-dropout RNG preparation from the C++ path, adds correctness coverage, and adds the profiling benchmark and repeated NVTX ranges.

These captures compare **warmed-cache runtime through the same PyTorch `DotProductAttention` API**. They include TE/PyTorch wrappers, autograd, tensor/buffer preparation, and cached cuDNN execution. They do not isolate frontend graph construction: all cuDNN graph construction during warmup is outside capture. `fwd_bwd` enables training, uses a forward graph with softmax statistics, and executes backward. This is not an inference-forward comparison.

## Environment

{table(['Component', 'Value'], [
    ('GPU', 'NVIDIA GB200, device 0; idle before the runs'),
    ('CPU / target architecture', 'ARM64 / AArch64, Linux SBSA'),
    ('NVIDIA-SMI driver version', '580.159.04'),
    ('Nsight Systems', manifest['environment']['nsys_version']),
    ('Nsight target binary', '`' + manifest['environment']['nsys_target_binary'] + '`'),
    ('CUDA toolkit / CUPTI', '13.4'),
    ('cuDNN runtime', '9.25.0 (`92500`)'),
    ('cuDNN frontend Python package', '1.30.0'),
    ('Python', '3.12.3'),
    ('PyTorch', manifest['environment']['pytorch_version']),
    ('TE Python source', '`/workspace/TransformerEngine/transformer_engine/`'),
    ('TE extension', '`transformer_engine_torch.cpython-312-aarch64-linux-gnu.so` from this checkout'),
])}

The extension was loaded from the checkout. Its exact compiled-source revision was not independently established by a rebuild in this session. No GPU clock locking or uninstrumented control runs were performed. Loaded CUDA driver-library names in the exports include `libcuda.so.615.65.02`; the table records the driver version reported by `nvidia-smi` rather than assuming those are identical.

## Methodology

### Workload and capture

{table(['Parameter', 'Value'], [
    ('Small shape (B, S, H, D)', '(1, 128, 1, 64)'),
    ('Large shape (B, S, H, D)', '(2, 4096, 128, 128)'),
    ('Data / layout / mask', 'BF16 / BSHD / top-left causal'),
    ('Dropout', '0'),
    ('Mode', 'Forward plus backward (`--mode fwd_bwd`)'),
    ('Warmups / captured iterations', '20 / 200 per scenario'),
    ('Execution', 'Queued iterations; no `--cpu-overhead` or per-iteration synchronization'),
    ('Capture boundary', '`cudaProfilerStart/Stop`, after warmup and before printing the summary'),
    ('CPU sampling', 'Process tree, DWARF, period 1,696,496 hardware cycles, backtrace on every sample'),
    ('Python sampling', '2,000 Hz'),
    ('Tracing', 'CUDA, NVTX, OS runtime, cuDNN, and PyTorch function ranges'),
    ('Run order', 'C++ small, Python small, C++ large, Python large; sequential'),
])}

The script forces TE's fused-attention backend by disabling FlashAttention and unfused attention, then selects `cpp` or `python`. It verifies that fused attention was selected. CUDA synchronization after the 20 warmups completes warmup work before capture starts. The final synchronization completes all 200 queued iterations before capture ends.

`--pytorch=functions-trace` adds module forward and selected PyTorch function ranges. The benchmark's own repeated `iteration`, `forward_call`, and `backward_call` ranges provide the paired timing boundaries. The installed PyTorch tracing preset omits `torch.autograd.grad`, which this benchmark uses; the explicit `backward_call` range covers it. No shape tracing or extra autograd instrumentation was requested. [Nsight PyTorch profiling documentation](https://docs.nvidia.com/nsight-systems/UserGuide/#pytorch-profiling)

### Exact profiling commands

Working directory: `/workspace/TransformerEngine`. The output directory was created beforehand. These are the four commands actually executed, with line breaks added for readability:
''']
sections.append('```bash\n' + '\n\n'.join(pretty_profile(command) for command in manifest['profile_commands']) + '\n```\n')
sections.append('''### Timing and attribution definitions

- **Wall time/step** is the complete outer benchmark NVTX duration divided by 200. It includes final GPU completion and the final synchronization return.
- **CUDA-event time/step** is the benchmark's aggregate stream event interval divided by 200, printed to three decimals. It includes GPU idle intervals between the events, not just kernel execution.
- **CPU iteration / forward / backward** are the durations of their corresponding NVTX ranges. These are host wall times, not exclusive on-core CPU time. The iteration includes both child calls and intervening bookkeeping; final queue drain is outside the individual iterations.
- **GPU work/step** is the sum of the 800 kernel durations divided by 200. Kernels are sequential on one device/stream in these exports. **GPU busy share** divides that sum by the outer wall duration; it is measured kernel occupancy of this window, not a device-wide utilization counter.
- **Native CPU function share** counts running IP samples at a function or in a recorded stack. Every exported `COMPOSITE_EVENTS.cpuCycles` weight is 1 in these captures, so percentages use sample counts. The configured trigger is hardware cycles, whose period is not a fixed wall-clock frequency. These are relative sampled CPU activity, not measured CPU milliseconds or core utilization.
- **Python leaf occupancy** counts samples whose deepest visible Python frame is the named function. Native work or waiting beneath that Python frame is included. **Python inclusive occupancy** counts samples with that function anywhere in the stack, once per sample even with recursion. Inclusive percentages overlap and must not be summed.

Native and Python streams have different sampling triggers and cannot be combined into a single percentage. Python sampler states such as `Running` and `Uninterruptible` are retained; they do not establish exclusive Python bytecode execution time.

The report uses two scopes. **Submission** keeps samples with timestamps inside one of the 200 outer iteration ranges. **Full measured window** also includes inter-iteration bookkeeping and the final queue drain. Both scopes include all captured process threads, including `main`, PyTorch's `pt_autograd_0` worker, and native samples from Nsight helper threads. Thread roots in the graphs make that distinction visible. Sampling only `main` would omit backward work performed by the autograd worker.

### Stack conversion and flamegraph generation

The unmodified NVIDIA helper was run on all four reports with full function names. It reads native CPU samples only; its SQL groups by leaf symbol and module and selects a representative caller stack for each group. Those original `*.cpu.folded` outputs are retained. **The attribution tables and graphs below use complete raw SQLite stacks**, preserving each sample's caller chain before folding identical full stacks. This avoids losing caller attribution in the original helper's leaf aggregation.

Python stacks are decoded from the `Python Periodic Sampling` JSON payloads in the NVTX export, resolving function/file IDs through `StringIds` and reversing leaf-first frames into root-first order. Function tables combine all lines of a function in the same file; graphs retain line numbers. Native graph labels shorten C++ signatures and template arguments using NVIDIA's name-shortening function; attribution by native function consequently combines those displayed overloads. Unknown addresses and broken-stack markers are retained.

The portable NVIDIA helper was extracted from the 2026.3.2 amd64 archive because the installed ARM CLI package lacks it. No package was installed and no amd64 binary ran. Every report and SQLite export used the installed ARM64 Nsight 2026.5.1 executable. The archive URL and SHA-256 are recorded in the artifact manifest.

The conversion and analysis commands, run from the repository root, are:

```bash
ARTIFACTS=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200

for scenario in cpp_small python_small cpp_large python_large; do
  nsys export --type=sqlite --include-json=true --dynamic-tables=all \\
    --quiet=true --force-overwrite=true \\
    --output="$ARTIFACTS/$scenario.sqlite" "$ARTIFACTS/$scenario.nsys-rep"
  python "$ARTIFACTS/tools/stackcollapse_nsys.py" \\
    --nsys=/usr/local/cuda/bin --full_function_names \\
    --out="$ARTIFACTS/$scenario.cpu.folded" "$ARTIFACTS/$scenario.nsys-rep"
done

python "$ARTIFACTS/tools/validate_and_fold_python.py" "$ARTIFACTS"
python "$ARTIFACTS/tools/analyze_report.py" "$ARTIFACTS"
python "$ARTIFACTS/tools/attribute_flamegraphs.py" "$ARTIFACTS"
python "$ARTIFACTS/tools/build_sampling_report.py" "$ARTIFACTS"
```

The last command uses Brendan Gregg's `flamegraph.pl` to render submission and full-window SVGs for each sampling stream. The exact 16 rendering commands are saved in `flamegraph_commands.sh`. For example, the C++ small native submission graph is rendered with:
''')
sections.append('```bash\n' + render_commands[0] + '\n```\n')
sections.append(f'''## Measured results

{table(['Shape', 'Implementation', 'Wall time/step (ms)', 'CUDA-event time/step (ms)', 'GPU work/step (us)', 'GPU busy share', 'Peak PyTorch allocation (GiB)'], [
    (result['name'].split('_')[1], 'C++' if result['name'].startswith('cpp') else 'Python',
     f'{result["nvtx_wall_step_ms"]:.6f}', f'{manifest["benchmark_printed_cuda_event_average_ms"][result["name"]]:.3f}',
     f'{result["gpu_kernel_us_per_step"]:.3f}', f'{result["gpu_kernel_busy_percent_of_wall"]:.2f}%',
     '0.000' if result['name'].endswith('small') else '3.508') for result in results])}

Small-shape `0.000 GiB` is the benchmark's rounded display, not zero memory use. All scenarios contain 800 kernels, or four per step. The kernel-name sets match between C++ and Python for each shape. No `extract_seed_and_offset` kernel was present. The two shape presets select different cuDNN kernels, so kernel names need not match across sizes.

### Inclusive CPU range times

{table(['Scenario', 'Iteration mean / median (us)', 'Forward mean / median (us)', 'Backward mean / median (us)', 'After last iteration (ms)'], [
    (result['name'],
     f'{result["cpu_nvtx"]["iteration"]["mean_us"]:.3f} / {result["cpu_nvtx"]["iteration"]["median_us"]:.3f}',
     f'{result["cpu_nvtx"]["forward_call"]["mean_us"]:.3f} / {result["cpu_nvtx"]["forward_call"]["median_us"]:.3f}',
     f'{result["cpu_nvtx"]["backward_call"]["mean_us"]:.3f} / {result["cpu_nvtx"]["backward_call"]["median_us"]:.3f}',
     f'{result["after_last_iteration_ms"]:.3f}') for result in results])}

“After last iteration” spans CUDA-event handling and the final synchronization until the outer NVTX range ends. For the large shape it is about 0.6 seconds: the host has submitted work faster than the GPU can execute it. That final wait explains why whole-window CPU/Python graphs emphasize synchronization rather than wrapper differences. CPU submission time and GPU time overlap, so adding them does not reconstruct step time.

## CPU attribution summary

### Sample coverage

{table(['Scenario', 'Native samples in report', 'Native submission samples', 'Python samples in report', 'Python submission samples', 'Named native leaf', 'Native stack has broken marker'], [
    (result['name'], result['cpu']['report_samples'], result['submission_cpu']['samples'],
     next(row['python_samples_in_report'] for row in validation if row['name'] == result['name']),
     result['submission_python']['samples'],
     percent_cell(result['submission_cpu']['named_leaf_samples'], result['submission_cpu']['samples']),
     percent_cell(result['submission_cpu']['broken_stack_samples'], result['submission_cpu']['samples'])) for result in results])}

The last two columns describe submission native samples. A broken marker means some caller ancestry could not be unwound; useful frames can still exist in the same stack. A named leaf excludes unresolved addresses, ARM mapping symbols such as `$x`, and synthetic markers. Both limitations constrain precise native function attribution. The roughly 250 native and 300–350 Python submission samples per scenario support broad call-path observations, not precise rankings of tiny functions.

### Native CPU: exclusive attribution by leaf module

Each native sample belongs to exactly one module group based on the module containing its instruction pointer. Module identity is usable even when the individual symbol is unresolved. Percentages use all submission native samples in that scenario and sum to approximately 100% after rounding. This includes profiling overhead visible inside the capture.
''')
native_groups = ['CPython interpreter', 'PyTorch native libraries', 'Transformer Engine native libraries',
                 'cuDNN native libraries and Python extension', 'CUDA driver and runtime',
                 'System runtime and vDSO', 'Nsight and CUPTI instrumentation', 'Other native modules']
sections.append(table(['Leaf module group', *[r['name'] for r in results]], [
    (group, *[f'{lookup(r["submission_cpu"], "exclusive_groups", group) * 100 / r["submission_cpu"]["samples"]:.1f}%' for r in results])
    for group in native_groups]) + '\n')
sections.append('''CPython accounts for about 40–43% of native submission samples in every scenario. The Python path shifts visible native activity from TE's compiled wrapper into PyTorch libraries and cuDNN's Python extension. That module-level shift is consistent with the implementation structure, but does not establish which specific helper caused the measured timing change. Nsight/CUPTI modules themselves account for approximately 9–14% here, showing that these traces contain substantial instrumentation cost.

### Native CPU: selected inclusive function observations

Counts below are samples whose recorded stack contains the named function, including descendants. Missing caller ancestry can undercount them. Zero means no matching recorded frame, not that a function never executed. The rows overlap and do not sum to 100%.
''')
native_functions = ['_PyEval_EvalFrameDefault', 'transformer_engine::pytorch::fused_attn_fwd',
                    'transformer_engine::pytorch::fused_attn_bwd',
                    'at::_ops::empty_memory_format::call', 'torch::autograd::Engine::execute']
sections.append(table(['Function family', *[r['name'] for r in results]], [
    ('`' + function + '`', *[percent_cell(lookup(r['submission_cpu'], 'inclusive_functions', function), r['submission_cpu']['samples']) for r in results])
    for function in native_functions]) + '\n')
sections.append('''### Python: exclusive leaf occupancy by function group

This is the deepest visible Python frame, including native work or waits underneath it. Each Python sample belongs to one row. In particular, a sample at `_engine_run_backward` on the main thread can coexist with a sample executing attention backward on `pt_autograd_0`; both are retained and thread-labelled. These percentages must not be read as exclusive CPU bytecode time.
''')
python_groups = ['PyTorch Python and autograd', 'TE attention and wrappers', 'TE Python cuDNN implementation',
                 'cuDNN frontend Python bindings', 'Other Python frames']
sections.append(table(['Deepest Python frame group', *[r['name'] for r in results]], [
    (group, *[f'{lookup(r["submission_python"], "exclusive_groups", group) * 100 / r["submission_python"]["samples"]:.1f}%' for r in results])
    for group in python_groups]) + '\n')
sections.append('''The Python implementation exposes `cudnn_attention.forward/backward`, `_execute_cudnn_graph`, and `cudnn._pygraph.execute` frames. The C++ path instead exposes `cpp_extensions.fused_attn.fused_attn_fwd/bwd` as its deepest Python entry into compiled work. Sample occupancy at those entry points includes execution below the Python boundary; it cannot distinguish argument preparation, buffer allocation, driver submission, and GPU waiting without the corresponding native stack or phase instrumentation.

### Why full-window graphs look different
''')
sync_function = 'synchronize (/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py)'
sections.append(table(['Scenario', 'Full-window native samples', 'Full-window Python samples', 'Python leaf at final CUDA synchronization'], [
    (r['name'], r['full_cpu']['samples'], r['full_python']['samples'],
     percent_cell(lookup(r['full_python'], 'exclusive_functions', sync_function), r['full_python']['samples']))
    for r in results]) + '\n')
sections.append('''The large runs spend most Python sampled occupancy in the final `torch.cuda.synchronize`. Native stacks often retain `cudaDeviceSynchronize` / `cuCtxSynchronize_v2` ancestors despite unresolved internal driver frames. That is queue-completion behavior after submission, not additional Python frontend graph-building overhead.

## Flame graphs and per-scenario function attribution

Each scenario embeds a native CPU submission graph and a Python submission graph. Their widths are independently normalized to that graph's sample total. Box width represents sampled stack weight; height represents stack depth; horizontal position is stack grouping, not chronological time. Colors identify functions and carry no timing or severity meaning. Full-window alternatives are linked below each scenario.

Markdown image embedding provides a static preview. Open an SVG directly in a browser for the renderer's function tooltips, click-to-zoom, and search. Whether those controls work inside a document preview depends on the renderer's SVG scripting policy. The adjacent links remain available when an application cannot embed SVGs. [FlameGraph renderer and interaction documentation](https://github.com/brendangregg/FlameGraph)
''')
for result in results:
    name = result['name']
    impl, shape = name.split('_')
    title = ('C++' if impl == 'cpp' else 'Python') + ' — ' + shape + ' shape'
    native = result['submission_cpu']
    python = result['submission_python']
    sections.append(f'### {title}\n\nNative submission graph: **{native["samples"]} running CPU samples**. Python submission graph: **{python["samples"]} Python stack samples**, including all sampler states.\n')
    if impl == 'cpp':
        sections.append('The Python entry points into compiled forward and backward are visible separately, with backward execution on `pt_autograd_0`. The native graph exposes compiled descendants where symbols and unwinding are available.\n')
    else:
        sections.append('The Python graph exposes the cached cuDNN graph execution path, including variant-pack/binding preparation and execution entry points. These are warmed-cache paths; the sampling does not show repeated graph construction.\n')
    rows = []
    for function in python['exclusive_functions'][:6]:
        inclusive = lookup(python, 'inclusive_functions', function['function'])
        rows.append((function_label(function['function']), percent_cell(function['samples'], python['samples']), percent_cell(inclusive, python['samples'])))
    sections.append(table(['Python function (file::function)', 'Leaf occupancy', 'Inclusive occupancy'], rows) + '\n')
    for kind, caption in (('cpu', 'Native CPU submission'), ('python', 'Python sampled submission')):
        relative = asset_dir.name + '/' + name + '.' + kind + '.submission.svg'
        sections.append(f'**{caption}**\n\n![{caption}: {title}]({relative})\n\n[Open SVG]({relative})\n')
    sections.append(f'Full measured-window views: [native CPU]({asset_dir.name}/{name}.cpu.full.svg), [Python stacks]({asset_dir.name}/{name}.python.full.svg).\n')

sections.append('''## Validation and limits

Every scenario has 200 completed iteration ranges, 200 forward ranges, 200 backward ranges, 800 GPU kernels, and 1,000 PyTorch function ranges. Both sampling streams are nonempty. Folded totals were checked against the underlying selected sample records. All 16 SVGs were parsed as valid XML and contain the renderer's interaction controls.

Nsight emitted warnings that some CUDA, NVTX, or OS runtime events might not have been collected, plus cuBLASLt symbol-loading warnings. The expected benchmark ranges and kernel counts are present, but that count check does not prove every ancillary event was collected. Native symbolization/unwinding is incomplete as quantified above. Detailed diagnostics are retained in `validation.json`.

Each scenario was captured once, in a fixed order, under sampling and tracing. No uninstrumented baseline, repeated trial confidence interval, or clock-controlled comparison is available. Consequently, the small-shape advantage is an observation in these matched instrumented captures, while the 0.12% large-shape difference is too small to rank the APIs reliably. Percentages from a few hundred submission samples have limited precision. Native addresses are process-specific and should not be compared by numeric address across reports.

The most useful follow-up for optimization would be repeated uninstrumented timings, followed by targeted instrumentation of cache-key construction, allocation, variant-pack/binding conversion, and graph execution. The present evidence locates broad call paths and the forward/backward timing split; it does not exclusively price each internal helper.

## Artifacts

Raw captures, SQLite exports, original NVIDIA-helper outputs, derived full-stack folds, metadata, and scripts are in [`../../profiling_artifacts/py_attn_f8815da0_w20_i200/`](../../profiling_artifacts/py_attn_f8815da0_w20_i200/). The relative links require that artifact directory to accompany this checkout. Embedded SVGs are stored next to this report and remain available independently of the raw captures.
''')
sections.append(table(['Scenario', 'Nsight report', 'CPU submission stacks', 'Python submission stacks'], [
    (r['name'], f'[report](../../profiling_artifacts/py_attn_f8815da0_w20_i200/{r["name"]}.nsys-rep)',
     f'[folded](../../profiling_artifacts/py_attn_f8815da0_w20_i200/{r["name"]}.cpu.submission.folded)',
     f'[folded](../../profiling_artifacts/py_attn_f8815da0_w20_i200/{r["name"]}.python.submission.folded)') for r in results]) + '\n')
sections.append('''Machine-readable results: [analysis](../../profiling_artifacts/py_attn_f8815da0_w20_i200/report_analysis.json), [validation](../../profiling_artifacts/py_attn_f8815da0_w20_i200/validation.json), [capture manifest](../../profiling_artifacts/py_attn_f8815da0_w20_i200/manifest.json), [report-generation provenance](../../profiling_artifacts/py_attn_f8815da0_w20_i200/report_provenance.json).

Command files: [profiling](../../profiling_artifacts/py_attn_f8815da0_w20_i200/profile_commands.sh), [original conversion](../../profiling_artifacts/py_attn_f8815da0_w20_i200/convert_commands.sh), [SVG rendering](../../profiling_artifacts/py_attn_f8815da0_w20_i200/flamegraph_commands.sh).
''')
report.write_text('\n'.join(sections))

# Check every local Markdown link, including embedded SVGs, before handing over.
provenance_path = directory / 'report_provenance.json'
for name in names:
    assert hashlib.sha256((directory / (name + '.nsys-rep')).read_bytes()).hexdigest() == manifest['artifacts'][name + '.nsys-rep']['sha256']
provenance = {'renderer_url': 'https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl',
              'renderer_sha256': hashlib.sha256(renderer.read_bytes()).hexdigest(),
              'report': str(report.relative_to(repository)), 'graphs': [],
              'source_report_sha256': {name: manifest['artifacts'][name + '.nsys-rep']['sha256'] for name in names},
              'analysis_sha256': hashlib.sha256((directory / 'report_analysis.json').read_bytes()).hexdigest(),
              'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
              'analysis_scripts_sha256': {name: hashlib.sha256((directory / 'tools' / name).read_bytes()).hexdigest()
                   for name in ('analyze_report.py', 'attribute_flamegraphs.py', 'build_sampling_report.py')},
              'derived_folded_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(directory.glob('*.folded')) if '.full.' in path.name or '.submission.' in path.name}}
for svg in sorted(asset_dir.glob('*.svg')):
    provenance['graphs'].append({'file': str(svg.relative_to(repository)), 'bytes': svg.stat().st_size,
                                 'sha256': hashlib.sha256(svg.read_bytes()).hexdigest()})
provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report.read_text())
for link in links:
    if not link.startswith(('https://', 'http://', '#')):
        assert (report.parent / link).exists(), f'Missing link: {link}'
assert len(list(asset_dir.glob('*.svg'))) == 16
assert report.read_text().count('![Native CPU submission:') == 4
assert report.read_text().count('![Python sampled submission:') == 4
print(json.dumps({'report': str(report), 'svg_count': 16, 'embedded_graphs': 8,
                  'local_links_checked': sum(not link.startswith(('http://', 'https://', '#')) for link in links)}))
