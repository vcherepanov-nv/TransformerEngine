"""Validate this benchmark's captures and fold its Python periodic samples.

Usage: python validate_and_fold_python.py <report-directory>
SQLite exports must include --include-json=true. Python stacks are leaf-first
in Nsight's payload and are reversed for FlameGraph's root-first format.
Each output weight is the number of Python samples, including all thread states.
Only samples inside the benchmark's outer measured NVTX range are included.
"""

from collections import Counter
import html
import json
from pathlib import Path
import sqlite3
import sys


def clean_frame(value):
    return html.unescape(str(value)).replace(';', ':').replace('\n', ' ').replace('\r', ' ')


def folded_weights(path):
    lines = path.read_text().splitlines()
    assert lines, f'Empty folded output: {path}'
    weights = []
    for line in lines:
        stack, weight = line.rsplit(' ', 1)
        assert stack and int(weight) > 0, f'Invalid folded line in {path}'
        weights.append(int(weight))
    return {'unique_stacks': len(lines), 'total_weight': sum(weights)}


def inspect(directory, name):
    with sqlite3.connect(directory / f'{name}.sqlite') as db:
        strings = dict(db.execute('SELECT id, value FROM StringIds'))
        ranges = list(db.execute('''
            SELECT n.start, n.end, COALESCE(n.text, s.value), n.domainId
            FROM NVTX_EVENTS AS n LEFT JOIN StringIds AS s ON s.id = n.textId
        '''))
        outer = [r for r in ranges if r[2] and r[2].startswith('fused_attention_')
                 and not r[2].endswith(('_iteration', '_forward_call', '_backward_call'))]
        assert len(outer) == 1 and outer[0][1] is not None, (name, outer)
        start, end, range_name, _ = outer[0]
        counts = {suffix: sum(r[2] == range_name + suffix and r[1] is not None for r in ranges)
                  for suffix in ('_iteration', '_forward_call', '_backward_call')}
        assert all(count == 200 for count in counts.values()), (name, counts)

        native_samples, native_weight = db.execute(
            'SELECT COUNT(*), SUM(cpuCycles) FROM COMPOSITE_EVENTS'
        ).fetchone()
        assert native_samples > 0, f'No native samples: {name}'
        stock = folded_weights(directory / f'{name}.cpu.folded')
        assert stock['total_weight'] == native_weight, (name, stock, native_weight)

        python_stacks = Counter()
        states = Counter()
        captured_python_samples = 0
        for timestamp, payload_text in db.execute('''
            SELECT start, COALESCE(n.jsonText, s.value)
            FROM NVTX_EVENTS AS n LEFT JOIN StringIds AS s ON s.id = n.jsonTextId
            WHERE COALESCE(n.jsonText, s.value) IS NOT NULL
        '''):
            payload = json.loads(payload_text)
            sample = payload.get('Python Periodic Sampling')
            if sample is None:
                continue
            captured_python_samples += 1
            if not start <= timestamp <= end:
                continue
            frames = sample['Frames']
            assert len(frames) == sample['Frames count'] and frames, (name, sample)
            stack = []
            for frame in reversed(frames):
                function = strings[frame['Function']]
                filename = strings[frame['File']]
                stack.append(clean_frame(f'{function} ({filename}:{frame["Line"]})'))
            python_stacks[';'.join(stack)] += 1
            states[sample['Thread state']] += 1
        assert python_stacks, f'No Python stacks inside measured range: {name}'
        python_path = directory / f'{name}.python.folded'
        with python_path.open('w') as output:
            for stack, count in sorted(python_stacks.items(), key=lambda item: (-item[1], item[0])):
                output.write(f'{stack} {count}\n')
        python_folded = folded_weights(python_path)
        assert python_folded['total_weight'] == sum(states.values())

        kernels = list(db.execute('''
            SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.demangledName
        '''))
        assert len(kernels) == 800, (name, len(kernels))
        assert not any('extract_seed_and_offset' in row[2] for row in kernels)
        pytorch_domains = {r[3] for r in ranges if r[2] == 'PyTorch'}
        pytorch_ranges = sum(r[3] in pytorch_domains and r[1] is not None for r in ranges)
        assert pytorch_ranges > 0, f'No PyTorch function ranges: {name}'
        diagnostics = [r[0] for r in db.execute('SELECT text FROM DIAGNOSTIC_EVENT WHERE severity >= 2')]
        return {
            'name': name,
            'measured_range': {'name': range_name, 'start_ns': start, 'end_ns': end},
            'measured_iteration_counts': counts,
            'gpu_kernel_count': len(kernels),
            'gpu_kernel_names': sorted({row[2] for row in kernels}),
            'native_cpu_samples': native_samples,
            'native_cpu_folded': stock,
            'python_samples_in_report': captured_python_samples,
            'python_samples_in_measured_range': python_folded['total_weight'],
            'python_unique_stacks': python_folded['unique_stacks'],
            'python_thread_states_in_measured_range': dict(states),
            'pytorch_function_ranges': pytorch_ranges,
            'diagnostic_warnings': diagnostics,
        }


directory = Path(sys.argv[1])
results = [inspect(directory, name) for name in ('cpp_small', 'python_small', 'cpp_large', 'python_large')]
(directory / 'validation.json').write_text(json.dumps(results, indent=2) + '\n')
for result in results:
    print(json.dumps({key: result[key] for key in (
        'name', 'measured_iteration_counts', 'gpu_kernel_count', 'native_cpu_samples',
        'python_samples_in_report', 'python_samples_in_measured_range', 'pytorch_function_ranges'
    )}))
