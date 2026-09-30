"""Derive benchmark metrics and function attribution from unaggregated Nsight data."""

from collections import Counter, defaultdict
import html
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys


directory = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('nvidia_stackcollapse', directory / 'tools/stackcollapse_nsys.py')
collapse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collapse)


def clean(value):
    return html.unescape(str(value)).replace(';', ':').replace('\n', ' ').replace('\r', ' ')


def short_native(symbol):
    symbol = clean(symbol)
    return clean(collapse.shorten_function_name_approximately(symbol))


def fold(path, stacks):
    with path.open('w') as output:
        for stack, count in sorted(stacks.items(), key=lambda item: (-item[1], item[0])):
            output.write(';'.join(stack) + f' {count}\n')


def ranked(counter, denominator, limit=15):
    return [{'function': name, 'samples': count, 'percent': count * 100 / denominator}
            for name, count in counter.most_common(limit)] if denominator else []


def duration_stats(events):
    durations = [(end - start) / 1000 for start, end in events]
    return {'count': len(durations), 'total_us': sum(durations),
            'mean_us': statistics.mean(durations), 'median_us': statistics.median(durations),
            'min_us': min(durations), 'max_us': max(durations)}


def inspect(name):
    with sqlite3.connect(directory / f'{name}.sqlite') as db:
        strings = {key: html.unescape(value) for key, value in db.execute('SELECT id, value FROM StringIds')}
        nvtx = list(db.execute('''SELECT n.start, n.end, COALESCE(n.text, s.value), n.globalTid
            FROM NVTX_EVENTS AS n LEFT JOIN StringIds AS s ON s.id = n.textId'''))
        outer = [event for event in nvtx if event[2] and event[2].startswith('fused_attention_')
                 and not event[2].endswith(('_iteration', '_forward_call', '_backward_call'))]
        assert len(outer) == 1
        start, end, range_name, main_tid = outer[0]
        phases = {suffix: sorted((a, b) for a, b, text, tid in nvtx if text == range_name + suffix)
                  for suffix in ('_iteration', '_forward_call', '_backward_call')}
        assert all(len(events) == 200 for events in phases.values())

        def phase_at(timestamp):
            for suffix in ('_forward_call', '_backward_call', '_iteration'):
                if any(a <= timestamp < b for a, b in phases[suffix]):
                    return suffix.removeprefix('_')
            if timestamp >= phases['_iteration'][-1][1]:
                return 'final_drain_and_event_handling'
            return 'iteration_bookkeeping'

        cpu_events = list(db.execute('SELECT id, start, globalTid, threadState, cpuCycles FROM COMPOSITE_EVENTS'))
        chains = defaultdict(list)
        for chain_id, symbol, module, unresolved, special, depth in db.execute('''
            SELECT id, symbol, module, unresolved, specialEntry, stackDepth
            FROM SAMPLING_CALLCHAINS ORDER BY id, stackDepth
        '''):
            chains[chain_id].append({'symbol': strings[symbol], 'module': strings[module],
                                     'unresolved': unresolved, 'special': special, 'depth': depth})

        cpu_stacks, cpu_self, cpu_inclusive, cpu_modules = Counter(), Counter(), Counter(), Counter()
        cpu_by_phase, cpu_by_tid, cpu_states = Counter(), Counter(), Counter()
        cpu_all_in_range = 0
        named_leaf_samples, broken_samples, unknown_samples = 0, 0, 0
        leaf_weights = []
        native_targets = Counter()
        for chain_id, timestamp, tid, state, weight in cpu_events:
            if not start <= timestamp <= end:
                continue
            cpu_all_in_range += 1
            cpu_by_tid[tid] += 1
            if tid != main_tid:
                continue
            cpu_states[state] += 1
            # Linux hardware IP samples represent running CPU execution here.
            # Explicitly require the exported state 'Running' (id=1).
            if state != 1:
                continue
            assert weight == 1, (name, weight)
            leaf_weights.append(weight)
            chain = chains[chain_id]
            assert chain and chain[0]['depth'] == 0
            leaf = chain[0]
            frames = tuple(short_native(frame['symbol']) for frame in reversed(chain)
                           if frame['symbol'] != '[Max depth]')
            cpu_stacks[frames] += weight
            cpu_self[short_native(leaf['symbol'])] += weight
            cpu_modules[Path(leaf['module']).name] += weight
            cpu_inclusive.update(set(frames))
            cpu_by_phase[phase_at(timestamp)] += weight
            if any('Broken backtraces' in frame['symbol'] for frame in chain):
                broken_samples += weight
            unknown_leaf = (bool(leaf['unresolved']) or bool(leaf['special'])
                            or bool(re.fullmatch(r'0x[0-9a-fA-F]+|\$[xd](?:\.\d+)?|\[.*\]', leaf['symbol'])))
            if unknown_leaf:
                unknown_samples += weight
            else:
                named_leaf_samples += weight
            for target in ('transformer_engine::pytorch::fused_attn_fwd',
                           'transformer_engine::pytorch::fused_attn_bwd',
                           'cudnn_frontend::graph::Graph::execute',
                           'cudnn_frontend::python_bindings::read_buffer_extent',
                           'TorchDLPackExchangeAPI::DLTensorFromPyObjectNoSync'):
                if any(target in frame['symbol'] for frame in chain):
                    native_targets[target] += weight

        native_total = sum(cpu_stacks.values())
        assert native_total > 0 and native_total == named_leaf_samples + unknown_samples
        fold(directory / f'{name}.cpu.main.folded', cpu_stacks)

        py_stacks, py_self, py_inclusive, py_states, py_phase = Counter(), Counter(), Counter(), Counter(), Counter()
        py_groups = Counter()
        py_function_locations = {}
        for timestamp, tid, text in db.execute('''
            SELECT start, globalTid, COALESCE(n.jsonText, s.value)
            FROM NVTX_EVENTS AS n LEFT JOIN StringIds AS s ON s.id = n.jsonTextId
            WHERE COALESCE(n.jsonText, s.value) IS NOT NULL
        '''):
            sample = json.loads(text).get('Python Periodic Sampling')
            if sample is None or not start <= timestamp <= end or tid != main_tid:
                continue
            frames = sample['Frames']
            assert len(frames) == sample['Frames count'] and frames
            functions = []
            rendered = []
            for frame in reversed(frames):
                function, filename = strings[frame['Function']], strings[frame['File']]
                # Function attribution groups all lines of the same function in a file.
                key = clean(f'{function} ({filename})')
                functions.append(key)
                py_function_locations[key] = {'function': function, 'file': filename}
                display_file = filename.removeprefix('/workspace/TransformerEngine/').removeprefix('/usr/local/lib/python3.12/dist-packages/')
                rendered.append(clean(f'{function} ({display_file}:{frame["Line"]})'))
            py_stacks[tuple(rendered)] += 1
            py_self[functions[-1]] += 1
            py_inclusive.update(set(functions))
            py_states[sample['Thread state']] += 1
            py_phase[phase_at(timestamp)] += 1
            filename, function = strings[frames[0]['File']], strings[frames[0]['Function']]
            if filename.endswith('/torch/cuda/__init__.py') and function == 'synchronize':
                group = 'torch.cuda.synchronize'
            elif '/cudnn/' in filename:
                group = 'cuDNN frontend Python bindings'
            elif filename.endswith('/cudnn_attention.py') or filename.endswith('/cudnn_frontend.py'):
                group = 'TE Python cuDNN implementation'
            elif '/transformer_engine/' in filename:
                group = 'TE attention and wrappers'
            elif '/torch/' in filename:
                group = 'PyTorch Python and autograd'
            elif '/PythonFunctionsTrace/' in filename or 'nsys' in filename.lower():
                group = 'Nsight Python instrumentation'
            else:
                group = 'Other Python frames'
            py_groups[group] += 1
        py_total = sum(py_stacks.values())
        assert py_total > 0
        fold(directory / f'{name}.python.main.folded', py_stacks)

        kernels = list(db.execute('SELECT start, end, deviceId, streamId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'))
        assert len(kernels) == 800
        assert len({(row[2], row[3]) for row in kernels}) == 1
        gpu_us = sum((row[1] - row[0]) / 1000 for row in kernels)
        gaps = [(b[0] - a[1]) / 1000 for a, b in zip(kernels, kernels[1:])]
        assert min(gaps) >= 0
        runtime = list(db.execute('SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME'))
        cuda_api = Counter()
        for a, b, function_id in runtime:
            cuda_api[strings[function_id]] += (b - a) / 1000

        return {
            'name': name, 'main_global_tid': main_tid, 'window_start_ns': start, 'window_end_ns': end,
            'nvtx_wall_step_ms': (end - start) / 1e6 / 200,
            'cpu_nvtx': {suffix.removeprefix('_'): duration_stats(events) for suffix, events in phases.items()},
            'after_last_iteration_ms': (end - phases['_iteration'][-1][1]) / 1e6,
            'gpu_kernel_us_per_step': gpu_us / 200,
            'gpu_kernel_busy_percent_of_wall': gpu_us * 100 / ((end - start) / 1000),
            'gpu_kernels': len(kernels),
            'gpu_gap_mean_us': statistics.mean(gaps), 'gpu_gap_median_us': statistics.median(gaps),
            'cuda_api_total_us': dict(cuda_api.most_common()),
            'cpu': {
                'report_samples': len(cpu_events), 'all_threads_in_measured_window': cpu_all_in_range,
                'samples_by_global_tid_in_window': dict(cpu_by_tid), 'main_thread_states_in_window': dict(cpu_states),
                'main_running_samples': native_total, 'named_leaf_samples': named_leaf_samples,
                'unknown_leaf_samples': unknown_samples, 'broken_stack_samples': broken_samples,
                'named_leaf_percent': named_leaf_samples * 100 / native_total,
                'broken_stack_percent': broken_samples * 100 / native_total,
                'exclusive_leaf_functions': ranked(cpu_self, native_total),
                'inclusive_functions': ranked(cpu_inclusive, native_total),
                'leaf_modules': ranked(cpu_modules, native_total),
                'phase_samples': dict(cpu_by_phase), 'selected_inclusive_functions': dict(native_targets),
            },
            'python': {
                'main_thread_samples': py_total, 'thread_states': dict(py_states),
                'phase_samples': dict(py_phase), 'exclusive_leaf_groups': ranked(py_groups, py_total),
                'exclusive_leaf_functions': ranked(py_self, py_total),
                'inclusive_functions': ranked(py_inclusive, py_total),
                'all_exclusive_leaf_functions': dict(py_self), 'all_inclusive_functions': dict(py_inclusive),
            },
        }


results = [inspect(name) for name in ('cpp_small', 'python_small', 'cpp_large', 'python_large')]
(directory / 'report_analysis.json').write_text(json.dumps(results, indent=2) + '\n')
for result in results:
    print(json.dumps({key: result[key] for key in ('name', 'nvtx_wall_step_ms', 'cpu_nvtx', 'after_last_iteration_ms', 'gpu_kernel_us_per_step', 'gpu_kernel_busy_percent_of_wall')}))
    print(json.dumps({'name': result['name'], 'cpu': result['cpu']}))
    print(json.dumps({'name': result['name'], 'python': {key: value for key, value in result['python'].items() if not key.startswith('all_')}}))
