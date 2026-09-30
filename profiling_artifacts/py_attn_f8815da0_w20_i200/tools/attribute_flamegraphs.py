"""Preserve complete stacks and include the autograd worker in attribution."""

from collections import Counter, defaultdict
import html
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import sys


directory = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('nvidia_stackcollapse', directory / 'tools/stackcollapse_nsys.py')
collapse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collapse)
results = json.loads((directory / 'report_analysis.json').read_text())


def clean(value):
    return html.unescape(str(value)).replace(';', ':').replace('\n', ' ').replace('\r', ' ')


def rank(counter, total):
    return [{'function': function, 'samples': count, 'percent': count * 100 / total}
            for function, count in counter.most_common()]


def write_folded(name, records):
    counts = Counter(row['stack'] for row in records)
    with (directory / f'{name}.folded').open('w') as output:
        for stack, count in sorted(counts.items(), key=lambda row: (-row[1], row[0])):
            output.write(';'.join(stack) + f' {count}\n')
    assert sum(counts.values()) == len(records) > 0


def summarize(records, native):
    total = len(records)
    self_counts, inclusive, groups, modules, threads, phases, states = [Counter() for _ in range(7)]
    for row in records:
        self_counts[row['functions'][-1]] += 1
        inclusive.update(set(row['functions']))
        groups[row['group']] += 1
        modules[row.get('module', '')] += 1
        threads[row['thread']] += 1
        phases[row['phase']] += 1
        states[row['state']] += 1
    summary = {'samples': total, 'exclusive_functions': rank(self_counts, total),
               'inclusive_functions': rank(inclusive, total), 'exclusive_groups': rank(groups, total),
               'samples_by_thread': dict(threads), 'samples_by_phase': dict(phases),
               'thread_states': dict(states)}
    if native:
        summary.update({'leaf_modules': rank(modules, total),
                        'named_leaf_samples': sum(not row['unknown'] for row in records),
                        'broken_stack_samples': sum(row['broken'] for row in records)})
    return summary


def native_group(module):
    if module == 'python3.12':
        return 'CPython interpreter'
    if module.startswith(('libtorch', 'libc10')):
        return 'PyTorch native libraries'
    if module.startswith(('libtransformer_engine', 'transformer_engine_torch')):
        return 'Transformer Engine native libraries'
    if module.startswith(('libcupti', 'libTools', 'libnsys')):
        return 'Nsight and CUPTI instrumentation'
    if module.startswith(('libcudnn', '_compiled_module')):
        return 'cuDNN native libraries and Python extension'
    if module.startswith(('libcuda.', 'libcudart.')):
        return 'CUDA driver and runtime'
    if module.startswith(('libc.', 'libstdc++', 'libgcc', 'libm.', 'ld-linux')) or module == '[vdso]':
        return 'System runtime and vDSO'
    return 'Other native modules'


for result in results:
    name = result['name']
    start, end = result['window_start_ns'], result['window_end_ns']
    with sqlite3.connect(directory / f'{name}.sqlite') as db:
        strings = {key: html.unescape(value) for key, value in db.execute('SELECT id, value FROM StringIds')}
        names = {tid: strings[key] for key, _, tid in db.execute('SELECT * FROM ThreadNames')}
        ranges = list(db.execute('''SELECT n.start, n.end, COALESCE(n.text, s.value)
            FROM NVTX_EVENTS AS n LEFT JOIN StringIds AS s ON s.id = n.textId
            WHERE n.end IS NOT NULL'''))
        phases = {suffix: sorted((a, b) for a, b, text in ranges
                                if text and text.startswith('fused_attention_') and text.endswith('_' + suffix))
                  for suffix in ('forward_call', 'backward_call', 'iteration')}
        last_iteration = phases['iteration'][-1][1]

        def phase_at(timestamp):
            for suffix in ('forward_call', 'backward_call', 'iteration'):
                if any(a <= timestamp < b for a, b in phases[suffix]):
                    return suffix
            return 'final_drain_and_event_handling' if timestamp >= last_iteration else 'iteration_bookkeeping'

        def thread_label(tid):
            return 'main' if tid == result['main_global_tid'] else names.get(tid, f'tid {tid & 0xffffff}')

        chains = defaultdict(list)
        for chain_id, symbol, module, unresolved, special, depth in db.execute('''
            SELECT id, symbol, module, unresolved, specialEntry, stackDepth
            FROM SAMPLING_CALLCHAINS ORDER BY id, stackDepth'''):
            chains[chain_id].append((strings[symbol], strings[module], unresolved, special, depth))
        native_records = []
        for chain_id, timestamp, tid, state, weight in db.execute('SELECT id, start, globalTid, threadState, cpuCycles FROM COMPOSITE_EVENTS'):
            if not start <= timestamp <= end or state != 1:
                continue
            assert weight == 1
            chain = chains[chain_id]
            assert chain[0][4] == 0
            leaf_symbol, leaf_module, unresolved, special, _ = chain[0]
            functions = [clean(collapse.shorten_function_name_approximately(frame[0])) for frame in reversed(chain) if frame[0] != '[Max depth]']
            module = Path(leaf_module).name
            native_records.append({'stack': tuple(['thread: ' + clean(thread_label(tid)), *functions]),
                'functions': functions, 'phase': phase_at(timestamp), 'thread': thread_label(tid),
                'state': 'Running', 'module': module, 'group': native_group(module),
                'unknown': bool(unresolved or special or re.fullmatch(r'0x[0-9a-fA-F]+|\$[xd](?:\.\d+)?|\[.*\]', leaf_symbol)),
                'broken': any('Broken backtraces' in frame[0] for frame in chain)})

        python_records = []
        for timestamp, tid, text in db.execute('''SELECT start, globalTid, COALESCE(n.jsonText, s.value)
            FROM NVTX_EVENTS AS n LEFT JOIN StringIds AS s ON s.id = n.jsonTextId
            WHERE COALESCE(n.jsonText, s.value) IS NOT NULL'''):
            sample = json.loads(text).get('Python Periodic Sampling')
            if sample is None or not start <= timestamp <= end:
                continue
            functions, rendered = [], []
            for frame in reversed(sample['Frames']):
                function, filename = strings[frame['Function']], strings[frame['File']]
                functions.append(clean(f'{function} ({filename})'))
                display_file = filename.removeprefix('/workspace/TransformerEngine/').removeprefix('/usr/local/lib/python3.12/dist-packages/')
                rendered.append(clean(f'{function} ({display_file}:{frame["Line"]})'))
            assert functions and len(functions) == sample['Frames count']
            leaf = sample['Frames'][0]
            filename, function = strings[leaf['File']], strings[leaf['Function']]
            if filename.endswith('/torch/cuda/__init__.py') and function == 'synchronize':
                group = 'torch.cuda.synchronize'
            elif '/cudnn/' in filename:
                group = 'cuDNN frontend Python bindings'
            elif filename.endswith(('/cudnn_attention.py', '/cudnn_frontend.py')):
                group = 'TE Python cuDNN implementation'
            elif '/transformer_engine/' in filename:
                group = 'TE attention and wrappers'
            elif '/torch/' in filename:
                group = 'PyTorch Python and autograd'
            elif '/PythonFunctionsTrace/' in filename or 'nsys' in filename.lower():
                group = 'Nsight Python instrumentation'
            else:
                group = 'Other Python frames'
            python_records.append({'stack': tuple(['thread: ' + clean(thread_label(tid)), *rendered]),
                'functions': functions, 'phase': phase_at(timestamp), 'thread': thread_label(tid),
                'state': sample['Thread state'], 'group': group})

        for kind, records in (('cpu', native_records), ('python', python_records)):
            submission = [row for row in records if row['phase'] in ('forward_call', 'backward_call', 'iteration')]
            result['full_' + kind] = summarize(records, kind == 'cpu')
            result['submission_' + kind] = summarize(submission, kind == 'cpu')
            write_folded(name + '.' + kind + '.full', records)
            write_folded(name + '.' + kind + '.submission', submission)

    print(json.dumps({'name': name, 'wall_ms': result['nvtx_wall_step_ms'],
        'native_submission': result['submission_cpu']['samples'],
        'python_submission': result['submission_python']['samples'],
        'cpu_threads': result['full_cpu']['samples_by_thread'],
        'python_threads': result['full_python']['samples_by_thread'],
        'native_leaf_groups': result['submission_cpu']['exclusive_groups'],
        'python_leaf_groups': result['submission_python']['exclusive_groups'],
        'python_top_self': result['submission_python']['exclusive_functions'][:8],
        'cpu_named_leaf': result['submission_cpu']['named_leaf_samples'],
        'cpu_broken': result['submission_cpu']['broken_stack_samples']}))
(directory / 'report_analysis.json').write_text(json.dumps(results, indent=2) + '\n')
