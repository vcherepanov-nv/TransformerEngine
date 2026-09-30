"""Record the profiling commands, environment, provenance, and artifact hashes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess


directory = Path(__file__).resolve().parent.parent
repository = directory.parent.parent
benchmark = 'benchmarks/attention/profile_fused_attention.py'
targets = ('cpp_small', 'python_small', 'cpp_large', 'python_large')
nsys_options = [
    '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
    '--trace=cuda,nvtx,osrt,cudnn', '--sample=process-tree', '--backtrace=dwarf',
    '--sampling-period=1696496', '--samples-per-backtrace=1',
    '--python-sampling=true', '--python-sampling-frequency=2000',
    '--pytorch=functions-trace',
]
commands = []
conversion_commands = []
for name in targets:
    impl, shape = name.split('_')
    argv = ['nsys', 'profile', *nsys_options, f'--output={directory / name}',
            'python', benchmark, '--impl', impl, '--shape', shape,
            '--measure', 'runtime', '--mode', 'fwd_bwd', '--dtype', 'bf16',
            '--mask', 'causal', '--device', '0', '--warmup', '20',
            '--iterations', '200', '--profile']
    commands.append(shlex.join(argv))
    rep = directory / f'{name}.nsys-rep'
    conversion_commands.extend([
        shlex.join(['nsys', 'export', '--type=sqlite', '--include-json=true',
                    '--dynamic-tables=all', '--quiet=true', '--force-overwrite=true',
                    f'--output={directory / (name + ".sqlite")}', str(rep)]),
        shlex.join(['python', str(directory / 'tools/stackcollapse_nsys.py'),
                    '--nsys=/usr/local/cuda/bin', '--full_function_names',
                    f'--out={directory / (name + ".cpu.folded")}', str(rep)]),
    ])
conversion_commands.append(shlex.join([
    'python', str(directory / 'tools/validate_and_fold_python.py'), str(directory)
]))
prefix = '#!/usr/bin/env bash\nset -eu\ncd ' + shlex.quote(str(repository)) + '\n\n'
(directory / 'profile_commands.sh').write_text(prefix + '\n\n'.join(commands) + '\n')
(directory / 'convert_commands.sh').write_text(prefix + '\n\n'.join(conversion_commands) + '\n')

target_binary = Path('/usr/local/cuda-13.4/NsightSystems-cli-2026.5.1/target-linux-sbsa-armv8/nsys')
header = target_binary.read_bytes()[:20]
assert header[:4] == b'\x7fELF', 'Expected an ELF Nsight target executable'
machine = int.from_bytes(header[18:20], 'little' if header[5] == 1 else 'big')
assert machine == 183 and platform.machine() == 'aarch64', (machine, platform.machine())

manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'branch': 'py-attn',
    'source_commit': 'f8815da06f194cbceba689987a176ec62172d710',
    'branch_analysis': [
        'Adds fused_attention_impl=cpp|python to select either TE C++ fused attention or the Python cuDNN frontend implementation.',
        'The Python path builds and caches forward and backward SDPA graphs and exposes a PyTorch autograd implementation.',
        'The branch also removes zero-dropout C++ RNG work, adds correctness tests, and adds runtime/graph-creation profiling with NVTX ranges.',
        'These captures measure warmed-cache runtime. Cold-cache cuDNN graph construction is excluded by the warmup and capture range.',
    ],
    'environment': {
        'architecture': platform.machine(), 'nsys_target_elf_machine': machine,
        'nsys_cli': shutil.which('nsys'), 'nsys_cli_resolved': str(Path(shutil.which('nsys')).resolve()),
        'nsys_target_binary': str(target_binary),
        'nsys_version': subprocess.check_output(['nsys', '--version'], text=True).strip(),
        'python_version': platform.python_version(),
        'pytorch_version': '2.14.0a0+4fdf77b940.nv26.08',
        'cudnn_runtime_version': 92500, 'cudnn_frontend_version': '1.30.0',
        'gpu': 'NVIDIA GB200', 'gpu_device': 0, 'driver_version': '580.159.04',
        'te_extension': '/workspace/TransformerEngine/transformer_engine_torch.cpython-312-aarch64-linux-gnu.so',
    },
    'workload': {
        'mode': 'fwd_bwd', 'measure': 'runtime', 'dtype': 'bf16', 'mask': 'causal',
        'layout': 'bshd', 'dropout': 0, 'warmups': 20, 'iterations': 200,
        'small_shape_bshd': [1, 128, 1, 64], 'large_shape_bshd': [2, 4096, 128, 128],
        'cpu_overhead_per_iteration_sync': False,
    },
    'profile_commands': commands,
    'conversion_commands': conversion_commands,
    'benchmark_printed_cuda_event_average_ms': {
        'cpp_small': 0.735, 'python_small': 0.695,
        'cpp_large': 3.730, 'python_large': 3.735,
    },
    'helper_provenance': {
        'url': 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2026.3.2_2026.3.2.476-1_amd64.deb',
        'package_sha256': 'e9bf0ff9c613ff5545095a4177676a9579cb666e78c8621c3792f3559e2c1690',
        'extracted_members': ['host-linux-x64/Scripts/Flamegraph/stackcollapse_nsys.py', 'host-linux-x64/Scripts/Flamegraph/README.md'],
        'package_installed': False, 'amd64_binaries_executed': False,
        'reason': 'Portable Python helper is absent from the existing ARM CLI-only installation. The helper executes the installed ARM nsys via --nsys=/usr/local/cuda/bin.',
    },
    'folded_data_notes': [
        '*.cpu.folded was generated by the unmodified NVIDIA stackcollapse_nsys.py with --full_function_names. It extracts native CPU samples only.',
        'The stock NVIDIA SQL groups samples by leaf symbol and module and emits one representative stack per group. Raw CPU stacks are retained in the SQLite and .nsys-rep files.',
        '*.python.folded is supplementary Python periodic sampling data, decoded from exported NVTX JSON payloads and folded by full root-to-leaf stack including file and line.',
        'Python folded weights are sample counts. All recorded Python thread states are included; these are not exclusively on-CPU measurements.',
        'Native backtraces contain some unresolved addresses, ARM mapping symbols, and broken-stack markers. Diagnostics are retained in validation.json.',
        'Nsight emitted possible event-incompleteness warnings. All 200 measured iterations, forward calls, backward calls, and 800 expected kernels were independently verified.',
        'The reported timings include profiling and sampling overhead, including PyTorch function tracing; they are not uninstrumented benchmark timings.',
    ],
    'artifacts': {},
}
for path in sorted(directory.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.name != 'manifest.json':
        manifest['artifacts'][str(path.relative_to(directory))] = {
            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'architecture': platform.machine(), 'nsys_target_elf_machine': machine,
                  'reports': len(list(directory.glob('*.nsys-rep'))),
                  'folded_files': len(list(directory.glob('*.folded'))),
                  'directory': str(directory)}))
