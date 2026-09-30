"""Check report totals, links, SVG weights and optionally render a preview."""

import importlib.util
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

directory = Path(sys.argv[1]).resolve()
report = directory.parent.parent / 'benchmarks/attention/fused_attention_python_vs_cpp_sampling_report.md'
assets = report.parent / 'fused_attention_python_vs_cpp_sampling_flamegraphs'
analysis = json.loads((directory / 'report_analysis.json').read_text())
for result in analysis:
    for kind in ('cpu', 'python'):
        for scope in ('submission', 'full'):
            data = result[scope + '_' + kind]
            assert sum(row['samples'] for row in data['exclusive_groups']) == data['samples']
            assert sum(data['samples_by_thread'].values()) == data['samples']
            assert sum(data['samples_by_phase'].values()) == data['samples']
            assert sum(row['samples'] for row in data['exclusive_functions']) == data['samples']
            folded = directory / f'{result["name"]}.{kind}.{scope}.folded'
            assert sum(int(line.rsplit(' ', 1)[1]) for line in folded.read_text().splitlines()) == data['samples']
            svg = assets / f'{result["name"]}.{kind}.{scope}.svg'
            root = ET.parse(svg).getroot()
            titles = [element.text for element in root.iter('{http://www.w3.org/2000/svg}title')]
            assert f'all ({data["samples"]:,} samples, 100%)' in titles, (svg, titles[-3:])
            assert (directory / f'{result["name"]}.nsys-rep').is_file()
    for phase in ('iteration', 'forward_call', 'backward_call'):
        assert result['cpu_nvtx'][phase]['count'] == 200
    assert result['gpu_kernels'] == 800

links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report.read_text())
for link in links:
    if not link.startswith(('https://', 'http://', '#')):
        assert (report.parent / link).exists(), link
previews = []
if importlib.util.find_spec('cairosvg'):
    import cairosvg
    for name in ('cpp_small.cpu.submission', 'python_small.python.submission'):
        target = Path('/tmp') / ('te_' + name + '.png')
        cairosvg.svg2png(url=str(assets / (name + '.svg')), write_to=str(target), output_width=1600)
        previews.append(str(target))
print(json.dumps({'report_bytes': report.stat().st_size, 'svg_files': len(list(assets.glob('*.svg'))),
                  'svg_weights_verified': 16, 'local_links_verified': True, 'previews': previews}))
