#!/usr/bin/env bash
set -eu
cd /workspace/TransformerEngine

nsys export --type=sqlite --include-json=true --dynamic-tables=all --quiet=true --force-overwrite=true --output=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/cpp_small.sqlite /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/cpp_small.nsys-rep

python /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/tools/stackcollapse_nsys.py --nsys=/usr/local/cuda/bin --full_function_names --out=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/cpp_small.cpu.folded /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/cpp_small.nsys-rep

nsys export --type=sqlite --include-json=true --dynamic-tables=all --quiet=true --force-overwrite=true --output=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/python_small.sqlite /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/python_small.nsys-rep

python /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/tools/stackcollapse_nsys.py --nsys=/usr/local/cuda/bin --full_function_names --out=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/python_small.cpu.folded /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/python_small.nsys-rep

nsys export --type=sqlite --include-json=true --dynamic-tables=all --quiet=true --force-overwrite=true --output=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/cpp_large.sqlite /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/cpp_large.nsys-rep

python /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/tools/stackcollapse_nsys.py --nsys=/usr/local/cuda/bin --full_function_names --out=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/cpp_large.cpu.folded /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/cpp_large.nsys-rep

nsys export --type=sqlite --include-json=true --dynamic-tables=all --quiet=true --force-overwrite=true --output=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/python_large.sqlite /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/python_large.nsys-rep

python /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/tools/stackcollapse_nsys.py --nsys=/usr/local/cuda/bin --full_function_names --out=/workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/python_large.cpu.folded /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/python_large.nsys-rep

python /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200/tools/validate_and_fold_python.py /workspace/TransformerEngine/profiling_artifacts/py_attn_f8815da0_w20_i200
