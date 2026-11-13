# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""External pytest collector: native FFI execution counters, no TE source changes."""
import atexit
from collections import Counter
import ctypes
import functools
import json
import os
from pathlib import Path

TARGETS = (
    "te_cudnn_graph_ffi",
    "te_fused_attn_forward_ffi",
    "te_fused_attn_backward_ffi",
    "te_fused_attn_score_mod_forward_ffi",
    "te_fused_attn_score_mod_backward_ffi",
)
_registered = []
_names = []
_score_mod = Counter()


def _report():
    result = {"pid": os.getpid(), "targets": {}, "main_score_mod_cache": dict(_score_mod)}
    for name in _registered:
        index = TARGETS.index(name)
        result["targets"][name] = {
            "executions": _library.Executions(index),
            "metadata_queries": _library.MetadataQueries(index),
            "errors": _library.Errors(index),
        }
    result["total_executions"] = sum(x["executions"] for x in result["targets"].values())
    Path(os.environ["TE_FFI_PROBE_OUTPUT"]).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


# Register before importing JAX so its later exit cleanup can complete first.
atexit.register(_report)
from jax import ffi  # pylint: disable=wrong-import-position

_library = ctypes.CDLL(str(Path(__file__).with_name("ffi_probe.so")))
_library.Install.argtypes = [ctypes.c_int, ctypes.c_void_p]
_library.Install.restype = ctypes.c_void_p
for _function in (_library.Executions, _library.MetadataQueries, _library.Errors):
    _function.argtypes = [ctypes.c_int]
    _function.restype = ctypes.c_uint64
_get_name = ctypes.pythonapi.PyCapsule_GetName
_get_name.argtypes = [ctypes.py_object]
_get_name.restype = ctypes.c_char_p
_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_get_pointer.restype = ctypes.c_void_p
_new_capsule = ctypes.pythonapi.PyCapsule_New
_new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_new_capsule.restype = ctypes.py_object
_register = ffi.register_ffi_target


def _register_target(name, handlers, *args, **kwargs):
    if name in TARGETS:
        assert isinstance(handlers, dict) and "execute" in handlers
        handlers = dict(handlers)
        original = handlers["execute"]
        capsule_name = _get_name(original)
        _names.append(capsule_name)  # PyCapsule retains this character pointer.
        pointer = _library.Install(TARGETS.index(name), _get_pointer(original, capsule_name))
        assert pointer, name
        handlers["execute"] = _new_capsule(pointer, capsule_name, None)
        _registered.append(name)
    return _register(name, handlers, *args, **kwargs)


ffi.register_ffi_target = _register_target


def pytest_configure(config):
    """Count the old main revision's otherwise uninstrumented score-mod cache."""
    del config
    if os.getenv("TE_PROBE_MAIN_SCORE_MOD") != "1":
        return
    from transformer_engine.jax.cpp_extensions import flex_attention as flex

    class Cache(dict):
        """Preserve dictionary behavior while recording actual lookups and inserts."""

        def __contains__(self, key):
            present = super().__contains__(key)
            _score_mod["HIT" if present else "MISS"] += 1
            return present

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            _score_mod["CACHE_GRAPH"] += 1

    assert not flex._score_mod_graph_cache
    flex._score_mod_graph_cache = Cache()

    def count_build(function):
        @functools.wraps(function)
        def build(*args, **kwargs):
            _score_mod["CREATE_GRAPH"] += 1
            graph = function(*args, **kwargs)
            _score_mod["BUILD_PLANS"] += 1
            return graph

        return build

    flex._build_score_mod_fwd_graph = count_build(flex._build_score_mod_fwd_graph)
    flex._build_score_mod_bwd_graph = count_build(flex._build_score_mod_bwd_graph)
