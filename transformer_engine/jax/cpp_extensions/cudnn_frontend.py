# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared helpers for JAX operations built with the cuDNN frontend Python API."""

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import transformer_engine_jax


@dataclass(frozen=True)
class SerializedCudnnGraph:
    """Serialized cuDNN graph and static metadata for JAX FFI execution."""

    serialized_graph: bytes
    graph_hash: Tuple[int, int]
    cudnn_frontend_version: int
    workspace_size: int
    input_uids: np.ndarray
    output_uids: np.ndarray
    scalar_uids: np.ndarray
    scalar_sizes: np.ndarray
    scalar_values: np.ndarray


def row_major_stride(shape: Sequence[int]) -> Tuple[int, ...]:
    """Return contiguous row-major strides for a shape."""
    stride = []
    running = 1
    for dim in reversed(tuple(shape)):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def bshd_as_bhsd_dim_stride(shape: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Describe a contiguous BSHD tensor as cuDNN frontend's logical BHSD format."""
    if len(shape) != 4:
        raise ValueError(f"cuDNN frontend attention requires rank-4 BSHD tensors, got {shape=}.")
    batch, seqlen, heads, head_dim = tuple(int(dim) for dim in shape)
    return (
        (batch, heads, seqlen, head_dim),
        (seqlen * heads * head_dim, head_dim, heads * head_dim, 1),
    )


def dtype_name(dtype: Any) -> str:
    """Return a stable dtype name for graph-cache keys."""
    return str(jnp.dtype(dtype))


def cudnn_data_type(cudnn, dtype):
    """Convert a JAX/NumPy dtype to a cuDNN frontend dtype."""
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float16:
        return cudnn.data_type.HALF
    if dtype == jnp.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == jnp.float32:
        return cudnn.data_type.FLOAT
    if dtype == jnp.float64:
        return cudnn.data_type.DOUBLE
    if dtype == jnp.int32:
        return cudnn.data_type.INT32
    if dtype == jnp.int64:
        return cudnn.data_type.INT64
    if dtype == jnp.uint8:
        return cudnn.data_type.UINT8
    if dtype == jnp.bool_:
        return cudnn.data_type.BOOLEAN
    raise ValueError(f"Unsupported cuDNN frontend tensor dtype: {dtype}.")


def encode_cudnn_frontend_version(version: str) -> int:
    """Encode a semantic cuDNN frontend version as its C++ integer form."""
    public_version = version.split("+", 1)[0].split("-", 1)[0]
    parts = public_version.split(".")
    if len(parts) < 3:
        raise RuntimeError(f"Could not parse cuDNN frontend Python version: {version!r}.")
    major, minor, patch = (int(part) for part in parts[:3])
    return major * 10000 + minor * 100 + patch


def check_cudnn_frontend_version_match(cudnn) -> int:
    """Require serialized Python graphs and the C++ executor to use matching headers."""
    python_version_string = getattr(cudnn, "__version__", None)
    if python_version_string is None:
        raise RuntimeError("cuDNN frontend Python package does not expose __version__.")
    python_version = encode_cudnn_frontend_version(python_version_string)
    cpp_version = int(transformer_engine_jax.get_cudnn_frontend_version())
    if python_version != cpp_version:
        raise RuntimeError(
            "cuDNN frontend Python/C++ version mismatch for graph serialization: "
            f"Python cudnn.__version__={python_version_string!r} encodes to {python_version}, "
            f"but Transformer Engine C++ was built with CUDNN_FRONTEND_VERSION={cpp_version}. "
            "Use matching cuDNN frontend Python package and C++ headers."
        )
    return python_version


def import_cudnn_frontend():
    """Import a cuDNN frontend Python package compatible with the C++ extension."""
    try:
        cudnn = importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            "cuDNN frontend Python package not found. "
            "Install it with: pip install nvidia-cudnn-frontend"
        ) from exc
    check_cudnn_frontend_version_match(cudnn)
    return cudnn


def graph_hash(serialized_graph: bytes) -> Tuple[int, int]:
    """Return a stable 128-bit cache identifier for a serialized graph."""
    digest = hashlib.sha256(serialized_graph).digest()
    return (
        int.from_bytes(digest[0:8], byteorder="little", signed=True),
        int.from_bytes(digest[8:16], byteorder="little", signed=True),
    )


def pack_scalar_values(scalar_values: Sequence[bytes]) -> Tuple[np.ndarray, np.ndarray]:
    """Pack pass-by-value scalar bytes into fixed-width FFI attributes."""
    scalar_sizes = np.asarray([len(value) for value in scalar_values], dtype=np.int64)
    packed_values = np.zeros((len(scalar_values), 16), dtype=np.uint8)
    for index, value in enumerate(scalar_values):
        if len(value) > 16:
            raise ValueError("cuDNN pass-by-value scalars must be at most 16 bytes.")
        packed_values[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    return scalar_sizes, packed_values.reshape(-1)


def make_serialized_graph(
    *,
    serialized_graph: bytes,
    cudnn_frontend_version: int,
    workspace_size: int,
    input_uids: Sequence[int],
    output_uids: Sequence[int],
    scalar_uids: Sequence[int] = (),
    scalar_values: Sequence[bytes] = (),
) -> SerializedCudnnGraph:
    """Construct the immutable metadata passed to the generic C++ FFI executor."""
    scalar_sizes, packed_scalar_values = pack_scalar_values(scalar_values)
    return SerializedCudnnGraph(
        serialized_graph=serialized_graph,
        graph_hash=graph_hash(serialized_graph),
        cudnn_frontend_version=int(cudnn_frontend_version),
        workspace_size=max(int(workspace_size), 1),
        input_uids=np.asarray(input_uids, dtype=np.int64),
        output_uids=np.asarray(output_uids, dtype=np.int64),
        scalar_uids=np.asarray(scalar_uids, dtype=np.int64),
        scalar_sizes=scalar_sizes,
        scalar_values=packed_scalar_values,
    )


def finalize_cudnn_graph(cudnn, graph, *, operation: str) -> Tuple[int, bytes, int]:
    """Validate, plan, and serialize a cuDNN frontend Python graph."""
    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as exc:
        raise RuntimeError(f"cuDNN frontend {operation} graph is not supported: {exc}") from exc
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    return (
        max(int(graph.get_workspace_size()), 1),
        bytes(graph.serialize()),
        check_cudnn_frontend_version_match(cudnn),
    )


def shape_dtype(value) -> jax.ShapeDtypeStruct:
    """Return the static shape/dtype portion of a JAX value or tracer."""
    return jax.ShapeDtypeStruct(tuple(value.shape), value.dtype)
