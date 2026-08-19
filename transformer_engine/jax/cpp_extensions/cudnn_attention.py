# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Ordinary fused attention built with the cuDNN frontend Python API."""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

from .cudnn_frontend import (
    SerializedCudnnGraph,
    bshd_as_bhsd_dim_stride,
    cudnn_data_type,
    dtype_name,
    finalize_cudnn_graph,
    import_cudnn_frontend,
    make_serialized_graph,
    row_major_stride,
    shape_dtype,
)

__all__ = [
    "CudnnAttentionConfig",
    "fused_attn_cudnn_fwd",
    "fused_attn_cudnn_bwd",
    "make_cudnn_attention_config",
    "reset_cudnn_attention_graph_cache",
    "validate_cudnn_attention",
]


@dataclass(frozen=True)
class CudnnAttentionConfig:
    """Static configuration for an ordinary Python-frontend SDPA graph."""

    scaling_factor: float
    is_training: bool
    deterministic: bool
    causal: bool


_UID_Q = 1
_UID_K = 2
_UID_V = 3
_UID_O = 4
_UID_STATS = 5
_UID_DO = 6
_UID_DQ = 7
_UID_DK = 8
_UID_DV = 9
_UID_ATTN_SCALE = 10

_cudnn_attention_graph_cache: Dict[Tuple[Any, ...], SerializedCudnnGraph] = {}


def _enum_name(value: Any) -> str:
    """Return the stable enum member name used by JAX attention configuration."""
    return getattr(value, "name", str(value))


def validate_cudnn_attention(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    sequence_descriptor: Optional[Any],
    seed: Optional[jnp.ndarray],
    attn_bias_type: Any,
    attn_mask_type: Any,
    qkv_layout: Any,
    softmax_type: Any,
    dropout_probability: float,
    max_segments_per_seq: int,
    window_size: Optional[Tuple[int, int]],
    context_parallel_strategy: Any,
    context_parallel_causal_load_balanced: bool,
    context_parallel_axis: str,
    softmax_offset: Optional[jnp.ndarray],
    stripe_size: int | None,
) -> None:
    """Validate the intentionally narrow first release of ordinary Python SDPA."""
    header = "fused_attention_impl='python'"
    unsupported = []

    if _enum_name(qkv_layout) != "BSHD_BSHD_BSHD":
        unsupported.append("QKV layout is not BSHD_BSHD_BSHD")
    if len(qkv) != 3:
        unsupported.append("query, key, and value are not separate tensors")
    elif any(tensor.ndim != 4 for tensor in qkv):
        unsupported.append("query, key, and value are not rank-4 BSHD tensors")
    else:
        q, k, v = qkv
        if q.dtype != k.dtype or q.dtype != v.dtype:
            unsupported.append("query, key, and value do not have the same dtype")
        if q.dtype not in (jnp.float16, jnp.bfloat16):
            unsupported.append(f"QKV dtype is {q.dtype}, expected FP16 or BF16")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            unsupported.append("query, key, and value batch dimensions do not match")
        if k.shape[1] != v.shape[1]:
            unsupported.append("key and value sequence lengths do not match")
        if k.shape[2] != v.shape[2]:
            unsupported.append("key and value head counts do not match")
        if k.shape[2] == 0 or q.shape[2] < k.shape[2] or q.shape[2] % k.shape[2] != 0:
            unsupported.append("query head count is not a multiple of the key/value head count")
        if q.shape[3] != k.shape[3]:
            unsupported.append("query and key head dimensions do not match")

    if bias is not None or _enum_name(attn_bias_type) != "NO_BIAS":
        unsupported.append("attention bias is enabled")
    if sequence_descriptor is not None:
        unsupported.append("a sequence descriptor or explicit mask was provided")
    if seed is not None:
        unsupported.append("a dropout seed was provided")
    if _enum_name(attn_mask_type) not in ("NO_MASK", "CAUSAL_MASK"):
        unsupported.append(f"attention mask is {_enum_name(attn_mask_type)}")
    if _enum_name(softmax_type) != "VANILLA_SOFTMAX" or softmax_offset is not None:
        unsupported.append("softmax is not vanilla softmax without an offset")
    if dropout_probability != 0.0:
        unsupported.append(f"attention dropout is {dropout_probability}, expected 0.0")
    if max_segments_per_seq != 1:
        unsupported.append("packed/ragged sequence metadata is enabled")
    if window_size not in (None, (-1, -1)):
        unsupported.append(f"sliding-window attention is enabled with {window_size=}")
    if _enum_name(context_parallel_strategy) != "DEFAULT":
        unsupported.append("context parallelism is enabled")
    if context_parallel_causal_load_balanced or context_parallel_axis:
        unsupported.append("context parallelism is enabled")
    if stripe_size is not None:
        unsupported.append("striped context parallelism is enabled")

    if unsupported:
        raise ValueError(
            f"{header} does not support this configuration: " + "; ".join(unsupported) + "."
        )


def make_cudnn_attention_config(
    scaling_factor: float,
    is_training: bool,
    attn_mask_type: Any,
) -> CudnnAttentionConfig:
    """Create the static config included in graph-cache and JAX trace keys."""
    return CudnnAttentionConfig(
        scaling_factor=float(scaling_factor),
        is_training=bool(is_training),
        deterministic=not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))),
        causal=_enum_name(attn_mask_type) == "CAUSAL_MASK",
    )


def reset_cudnn_attention_graph_cache() -> None:
    """Clear serialized ordinary-attention graphs, primarily for tests and profiling."""
    _cudnn_attention_graph_cache.clear()


def _graph_cache_key(
    direction: str,
    config: CudnnAttentionConfig,
    avals: Sequence[Any],
) -> Tuple[Any, ...]:
    """Create a static cache key for a Python-built cuDNN graph."""
    return (
        direction,
        config,
        tuple((tuple(aval.shape), dtype_name(aval.dtype)) for aval in avals),
    )


def _attn_scale_tensor(cudnn, graph, scaling_factor: float):
    """Create the pass-by-value FP32 attention scale and its packed host value."""
    tensor = graph.tensor(
        name="attn_scale",
        dim=(1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
        uid=_UID_ATTN_SCALE,
    )
    value = np.full((1, 1, 1, 1), scaling_factor, dtype=np.float32).tobytes()
    return tensor, value


def _mask_kwargs(cudnn, config: CudnnAttentionConfig) -> Dict[str, Any]:
    """Map the supported JAX mask types to preferred cuDNN diagonal-band arguments."""
    if not config.causal:
        return {}
    return {
        "diagonal_alignment": cudnn.diagonal_alignment.TOP_LEFT,
        "diagonal_band_right_bound": 0,
    }


def _build_fwd_graph(q_aval, k_aval, v_aval, config: CudnnAttentionConfig):
    """Build and serialize ordinary SDPA forward."""
    cudnn = import_cudnn_frontend()
    io_data_type = cudnn_data_type(cudnn, q_aval.dtype)
    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    def tensor(name, aval, uid):
        dim, stride = bshd_as_bhsd_dim_stride(aval.shape)
        return graph.tensor(
            name=name,
            dim=dim,
            stride=stride,
            data_type=io_data_type,
            uid=uid,
        )

    q = tensor("q", q_aval, _UID_Q)
    k = tensor("k", k_aval, _UID_K)
    v = tensor("v", v_aval, _UID_V)
    attn_scale, attn_scale_value = _attn_scale_tensor(cudnn, graph, config.scaling_factor)

    output, stats = graph.sdpa(
        name="te_jax_python_frontend_sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=config.is_training,
        attn_scale=attn_scale,
        **_mask_kwargs(cudnn, config),
    )

    batch, q_seqlen, q_heads, _ = q_aval.shape
    v_head_dim = v_aval.shape[-1]
    output_dim, output_stride = bshd_as_bhsd_dim_stride((batch, q_seqlen, q_heads, v_head_dim))
    output.set_output(True).set_uid(_UID_O).set_dim(output_dim).set_stride(output_stride)
    output.set_data_type(io_data_type)

    output_uids = [_UID_O]
    if config.is_training:
        stats_shape = (batch, q_heads, q_seqlen, 1)
        stats.set_output(True).set_uid(_UID_STATS).set_dim(stats_shape).set_stride(
            row_major_stride(stats_shape)
        )
        stats.set_data_type(cudnn.data_type.FLOAT)
        output_uids.append(_UID_STATS)

    workspace_size, serialized_graph, frontend_version = finalize_cudnn_graph(
        cudnn, graph, operation="ordinary SDPA forward"
    )
    return make_serialized_graph(
        serialized_graph=serialized_graph,
        cudnn_frontend_version=frontend_version,
        workspace_size=workspace_size,
        input_uids=[_UID_Q, _UID_K, _UID_V],
        output_uids=output_uids,
        scalar_uids=[_UID_ATTN_SCALE],
        scalar_values=[attn_scale_value],
    )


def _build_bwd_graph(
    q_aval,
    k_aval,
    v_aval,
    output_aval,
    doutput_aval,
    stats_aval,
    config: CudnnAttentionConfig,
):
    """Build and serialize ordinary SDPA backward."""
    cudnn = import_cudnn_frontend()
    io_data_type = cudnn_data_type(cudnn, q_aval.dtype)
    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    def tensor(name, aval, uid):
        dim, stride = bshd_as_bhsd_dim_stride(aval.shape)
        return graph.tensor(
            name=name,
            dim=dim,
            stride=stride,
            data_type=io_data_type,
            uid=uid,
        )

    q = tensor("q", q_aval, _UID_Q)
    k = tensor("k", k_aval, _UID_K)
    v = tensor("v", v_aval, _UID_V)
    output = tensor("o", output_aval, _UID_O)
    doutput = tensor("dO", doutput_aval, _UID_DO)
    stats = graph.tensor(
        name="stats",
        dim=tuple(int(dim) for dim in stats_aval.shape),
        stride=row_major_stride(stats_aval.shape),
        data_type=cudnn.data_type.FLOAT,
        uid=_UID_STATS,
    )
    attn_scale, attn_scale_value = _attn_scale_tensor(cudnn, graph, config.scaling_factor)

    dq, dk, dv = graph.sdpa_backward(
        name="te_jax_python_frontend_sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=output,
        dO=doutput,
        stats=stats,
        attn_scale=attn_scale,
        use_deterministic_algorithm=config.deterministic,
        **_mask_kwargs(cudnn, config),
    )
    for grad, aval, uid in (
        (dq, q_aval, _UID_DQ),
        (dk, k_aval, _UID_DK),
        (dv, v_aval, _UID_DV),
    ):
        dim, stride = bshd_as_bhsd_dim_stride(aval.shape)
        grad.set_output(True).set_uid(uid).set_dim(dim).set_stride(stride)
        grad.set_data_type(io_data_type)

    workspace_size, serialized_graph, frontend_version = finalize_cudnn_graph(
        cudnn, graph, operation="ordinary SDPA backward"
    )
    return make_serialized_graph(
        serialized_graph=serialized_graph,
        cudnn_frontend_version=frontend_version,
        workspace_size=workspace_size,
        input_uids=[_UID_Q, _UID_K, _UID_V, _UID_O, _UID_DO, _UID_STATS],
        output_uids=[_UID_DQ, _UID_DK, _UID_DV],
        scalar_uids=[_UID_ATTN_SCALE],
        scalar_values=[attn_scale_value],
    )


def _ffi_attrs(graph: SerializedCudnnGraph) -> Dict[str, Any]:
    """Return static attributes shared by forward and backward FFI calls."""
    return {
        "serialized_graph": graph.serialized_graph,
        "graph_hash0": graph.graph_hash[0],
        "graph_hash1": graph.graph_hash[1],
        "cudnn_frontend_version": graph.cudnn_frontend_version,
        "input_uids": graph.input_uids,
        "output_uids": graph.output_uids,
        "scalar_uids": graph.scalar_uids,
        "scalar_sizes": graph.scalar_sizes,
        "scalar_values": graph.scalar_values,
    }


def fused_attn_cudnn_fwd(
    qkv: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: CudnnAttentionConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Execute an ordinary Python-built SDPA forward graph through JAX FFI."""
    q, k, v = qkv
    q_aval, k_aval, v_aval = map(shape_dtype, (q, k, v))
    key = _graph_cache_key("fwd", config, (q_aval, k_aval, v_aval))
    graph = _cudnn_attention_graph_cache.get(key)
    if graph is None:
        graph = _build_fwd_graph(q_aval, k_aval, v_aval, config)
        _cudnn_attention_graph_cache[key] = graph

    batch, q_seqlen, q_heads, _ = q.shape
    output = jax.ShapeDtypeStruct((batch, q_seqlen, q_heads, v.shape[-1]), q.dtype)
    stats_shape = (batch, q_heads, q_seqlen, 1) if config.is_training else (0,)
    stats = jax.ShapeDtypeStruct(stats_shape, jnp.float32)
    workspace = jax.ShapeDtypeStruct((graph.workspace_size,), jnp.uint8)
    output, softmax_stats, _ = ffi.ffi_call(
        "te_cudnn_frontend_attn_forward_ffi",
        (output, stats, workspace),
    )(q, k, v, **_ffi_attrs(graph))
    return output, softmax_stats


def fused_attn_cudnn_bwd(
    qkv: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    softmax_stats: jnp.ndarray,
    config: CudnnAttentionConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Execute an ordinary Python-built SDPA backward graph through JAX FFI."""
    if not config.is_training:
        raise RuntimeError("Python-frontend attention backward requires is_training=True.")

    q, k, v = qkv
    avals = tuple(shape_dtype(arg) for arg in (q, k, v, output, doutput, softmax_stats))
    key = _graph_cache_key("bwd", config, avals)
    graph = _cudnn_attention_graph_cache.get(key)
    if graph is None:
        graph = _build_bwd_graph(*avals, config)
        _cudnn_attention_graph_cache[key] = graph

    dq = jax.ShapeDtypeStruct(q.shape, q.dtype)
    dk = jax.ShapeDtypeStruct(k.shape, k.dtype)
    dv = jax.ShapeDtypeStruct(v.shape, v.dtype)
    workspace = jax.ShapeDtypeStruct((graph.workspace_size,), jnp.uint8)
    dq, dk, dv, _ = ffi.ffi_call(
        "te_cudnn_frontend_attn_backward_ffi",
        (dq, dk, dv, workspace),
    )(q, k, v, output, doutput, softmax_stats, **_ffi_attrs(graph))
    return dq, dk, dv
