# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Ordinary fused attention implemented with the cuDNN frontend Python API."""

from dataclasses import dataclass
from typing import Any

import torch

from transformer_engine.pytorch.attention.dot_product_attention.cudnn_frontend import (
    _bhsd_dim_stride,
    _bhsd_graph_tensor,
    _build_cudnn_pygraph,
    _execute_cudnn_graph,
    _finalize_cudnn_graph,
    _import_cudnn_frontend,
)

_cudnn_attention_graph_cache: dict[tuple[Any, ...], Any] = {}


def _device_key(device: torch.device) -> tuple[Any, ...]:
    """Normalize a tensor device for graph cache keys."""
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return (device.type, index)
    return (device.type, device.index)


def _tensor_metadata(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Describe tensor metadata that can affect cuDNN graph construction."""
    return (
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.dtype,
        _device_key(tensor.device),
    )


def _bhsd_tensor_metadata(tensor: torch.Tensor, tensor_format: str) -> tuple[Any, ...]:
    """Describe an SBHD/BSHD runtime tensor as a cuDNN BHSD graph tensor."""
    dim, stride = _bhsd_dim_stride(tensor, tensor_format)
    return (dim, stride, tensor.dtype, _device_key(tensor.device))


def _sdpa_mask_kwargs(attn_mask_type: str) -> dict[str, Any]:
    """Convert a supported TE mask type to cuDNN frontend SDPA arguments."""
    if attn_mask_type == "no_mask":
        return {}
    if attn_mask_type == "causal":
        cudnn = _import_cudnn_frontend()
        return {
            "diagonal_alignment": cudnn.diagonal_alignment.TOP_LEFT,
            "diagonal_band_right_bound": 0,
        }
    raise ValueError(
        "cuDNN frontend Python attention initially supports only attn_mask_type "
        f"'no_mask' and 'causal', got {attn_mask_type!r}."
    )


@dataclass
class _CudnnAttentionFwdGraphEntry:
    """Cached cuDNN frontend graph and tensor handles for ordinary SDPA fprop."""

    graph: Any
    q: Any
    k: Any
    v: Any
    output: Any
    stats: Any | None
    workspace_size: int


@dataclass
class _CudnnAttentionBwdGraphEntry:
    """Cached cuDNN frontend graph and tensor handles for ordinary SDPA bprop."""

    graph: Any
    q: Any
    k: Any
    v: Any
    output: Any
    d_output: Any
    stats: Any
    dq: Any
    dk: Any
    dv: Any
    workspace_size: int


def _cudnn_attention_fwd_cache_key(
    is_training: bool,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    attn_mask_type: str,
    output_layer: torch.Tensor,
    stats: torch.Tensor | None,
) -> tuple[Any, ...]:
    """Create a pre-build cache key for ordinary SDPA fprop execution plans."""
    return (
        "ordinary_fwd",
        is_training,
        q_format,
        kv_format,
        attn_scale,
        attn_mask_type,
        _bhsd_tensor_metadata(query_layer, q_format),
        _bhsd_tensor_metadata(key_layer, kv_format),
        _bhsd_tensor_metadata(value_layer, kv_format),
        _bhsd_tensor_metadata(output_layer, q_format),
        _tensor_metadata(stats) if stats is not None else None,
    )


def _cudnn_attention_bwd_cache_key(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    output_layer: torch.Tensor,
    d_out: torch.Tensor,
    stats: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    attn_mask_type: str,
    deterministic: bool,
) -> tuple[Any, ...]:
    """Create a pre-build cache key for ordinary SDPA bprop execution plans."""
    return (
        "ordinary_bwd",
        q_format,
        kv_format,
        attn_scale,
        attn_mask_type,
        deterministic,
        _bhsd_tensor_metadata(query_layer, q_format),
        _bhsd_tensor_metadata(key_layer, kv_format),
        _bhsd_tensor_metadata(value_layer, kv_format),
        _bhsd_tensor_metadata(output_layer, q_format),
        _bhsd_tensor_metadata(d_out, q_format),
        _tensor_metadata(stats),
    )


def _build_cudnn_attention_fwd_graph(
    is_training: bool,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    attn_mask_type: str,
    output_layer: torch.Tensor,
    stats: torch.Tensor | None,
) -> _CudnnAttentionFwdGraphEntry:
    """Build a cuDNN frontend Python graph for ordinary SDPA fprop."""
    cudnn = _import_cudnn_frontend()
    graph = _build_cudnn_pygraph(query_layer.dtype, query_layer.device)
    q = _bhsd_graph_tensor(graph, query_layer, q_format)
    k = _bhsd_graph_tensor(graph, key_layer, kv_format)
    v = _bhsd_graph_tensor(graph, value_layer, kv_format)

    output_dim, output_stride = _bhsd_dim_stride(output_layer, q_format)
    output, stats_tensor = graph.sdpa(
        name="te_python_frontend_sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=is_training,
        attn_scale=attn_scale,
        **_sdpa_mask_kwargs(attn_mask_type),
    )
    output.set_output(True).set_dim(output_dim).set_stride(output_stride)

    if is_training:
        assert stats is not None
        stats_tensor.set_output(True).set_dim(stats.size()).set_stride(
            stats.stride()
        ).set_data_type(cudnn.data_type.FLOAT)
    else:
        stats_tensor = None

    workspace_size = _finalize_cudnn_graph(graph)
    return _CudnnAttentionFwdGraphEntry(
        graph=graph,
        q=q,
        k=k,
        v=v,
        output=output,
        stats=stats_tensor,
        workspace_size=workspace_size,
    )


def _get_cudnn_attention_fwd_graph(
    is_training: bool,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    attn_mask_type: str,
    output_layer: torch.Tensor,
    stats: torch.Tensor | None,
) -> _CudnnAttentionFwdGraphEntry:
    """Return a cached cuDNN frontend Python graph for ordinary SDPA fprop."""
    build_args = (
        is_training,
        query_layer,
        key_layer,
        value_layer,
        q_format,
        kv_format,
        attn_scale,
        attn_mask_type,
        output_layer,
        stats,
    )
    key = _cudnn_attention_fwd_cache_key(*build_args)
    entry = _cudnn_attention_graph_cache.get(key)
    if entry is None:
        entry = _build_cudnn_attention_fwd_graph(*build_args)
        _cudnn_attention_graph_cache[key] = entry
    return entry


def _build_cudnn_attention_bwd_graph(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    output_layer: torch.Tensor,
    d_out: torch.Tensor,
    stats: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    attn_mask_type: str,
    deterministic: bool,
) -> _CudnnAttentionBwdGraphEntry:
    """Build a cuDNN frontend Python graph for ordinary SDPA bprop."""
    graph = _build_cudnn_pygraph(query_layer.dtype, query_layer.device)
    q = _bhsd_graph_tensor(graph, query_layer, q_format)
    k = _bhsd_graph_tensor(graph, key_layer, kv_format)
    v = _bhsd_graph_tensor(graph, value_layer, kv_format)
    output = _bhsd_graph_tensor(graph, output_layer, q_format)
    d_output = _bhsd_graph_tensor(graph, d_out, q_format)
    stats_tensor = graph.tensor_like(stats)

    dq_layer = torch.empty_like(query_layer)
    dk_layer = torch.empty_like(key_layer)
    dv_layer = torch.empty_like(value_layer)
    dq_dim, dq_stride = _bhsd_dim_stride(dq_layer, q_format)
    dk_dim, dk_stride = _bhsd_dim_stride(dk_layer, kv_format)
    dv_dim, dv_stride = _bhsd_dim_stride(dv_layer, kv_format)
    dq, dk, dv = graph.sdpa_backward(
        name="te_python_frontend_sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=output,
        dO=d_output,
        stats=stats_tensor,
        attn_scale=attn_scale,
        use_deterministic_algorithm=deterministic,
        **_sdpa_mask_kwargs(attn_mask_type),
    )
    dq.set_output(True).set_dim(dq_dim).set_stride(dq_stride)
    dk.set_output(True).set_dim(dk_dim).set_stride(dk_stride)
    dv.set_output(True).set_dim(dv_dim).set_stride(dv_stride)

    workspace_size = _finalize_cudnn_graph(graph)
    return _CudnnAttentionBwdGraphEntry(
        graph=graph,
        q=q,
        k=k,
        v=v,
        output=output,
        d_output=d_output,
        stats=stats_tensor,
        dq=dq,
        dk=dk,
        dv=dv,
        workspace_size=workspace_size,
    )


def _get_cudnn_attention_bwd_graph(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    output_layer: torch.Tensor,
    d_out: torch.Tensor,
    stats: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    attn_mask_type: str,
    deterministic: bool,
) -> _CudnnAttentionBwdGraphEntry:
    """Return a cached cuDNN frontend Python graph for ordinary SDPA bprop."""
    build_args = (
        query_layer,
        key_layer,
        value_layer,
        output_layer,
        d_out,
        stats,
        q_format,
        kv_format,
        attn_scale,
        attn_mask_type,
        deterministic,
    )
    key = _cudnn_attention_bwd_cache_key(*build_args)
    entry = _cudnn_attention_graph_cache.get(key)
    if entry is None:
        entry = _build_cudnn_attention_bwd_graph(*build_args)
        _cudnn_attention_graph_cache[key] = entry
    return entry


class FusedAttentionWithPythonFrontendFunc(torch.autograd.Function):
    """Ordinary fused SDPA using the cuDNN frontend Python API."""

    @staticmethod
    def forward(
        ctx,
        is_training: bool,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        q_format: str,
        kv_format: str,
        attn_scale: float,
        attn_mask_type: str,
        deterministic: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        q_bhsd_dim, _ = _bhsd_dim_stride(query_layer, q_format)
        output_shape = (*query_layer.shape[:-1], value_layer.shape[-1])
        output_layer = torch.empty(
            output_shape, device=query_layer.device, dtype=query_layer.dtype
        )
        if is_training:
            stats = torch.empty(
                (*q_bhsd_dim[:-1], 1),
                device=query_layer.device,
                dtype=torch.float32,
            )
        else:
            stats = None

        entry = _get_cudnn_attention_fwd_graph(
            is_training,
            query_layer,
            key_layer,
            value_layer,
            q_format,
            kv_format,
            attn_scale,
            attn_mask_type,
            output_layer,
            stats,
        )
        variant_pack = {
            entry.q: query_layer,
            entry.k: key_layer,
            entry.v: value_layer,
            entry.output: output_layer,
        }
        if is_training:
            variant_pack[entry.stats] = stats

        _execute_cudnn_graph(
            entry.graph,
            variant_pack,
            entry.workspace_size,
            query_layer.device,
        )

        ctx.is_training = is_training
        ctx.q_format = q_format
        ctx.kv_format = kv_format
        ctx.attn_scale = attn_scale
        ctx.attn_mask_type = attn_mask_type
        ctx.deterministic = deterministic
        if is_training:
            ctx.save_for_backward(
                query_layer,
                key_layer,
                value_layer,
                output_layer,
                stats,
            )
        else:
            ctx.save_for_backward(query_layer, key_layer, value_layer, output_layer)

        return output_layer

    @staticmethod
    def backward(ctx, d_out: torch.Tensor):
        # pylint: disable=missing-function-docstring
        if not ctx.is_training:
            raise RuntimeError(
                "cuDNN frontend Python attention backward requires "
                "DotProductAttention to be in training mode."
            )

        query_layer, key_layer, value_layer, output_layer, stats = ctx.saved_tensors
        d_out = d_out.contiguous()
        dq_layer = torch.empty_like(query_layer)
        dk_layer = torch.empty_like(key_layer)
        dv_layer = torch.empty_like(value_layer)

        entry = _get_cudnn_attention_bwd_graph(
            query_layer,
            key_layer,
            value_layer,
            output_layer,
            d_out,
            stats,
            ctx.q_format,
            ctx.kv_format,
            ctx.attn_scale,
            ctx.attn_mask_type,
            ctx.deterministic,
        )
        variant_pack = {
            entry.q: query_layer,
            entry.k: key_layer,
            entry.v: value_layer,
            entry.output: output_layer,
            entry.d_output: d_out,
            entry.stats: stats,
            entry.dq: dq_layer,
            entry.dk: dk_layer,
            entry.dv: dv_layer,
        }

        _execute_cudnn_graph(
            entry.graph,
            variant_pack,
            entry.workspace_size,
            query_layer.device,
        )

        return (
            None,
            dq_layer,
            dk_layer,
            dv_layer,
            None,
            None,
            None,
            None,
            None,
        )
