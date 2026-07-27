# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared helpers for attention implemented with the cuDNN frontend Python API."""

import importlib
from typing import Any

import torch

_cudnn_handles: dict[torch.device, Any] = {}


def _import_cudnn_frontend():
    """Import the cuDNN frontend Python package."""
    try:
        return importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            "cuDNN frontend Python package not found. "
            "Install it with: pip install nvidia-cudnn-frontend"
        ) from exc


def _bhsd_dim_stride(
    tensor: torch.Tensor, tensor_format: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Describe an SBHD/BSHD tensor as cuDNN frontend's logical BHSD format."""
    if tensor_format == "sbhd":
        return (
            (tensor.shape[1], tensor.shape[2], tensor.shape[0], tensor.shape[3]),
            (tensor.stride(1), tensor.stride(2), tensor.stride(0), tensor.stride(3)),
        )
    if tensor_format == "bshd":
        return (
            (tensor.shape[0], tensor.shape[2], tensor.shape[1], tensor.shape[3]),
            (tensor.stride(0), tensor.stride(2), tensor.stride(1), tensor.stride(3)),
        )
    raise ValueError(
        f"cuDNN frontend Python attention only supports SBHD/BSHD tensor formats, "
        f"got {tensor_format}."
    )


def _bhsd_graph_tensor(graph, tensor: torch.Tensor, tensor_format: str):
    """Create a cuDNN graph tensor with BHSD dims and TE-layout strides."""
    dim, stride = _bhsd_dim_stride(tensor, tensor_format)
    return graph.tensor(dim=dim, stride=stride, data_type=tensor.dtype)


def _get_cudnn_current_stream_handle(cudnn, device: torch.device):
    """Return a cuDNN handle for device, bound to PyTorch's current stream."""
    if device.type != "cuda":
        raise ValueError(
            f"cuDNN frontend Python attention only supports CUDA tensors, got device {device}."
        )
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    handle = _cudnn_handles.get(device)
    with torch.cuda.device(device):
        if handle is None:
            handle = cudnn.create_handle()
            _cudnn_handles[device] = handle

        stream = torch.cuda.current_stream(device).cuda_stream
        cudnn.set_stream(handle=handle, stream=stream)
    return handle


def _build_cudnn_pygraph(dtype: torch.dtype, device: torch.device):
    """Create a cuDNN frontend Python graph for F16/BF16 SDPA."""
    cudnn = _import_cudnn_frontend()

    if dtype == torch.float16:
        io_data_type = cudnn.data_type.HALF
    elif dtype == torch.bfloat16:
        io_data_type = cudnn.data_type.BFLOAT16
    else:
        raise ValueError(
            f"cuDNN frontend Python attention only supports FP16/BF16 tensors, got {dtype}."
        )

    return cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_current_stream_handle(cudnn, device),
    )


def _finalize_cudnn_graph(graph) -> int:
    """Build a cuDNN frontend Python graph and return its workspace size."""
    cudnn = _import_cudnn_frontend()

    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as exc:
        raise RuntimeError(
            f"cuDNN frontend Python attention graph is not supported: {exc}"
        ) from exc
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    return max(graph.get_workspace_size(), 1)


def _execute_cudnn_graph(
    graph,
    variant_pack: dict[Any, torch.Tensor],
    workspace_size: int,
    device: torch.device,
):
    """Execute a built cuDNN frontend Python graph."""
    cudnn = _import_cudnn_frontend()

    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    workspace = torch.empty(
        workspace_size,
        device=device,
        dtype=torch.uint8,
    )
    graph.execute(
        variant_pack,
        workspace,
        handle=_get_cudnn_current_stream_handle(cudnn, device),
    )
