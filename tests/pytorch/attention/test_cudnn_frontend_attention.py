# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for ordinary attention through the cuDNN frontend Python API."""

import pytest
import torch

from transformer_engine.pytorch import DotProductAttention, is_bf16_available
from transformer_engine.pytorch.attention.dot_product_attention import cudnn_attention
from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    FusedAttention,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend
from transformer_engine.pytorch.utils import get_cudnn_version
import transformer_engine_torch as tex


def _cpu_inputs():
    q = torch.empty((2, 4, 3, 8), dtype=torch.float16)
    k = torch.empty((2, 4, 3, 8), dtype=torch.float16)
    v = torch.empty((2, 4, 3, 8), dtype=torch.float16)
    o = torch.empty((2, 4, 3, 8), dtype=torch.float16)
    stats = torch.empty((2, 3, 4, 1), dtype=torch.float32)
    return q, k, v, o, stats


def test_fused_attention_impl_validation():
    """The implementation selector should be strict and default to C++."""
    assert FusedAttention(1.0).fused_attention_impl == "cpp"
    assert (
        FusedAttention(1.0, fused_attention_impl="python").fused_attention_impl
        == "python"
    )
    with pytest.raises(ValueError, match="fused_attention_impl"):
        FusedAttention(1.0, fused_attention_impl="auto")


def test_cudnn_attention_cache_keys_include_mask_and_determinism():
    """Graph topology options should separate ordinary attention cache entries."""
    q, k, v, o, stats = _cpu_inputs()
    fwd_no_mask = cudnn_attention._cudnn_attention_fwd_cache_key(
        True, q, k, v, "bshd", "bshd", 0.5, "no_mask", o, stats
    )
    fwd_causal = cudnn_attention._cudnn_attention_fwd_cache_key(
        True, q, k, v, "bshd", "bshd", 0.5, "causal", o, stats
    )
    assert fwd_no_mask != fwd_causal

    bwd_nondeterministic = cudnn_attention._cudnn_attention_bwd_cache_key(
        q, k, v, o, o, stats, "bshd", "bshd", 0.5, "no_mask", False
    )
    bwd_deterministic = cudnn_attention._cudnn_attention_bwd_cache_key(
        q, k, v, o, o, stats, "bshd", "bshd", 0.5, "no_mask", True
    )
    assert bwd_nondeterministic != bwd_deterministic


def test_python_frontend_graph_build_nvtx_ranges(monkeypatch):
    """Graph builders should emit nested, direction-specific NVTX ranges."""

    class FakeTensor:
        def set_output(self, *_args):
            return self

        def set_dim(self, *_args):
            return self

        def set_stride(self, *_args):
            return self

        def set_data_type(self, *_args):
            return self

    class FakeGraph:
        def __init__(self):
            self.sdpa_kwargs = None
            self.sdpa_backward_kwargs = None

        def tensor(self, **_kwargs):
            return FakeTensor()

        def tensor_like(self, _tensor):
            return FakeTensor()

        def sdpa(self, **kwargs):
            self.sdpa_kwargs = kwargs
            return FakeTensor(), FakeTensor()

        def sdpa_backward(self, **kwargs):
            self.sdpa_backward_kwargs = kwargs
            return FakeTensor(), FakeTensor(), FakeTensor()

        def validate(self):
            return None

        def get_workspace_size(self):
            return 1

    class FakeCudnn:
        class data_type:
            FLOAT = object()

        class heur_mode:
            A = object()

    events = []

    class FakeRange:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            events.append(("enter", self.name))

        def __exit__(self, *_args):
            events.append(("exit", self.name))

    monkeypatch.setattr(torch.cuda.nvtx, "range", FakeRange)
    monkeypatch.setattr(cudnn_attention, "_import_cudnn_frontend", FakeCudnn)

    graphs = []

    def fake_build_graph(*_args):
        graph = FakeGraph()
        graphs.append(graph)
        return graph

    monkeypatch.setattr(cudnn_attention, "_build_cudnn_pygraph", fake_build_graph)

    heuristic_calls = []

    def fake_build_plans(_graph, heuristic_modes, *, stage_range):
        heuristic_calls.append(heuristic_modes)
        for stage in (
            "build_operation_graph",
            "create_execution_plans",
            "check_support",
            "build_plans",
        ):
            with stage_range(stage):
                pass

    monkeypatch.setattr(cudnn_attention, "_build_cudnn_graph_plans", fake_build_plans)

    q, k, v, o, stats = _cpu_inputs()
    cudnn_attention._set_cudnn_attention_graph_build_profiling(True)
    cudnn_attention._set_cudnn_attention_cpp_comparable_graph_build(True)
    try:
        fwd_entry = cudnn_attention._build_cudnn_attention_fwd_graph(
            True, q, k, v, "bshd", "bshd", 0.5, "no_mask", o, stats
        )
        bwd_entry = cudnn_attention._build_cudnn_attention_bwd_graph(
            q, k, v, o, o, stats, "bshd", "bshd", 0.5, "no_mask", False
        )
    finally:
        cudnn_attention._set_cudnn_attention_cpp_comparable_graph_build(False)
        cudnn_attention._set_cudnn_attention_graph_build_profiling(False)

    assert events == [
        ("enter", "cudnn_graph_build_python_fwd"),
        ("enter", "cudnn_graph_materialization_and_validation_python_fwd"),
        ("enter", "cudnn_graph_ir_definition_python_fwd"),
        ("exit", "cudnn_graph_ir_definition_python_fwd"),
        ("enter", "cudnn_graph_lowering_and_native_validation_python_fwd"),
        ("exit", "cudnn_graph_lowering_and_native_validation_python_fwd"),
        ("exit", "cudnn_graph_materialization_and_validation_python_fwd"),
        ("enter", "cudnn_graph_build_operation_graph_python_fwd"),
        ("exit", "cudnn_graph_build_operation_graph_python_fwd"),
        ("enter", "cudnn_graph_create_execution_plans_python_fwd"),
        ("exit", "cudnn_graph_create_execution_plans_python_fwd"),
        ("enter", "cudnn_graph_check_support_python_fwd"),
        ("exit", "cudnn_graph_check_support_python_fwd"),
        ("enter", "cudnn_graph_build_plans_python_fwd"),
        ("exit", "cudnn_graph_build_plans_python_fwd"),
        ("exit", "cudnn_graph_build_python_fwd"),
        ("enter", "cudnn_graph_build_python_bwd"),
        ("enter", "cudnn_graph_materialization_and_validation_python_bwd"),
        ("enter", "cudnn_graph_ir_definition_python_bwd"),
        ("exit", "cudnn_graph_ir_definition_python_bwd"),
        ("enter", "cudnn_graph_lowering_and_native_validation_python_bwd"),
        ("exit", "cudnn_graph_lowering_and_native_validation_python_bwd"),
        ("exit", "cudnn_graph_materialization_and_validation_python_bwd"),
        ("enter", "cudnn_graph_build_operation_graph_python_bwd"),
        ("exit", "cudnn_graph_build_operation_graph_python_bwd"),
        ("enter", "cudnn_graph_create_execution_plans_python_bwd"),
        ("exit", "cudnn_graph_create_execution_plans_python_bwd"),
        ("enter", "cudnn_graph_check_support_python_bwd"),
        ("exit", "cudnn_graph_check_support_python_bwd"),
        ("enter", "cudnn_graph_build_plans_python_bwd"),
        ("exit", "cudnn_graph_build_plans_python_bwd"),
        ("exit", "cudnn_graph_build_python_bwd"),
    ]
    assert graphs[0].sdpa_kwargs["attn_scale"] is fwd_entry.attn_scale
    assert graphs[1].sdpa_backward_kwargs["attn_scale"] is bwd_entry.attn_scale
    for entry in (fwd_entry, bwd_entry):
        assert entry.attn_scale_value.device.type == "cpu"
        assert entry.attn_scale_value.dtype == torch.float32
        assert entry.attn_scale_value.shape == (1, 1, 1, 1)
        assert entry.attn_scale_value.item() == 0.5
    assert heuristic_calls == [
        [FakeCudnn.heur_mode.A],
        [FakeCudnn.heur_mode.A],
    ]

    events.clear()
    cudnn_attention._build_cudnn_attention_fwd_graph(
        True, q, k, v, "bshd", "bshd", 0.5, "no_mask", o, stats
    )
    assert not events
    assert heuristic_calls[-1] is None


def test_python_frontend_graph_cache_reset():
    """The cache-reset hook should invalidate cached Python graphs."""
    cache = cudnn_attention._cudnn_attention_graph_cache
    saved_cache = dict(cache)
    try:
        cache.clear()
        cache["sentinel"] = object()
        cudnn_attention._reset_cudnn_attention_graph_cache()
        assert not cache
    finally:
        cache.clear()
        cache.update(saved_cache)


def test_cpp_frontend_graph_profile_bindings():
    """The private profiling bindings should be callable without CUDA work."""
    tex._set_fused_attn_graph_build_profiling(True)
    try:
        tex._reset_fused_attn_graph_caches()
    finally:
        tex._set_fused_attn_graph_build_profiling(False)


def test_python_frontend_autograd_plumbing(monkeypatch):
    """Forward and backward should bind runtime tensors to their cached graph handles."""

    class FwdEntry:
        graph = object()
        q = object()
        k = object()
        v = object()
        attn_scale = object()
        attn_scale_value = torch.tensor(0.5)
        output = object()
        stats = object()
        workspace_size = 1

    class BwdEntry:
        graph = object()
        q = object()
        k = object()
        v = object()
        output = object()
        d_output = object()
        stats = object()
        attn_scale = object()
        attn_scale_value = torch.tensor(0.5)
        dq = object()
        dk = object()
        dv = object()
        workspace_size = 1

    monkeypatch.setattr(
        cudnn_attention,
        "_get_cudnn_attention_fwd_graph",
        lambda *args: FwdEntry(),
    )
    monkeypatch.setattr(
        cudnn_attention,
        "_get_cudnn_attention_bwd_graph",
        lambda *args: BwdEntry(),
    )

    def fake_execute(_graph, variant_pack, _workspace_size, _device):
        if BwdEntry.dq in variant_pack:
            assert variant_pack[BwdEntry.attn_scale] is BwdEntry.attn_scale_value
            variant_pack[BwdEntry.dq].fill_(1.0)
            variant_pack[BwdEntry.dk].fill_(2.0)
            variant_pack[BwdEntry.dv].fill_(3.0)
        else:
            assert variant_pack[FwdEntry.attn_scale] is FwdEntry.attn_scale_value
            variant_pack[FwdEntry.output].copy_(variant_pack[FwdEntry.q])

    monkeypatch.setattr(cudnn_attention, "_execute_cudnn_graph", fake_execute)

    q, k, v, _, _ = _cpu_inputs()
    q = q.requires_grad_()
    k = k.requires_grad_()
    v = v.requires_grad_()
    out = cudnn_attention.FusedAttentionWithPythonFrontendFunc.apply(
        True,
        q,
        k,
        v,
        "bshd",
        "bshd",
        0.5,
        "no_mask",
        False,
    )
    torch.testing.assert_close(out, q)

    out.sum().backward()
    torch.testing.assert_close(q.grad, torch.ones_like(q))
    torch.testing.assert_close(k.grad, torch.full_like(k, 2.0))
    torch.testing.assert_close(v.grad, torch.full_like(v, 3.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_dot_product_attention_propagates_fused_attention_impl():
    """The public selector should be retained by the internal fused backend."""
    attention = DotProductAttention(
        num_attention_heads=2,
        kv_channels=8,
        qkv_format="bshd",
        attn_mask_type="no_mask",
        fused_attention_impl="python",
    )
    assert attention.fused_attention_impl == "python"
    assert attention.fused_attention.fused_attention_impl == "python"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_python_frontend_does_not_fall_back_for_unsupported_configuration():
    """An explicitly selected Python path should reject unsupported dropout."""
    batch_size, seqlen, num_heads, head_dim = 2, 16, 2, 64
    q, k, v = [
        torch.randn(
            (batch_size, seqlen, num_heads, head_dim),
            dtype=torch.float16,
            device="cuda",
        )
        for _ in range(3)
    ]
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        seqlen,
        dtype=torch.int32,
        device="cuda",
    )
    attention = FusedAttention(
        head_dim**-0.5,
        attention_dropout=0.1,
        fused_attention_impl="python",
    ).cuda()

    with pytest.raises(ValueError, match="attention dropout"):
        attention(
            q,
            k,
            v,
            qkv_layout="bshd_bshd_bshd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            attn_mask_type="no_mask",
            window_size=(-1, -1),
            fused_attention_backend=FusedAttnBackend["F16_arbitrary_seqlen"],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.skipif(get_cudnn_version() < (9, 6, 0), reason="cuDNN 9.6.0+ is required.")
@pytest.mark.parametrize("is_training", [True, False])
def test_cpp_zero_dropout_does_not_advance_cuda_rng(is_training):
    """Zero-dropout C++ fused attention should not reserve Philox state."""
    batch_size, seqlen, num_heads, head_dim = 2, 16, 2, 64
    q, k, v = [
        torch.ones(
            (batch_size, seqlen, num_heads, head_dim),
            dtype=torch.float16,
            device="cuda",
            requires_grad=is_training,
        )
        for _ in range(3)
    ]
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        seqlen,
        dtype=torch.int32,
        device="cuda",
    )
    attention = (
        FusedAttention(
            head_dim**-0.5,
            attention_dropout=0.0,
            fused_attention_impl="cpp",
        )
        .cuda()
        .train(is_training)
    )
    kwargs = {
        "qkv_layout": "bshd_bshd_bshd",
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_kv": cu_seqlens,
        "attn_mask_type": "no_mask",
        "window_size": (-1, -1),
        "fused_attention_backend": FusedAttnBackend["F16_arbitrary_seqlen"],
    }

    # Warm up graph construction before observing the generator state.
    attention(q, k, v, **kwargs)
    rng_state_before = torch.cuda.get_rng_state()
    attention(q, k, v, **kwargs)
    rng_state_after = torch.cuda.get_rng_state()

    torch.testing.assert_close(rng_state_after, rng_state_before, rtol=0, atol=0)


_gpu_dtypes = [torch.float16]
if torch.cuda.is_available() and is_bf16_available():
    _gpu_dtypes.append(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@pytest.mark.skipif(get_cudnn_version() < (9, 6, 0), reason="cuDNN 9.6.0+ is required.")
@pytest.mark.parametrize("dtype", _gpu_dtypes)
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
@pytest.mark.parametrize("attn_mask_type", ["no_mask", "causal"])
@pytest.mark.parametrize("is_training", [True, False])
def test_python_frontend_matches_cpp_fused_attention(
    dtype, qkv_format, attn_mask_type, is_training
):
    """Compare basic Python-frontend attention fprop and bprop with the C++ path."""
    try:
        cudnn_attention._import_cudnn_frontend()
    except ImportError:
        pytest.skip("cuDNN frontend Python package is required.")

    torch.manual_seed(1234)
    batch_size, seqlen, num_heads, head_dim = 2, 32, 4, 64
    if qkv_format == "bshd":
        shape = (batch_size, seqlen, num_heads, head_dim)
    else:
        shape = (seqlen, batch_size, num_heads, head_dim)

    q_cpp, k_cpp, v_cpp = [
        (0.1 * torch.randn(shape, dtype=dtype, device="cuda")).requires_grad_(
            is_training
        )
        for _ in range(3)
    ]
    q_python, k_python, v_python = [
        tensor.detach().clone().requires_grad_(is_training)
        for tensor in (q_cpp, k_cpp, v_cpp)
    ]

    common_kwargs = {
        "qkv_layout": f"{qkv_format}_{qkv_format}_{qkv_format}",
        "cu_seqlens_q": torch.arange(
            0,
            (batch_size + 1) * seqlen,
            seqlen,
            dtype=torch.int32,
            device="cuda",
        ),
        "cu_seqlens_kv": torch.arange(
            0,
            (batch_size + 1) * seqlen,
            seqlen,
            dtype=torch.int32,
            device="cuda",
        ),
        "attn_mask_type": attn_mask_type,
        "window_size": (-1, 0) if attn_mask_type == "causal" else (-1, -1),
        "bottom_right_diagonal": False,
        "fused_attention_backend": FusedAttnBackend["F16_arbitrary_seqlen"],
    }
    scale = head_dim**-0.5
    cpp_attention = (
        FusedAttention(scale, fused_attention_impl="cpp").cuda().train(is_training)
    )
    python_attention = (
        FusedAttention(scale, fused_attention_impl="python").cuda().train(is_training)
    )

    with torch.set_grad_enabled(is_training):
        out_cpp = cpp_attention(q_cpp, k_cpp, v_cpp, **common_kwargs)
        out_python = python_attention(q_python, k_python, v_python, **common_kwargs)

    tolerances = {"atol": 5e-2, "rtol": 5e-2}
    torch.testing.assert_close(out_python, out_cpp, **tolerances)
    if is_training:
        d_out = torch.randn_like(out_cpp)
        out_cpp.backward(d_out)
        out_python.backward(d_out)
        torch.testing.assert_close(q_python.grad, q_cpp.grad, **tolerances)
        torch.testing.assert_close(k_python.grad, k_cpp.grad, **tolerances)
        torch.testing.assert_close(v_python.grad, v_cpp.grad, **tolerances)
