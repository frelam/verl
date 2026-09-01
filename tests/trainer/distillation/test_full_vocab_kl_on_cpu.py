# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU correctness tests for full-vocab OPD/MOPD distillation kernels.

Covers (see MOPD_full_vocab_design.md §7):

1. Chunked online-softmax forward/reverse KL vs the naive materialized ``[N, V]``
   reference, including temperature and ``log_prob_min_clamp``.
2. Backward gradients of both autograd Functions vs the naive autograd reference.
3. Multi-teacher (MOPD) scheduling: mixed-teacher micro batches produce the same
   per-token KL as computing each teacher separately; uncovered rows stay at 0.
4. Teacher lm_head loading (untied / tied / sharded safetensors), vocab-parallel
   sharding with zero-row padding, and ``TeacherLmHeadStore`` residency/LRU policy.
5. Artifact normalization / grouping / hidden-row fetching (TQ fetch monkeypatched).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Keep the chunk-update functions in eager mode: torch.compile on CPU test runners
# is slow and adds no coverage value here.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tensordict import NonTensorData

from verl.trainer.distillation import full_vocab_kl as fvkl
from verl.trainer.distillation import full_vocab_tq

_ATOL = 2e-5


# ---------------------------------------------------------------------------
# Naive references (materialize [N, V] teacher logits)
# ---------------------------------------------------------------------------


def _naive_fwd_kl(zt: torch.Tensor, zs: torch.Tensor, temperature: float, clamp: float | None) -> torch.Tensor:
    zt = zt.float() / temperature
    zs = zs.float() / temperature
    lse_t = zt.logsumexp(-1)
    lse_s = zs.logsumexp(-1)
    log_pt = zt - lse_t.unsqueeze(-1)
    pt = log_pt.exp()
    # The kernel treats the clamp floor as a per-token constant (detached), so the
    # reference must detach it too for the gradients to be comparable.
    zs_c = zs if clamp is None else torch.maximum(zs, (lse_s.detach() + clamp).unsqueeze(-1))
    return (pt * (log_pt - (zs_c - lse_s.unsqueeze(-1)))).sum(-1)


def _naive_rev_kl(zt: torch.Tensor, zs: torch.Tensor, temperature: float) -> torch.Tensor:
    zt = zt.float() / temperature
    zs = zs.float() / temperature
    lse_t = zt.logsumexp(-1)
    lse_s = zs.logsumexp(-1)
    log_ps = zs - lse_s.unsqueeze(-1)
    log_pt = zt - lse_t.unsqueeze(-1)
    return (log_ps.exp() * (log_ps - log_pt)).sum(-1)


def _make_inputs(n=7, v=24, h=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, h, generator=g)
    w = torch.randn(v, h, generator=g) / h**0.5
    s = torch.randn(n, v, generator=g)
    return z, w, s


# ---------------------------------------------------------------------------
# 1. Forward value correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_vocab", [3, 5, 24, 1000])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("clamp", [None, -5.0])
def test_fwd_kl_chunked_matches_naive(chunk_vocab, temperature, clamp):
    z, w, s = _make_inputs()
    kl = fvkl.FullVocabForwardKL.apply(z, w, w, s, chunk_vocab, temperature, clamp)
    ref = _naive_fwd_kl(z @ w.T, s, temperature, clamp)
    assert torch.allclose(kl, ref, atol=_ATOL), (kl - ref).abs().max()


@pytest.mark.parametrize("chunk_vocab", [3, 5, 24, 1000])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_rev_kl_chunked_matches_naive(chunk_vocab, temperature):
    z, w, s = _make_inputs()
    kl = fvkl.FullVocabReverseKL.apply(z, w, w, s, chunk_vocab, temperature)
    ref = _naive_rev_kl(z @ w.T, s, temperature)
    assert torch.allclose(kl, ref, atol=_ATOL), (kl - ref).abs().max()


def test_kl_nonnegative_for_identical_distributions():
    """KL(p||p) == 0: a sentinel for accumulator rescaling bugs."""
    z, w, _ = _make_inputs()
    s = z @ w.T  # student distribution identical to teacher's
    kl_fwd = fvkl.FullVocabForwardKL.apply(z, w, w, s.clone(), 4, 1.0, None)
    kl_rev = fvkl.FullVocabReverseKL.apply(z, w, w, s.clone(), 4, 1.0)
    assert torch.allclose(kl_fwd, torch.zeros_like(kl_fwd), atol=_ATOL)
    assert torch.allclose(kl_rev, torch.zeros_like(kl_rev), atol=_ATOL)


# ---------------------------------------------------------------------------
# 2. Backward gradient correctness (teacher side is stop-gradient)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_vocab", [3, 24])
@pytest.mark.parametrize("temperature", [1.0, 1.5])
@pytest.mark.parametrize("clamp", [None, -5.0])
def test_fwd_kl_backward_matches_naive(chunk_vocab, temperature, clamp):
    z, w, s = _make_inputs()
    s_kernel = s.clone().requires_grad_(True)
    kl = fvkl.FullVocabForwardKL.apply(z, w, w, s_kernel, chunk_vocab, temperature, clamp)
    kl.sum().backward()

    s_ref = s.clone().requires_grad_(True)
    ref = _naive_fwd_kl(z @ w.T, s_ref, temperature, clamp)
    ref.sum().backward()
    assert torch.allclose(s_kernel.grad, s_ref.grad, atol=_ATOL), (s_kernel.grad - s_ref.grad).abs().max()


@pytest.mark.parametrize("chunk_vocab", [3, 24])
@pytest.mark.parametrize("temperature", [1.0, 1.5])
def test_rev_kl_backward_matches_naive(chunk_vocab, temperature):
    z, w, s = _make_inputs()
    s_kernel = s.clone().requires_grad_(True)
    kl = fvkl.FullVocabReverseKL.apply(z, w, w, s_kernel, chunk_vocab, temperature)
    kl.sum().backward()

    s_ref = s.clone().requires_grad_(True)
    ref = _naive_rev_kl(z @ w.T, s_ref, temperature)
    ref.sum().backward()
    assert torch.allclose(s_kernel.grad, s_ref.grad, atol=_ATOL), (s_kernel.grad - s_ref.grad).abs().max()


def test_teacher_side_is_stop_gradient():
    """Teacher hidden states and the lm_head are stop-gradient: even when they
    require grad, the backward must not populate their .grad (the Function
    returns None for those inputs), and only the student logits differentiate."""
    z, w, s = _make_inputs()
    z_req = z.clone().requires_grad_(True)
    w_req = w.clone().requires_grad_(True)
    s_req = s.clone().requires_grad_(True)
    kl = fvkl.FullVocabForwardKL.apply(z_req, w_req, w_req, s_req, 4, 1.0, None)
    kl.sum().backward()
    assert z_req.grad is None
    assert w_req.grad is None
    assert s_req.grad is not None and s_req.grad.abs().max() > 0


# ---------------------------------------------------------------------------
# 3. Artifact helpers
# ---------------------------------------------------------------------------


def _artifact(teacher: str, key: str, seq_len: int, hidden_size: int = 16) -> dict:
    return full_vocab_tq.build_artifact(
        teacher_name=teacher,
        step=3,
        uid=key,
        seq_len=seq_len,
        hidden_size=hidden_size,
        dtype=torch.bfloat16,
        prefix="test_exp",
    )


def test_normalize_artifacts_unwraps_non_tensor_data():
    arts = [_artifact("t", "a", 5), _artifact("t", "b", 4)]
    raw = [NonTensorData(a) for a in arts]
    out = fvkl.normalize_artifacts(raw, batch_size=2)
    assert out == arts


def test_normalize_artifacts_none_becomes_uncovered():
    out = fvkl.normalize_artifacts([_artifact("t", "a", 5), None], batch_size=2)
    assert out[1] is None


def test_normalize_artifacts_missing_column_raises():
    with pytest.raises(KeyError, match="teacher_full_vocab_artifact"):
        fvkl.normalize_artifacts(None, batch_size=2)


def test_normalize_artifacts_length_mismatch_raises():
    with pytest.raises(RuntimeError, match="entries but the micro batch"):
        fvkl.normalize_artifacts([_artifact("t", "a", 5)], batch_size=2)


def test_group_artifacts_by_teacher_sorted_order():
    arts = [
        _artifact("zebra", "a", 5),
        None,
        _artifact("apple", "b", 4),
        _artifact("zebra", "c", 3),
    ]
    groups = fvkl.group_artifacts_by_teacher(arts)
    # Keys must be sorted so every rank iterates teachers in the same order.
    assert list(groups) == ["apple", "zebra"]
    assert groups["apple"] == [2]
    assert groups["zebra"] == [0, 3]


def test_fetch_hidden_rows_mixed_coverage(monkeypatch):
    h = 4
    arts = [
        _artifact("tb", "k0", 3, h),
        None,
        _artifact("ta", "k2", 2, h),
        _artifact("tb", "k3", 1, h),
    ]
    hidden_by_key = {a["uid"]: torch.full((a["seq_len"], h), float(i + 1)) for i, a in enumerate(arts) if a}

    def fake_fetch(batch_artifacts):
        return [hidden_by_key[a["uid"]] for a in batch_artifacts]

    monkeypatch.setattr(full_vocab_tq, "fetch_hidden_batch", fake_fetch)
    seq_lens = [3, 2, 2, 1]
    rows, teacher_index, teacher_keys = fvkl.fetch_hidden_rows(arts, seq_lens)

    assert teacher_keys == ["ta", "tb"]  # sorted
    assert teacher_index == [1, -1, 0, 1]
    for i, art in enumerate(arts):
        assert rows[i].shape == (seq_lens[i], h)
        if art is None:
            assert torch.all(rows[i] == 0)
        else:
            assert torch.allclose(rows[i], hidden_by_key[art["uid"]])


def test_fetch_hidden_rows_length_mismatch_raises(monkeypatch):
    arts = [_artifact("t", "k0", 3, 4)]

    def fake_fetch(batch_artifacts):
        return [torch.zeros(2, 4)]  # wrong length

    monkeypatch.setattr(full_vocab_tq, "fetch_hidden_batch", fake_fetch)
    with pytest.raises(RuntimeError, match="teacher hidden length"):
        fvkl.fetch_hidden_rows(arts, [3])


def test_fetch_hidden_rows_all_uncovered_uses_dummy_width():
    rows, teacher_index, teacher_keys = fvkl.fetch_hidden_rows([None, None], [3, 2])
    assert teacher_keys == []
    assert teacher_index == [-1, -1]
    assert rows[0].shape == (3, 1) and rows[1].shape == (2, 1)
    assert all(torch.all(r == 0) for r in rows)


# ---------------------------------------------------------------------------
# 4. Multi-teacher (MOPD) pass scheduling
# ---------------------------------------------------------------------------


class _FakeStore:
    """In-memory stand-in for TeacherLmHeadStore (CPU residency semantics)."""

    def __init__(self, weights: dict[str, torch.Tensor]):
        self.weights = weights
        self.acquired: list[str] = []

    @contextmanager
    def acquire(self, teacher_key, device):
        self.acquired.append(teacher_key)
        w = self.weights[teacher_key]
        yield w, w


def _loss_config(**overrides):
    cfg = dict(
        full_vocab_chunk_vocab=5,
        full_vocab_chunk_tokens=4,
        full_vocab_max_tokens_per_pass=None,
        kd_temperature=1.0,
        log_prob_min_clamp=None,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _run_passes(z, idx, keys, s, cfg, reverse=False):
    store = _FakeStore({k: _W[k] for k in keys})
    kl, norm = fvkl.run_teacher_passes(
        z,
        idx,
        keys,
        s,
        store=store,
        loss_config=cfg,
        reverse=reverse,
        kernel_fwd=fvkl.FullVocabForwardKL.apply,
        kernel_rev=fvkl.FullVocabReverseKL.apply,
    )
    return kl, norm, store


_W: dict[str, torch.Tensor] = {}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("chunk_tokens", [2, 4, 100])
def test_run_teacher_passes_multi_teacher_matches_per_teacher(reverse, chunk_tokens):
    torch.manual_seed(0)
    h, v = 16, 24
    _W["apple"] = torch.randn(v, h) / h**0.5
    _W["zebra"] = torch.randn(v, h) / h**0.5
    # Interleaved teachers with uncovered rows: [zebra, apple, -, zebra, apple, zebra, -]
    gid = torch.tensor([1, 0, -1, 1, 0, 1, -1])
    keys = ["apple", "zebra"]
    n = gid.numel()
    z = torch.randn(n, h)
    s = torch.randn(n, v)

    cfg = _loss_config(full_vocab_chunk_tokens=chunk_tokens)
    kl, norm, store = _run_passes(z, gid, keys, s, cfg, reverse=reverse)

    # Per-token reference computed teacher by teacher with the naive formula.
    ref = torch.zeros(n)
    for i in range(n):
        g = int(gid[i])
        if g < 0:
            continue
        w = _W[keys[g]]
        zt_i = (z[i : i + 1] @ w.T).squeeze(0)
        ref[i] = (
            _naive_rev_kl(zt_i, s[i], cfg.kd_temperature)
            if reverse
            else _naive_fwd_kl(zt_i, s[i], cfg.kd_temperature, cfg.log_prob_min_clamp)
        )

    assert torch.allclose(kl, ref, atol=_ATOL), (kl - ref).abs().max()
    # Uncovered rows: zero loss and zero norm; covered rows carry the hidden norm.
    assert kl[gid < 0].abs().max() == 0
    assert norm[gid < 0].abs().max() == 0
    assert torch.allclose(norm[gid >= 0], z[gid >= 0].float().norm(dim=-1), atol=_ATOL)
    # Both teachers were acquired exactly once, in sorted key order, regardless of
    # how their rows are interleaved in the micro batch.
    assert store.acquired == ["apple", "zebra"]


def test_run_teacher_passes_no_teachers_returns_zeros():
    z = torch.zeros(3, 4)
    s = torch.randn(3, 8)
    kl, norm, store = _run_passes(z, torch.full((3,), -1), [], s, _loss_config())
    assert torch.all(kl == 0) and torch.all(norm == 0)
    assert store.acquired == []


def test_run_teacher_passes_max_tokens_per_pass_caps_chunks():
    """full_vocab_max_tokens_per_pass tightens the token chunking without changing values."""
    torch.manual_seed(1)
    h, v = 8, 16
    _W["solo"] = torch.randn(v, h)
    n = 10
    z, s = torch.randn(n, h), torch.randn(n, v)
    gid = torch.zeros(n, dtype=torch.int64)
    cfg_base = _loss_config(full_vocab_chunk_tokens=100)
    cfg_capped = _loss_config(full_vocab_chunk_tokens=100, full_vocab_max_tokens_per_pass=3)
    kl_base, _, _ = _run_passes(z, gid, ["solo"], s, cfg_base)
    kl_capped, _, _ = _run_passes(z, gid, ["solo"], s, cfg_capped)
    assert torch.allclose(kl_base, kl_capped, atol=_ATOL)


def test_run_teacher_passes_gradient_only_for_covered_rows():
    torch.manual_seed(2)
    h, v = 8, 12
    _W["solo"] = torch.randn(v, h)
    z = torch.randn(4, h)
    s = torch.randn(4, v, requires_grad=True)
    gid = torch.tensor([0, -1, 0, -1])
    kl, _, _ = _run_passes(z, gid, ["solo"], s, _loss_config())
    kl.sum().backward()
    assert s.grad[1].abs().max() == 0 and s.grad[3].abs().max() == 0
    assert s.grad[0].abs().max() > 0 and s.grad[2].abs().max() > 0


# ---------------------------------------------------------------------------
# 5. lm_head loading / sharding / store
# ---------------------------------------------------------------------------


def test_shard_lm_head_zero_padding_and_tp_split():
    v, h, padded, tp = 10, 4, 16, 2
    w = torch.randn(v, h)
    shard0 = fvkl.shard_lm_head(w, padded, tp_rank=0, tp_size=tp)
    shard1 = fvkl.shard_lm_head(w, padded, tp_rank=1, tp_size=tp)
    assert shard0.shape == (8, h) and shard1.shape == (8, h)
    # Rank 0 holds real rows 0..7; rank 1 holds real rows 8..9 then zero pad.
    assert torch.allclose(shard0, w[:8])
    assert torch.allclose(shard1[:2], w[8:10])
    assert torch.all(shard1[2:] == 0)
    # The concatenation of shards reproduces the padded weight.
    assert torch.allclose(torch.cat([shard0, shard1])[:v], w)


def test_shard_lm_head_rejects_bad_layout():
    w = torch.randn(10, 4)
    with pytest.raises(ValueError, match="divisible"):
        fvkl.shard_lm_head(w, 15, tp_rank=0, tp_size=2)
    with pytest.raises(ValueError, match="exceeds the student's padded vocab"):
        fvkl.shard_lm_head(w, 8, tp_rank=0, tp_size=1)


def _write_ckpt(folder, tensors: dict[str, torch.Tensor], tie: bool, sharded: bool):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "tie_word_embeddings": tie,
                "hidden_size": 8,
                "vocab_size": 12,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 16,
            }
        )
    )
    if not sharded:
        save_file(tensors, str(folder / "model.safetensors"))
        return
    # Two shards; the target weight lives in the second one to exercise the index path.
    target_key = next(iter(tensors))
    shard_a = {"model.layers.0.mlp.down_proj.weight": torch.randn(4, 4)}
    save_file(shard_a, str(folder / "model-00001-of-00002.safetensors"))
    save_file(tensors, str(folder / "model-00002-of-00002.safetensors"))
    weight_map = {
        **{k: "model-00001-of-00002.safetensors" for k in shard_a},
        target_key: "model-00002-of-00002.safetensors",
    }
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize("sharded", [False, True])
def test_load_teacher_lm_head_untied(tmp_path, sharded):
    w = torch.randn(12, 8)
    ckpt = tmp_path / "untied"
    _write_ckpt(ckpt, {"lm_head.weight": w}, tie=False, sharded=sharded)
    loaded = fvkl.load_teacher_lm_head(str(ckpt))
    assert torch.allclose(loaded, w)


def test_load_teacher_lm_head_tied_uses_embed_tokens(tmp_path):
    w = torch.randn(12, 8)
    ckpt = tmp_path / "tied"
    _write_ckpt(ckpt, {"model.embed_tokens.weight": w}, tie=True, sharded=False)
    loaded = fvkl.load_teacher_lm_head(str(ckpt))
    assert torch.allclose(loaded, w)


def test_load_teacher_lm_head_missing_weight_raises(tmp_path):
    ckpt = tmp_path / "empty"
    _write_ckpt(ckpt, {"some.other.weight": torch.randn(3, 3)}, tie=False, sharded=False)
    with pytest.raises(ValueError, match="Could not find lm_head"):
        fvkl.load_teacher_lm_head(str(ckpt))


def _store(tmp_path, residency, max_resident=1):
    w_a = torch.randn(12, 8)
    w_b = torch.randn(12, 8)
    _write_ckpt(tmp_path / "ta", {"lm_head.weight": w_a}, tie=False, sharded=False)
    _write_ckpt(tmp_path / "tb", {"lm_head.weight": w_b}, tie=False, sharded=False)
    store = fvkl.TeacherLmHeadStore(
        teacher_checkpoints={"ta": str(tmp_path / "ta"), "tb": str(tmp_path / "tb")},
        vocab_size_padded=12,
        residency=residency,
        max_resident_teachers=max_resident,
        pin_memory=False,
    )
    return store, w_a, w_b


def test_store_cpu_residency_lazy_load_and_cache(tmp_path):
    store, w_a, _ = _store(tmp_path, residency="cpu")
    assert store.teacher_keys == ["ta", "tb"]
    with store.acquire("ta", torch.device("cpu")) as (w_fwd, w_bwd):
        # CPU residency: forward reads the CPU shard directly, backward the same reference.
        assert w_fwd is w_bwd
        assert not w_fwd.is_cuda
        assert torch.allclose(w_fwd, w_a)
    # Second acquire hits the cached CPU shard (no reload).
    assert store.cpu_shard("ta") is store.cpu_shard("ta")
    with pytest.raises(KeyError, match="unknown teacher key"):
        store.cpu_shard("unknown")


def test_store_gpu_residency_lru_bound(tmp_path):
    store, w_a, w_b = _store(tmp_path, residency="gpu", max_resident=1)
    dev = torch.device("cpu")  # CPU test box: "device" copies are still plain tensors.
    with store.acquire("ta", dev) as (w_fwd, _):
        assert torch.allclose(w_fwd, w_a)
    assert list(store._gpu_cache) == ["ta"]
    # Acquiring the second teacher evicts the first under the max_resident=1 bound.
    with store.acquire("tb", dev) as (w_fwd, _):
        assert torch.allclose(w_fwd, w_b)
    assert list(store._gpu_cache) == ["tb"]
    # CPU shards stay resident; only the device copies are evicted.
    assert set(store._cpu_shards) == {"ta", "tb"}
    store.release_all()
    assert not store._gpu_cache


def test_store_rejects_bad_config(tmp_path):
    with pytest.raises(ValueError, match="residency"):
        fvkl.TeacherLmHeadStore(
            teacher_checkpoints={"t": str(tmp_path)}, vocab_size_padded=8, residency="tpu"
        )
    with pytest.raises(ValueError, match="max_resident"):
        fvkl.TeacherLmHeadStore(
            teacher_checkpoints={"t": str(tmp_path)}, vocab_size_padded=8, max_resident_teachers=0
        )
    with pytest.raises(ValueError, match="at least one teacher"):
        fvkl.TeacherLmHeadStore(teacher_checkpoints={}, vocab_size_padded=8)


# ---------------------------------------------------------------------------
# 6. TP online-softmax merge (gloo, two processes)
# ---------------------------------------------------------------------------

try:
    from verl.trainer.distillation.megatron.full_vocab_kl import _merge_tp_online, _vocab_parallel_lse

    _MEGATRON_ENTRY_OK = True
except Exception:  # megatron package not installed on this runner
    _MEGATRON_ENTRY_OK = False


def _tp_merge_worker(rank, world_size, init_file, z, w, s, chunk_vocab, result_dir):
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    group = dist.group.WORLD
    per_rank = s.shape[1] // world_size
    w_shard = w[rank * per_rank : (rank + 1) * per_rank]
    s_shard = s[:, rank * per_rank : (rank + 1) * per_rank]

    s_lse = _vocab_parallel_lse(s_shard.float(), group)
    mt, st, tt, ut = fvkl._fwd_local_accumulators(
        z.float(), w_shard, s_shard.float(), chunk_vocab, fvkl._student_floor(s_lse, None, s.device), s.device
    )
    mt, (st, tt, ut) = _merge_tp_online(mt, [st, tt, ut], group)
    t_lse = mt + st.log()
    kl = tt / st - t_lse - ut / st + s_lse

    if rank == 0:
        torch.save({"kl": kl, "s_lse": s_lse}, result_dir / "tp_merge_result.pt")
    dist.destroy_process_group()


@pytest.mark.skipif(not _MEGATRON_ENTRY_OK, reason="megatron package is not installed on this runner")
def test_tp_merge_two_process_gloo(tmp_path):
    """Two gloo ranks holding disjoint vocab shards must reconstruct, after the exact
    online-softmax merge, the same KL as a single process over the full vocab."""
    import torch.multiprocessing as mp

    z, w, s = _make_inputs(n=5, v=24, h=16, seed=7)
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _tp_merge_worker,
        args=(2, init_file, z, w, s, 4, tmp_path),
        nprocs=2,
        join=True,
    )
    result = torch.load(tmp_path / "tp_merge_result.pt", weights_only=True)
    ref_s_lse = s.float().logsumexp(-1)
    ref_kl = _naive_fwd_kl(z @ w.T, s, 1.0, None)
    assert torch.allclose(result["s_lse"], ref_s_lse, atol=_ATOL)
    assert torch.allclose(result["kl"], ref_kl, atol=_ATOL)
