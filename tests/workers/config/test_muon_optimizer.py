import math

import pytest
import torch
import torch.nn as nn

from verl.utils.muon import Muon, MuonWithAdamW, newton_schulz
from verl.workers.config.optimizer import MuonOptimizerConfig, _should_use_muon, build_optimizer


class TestNewtonSchulz:
    def test_orthogonalizes_identity(self):
        M = torch.eye(4, dtype=torch.float32)
        result = newton_schulz(M, steps=5)
        assert result.shape == (4, 4)
        gram = result.T @ result
        diag = torch.diag(gram)
        assert (diag > 0.5).all(), f"Singular values should be close to 1, got {diag}"

    def test_orthogonalizes_random_matrix(self):
        torch.manual_seed(42)
        M = torch.randn(8, 4, dtype=torch.float32)
        result = newton_schulz(M, steps=5)
        assert result.shape == (8, 4)
        gram = result.T @ result
        diag = torch.diag(gram)
        off_diag = gram - torch.diag(diag)
        assert (diag > 0.3).all(), f"Diagonal should be significant, got {diag}"
        assert off_diag.abs().max() < 0.5, f"Off-diagonal should be small, got {off_diag.abs().max()}"

    def test_preserves_rank(self):
        torch.manual_seed(42)
        M = torch.randn(8, 4, dtype=torch.float32)
        result = newton_schulz(M, steps=5)
        assert torch.linalg.matrix_rank(result.float()) == 4

    def test_requires_2d(self):
        with pytest.raises(ValueError, match="2D"):
            newton_schulz(torch.randn(10), steps=5)

    def test_wide_matrix(self):
        torch.manual_seed(42)
        M = torch.randn(4, 8, dtype=torch.float32)
        result = newton_schulz(M, steps=5)
        assert result.shape == (4, 8)
        gram = result @ result.T
        diag = torch.diag(gram)
        assert (diag > 0.3).all(), f"Diagonal should be significant, got {diag}"

    def test_float32_input(self):
        torch.manual_seed(42)
        M = torch.randn(8, 4, dtype=torch.float32)
        result = newton_schulz(M, steps=5)
        assert result.dtype == torch.float32

    def test_bfloat16_input(self):
        torch.manual_seed(42)
        M = torch.randn(8, 4, dtype=torch.bfloat16)
        result = newton_schulz(M, steps=5)
        assert result.dtype == torch.bfloat16


class TestMuon:
    def _make_model(self):
        return nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.Linear(16, 4, bias=False),
        )

    def test_basic_step(self):
        model = self._make_model()
        optimizer = Muon(model.parameters(), lr=1e-3)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_momentum_buffer_created(self):
        model = self._make_model()
        optimizer = Muon(model.parameters(), lr=1e-3)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        for p in model.parameters():
            if p.ndim >= 2 and p in optimizer.state:
                assert "momentum_buffer" in optimizer.state[p]

    def test_weight_decay(self):
        model = self._make_model()
        optimizer = Muon(model.parameters(), lr=1e-3, weight_decay=0.01)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        p_before = model[0].weight.clone()
        optimizer.step()
        assert not torch.equal(model[0].weight, p_before)

    def test_rms_scale(self):
        model = self._make_model()
        optimizer = Muon(model.parameters(), lr=1e-3, rms_scale=0.2)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    def test_skips_1d_param(self):
        model = nn.Linear(8, 4)
        optimizer = Muon(model.parameters(), lr=1e-3)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        assert model.bias not in optimizer.state


class TestMuonWithAdamW:
    def _make_model(self):
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def test_param_split(self):
        model = self._make_model()
        muon_params = [model[0].weight, model[2].weight]
        adamw_params = [model[0].bias, model[2].bias]

        optimizer = MuonWithAdamW(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=1e-3,
        )
        assert len(optimizer.muon.param_groups[0]["params"]) == 2
        assert len(optimizer.adamw.param_groups[0]["params"]) == 2

    def test_step(self):
        model = self._make_model()
        muon_params = [model[0].weight, model[2].weight]
        adamw_params = [model[0].bias, model[2].bias]

        optimizer = MuonWithAdamW(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=1e-3,
        )

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_state_dict(self):
        model = self._make_model()
        muon_params = [model[0].weight, model[2].weight]
        adamw_params = [model[0].bias, model[2].bias]

        optimizer = MuonWithAdamW(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=1e-3,
        )
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        sd = optimizer.state_dict()
        assert "muon" in sd
        assert "adamw" in sd

    def test_adamw_lr_override(self):
        model = self._make_model()
        muon_params = [model[0].weight]
        adamw_params = [model[0].bias]

        optimizer = MuonWithAdamW(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=1e-3,
            adamw_lr=5e-3,
        )
        assert optimizer.adamw.param_groups[0]["lr"] == 5e-3


class TestMuonParamFilter:
    def test_hidden_layer_weight(self):
        p = nn.Parameter(torch.randn(16, 8))
        assert _should_use_muon("layers.0.self_attn.q_proj.weight", p, "hidden") is True

    def test_embedding_excluded(self):
        p = nn.Parameter(torch.randn(8, 16))
        assert _should_use_muon("model.embed_tokens.weight", p, "hidden") is False

    def test_lm_head_excluded(self):
        p = nn.Parameter(torch.randn(8, 16))
        assert _should_use_muon("lm_head.weight", p, "hidden") is False

    def test_norm_excluded(self):
        p = nn.Parameter(torch.randn(16, 8))
        assert _should_use_muon("layers.0.input_layernorm.weight", p, "hidden") is False

    def test_bias_excluded(self):
        p = nn.Parameter(torch.randn(16, 8))
        assert _should_use_muon("layers.0.self_attn.q_proj.bias", p, "hidden") is False

    def test_1d_always_excluded(self):
        p = nn.Parameter(torch.randn(16))
        assert _should_use_muon("layers.0.self_attn.q_proj.weight", p, "hidden") is False

    def test_all_2d_includes_embedding(self):
        p = nn.Parameter(torch.randn(8, 16))
        assert _should_use_muon("model.embed_tokens.weight", p, "all_2d") is True

    def test_all_2d_still_excludes_1d(self):
        p = nn.Parameter(torch.randn(16))
        assert _should_use_muon("layers.0.self_attn.q_proj.weight", p, "all_2d") is False


class TestMuonOptimizerConfig:
    def test_default_config(self):
        config = MuonOptimizerConfig(lr=2e-3)
        assert config.momentum == 0.95
        assert config.ns_steps == 5
        assert config.rms_scale == 0.2
        assert config.muon_param_filter == "hidden"
        assert config.optimizer == "MuonWithAdamW"
        assert config.optimizer_impl == "verl.utils.muon"

    def test_invalid_param_filter(self):
        with pytest.raises(AssertionError):
            MuonOptimizerConfig(lr=2e-3, muon_param_filter="invalid")


class TestBuildMuonOptimizer:
    def test_build_with_named_parameters(self):
        model = nn.Sequential(
            nn.Embedding(100, 8),
            nn.Linear(8, 16),
            nn.LayerNorm(16),
            nn.Linear(16, 4),
        )

        config = MuonOptimizerConfig(lr=2e-3)
        optimizer = build_optimizer(
            model.parameters(),
            config,
            named_parameters=model.named_parameters(),
        )
        assert isinstance(optimizer, MuonWithAdamW)

        muon_param_ids = {id(p) for p in optimizer.muon.param_groups[0]["params"]}
        adamw_param_ids = {id(p) for p in optimizer.adamw.param_groups[0]["params"]}

        for name, p in model.named_parameters():
            if _should_use_muon(name, p, "hidden"):
                assert id(p) in muon_param_ids, f"{name} should be in Muon"
            else:
                assert id(p) in adamw_param_ids, f"{name} should be in AdamW"

    def test_build_without_named_parameters(self):
        model = nn.Linear(8, 4)
        config = MuonOptimizerConfig(lr=2e-3, muon_param_filter="all_2d")
        optimizer = build_optimizer(model.parameters(), config)
        assert isinstance(optimizer, MuonWithAdamW)

    def test_build_non_muon_optimizer(self):
        config = MuonOptimizerConfig(lr=2e-3)
        from verl.workers.config.optimizer import FSDPOptimizerConfig

        fsdp_config = FSDPOptimizerConfig(lr=1e-3)
        optimizer = build_optimizer(nn.Linear(4, 4).parameters(), fsdp_config)
        assert type(optimizer).__name__ == "AdamW"
