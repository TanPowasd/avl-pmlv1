"""Configuration schema and YAML loader for AVL-PMLv1."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import List
import yaml


@dataclass
class AVLPMLConfig:
    model_name: str = "avl-pmlv1-40m"
    tier: str = "stage0_40m"
    base_model: str = ""

    num_layers: int = 12
    hidden_size: int = 512
    num_q_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    dense_intermediate_size: int = 768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_qk_norm: bool = False
    use_block_idx_embed: bool = True
    cold_start_graft: bool = False
    enable_copy_gate: bool = False
    continuous_esc_whiteboard: bool = True

    sym_vocab_size: int = 4096
    nl_vocab_size: int = 2048
    op_vocab_size: int = 256
    fsq_levels: List[int] = field(default_factory=lambda: [8, 4, 4, 4, 2])
    fsq_residual_rank: int = 16
    lit_prim_size: int = 512
    lit_escape_size: int = 1216
    num_ro_slots: int = 32
    num_rw_slots: int = 32

    moe_layer_indices: List[int] = field(default_factory=lambda: list(range(1, 10)))
    shared_expert_intermediate_size: int = 256
    num_routed_experts: int = 8
    routed_expert_intermediate_size: int = 224
    probe_rank: int = 8
    k_max: int = 2
    k_min_guard: int = 1
    g_min_guard: float = 0.25
    c_pain_max: float = 0.60
    gamma_pain: float = 0.85
    kappa_gate: float = 0.50
    router_temp: float = 0.70

    pml_layer_indices: List[int] = field(default_factory=lambda: [1, 4, 7, 10])
    pml_state_dim: int = 32

    engram_layer_indices: List[int] = field(default_factory=lambda: [1, 5])
    engram_num_heads: int = 4
    engram_num_buckets: int = 4097
    engram_dim: int = 64

    dir_vector_layer_index: int = 3
    atomic_chunk_size: int = 16
    default_block_size: int = 32
    wet_cement_slots: int = 16
    max_circuit_breakers: int = 2
    sparse_snapshot_interval: int = 4
    pll_guard_margin: float = 0.08

    @property
    def single_fsq_size(self) -> int:
        return math.prod(self.fsq_levels)

    @property
    def total_fsq_size(self) -> int:
        return 2 * self.single_fsq_size

    @property
    def total_slots(self) -> int:
        return self.num_ro_slots + self.num_rw_slots

    @property
    def ro_token_start(self) -> int:
        return self.sym_vocab_size - self.total_slots

    @property
    def rw_token_start(self) -> int:
        return self.sym_vocab_size - self.num_rw_slots

    def validate(self) -> None:
        expected_sym = (
            self.op_vocab_size
            + self.total_fsq_size
            + self.lit_prim_size
            + self.lit_escape_size
            + self.total_slots
        )
        if expected_sym != self.sym_vocab_size:
            raise ValueError(
                f"Symbolic vocab partition sum {expected_sym} != sym_vocab_size {self.sym_vocab_size}"
            )
        if self.atomic_chunk_size not in (8, 16):
            raise ValueError(f"Physical atomic chunk size must be 8 or 16, got {self.atomic_chunk_size}.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AVLPMLConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls(**data)
        cfg.validate()
        return cfg
