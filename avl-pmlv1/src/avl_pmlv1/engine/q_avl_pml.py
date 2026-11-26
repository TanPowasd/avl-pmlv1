"""Q-AVL-PML (Blackwell-Native Group-Wise INT4 + BF16 LoRA Hybrid) & Master Manual §3.2 5-Group Optimizer.

Enables full 4.54B AVL-PMLv1.2 surgery, verification, and local Stage 2/3 training on a single 16 GB RTX 5080
(< 9 GB peak VRAM, zero external DLL dependencies) while seamlessly sharing the exact same 5-group optimizer
specification for 48 GB L40S / 80 GB H100 full-parameter cloud runs.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear4Bit(nn.Module):
    """Symmetric group-wise 4-bit packed linear layer (0.53 bytes/param, Blackwell-native pure PyTorch)."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 64) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.padded_in = ((in_features + group_size - 1) // group_size) * group_size
        self.num_groups = self.padded_in // group_size

        self.register_buffer("qweight", torch.empty((out_features, self.padded_in // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.empty((out_features, self.num_groups), dtype=torch.bfloat16))

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, group_size: int = 64) -> "QuantizedLinear4Bit":
        w = linear.weight.detach()
        out_f, in_f = w.shape
        qmod = cls(in_f, out_f, group_size=group_size)
        wf = w.float()
        if qmod.padded_in > in_f:
            wf = F.pad(wf, (0, qmod.padded_in - in_f))
        wg = wf.view(out_f, qmod.num_groups, group_size)
        amax = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (amax / 7.0).squeeze(-1)  # [out_f, num_groups]
        q = torch.round(wg / (amax / 7.0)).clamp(-8, 7).to(torch.int16) + 8  # unsigned [0..15]
        q_flat = q.view(out_f, qmod.padded_in).to(torch.uint8)
        packed = (q_flat[:, 0::2] & 0x0F) | ((q_flat[:, 1::2] & 0x0F) << 4)
        qmod.qweight.copy_(packed)
        qmod.scales.copy_(scale.to(torch.bfloat16))
        return qmod

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        low = (self.qweight & 0x0F).to(torch.int8) - 8
        high = ((self.qweight >> 4) & 0x0F).to(torch.int8) - 8
        unpacked = torch.stack([low, high], dim=-1).view(self.out_features, self.num_groups, self.group_size)
        w = (unpacked.to(dtype) * self.scales.to(dtype).unsqueeze(-1)).view(self.out_features, self.padded_in)
        if self.padded_in > self.in_features:
            w = w[:, : self.in_features]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize(x.dtype)
        return F.linear(x, w, None)


class LoRAQuantizedLinear(nn.Module):
    """Frozen 4-bit quantized base weight + BF16 trainable LoRA adapter (r=16, alpha=32)."""

    def __init__(self, base_q: QuantizedLinear4Bit, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base_q
        self.rank = rank
        self.scaling = float(alpha) / float(max(1, rank))
        self.lora_a = nn.Parameter(torch.empty(rank, base_q.in_features, dtype=torch.bfloat16))
        self.lora_b = nn.Parameter(torch.zeros(base_q.out_features, rank, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.rank > 0:
            dx = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
            y = y + dx * self.scaling
        return y


@torch.no_grad()
def quantize_to_q_avl_pml(
    model: nn.Module,
    lora_rank_gqa: int = 16,
    lora_rank_dense: int = 16,
    lora_rank_routed: int = 0,
    group_size: int = 64,
    freeze_nl_embed: bool = True,
) -> Dict[str, int]:
    """Convert 4.54B AVL-PMLv1.2 into Tier-Ref-5080 Q-AVL-PML hybrid state:
    - GQA (q/k/v/o) + DenseSwiGLU (L0/L35): 4-bit quantized + BF16 LoRA (Group A);
    - Routed Experts (34 x 8): 4-bit quantized (+ optional LoRA in Stage 3+);
    - Shared Micro-Experts (512), Track-1 PML, Static Engram, Factored FSQ, Copy-Gate, Routers,
      Bipolar Graphs, Hyper-Router, Dir-Vector & Norms: kept in 100% full BF16 / FP32!
    """
    from ..moe_track2.tri_state_moe import DenseSwiGLU, TriStateMoELayer

    q_count = 0
    lora_count = 0

    def _wrap_linear(parent: nn.Module, attr: str, rank: int) -> None:
        nonlocal q_count, lora_count
        lin = getattr(parent, attr)
        if not isinstance(lin, nn.Linear):
            return
        q4 = QuantizedLinear4Bit.from_linear(lin, group_size=group_size)
        q_count += 1
        if rank > 0:
            setattr(parent, attr, LoRAQuantizedLinear(q4, rank=rank, alpha=2.0 * rank))
            lora_count += 1
        else:
            setattr(parent, attr, q4)

    for attn_l in model.attn_layers:  # type: ignore[attr-defined]
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _wrap_linear(attn_l.gqa, proj_name, lora_rank_gqa)

    for ffn_l in model.ffn_layers:  # type: ignore[attr-defined]
        if isinstance(ffn_l, DenseSwiGLU):
            for proj_name in ("w1", "w2", "w3"):
                _wrap_linear(ffn_l.ffn, proj_name, lora_rank_dense)
        elif isinstance(ffn_l, TriStateMoELayer):
            for exp in ffn_l.routed_experts:
                for proj_name in ("w1", "w2", "w3"):
                    _wrap_linear(exp, proj_name, lora_rank_routed)

    if freeze_nl_embed and hasattr(model, "nl_embed"):
        model.nl_embed.weight.requires_grad_(False)  # type: ignore[attr-defined]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "quantized_linears": q_count,
        "lora_adapters": lora_count,
        "trainable_params": trainable,
        "quantized_buffer_mb": int(round(q_bytes / (1024 * 1024))),
    }


@torch.no_grad()
def merge_lora_delta_into_bf16_model(
    model: nn.Module,
    delta_sd: Dict[str, torch.Tensor],
    lora_scaling: float = 2.0,
) -> Dict[str, int]:
    """Merge a Q-AVL-PML LoRA+trainable delta state_dict into an unquantized BF16/FP32 AVLPMLModel in-place.
    - Non-LoRA trained tensors (SymEmbed, Engram, Track-1 PML, Norms, DirHead, etc.) are loaded directly;
    - LoRA adapter pairs (<prefix>.lora_a, <prefix>.lora_b) are folded into <prefix>.weight via
      W += lora_scaling * (lora_b @ lora_a), eliminating 4-bit eager dequant overhead during inference!
    """
    lora_a_map: Dict[str, torch.Tensor] = {}
    lora_b_map: Dict[str, torch.Tensor] = {}
    regular_sd: Dict[str, torch.Tensor] = {}

    for k, v in delta_sd.items():
        if k == "nl_head.weight":
            continue
        if k.endswith(".lora_a"):
            lora_a_map[k[: -len(".lora_a")]] = v
        elif k.endswith(".lora_b"):
            lora_b_map[k[: -len(".lora_b")]] = v
        else:
            regular_sd[k] = v

    if regular_sd:
        model.load_state_dict(regular_sd, strict=False)

    param_map = dict(model.named_parameters())
    merged_count = 0
    for prefix, la in lora_a_map.items():
        lb = lora_b_map.get(prefix)
        w_key = f"{prefix}.weight"
        if lb is None or w_key not in param_map:
            continue
        w_param = param_map[w_key]
        delta = (
            lb.to(device=w_param.device, dtype=torch.float32) @ la.to(device=w_param.device, dtype=torch.float32)
        ) * float(lora_scaling)
        w_param.data.add_(delta.to(dtype=w_param.dtype))
        merged_count += 1

    if hasattr(model, "nl_head") and hasattr(model, "nl_embed"):
        model.nl_head.weight = model.nl_embed.weight  # type: ignore[attr-defined]

    return {"merged_lora_layers": merged_count, "loaded_regular_tensors": len(regular_sd)}


class FP32MasterAdamW(torch.optim.AdamW):
    """Mixed-precision AdamW with automatic FP32 master parameter copies for BF16/FP16 model weights.
    Eliminates BF16 7-bit mantissa underflow (where w + lr*step rounds back to w on RMSNorm & Linear weights)
    while keeping forward/backward weights in BF16 and preserving tied parameter references in-place!
    """

    def __init__(
        self,
        param_groups: List[Dict[str, object]],
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        use_fp32_master: bool = True,
    ) -> None:
        self._model_to_master: List[Tuple[nn.Parameter, nn.Parameter]] = []
        self._all_model_params: List[nn.Parameter] = []
        seen_ids: set[int] = set()
        master_groups: List[Dict[str, object]] = []

        for grp in param_groups:
            orig_params: List[nn.Parameter] = list(grp["params"])  # type: ignore[arg-type]
            new_params: List[nn.Parameter] = []
            for p in orig_params:
                if id(p) in seen_ids:
                    continue
                seen_ids.add(id(p))
                self._all_model_params.append(p)
                if use_fp32_master and p.dtype in (torch.bfloat16, torch.float16):
                    p_m = nn.Parameter(p.detach().float())
                    self._model_to_master.append((p, p_m))
                    new_params.append(p_m)
                else:
                    new_params.append(p)
            if new_params:
                g_copy = dict(grp)
                g_copy["params"] = new_params
                master_groups.append(g_copy)

        super().__init__(master_groups, betas=betas, eps=eps)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        for p_model, p_master in self._model_to_master:
            if p_master.device != p_model.device:
                p_master.data = p_master.data.to(p_model.device)
            if p_model.grad is not None:
                p_master.grad = p_model.grad.detach().float()
            else:
                p_master.grad = None
        loss = super().step(closure)
        for p_model, p_master in self._model_to_master:
            p_model.data.copy_(p_master.data.to(p_model.dtype))
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p_model in self._all_model_params:
            if p_model.grad is not None:
                if set_to_none:
                    p_model.grad = None
                else:
                    p_model.grad.zero_()
        super().zero_grad(set_to_none=set_to_none)


def build_five_group_optimizer(
    model: nn.Module,
    stage: str = "B",
    is_lora_hybrid: bool = False,
    use_fp32_master: bool = True,
) -> Tuple[torch.optim.Optimizer, Dict[str, Dict[str, float]]]:
    """Master Manual §3.2 exact 5-Group Heterogeneous AdamW (beta1=0.9, beta2=0.95) + FP32 Master Weights:
      - Group A: GQA, Shared Micro-Experts, DenseSwiGLU L0/L35, V_nl, Norms (LoRA gets x5 LR!)
      - Group B: Spectral-Clustered Routed Full-Rank Experts
      - Group C: Factored FSQ, Copy-Gate, Router, probe_u, Attn-PML, 4D Dir/FiLM, Hyper-Router, Engram proj/conv
      - Group D: Bipolar Graphs, tau_0/w_t/w_e/w_u/b_stat/lambda_pos/lambda_neg scalars (STRICT FP32, WD=0.0!)
      - Group E: Static Engram sparse hash tables (5x Group C LR, WD=0.0!)
    """
    stage_lrs = {
        "B": {"A": 3e-5, "B": 5e-5, "C": 5e-4, "D": 1.5e-4, "E": 2.5e-3},
        "C": {"A": 2e-5, "B": 8e-5, "C": 3e-4, "D": 1.0e-4, "E": 1.5e-3},
        "D": {"A": 5e-6, "B": 8e-6, "C": 1.5e-5, "D": 5e-6, "E": 5e-5},
    }
    lrs = stage_lrs[stage]

    # Freeze copy_gate when disabled in config so dead parameters do not allocate optimizer state
    cfg = getattr(model, "cfg", None)
    if cfg is not None and not getattr(cfg, "enable_copy_gate", False) and hasattr(model, "copy_gate"):
        for p_cg in model.copy_gate.parameters():  # type: ignore[union-attr]
            p_cg.requires_grad_(False)

    buckets: Dict[str, List[nn.Parameter]] = {"A": [], "A_lora": [], "B": [], "C_emb": [], "C": [], "D": [], "E": []}

    group_d_tokens = (
        ".graph.",
        ".b_stat",
        ".tau_0",
        ".w_t",
        ".w_e",
        ".w_u",
        ".lambda_pos",
        ".lambda_neg",
        ".alpha_calm",
        ".alpha_vres",
        ".alpha_span_tag",
        ".logit_scale",
    )
    group_c_emb_tokens = (
        "sym_embed_head.",
        "chunk_slot_embed.",
        "block_idx_embed.",
        "in_span_tag.",
        "sym_final_norm.",
    )
    group_c_tokens = (
        "copy_gate.",
        "sc_proj.",
        ".pml.",
        ".film_proj.",
        ".film_tau.",
        ".static_router.",
        ".probe_u",
        ".w_lk.",
        "dir_head.",
        "hyper_router.",
        "engrams.",
    )

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "engrams." in name and ".tables." in name:
            buckets["E"].append(p)
        elif any(tok in name for tok in group_d_tokens):
            # Force Group D manifold/threshold scalars to strict FP32!
            if p.dtype != torch.float32:
                p.data = p.data.float()
            buckets["D"].append(p)
        elif ".routed_experts." in name:
            buckets["B"].append(p)
        elif any(tok in name for tok in group_c_emb_tokens):
            buckets["C_emb"].append(p)
        elif any(tok in name for tok in group_c_tokens):
            buckets["C"].append(p)
        elif ".lora_" in name:
            buckets["A_lora"].append(p)
        else:
            buckets["A"].append(p)

    param_groups = []
    summary: Dict[str, Dict[str, float]] = {}
    spec = [
        ("A", lrs["A"], 0.01),
        ("A_lora", lrs["A"] * (5.0 if is_lora_hybrid else 1.0), 0.01),
        ("B", lrs["B"], 0.01),
        ("C_emb", min(lrs["C"], 1.5e-4), 0.01),
        ("C", lrs["C"], 0.01),
        ("D", lrs["D"], 0.0),
        ("E", lrs["E"], 0.0),
    ]
    for key, lr_val, wd_val in spec:
        params = buckets[key]
        if params:
            param_groups.append({"name": key, "params": params, "lr": lr_val, "weight_decay": wd_val})
        summary[key] = {
            "num_tensors": float(len(params)),
            "num_params": float(sum(p.numel() for p in params)),
            "lr": lr_val,
            "weight_decay": wd_val,
        }

    opt = FP32MasterAdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, use_fp32_master=use_fp32_master)
    return opt, summary


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.10, min_lr_ratio: float = 0.10
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def _lr_fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_fn)
