"""Stage 2 (M3) Activation-Aware Co-Firing Spectral Upcycling Surgery (Qwen3-4B-Base -> AVL-PMLv1.2-4.5B).

Memory-safe (< 2.0 GB GPU VRAM, single-copy BF16 CPU RAM) layer-by-layer surgical transplant:
  1. Loads Qwen/Qwen3-4B-Base safetensors from E:/avl-list/hf_cache.
  2. Preserves V_nl (151,936 tied embed/head), 36x Left-Brain GQA (+ per-head q_norm/k_norm, rope_theta=1e6),
     and L0/L35 DenseSwiGLU bit-for-bit.
  3. Exact zero-waste neuron-for-neuron partition of all 34 MoE layers (L1..L34):
       9,728 original SwiGLU neurons == 512 Activation-Energy Shared + 8 x 1,152 Co-Firing Spectral Routed!
  4. Closed-form GPU Ridge-Regression static_router initialization (targeting exact per-expert alignment to
     y_orig - y_shared so Top-2 active experts fire cleanly at Step 0 above tau_0!) + GPU SVD probe_u (rank=16)
     + exact closed-form least-squares alpha_upcycle calibration!
  5. Cold-start grafting: alpha_calm=0.08, Track-1 W_O std=0.01, StaticEngram out_proj=0, pure relative RoPE.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.engine.model import AVLPMLModel
from avl_pmlv1.moe_track2.tri_state_moe import DenseSwiGLU, TriStateMoELayer
from avl_pmlv1.vocab.sym_vocab import CognitivePhase


class ShardedSafetensorReader:
    """Memory-mapped on-demand tensor loader for multi-shard HuggingFace safetensors."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            idx = json.load(open(index_path, "r", encoding="utf-8"))
            self.weight_map: Dict[str, str] = idx["weight_map"]
        else:
            single = model_dir / "model.safetensors"
            with safe_open(str(single), framework="pt", device="cpu") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._handles: Dict[str, object] = {}

    def get(self, key: str, device: str = "cpu") -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
        t = self._handles[shard].get_tensor(key)  # type: ignore[attr-defined]
        return t.to(device) if device != "cpu" else t

    def close(self) -> None:
        self._handles.clear()


@torch.no_grad()
def activation_aware_spectral_partition(
    x_flat: torch.Tensor,  # [N_tok, 2560] BF16 on GPU
    w_gate: torch.Tensor,  # [9728, 2560] BF16 on GPU
    w_up: torch.Tensor,    # [9728, 2560] BF16 on GPU
    w_down: torch.Tensor,  # [2560, 9728] BF16 on GPU
    n_shared: int,
    n_experts: int,
    n_per_expert: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Exact 9728 -> 512 activation-energy shared + 8 x 1152 co-firing spectral-clustered routed experts!"""
    n_tot = w_gate.shape[0]
    assert n_shared + n_experts * n_per_expert == n_tot

    # 1. Exact SwiGLU per-neuron output contribution matrix A [N_tok, 9728]
    #    h_u(t) = SiLU(x_t @ w_gate_u) * (x_t @ w_up_u); output energy = |h_u(t)| * ||w_down[:, u]||_2
    h_act = F.silu(F.linear(x_flat, w_gate)) * F.linear(x_flat, w_up)  # [N_tok, 9728] BF16
    down_norm = w_down.float().norm(dim=0)                              # [9728] FP32
    contrib = h_act.float().abs() * down_norm.unsqueeze(0)              # [N_tok, 9728] FP32

    # Top-512 highest RMS activation-energy universal backbone neurons -> Slim Shared Micro-Expert!
    neuron_energy = contrib.pow(2).mean(dim=0)                          # [9728]
    order = torch.argsort(neuron_energy, descending=True)
    shared_idx = order[:n_shared].sort().values
    rem_idx = order[n_shared:]                                          # [9216]

    # 2. Co-firing signature: each neuron represented by its normalized token-activation profile [9216, N_tok]
    #    Neurons that fire on the same tokens get grouped into the SAME expert!
    act_prof = F.normalize(contrib.index_select(1, rem_idx).t(), dim=-1, eps=1e-8)  # [9216, N_tok]
    # Add light static gate direction prior for rare/untriggered neurons
    g_prof = F.normalize(w_gate.float().index_select(0, rem_idx), dim=-1, eps=1e-8) # [9216, 2560]
    feat = torch.cat([act_prof, 0.35 * g_prof], dim=-1)

    u_spec, _, _ = torch.svd_lowrank(feat, q=20, niter=4)
    emb = F.normalize(u_spec[:, :16], dim=-1, eps=1e-8)  # [9216, 16]

    # 3. Spherical K-Means++ (15 iters on GPU)
    gen = torch.Generator(device=feat.device).manual_seed(seed)
    n_rem = emb.shape[0]
    c_indices = [int(torch.randint(0, n_rem, (1,), generator=gen, device=feat.device).item())]
    for _ in range(1, n_experts):
        max_sim = torch.matmul(emb, emb[c_indices].t()).max(dim=-1).values
        c_indices.append(int(torch.argmin(max_sim).item()))
    centroids = emb[c_indices].clone()

    for _ in range(15):
        assign = torch.matmul(emb, centroids.t()).argmax(dim=-1)
        for e in range(n_experts):
            mask = assign == e
            if mask.any():
                centroids[e] = F.normalize(emb[mask].mean(dim=0), dim=-1, eps=1e-8)

    # 4. Exact capacity-balanced assignment (exact 1,152 neurons per expert, highest-margin specialists first!)
    sims = torch.matmul(emb, centroids.t()).cpu()
    top2_v, _ = torch.topk(sims, k=2, dim=-1)
    margins = top2_v[:, 0] - top2_v[:, 1]
    prio_order = torch.argsort(margins, descending=True).tolist()
    pref_rank = torch.argsort(sims, dim=-1, descending=True).tolist()
    rem_cpu = rem_idx.cpu().tolist()

    buckets: List[List[int]] = [[] for _ in range(n_experts)]
    for pos in prio_order:
        for cand_e in pref_rank[pos]:
            if len(buckets[cand_e]) < n_per_expert:
                buckets[cand_e].append(rem_cpu[pos])
                break

    expert_indices = [torch.tensor(sorted(b), dtype=torch.long, device=w_gate.device) for b in buckets]
    assert all(len(idx) == n_per_expert for idx in expert_indices)
    return shared_idx, expert_indices


@torch.no_grad()
def fit_ridge_router(
    x_flat: torch.Tensor,       # [N_tok, D] FP32/BF16 on GPU
    y_rem_flat: torch.Tensor,   # [N_tok, D] exact y_orig - y_shared on GPU
    exp_outs: List[torch.Tensor],  # E x [N_tok, D] on GPU
    ridge_lambda: float = 10.0,
) -> torch.Tensor:
    """Closed-form Ridge Regression W_router [E, D] predicting each expert's alignment to (y_orig - y_shared)!"""
    xf = x_flat.float()
    yf = y_rem_flat.float()
    y_norm = yf.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Exact per-token alignment score of expert e to residual target y_rem
    e_scores = torch.stack([(ye.float() * yf).sum(dim=-1) / y_norm.squeeze(-1) for ye in exp_outs], dim=-1)  # [N_tok, E]

    # Standardize per-token & shift so top-2 experts target ~+1.6..+2.4 (cleanly above tau_0=0.15!) and bottom experts < 0
    mu = e_scores.mean(dim=-1, keepdim=True)
    sd = e_scores.std(dim=-1, keepdim=True).clamp(min=1e-5)
    target_logits = 1.25 * (e_scores - mu) / sd + 0.35  # [N_tok, E]

    # Solve (X^T X + lambda I) W^T = X^T Y on GPU (D=2560 -> ~3 ms!)
    xtx = torch.matmul(xf.t(), xf)
    xtx.diagonal().add_(ridge_lambda)
    xty = torch.matmul(xf.t(), target_logits)  # [D, E]
    w_t = torch.linalg.solve(xtx, xty)         # [D, E]
    return w_t.t().contiguous()                # [E, D]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--hf-cache", type=str, default=r"e:\avl-list\hf_cache")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "model_4_5b.yaml"))
    parser.add_argument("--stage1-data", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "stage1" / "stage1_30k_fsq_aligned.jsonl"))
    parser.add_argument("--out-ckpt", type=str, default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "avl_pmlv12_4_5b_step0_bf16.safetensors"))
    parser.add_argument("--out-report", type=str, default=str(Path(__file__).resolve().parents[2] / "outputs" / "stage2_surgery_report.json"))
    parser.add_argument("--calib-samples", type=int, default=24)
    parser.add_argument("--preserve-dense-mlp", action="store_true", help="Keep 100%% of Qwen3-4B native 9728 DenseSwiGLU FFN unpruned (Converged Step 1).")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    Path(args.hf_cache).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.perf_counter()
    print(f"=== [Stage 2 M3 Activation-Aware Surgery] Resolving {args.base_model} ===", flush=True)
    local_snap = Path(args.hf_cache) / "models--Qwen--Qwen3-4B-Base" / "snapshots" / "906bfd4b4dc7f14ee4320094d8b41684abff8539"
    if args.base_model == "Qwen/Qwen3-4B-Base" and local_snap.exists():
        model_dir = local_snap
    elif Path(args.base_model).exists():
        model_dir = Path(args.base_model)
    else:
        model_dir = Path(
            snapshot_download(
                repo_id=args.base_model,
                cache_dir=args.hf_cache,
                allow_patterns=["*.safetensors", "*.json", "vocab.*", "merges.*", "tokenizer*"],
            )
        )
    print(f"[Cache Ready in {time.perf_counter() - t_start:.1f}s] {model_dir}", flush=True)

    # Diverse calibration batch spanning Math + Code + Logic + General English prose!
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    calib_texts: List[str] = [
        "Solve step by step: A car travels 180 kilometers in 2.5 hours. What is its average speed in km/h?",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n",
        "In propositional logic, if P implies Q and Q implies R, then by hypothetical syllogism P implies R.",
        "The European Union consists of twenty-seven member states located primarily in Europe, with Paris and Berlin.",
    ]
    if Path(args.stage1_data).exists():
        by_dom_seen = {"math": 0, "code": 0, "logic": 0}
        with open(args.stage1_data, "r", encoding="utf-8") as f:
            for line in f:
                if len(calib_texts) >= args.calib_samples:
                    break
                r = json.loads(line)
                d = r["domain"]
                if by_dom_seen.get(d, 0) < (args.calib_samples // 3):
                    calib_texts.append(r["question"][:420])
                    by_dom_seen[d] = by_dom_seen.get(d, 0) + 1
    tok_batch = tokenizer(calib_texts, return_tensors="pt", padding=True, truncation=True, max_length=96)
    calib_ids = tok_batch["input_ids"].to(device)
    calib_mask = tok_batch["attention_mask"].to(device).bool()
    bsz_c, seq_c = calib_ids.shape
    calib_pos = torch.arange(seq_c, device=device).unsqueeze(0).expand(bsz_c, -1)
    print(f"[Calibration Batch] shape={tuple(calib_ids.shape)} (non-pad tokens={int(calib_mask.sum().item())})", flush=True)

    cfg = AVLPMLConfig.from_yaml(args.config)
    if args.preserve_dense_mlp:
        cfg.moe_layer_indices = []
    print(f"[Constructing {cfg.model_name} (moe_layers={len(cfg.moe_layer_indices)}) skeleton in BF16 on CPU...]", flush=True)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = AVLPMLModel(cfg)
    torch.set_default_dtype(prev_dtype)

    for name, p in model.named_parameters():
        if any(k in name for k in (".graph.", ".b_stat", ".tau_0", ".w_t", ".w_e", ".w_u", ".lambda_pos", ".lambda_neg", ".alpha_calm", ".alpha_vres", ".alpha_span_tag", ".logit_scale")):
            p.data = p.data.float()

    reader = ShardedSafetensorReader(model_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    has_moe_surgery = len(cfg.moe_layer_indices) > 0
    with torch.no_grad():
        emb_w = reader.get("model.embed_tokens.weight").to(torch.bfloat16)
        model.nl_embed.weight.copy_(emb_w)
        model.nl_head.weight = model.nl_embed.weight
        model.final_norm.weight.copy_(reader.get("model.norm.weight").to(torch.bfloat16))
        h_qwen = F.embedding(calib_ids, emb_w.to(device)) if has_moe_surgery else None

    layer_metrics: List[Dict[str, float]] = []
    t_surg = time.perf_counter()

    with torch.no_grad():
        for l_idx in range(cfg.num_layers):
            prefix = f"model.layers.{l_idx}"
            attn_l = model.attn_layers[l_idx]
            ffn_l = model.ffn_layers[l_idx]

            attn_l.norm.weight.copy_(reader.get(f"{prefix}.input_layernorm.weight").to(torch.bfloat16))
            attn_l.gqa.q_proj.weight.copy_(reader.get(f"{prefix}.self_attn.q_proj.weight").to(torch.bfloat16))
            attn_l.gqa.k_proj.weight.copy_(reader.get(f"{prefix}.self_attn.k_proj.weight").to(torch.bfloat16))
            attn_l.gqa.v_proj.weight.copy_(reader.get(f"{prefix}.self_attn.v_proj.weight").to(torch.bfloat16))
            attn_l.gqa.o_proj.weight.copy_(reader.get(f"{prefix}.self_attn.o_proj.weight").to(torch.bfloat16))
            if cfg.use_qk_norm and attn_l.gqa.q_norm is not None and attn_l.gqa.k_norm is not None:
                attn_l.gqa.q_norm.weight.copy_(reader.get(f"{prefix}.self_attn.q_norm.weight").to(torch.bfloat16))
                attn_l.gqa.k_norm.weight.copy_(reader.get(f"{prefix}.self_attn.k_norm.weight").to(torch.bfloat16))

            if not has_moe_surgery:
                assert isinstance(ffn_l, DenseSwiGLU)
                ffn_l.norm.weight.copy_(reader.get(f"{prefix}.post_attention_layernorm.weight").to(torch.bfloat16))
                ffn_l.ffn.w1.weight.copy_(reader.get(f"{prefix}.mlp.gate_proj.weight").to(torch.bfloat16))
                ffn_l.ffn.w3.weight.copy_(reader.get(f"{prefix}.mlp.up_proj.weight").to(torch.bfloat16))
                ffn_l.ffn.w2.weight.copy_(reader.get(f"{prefix}.mlp.down_proj.weight").to(torch.bfloat16))
                layer_metrics.append({"layer": l_idx, "type": "dense", "full_sum_cos": 1.0, "shared_only_cos": 1.0, "sparse_k2_cos": 1.0, "alpha_upcycle": 1.0, "active_experts": 0.0})
                if (l_idx + 1) % 6 == 0 or l_idx == cfg.num_layers - 1:
                    print(f"  [Layer {l_idx+1:02d}/{cfg.num_layers}] dense | exact 1:1 unpruned copy | {time.perf_counter()-t_surg:.1f}s", flush=True)
                continue

            assert h_qwen is not None
            attn_l.to(device)
            h_qwen, _, _, _, _ = attn_l(
                h_qwen, token_ids=calib_ids, positions=calib_pos, wb_state=None, attn_mask=None, read_only_pml=True
            )
            attn_l.to("cpu")

            post_ln_w = reader.get(f"{prefix}.post_attention_layernorm.weight").to(torch.bfloat16)
            w_gate = reader.get(f"{prefix}.mlp.gate_proj.weight", device=str(device)).to(torch.bfloat16)
            w_up = reader.get(f"{prefix}.mlp.up_proj.weight", device=str(device)).to(torch.bfloat16)
            w_down = reader.get(f"{prefix}.mlp.down_proj.weight", device=str(device)).to(torch.bfloat16)

            ffn_l.norm.weight.copy_(post_ln_w)
            x_norm_gpu = ffn_l.norm.to(device)(h_qwen)
            y_orig = F.linear(F.silu(F.linear(x_norm_gpu, w_gate)) * F.linear(x_norm_gpu, w_up), w_down)

            if isinstance(ffn_l, DenseSwiGLU):
                ffn_l.ffn.w1.weight.copy_(w_gate.cpu())
                ffn_l.ffn.w3.weight.copy_(w_up.cpu())
                ffn_l.ffn.w2.weight.copy_(w_down.cpu())
                ffn_l.norm.to("cpu")
                h_qwen = h_qwen + y_orig
                layer_metrics.append({"layer": l_idx, "type": "dense", "full_sum_cos": 1.0, "shared_only_cos": 1.0, "sparse_k2_cos": 1.0, "alpha_upcycle": 1.0, "active_experts": 0.0})
            elif isinstance(ffn_l, TriStateMoELayer):
                # Extract non-pad calibration activations [N_tok, 2560]
                x_flat = x_norm_gpu[calib_mask]
                y_orig_flat = y_orig[calib_mask]

                shared_idx, exp_indices = activation_aware_spectral_partition(
                    x_flat,
                    w_gate,
                    w_up,
                    w_down,
                    n_shared=cfg.shared_expert_intermediate_size,
                    n_experts=cfg.num_routed_experts,
                    n_per_expert=cfg.routed_expert_intermediate_size,
                    seed=2026 + l_idx,
                )

                # Copy shared micro-expert & measure shared-only output
                wg_s = w_gate.index_select(0, shared_idx)
                wu_s = w_up.index_select(0, shared_idx)
                wd_s = w_down.index_select(1, shared_idx)
                ffn_l.shared_expert.w1.weight.copy_(wg_s.cpu())
                ffn_l.shared_expert.w3.weight.copy_(wu_s.cpu())
                ffn_l.shared_expert.w2.weight.copy_(wd_s.cpu())
                y_shared_flat = F.linear(F.silu(F.linear(x_flat, wg_s)) * F.linear(x_flat, wu_s), wd_s)
                cos_shared = float(F.cosine_similarity(y_shared_flat.float(), y_orig_flat.float(), dim=-1).mean().item())
                y_rem_flat = y_orig_flat - y_shared_flat

                # Copy routed experts, compute exact GPU SVD probe_u, & collect expert outputs for Ridge router!
                exp_outs_flat: List[torch.Tensor] = []
                for e_id, e_idx in enumerate(exp_indices):
                    wg_e = w_gate.index_select(0, e_idx)
                    wu_e = w_up.index_select(0, e_idx)
                    wd_e = w_down.index_select(1, e_idx)
                    ffn_l.routed_experts[e_id].w1.weight.copy_(wg_e.cpu())
                    ffn_l.routed_experts[e_id].w3.weight.copy_(wu_e.cpu())
                    ffn_l.routed_experts[e_id].w2.weight.copy_(wd_e.cpu())

                    u_svd, _, _ = torch.svd_lowrank(wg_e.t().float(), q=cfg.probe_rank + 4, niter=3)
                    ffn_l.probe_u.data[e_id].copy_(u_svd[:, : cfg.probe_rank].to(ffn_l.probe_u.dtype).cpu())
                    ye = F.linear(F.silu(F.linear(x_flat, wg_e)) * F.linear(x_flat, wu_e), wd_e)
                    exp_outs_flat.append(ye)

                # Closed-form Ridge-Regression Router Calibration!
                w_router = fit_ridge_router(x_flat, y_rem_flat, exp_outs_flat, ridge_lambda=15.0)
                ffn_l.static_router.weight.copy_(w_router.to(torch.bfloat16).cpu())
                # Keep tau_0 cleanly at 0.15 so ridge-calibrated top-2 experts fire above threshold!
                ffn_l.tau_0.data.fill_(0.15)

                # Evaluate exact conservation + closed-form least-squares alpha_upcycle on GPU!
                ffn_l.to(device)
                h_full_out, _ = ffn_l(h_qwen, phase=CognitivePhase.VERBALIZE, full_sum_eval=True)
                y_full = (h_full_out - h_qwen)[calib_mask]
                cos_full = float(F.cosine_similarity(y_full.float(), y_orig_flat.float(), dim=-1).mean().item())

                ffn_l.alpha_upcycle.fill_(1.0)
                h_k2_unit, aux_k2 = ffn_l(h_qwen, phase=CognitivePhase.VERBALIZE, t_k=0.0, static_top2_mode=False)
                y_routed_unit = ((h_k2_unit - h_qwen)[calib_mask]) - y_shared_flat

                num = (y_routed_unit.float() * y_rem_flat.float()).sum()
                den = y_routed_unit.float().pow(2).sum().clamp(min=1e-8)
                alpha_opt = float(torch.clamp(num / den, min=0.5, max=8.0).item())
                ffn_l.alpha_upcycle.fill_(alpha_opt)

                y_k2_cal = y_shared_flat + alpha_opt * y_routed_unit
                cos_k2 = float(F.cosine_similarity(y_k2_cal.float(), y_orig_flat.float(), dim=-1).mean().item())
                act_k2 = float(aux_k2["active_count"][calib_mask].mean().item())

                ffn_l.to("cpu")
                h_qwen = h_qwen + y_orig
                layer_metrics.append(
                    {
                        "layer": l_idx,
                        "type": "moe",
                        "full_sum_cos": round(cos_full, 6),
                        "shared_only_cos": round(cos_shared, 4),
                        "sparse_k2_cos": round(cos_k2, 4),
                        "alpha_upcycle": round(alpha_opt, 4),
                        "active_experts": round(act_k2, 2),
                    }
                )

            del w_gate, w_up, w_down, y_orig, x_norm_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if (l_idx + 1) % 6 == 0 or l_idx == cfg.num_layers - 1:
                m = layer_metrics[-1]
                print(
                    f"  [Layer {l_idx+1:02d}/{cfg.num_layers}] {m['type']:5s} | full={m['full_sum_cos']:.6f} | "
                    f"shared={m['shared_only_cos']:.4f} -> k2={m['sparse_k2_cos']:.4f} (act={m['active_experts']:.2f}, "
                    f"alpha={m['alpha_upcycle']:.3f}) | {time.perf_counter()-t_surg:.1f}s",
                    flush=True,
                )

    reader.close()
    peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
    param_summary = model.parameter_summary()

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items() if k != "nl_head.weight"}
    t_save = time.perf_counter()
    save_file(state_dict, str(out_ckpt))
    save_s = round(time.perf_counter() - t_save, 2)
    ckpt_gb = round(out_ckpt.stat().st_size / (1024**3), 2)

    moe_ms = [m for m in layer_metrics if m["type"] == "moe"]
    n_moe = len(moe_ms)
    report = {
        "model_name": cfg.model_name,
        "base_model": args.base_model,
        "base_model_dir": str(model_dir),
        "out_ckpt": str(out_ckpt),
        "ckpt_size_gb": ckpt_gb,
        "param_summary": {k: int(v) for k, v in param_summary.items()},
        "param_summary_billions": {k: round(v / 1e9, 3) for k, v in param_summary.items()},
        "surgery_sec": round(time.perf_counter() - t_surg, 2),
        "save_sec": save_s,
        "total_sec": round(time.perf_counter() - t_start, 2),
        "peak_surgery_vram_gb": peak_vram_gb,
        "mean_moe_full_sum_cos": round(sum(m["full_sum_cos"] for m in moe_ms) / n_moe, 6) if n_moe else 1.0,
        "min_moe_full_sum_cos": round(min(m["full_sum_cos"] for m in moe_ms), 6) if n_moe else 1.0,
        "mean_moe_shared_only_cos": round(sum(m["shared_only_cos"] for m in moe_ms) / n_moe, 4) if n_moe else 1.0,
        "mean_moe_sparse_k2_cos": round(sum(m["sparse_k2_cos"] for m in moe_ms) / n_moe, 4) if n_moe else 1.0,
        "mean_active_experts": round(sum(m["active_experts"] for m in moe_ms) / n_moe, 2) if n_moe else 0.0,
        "mean_alpha_upcycle": round(sum(m["alpha_upcycle"] for m in moe_ms) / n_moe, 4) if n_moe else 1.0,
        "layer_metrics": layer_metrics,
    }
    out_rep = Path(args.out_report)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== [Stage 2 M3 Activation-Aware Qwen3-4B Surgery Complete] ===", flush=True)
    print(f"Params: {report['param_summary_billions']} | Ckpt: {ckpt_gb} GB | Peak VRAM: {peak_vram_gb} GB", flush=True)
    print(
        f"Full-Partition Cos={report['mean_moe_full_sum_cos']:.6f} (min={report['min_moe_full_sum_cos']:.6f}) | "
        f"Shared-Only Cos={report['mean_moe_shared_only_cos']:.4f} -> Sparse-K2 Cos={report['mean_moe_sparse_k2_cos']:.4f} "
        f"(active={report['mean_active_experts']:.2f}/2, mean alpha_upcycle={report['mean_alpha_upcycle']:.3f})",
        flush=True,
    )
    gc.collect()


if __name__ == "__main__":
    main()
