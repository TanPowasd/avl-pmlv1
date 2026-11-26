"""Mandatory Test #2: TF32-Off Bit-Level Equivalence (< 1e-5) Between Training Scan & Inference Block Commit."""

from __future__ import annotations

from pathlib import Path
import torch

from avl_pmlv1.config import AVLPMLConfig
from avl_pmlv1.diffusion.wet_cement import build_block_diffusion_attn_mask
from avl_pmlv1.engine.model import AVLPMLModel
from avl_pmlv1.vocab.dsl_compiler import LinearScanDSLCompiler, MicroStepIR
from avl_pmlv1.vocab.sym_vocab import CognitivePhase, SymbolicVocab


def test_tf32_off_train_vs_inference_commit_equivalence() -> None:
    """Disable TF32 and verify multi-block training sequential scan matches block-by-block inference commit (< 1e-5)."""
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        root = Path(__file__).resolve().parents[1]
        cfg = AVLPMLConfig.from_yaml(root / "configs" / "model_40m.yaml")
        torch.manual_seed(2026)
        model = AVLPMLModel(cfg).float()
        model.eval()

        compiler = LinearScanDSLCompiler(cfg, model.vocab)
        # Build 3 blocks x 16 slots = 48 tokens with multiple intra-chunk and cross-chunk ⟨BIND⟩ writes
        steps = [
            MicroStepIR("⟨HYP⟩", in_refs=["⟨IN_0⟩"], fsq_pairs=[(11, 2), (15, 3)], lit_tokens=["1"], out_var="$a"),
            MicroStepIR("⟨DEDUCE⟩", var_deps=["$a"], fsq_pairs=[(22, 1)], lit_tokens=["+", "2"], out_var="$b"),
            MicroStepIR("⟨DEDUCE⟩", var_deps=["$b"], fsq_pairs=[(33, 4), (44, 1)], lit_tokens=["*", "3"], out_var="$c"),
            MicroStepIR("⟨DEDUCE⟩", var_deps=["$a", "$c"], fsq_pairs=[(55, 2)], lit_tokens=["-", "1"], out_var="$d"),
            MicroStepIR("⟨COMMIT⟩", var_deps=["$d"], fsq_pairs=[(66, 0)], lit_tokens=["7"], out_var="$e", flag="⟨EOT⟩"),
        ]
        comp = compiler.compile(steps, final_live_vars=["$e"])
        pad_len = (16 - (len(comp.token_ids) % 16)) % 16
        seq = comp.token_ids + [SymbolicVocab.PAD_ID] * pad_len
        rmask = comp.reassert_mask + [0] * pad_len
        total_len = len(seq)
        num_chunks = total_len // 16

        tokens_t = torch.tensor([seq], dtype=torch.long)
        rmask_t = torch.tensor([rmask], dtype=torch.long)

        # Path A: Full-sequence parallel block-causal training pass (using shared sequential scan kernel)
        wb_train = model.init_whiteboard_states(1, torch.device("cpu"), dtype=torch.float32)
        exp_shared = model.init_experience_state(1, torch.device("cpu"), dtype=torch.float32)
        block_mask = build_block_diffusion_attn_mask([16] * num_chunks, wet_seam_flags=[False] * num_chunks)

        with torch.no_grad():
            out_train = model.forward_symbolic(
                tokens_t,
                phase=CognitivePhase.FORMAL,
                attn_mask=block_mask,
                wb_states=wb_train,
                exp_state=exp_shared,
                t_k=0.0,
                reassert_flags=rmask_t,
                write_to_draft=False,
            )

        # Path B: Inference frame-by-frame atomic boundary commit on PML Track-1 modules
        wb_infer = model.init_whiteboard_states(1, torch.device("cpu"), dtype=torch.float32)
        # Feed identical per-chunk normalized inputs into PML Track-1 frame-by-frame vs full-sequence
        x_test = torch.randn(1, total_len, cfg.hidden_size, dtype=torch.float32)
        pml_mod = model.attn_layers[cfg.pml_layer_indices[0]].pml
        assert pml_mod is not None

        wb_full_scan = model.init_whiteboard_states(1, torch.device("cpu"), dtype=torch.float32)[0]
        wb_step_commit = model.init_whiteboard_states(1, torch.device("cpu"), dtype=torch.float32)[0]

        with torch.no_grad():
            h_full, u_epi_full, u_ac_full = pml_mod.sequential_scan_forward(
                x_test, tokens_t, wb_full_scan, reassert_flags=rmask_t, write_to_draft=False
            )

            h_parts, epi_parts, ac_parts = [], [], []
            prev_tok = None
            for c in range(num_chunks):
                s, e = c * 16, (c + 1) * 16
                hc, epic, acc = pml_mod.sequential_scan_forward(
                    x_test[:, s:e, :],
                    tokens_t[:, s:e],
                    wb_step_commit,
                    reassert_flags=rmask_t[:, s:e],
                    write_to_draft=True,
                    prev_last_token=prev_tok,
                )
                prev_tok = tokens_t[:, e - 1]
                wb_step_commit.lazy_commit_previous_tail()
                h_parts.append(hc)
                epi_parts.append(epic)
                ac_parts.append(acc)

            h_infer = torch.cat(h_parts, dim=1)
            u_epi_infer = torch.cat(epi_parts, dim=1)
            u_ac_infer = torch.cat(ac_parts, dim=1)

        max_state_err = (wb_full_scan.s_rw_commit - wb_step_commit.s_rw_commit).abs().max().item()
        max_h_err = (h_full - h_infer).abs().max().item()
        max_epi_err = (u_epi_full - u_epi_infer).abs().max().item()
        max_ac_err = (u_ac_full - u_ac_infer).abs().max().item()

        assert max_state_err < 1e-5, f"State mismatch {max_state_err} >= 1e-5"
        assert max_h_err < 1e-5, f"PML output mismatch {max_h_err} >= 1e-5"
        assert max_epi_err < 1e-5, f"Yellow light mismatch {max_epi_err} >= 1e-5"
        assert max_ac_err < 1e-5, f"Red light mismatch {max_ac_err} >= 1e-5"
        assert torch.isfinite(out_train["logits"]).all()  # type: ignore[arg-type]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
