# AVL-PMLv1 (`avl-pmlv1`)

**Pure-Forward Dual-Track Symbolic & Native-NL Block-Diffusion Architecture with Hard-Indexed PML Whiteboard, Static Engram, and Adaptive Tri-State MoE (Research Branch)**

Built on `Qwen/Qwen3-4B-Base`, **AVL-PMLv1.2** investigates decoupling structured multi-step reasoning from surface verbalization via:

- **Left-Brain Full GQA ∥ Right-Brain Track-1 PML Whiteboard**: 8 hybrid Attn-PML layers (`l ∈ {4, 8, 12, 16, 20, 24, 28, 32}`) pairing full GQA with a pure-forward read-only fact core (`S^RO`) and explicit hard-indexed event whiteboard (`S^RW`).
- **Dual-Stream BD3-LM 8-Slot Block Diffusion**: Parallel 8-slot micro-block denoising with exact Prefix-KV caching, wet-cement seam refinement (`C_wet = 16`), and factor-isolated comparison across both ternary symbolic vocabulary (`V_sym`) and native `Qwen3` vocabulary (`V_nl`).
- **Three-Phase Thinking (`Pre → Formal → Post`)**: Explicit skeleton planning, structured derivation, and counterfactual verification/self-correction before final answer verbalization.
- **Anti-`[MASK]` Static Engram & Adaptive Top-K MoE (Research Branch)**: Prime-bucket static lookup tables (`l ∈ {4, 16}`) and Track-2 bipolar association graphs (`M_syn / M_adv`) retained alongside the converged **4.111B Dense FFN mainline** (`configs/model_4b_dense_pml.yaml`).
- **Single-GPU `Q-AVL-PML` Training + Pure-BF16 Merged Eval**: 4-bit NF4 LoRA (`r=16`, FP32 master optimizer) training within 16 GB VRAM (RTX 5080), folded back into pure native BF16 weights (`merge_lora_delta_into_bf16_model`) for zero-quantization-overhead inference.

## Repository Structure

```text
.
├── docs/
│   ├── avl_pml_v1_master_manual.md      # Architecture & engineering master manual (ZH)
│   └── avl_pml_v1_math_stage0.md        # Formal math interfaces & Stage 0 spec (ZH)
└── avl-pmlv1/
    ├── pyproject.toml                   # Python package metadata & pytest config
    ├── configs/                         # Model configs (40M, 4B-Dense mainline, 4.5B/8.9B MoE)
    ├── src/avl_pmlv1/
    │   ├── config.py                    # Unified architecture configuration
    │   ├── engine/                      # Core model, four-phase runner, Q-AVL-PML, native V_nl Block-dLLM
    │   ├── pml_track1/                  # Attn-PML hybrid layers & hard-indexed whiteboard (S^RO / S^RW)
    │   ├── moe_track2/                  # Tri-state MoE, Hyper-Router, bipolar association graph
    │   ├── diffusion/                   # 4D direction vector, wet cement, post-surgery
    │   ├── engram/                      # Anti-[MASK] Static Engram
    │   └── vocab/                       # V_sym codec, FSQ, DSL compiler, native V_nl Three-Think formatter
    ├── scripts/
    │   ├── stage0_falsify/              # 40M Stage 0 Gate 1 falsification & hard-stress suites
    │   ├── stage1_data/                 # Mega corpus seed fetching, DSL compilation, FSQ & error sets
    │   ├── stage2_surgery/              # Qwen3-4B lossless transplant & M3 gate verification
    │   └── stage3_cloud/                # Gate 2 controlled multi-arm training & pure-BF16 evaluation
    ├── tests/                           # Unit & regression tests (pytest)
    └── outputs/                         # Experiment JSON/Markdown reports (checkpoints & datasets gitignored)
```

## Quick Start

```bash
# Create virtual environment and install in editable mode
cd avl-pmlv1
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Run unit & regression test suite
pytest
```

> **Note on Weights & Datasets**: Model checkpoints (`checkpoints/`, `*.safetensors`, `*.pt`), Hugging Face cache (`hf_cache/`), and raw/compiled corpora (`avl-pmlv1/data/`) are intentionally excluded from Git. Use `scripts/stage1_data/` and `scripts/stage2_surgery/` to reproduce the datasets and Qwen3-4B transplant checkpoints locally.

## Project Status

- **Stage 0 (40M Falsification / Gate 1)**: Completed — P0 regression fixes verified, Prefix-KV parity & hard-indexed PML stress tests green.
- **Stage 1 (Data & DSL Compilation)**: Completed — Mega Math + Logic DSL corpus compiled and split into train/val/blind-test.
- **Stage 2 (Qwen3-4B Dense Transplant / M3)**: Completed — Converged `4.111B` unpruned dense FFN + 8-layer Attn-PML transplant passed M3 verification.
- **Stage 3 (Gate 2 Controlled Multi-Arm Matrix, 2,000 steps / 16k examples per arm)**: Ongoing on strictly unseen validation slices (`Val40` GSM8K+ProofWriter+K&K & `Val60` step-bucket mixed Math+Logic), comparing zero-shot base, `V_nl` AR SFT, `V_nl` Three-Think AR SFT, `V_nl` Three-Think 8-Slot Block-dLLM, `V_sym` AR (`PML OFF/ON`), and `V_sym` 8-Slot Block-dLLM (`PML OFF/ON`). See [`docs/avl_pml_v1_master_manual.md`](docs/avl_pml_v1_master_manual.md) and [`avl-pmlv1/outputs/`](avl-pmlv1/outputs/) for detailed reports.
