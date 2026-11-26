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
    │   ├── engine/                      # Core model, four-phase runner, Q-AVL-PML, V_nl Block-dLLM & Backward Revision
    │   ├── pml_track1/                  # Attn-PML hybrid layers & hard-indexed whiteboard (S^RO / S^RW)
    │   ├── moe_track2/                  # Tri-state MoE, Hyper-Router, bipolar association graph
    │   ├── diffusion/                   # 4D direction vector, wet cement, post-surgery
    │   ├── engram/                      # Anti-[MASK] Static Engram
    │   └── vocab/                       # V_sym codec, FSQ, DSL compiler, native V_nl Three-Think formatter
    ├── scripts/
    │   ├── stage0_falsify/              # 40M Stage 0 Gate 1 falsification & hard-stress suites
    │   ├── stage1_data/                 # Mega corpus seed fetching, DSL compilation, FSQ & error sets
    │   ├── stage2_surgery/              # Qwen3-4B lossless transplant & M3 gate verification
    │   └── stage3_cloud/                # Gate 2 multi-arm training, Stage-C backward revision & pure-BF16 eval
    ├── tests/                           # 30 unit & regression tests (pytest 30/30 passing)
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

# Run unit & regression test suite (30/30 passing)
pytest
```

> **Note on Weights & Datasets**: Model checkpoints (`checkpoints/`, `*.safetensors`, `*.pt`), Hugging Face cache (`hf_cache/`), and raw/compiled corpora (`avl-pmlv1/data/`) are intentionally excluded from Git. Use `scripts/stage1_data/` and `scripts/stage2_surgery/` to reproduce the datasets and Qwen3-4B transplant checkpoints locally.

## Project Status & Gate 2 Results (4.111B Dense Mainline, Pure BF16 on RTX 5080)

All 8 controlled Gate 2 arms (`2,000` steps × `grad_accum=8` = `16,000` `N^0.5`-balanced math+logic examples, `seed=2026`) and Stage-C Backward Revision (`600` steps) have completed pure-BF16 evaluation on **100 strictly unseen validation problems** (`Val40`: 20 GSM8K + 10 ProofWriter + 10 K&K; `Val60`: step-bucket mixed Math+Logic; zero blind-test contamination):

| Arm | Vocab / Paradigm | Val40 EM | Val60 EM | Combined (n=100) | GSM8K (n=24) | ProofWriter (n=26) | Avg E2E (Val40/60) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `nl_cot_ref` | Native `V_nl` Zero-Shot AR | 30.0% | 10.0% | 18.0% | 58.3% | 0.0% | 9.22s / 9.80s |
| `nl_cot_sft` | Native `V_nl` 1-Phase AR SFT | 47.5% | 41.7% | 44.0% | 79.2% | 69.2% | 7.63s / 10.45s |
| **`nl_three_think_sft`** | Native `V_nl` Three-Think AR SFT | **55.0%** | 35.0%* | 43.0% | **87.5%** | 61.5% | 9.79s / 9.55s |
| **`nl_three_think_dllm`** (`t=384`) | Native `V_nl` Three-Think 8-Slot dLLM | 42.5% | 45.0% | 44.0% | 33.3% | **100.0%** | 4.64s / 6.18s |
| `ar_pml_off` | `V_sym` AR (PML OFF) | 27.5% | 35.0% | 32.0% | 12.5% | 84.6% | 8.14s / 9.50s |
| `ar_pml_on` | `V_sym` AR (PML ON) | 35.0% | 31.7% | 33.0% | 20.8% | 69.2% | 8.56s / 12.42s |
| `block_dllm_pml_off` | `V_sym` 8-Slot dLLM (PML OFF) | 40.0% | 50.0% | 46.0% | 29.2% | 92.3% | 2.94s / 5.27s |
| **`block_dllm_pml_on`** | `V_sym` 8-Slot dLLM (PML ON) | 37.5% | **60.0%** | **51.0%** | 20.8% | **100.0%** | **2.88s / 4.42s** |

- **Stage-C Backward Revision (`nl_three_think_dllm_rev`, `t=384`)**: Online **16-Slot Wet-Cement** delivers **`+5.0%` net gain on BOTH Val40 (`27.5% → 32.5%`) and Val60 (`43.3% → 48.3%`, Math `+9.3%`)** (`Fix/Brk/Net = 11/6/+5` across 100 problems, `+0.76s–1.46s` overhead), while 4-fold bidirectional masked accuracy jumped `69.0% → 92.3%`.
- See [`docs/avl_pml_v1_master_manual.md`](docs/avl_pml_v1_master_manual.md) and [`avl-pmlv1/outputs/`](avl-pmlv1/outputs/) for full factor-isolated analysis and reports.
