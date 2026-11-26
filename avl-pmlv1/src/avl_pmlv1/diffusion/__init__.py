"""Diffusion control modules: 4D direction vector, 16-slot wet cement, and 4-fold PLL surgery."""

from .dir_vector import DirectionVectorHead
from .wet_cement import (
    select_wet_cement_remask_slots,
    build_block_diffusion_attn_mask,
    hierarchical_three_beat_unmask,
)
from .post_surgery import build_4fold_complementary_masks, PostThinkingSurgeryGuard

__all__ = [
    "DirectionVectorHead",
    "select_wet_cement_remask_slots",
    "build_block_diffusion_attn_mask",
    "hierarchical_three_beat_unmask",
    "build_4fold_complementary_masks",
    "PostThinkingSurgeryGuard",
]
