"""Block rendering uses the original cell renderer shared with Orb."""
from dataclasses import replace
from .orb import PRESETS, OrbConfig, render_batch, decode_batch

def block_config(patch_size=8):
    return replace(PRESETS["orb36_p8"], content="block", patch_px=patch_size)
