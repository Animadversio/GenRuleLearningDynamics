# Landscape dynamics animation

[Open the original animation at full resolution](landscape-dynamics.gif).

This is the original research GIF, copied byte-for-byte; no fields or labels have been regenerated. It contains 16 training checkpoints, from ep 49 to ep 701704, for `DiT_mini_parity_N4096_D36_G3_even_rep2`.

The plane passes through a training sample, a valid novel sample obtained by flipping two bits in one parity group, and an invalid sample obtained by flipping one bit. Its axes use Euclidean distance. Columns use noise scales 0.2, 0.5, 1.0, and 2.0. The upper row colors the score norm; the lower row colors the denoiser component along the training-to-novel direction. The lower arrows show the denoising displacement, parallel to the score. Color and arrow scales adapt between panels/checkpoints, so compare geometry rather than interpreting identical colors as identical magnitudes. This is a two-dimensional slice, not the full probability landscape.

Research sources:

- Original asset: `DiffusionAttnConsistency/figures/vector_field/vf_v3_animation.gif`.
- [Plane construction and `plot_ckpt` (v3) plotting code](https://github.com/Animadversio/DiffusionAttnConsistency/blob/llm-switch-rule/notebooks/20260429_G3_vector_field_three_sample_plane.ipynb).
- [Field evaluation helpers](https://github.com/Animadversio/DiffusionAttnConsistency/blob/llm-switch-rule/core/vector_field_lib.py).

The notebook contains the underlying field plotting code; the historical GIF assembly command has not been located. Reusing this saved asset does not rerun model inference.
