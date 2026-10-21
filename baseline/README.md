# Baselines

Vendored (not submodules — `.git` removed, tracked directly in this repo)
copies of external baselines, adapted to condition on RAD-DINO and train on
the combined MIMIC-CXR/NIH/CheXpert dataset for comparison against RCT.

- `RCDM/` — originally from https://github.com/facebookresearch/RCDM,
  commit `71daaf10a73bb2012864f0827c68d209fc92b0a5`. Pooled/global
  conditioning (adaLN-style), analogous to RCT's `cls_only` mode.
- `FeatInv/` — originally from https://github.com/AI4HealthUOL/FeatInv,
  commit `877c8d99b3f9df48825f68d65d018834b6aae82a`. Dense/spatial feature
  conditioning (ControlNet-based), analogous to RCT's `patch_only` mode.

See each subdirectory's own README for the original project's documentation
and citation. RCT-specific adaptations are documented in
`docs/superpowers/specs/` (see specs with `-rcdm-` or `-featinv-` in the name).
