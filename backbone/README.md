# Backbone

Vendored (not a submodule — `.git` removed, tracked directly in this repo)
copy of an external SSL backbone under consideration as RAD-DINO's
replacement for CXR conditioning.

- `CheXWorld/` — originally from https://github.com/LeapLabTHU/CheXWorld,
  commit `090102758801dc097f53c49d135b835570c8d173`. Self-supervised
  (I-JEPA-style) ViT-Base/16 pretrained on MIMIC+NIH+CheXpert (the same
  combined dataset RCT trains on). No CLS token — patch-tokens-only
  output. `weights/chexworld_pretrained.tar` (gitignored, 700MB) is the
  pretrain checkpoint (epoch 229, `encoder`/`predictor`/`target_encoder`
  state dicts).

See `docs/superpowers/specs/` for RCT-specific adaptation specs.
