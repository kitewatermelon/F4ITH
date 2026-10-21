# Verbatim copies of backbone/CheXWorld/models/{jepa_vit,jepa_vit_3d,pos_embed,utils}.py.
# backbone/CheXWorld/models/__init__.py has a broken import chain (imports
# unet_adapter_conv -> a missing models.dinov2 submodule) unrelated to what
# this repo needs, so these 4 files are copied here instead of imported
# from backbone/CheXWorld/ directly. If backbone/CheXWorld/models/ is ever
# updated upstream, these copies will NOT automatically pick up the change
# — re-copy manually if needed.
