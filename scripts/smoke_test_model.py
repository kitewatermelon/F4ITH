"""
Smoke test: instantiate CrossAttnJiT_S_16 and CrossAttnJiT_B_16 directly
(no RAD-DINO download — dummy context tensors), for both conditioning
modes ("cls_only", "patch_only"), verify forward output shape, the
presence/absence of the cross-attention sublayer and its zero-init
output projection, and print parameter counts.

Run from the RCT repo root: python scripts/smoke_test_model.py
"""
import torch

from rct.model import CrossAttnJiT_S_16, CrossAttnJiT_B_16


def check_variant(name, ctor, hidden_size, num_ctx=256, ctx_dim=768):
    B = 2
    for mode in ("cls_only", "patch_only"):
        model = ctor(input_size=224, conditioning_mode=mode)
        model.eval()

        x = torch.randn(B, 3, 224, 224)
        t = torch.rand(B)
        cond_emb = torch.randn(B, hidden_size) if mode == "cls_only" else None
        patch_tokens = torch.randn(B, num_ctx, ctx_dim) if mode == "patch_only" else None

        with torch.no_grad():
            out = model(x, t, cond_emb, patch_tokens)
        assert out.shape == x.shape, f"{name} [{mode}]: unexpected output shape {out.shape}"

        for i, block in enumerate(model.blocks):
            if mode == "patch_only":
                assert torch.all(block.cross_attn.out_proj.weight == 0), \
                    f"{name} [{mode}] block {i}: cross_attn.out_proj.weight not zero-init"
                assert torch.all(block.cross_attn.out_proj.bias == 0), \
                    f"{name} [{mode}] block {i}: cross_attn.out_proj.bias not zero-init"
            else:
                assert not hasattr(block, "cross_attn"), \
                    f"{name} [{mode}] block {i}: unexpected cross_attn sublayer in cls_only mode"

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"OK: {name} [{mode}] out={tuple(out.shape)} total_params={total:,} trainable_params={trainable:,}")


check_variant("S_16", CrossAttnJiT_S_16, hidden_size=384)
check_variant("B_16", CrossAttnJiT_B_16, hidden_size=768)
