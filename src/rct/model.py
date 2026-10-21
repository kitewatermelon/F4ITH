import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from rct.rope_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """ Embeds scalar timesteps into vector representations. """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    """Thin wrapper around PyTorch's fused SDPA (flash-attention / memory-
    efficient backend on supported hardware). Never materializes the full
    (L, S) attention matrix, unlike a naive matmul+softmax implementation --
    this matters a lot at longer sequence lengths (e.g. 256 patch tokens),
    where that matrix was previously kept in float32 for backward across
    every block. Numerically, fused SDPA kernels accumulate softmax in
    float32 internally even when inputs are bf16, so this isn't a precision
    regression relative to the old explicit .float() cast.
    """
    return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """ The final layer of CrossAttnJiT.

    use_adaln=True (default): DiT-style, shift/scale modulation driven by `c`.
    use_adaln=False (t_embed_mode="token"): plain pre-norm — timestep already
      reached every token via the prepended time token, so no adaLN needed here.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c):
        if self.use_adaln:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x


class CrossAttnJiTBlock(nn.Module):
    """
    self-attention (RoPE, adaLN-modulated)
      -> [cross-attention (plain pre-norm, zero-init output projection) -- patch_only mode only]
      -> MLP (adaLN-modulated)

    conditioning_mode="cls_only": no cross-attention sublayer at all; the frozen
      encoder's pooled embedding drives adaLN (via `c`) for self-attention and MLP.
    conditioning_mode="patch_only": a cross-attention sublayer attends to
      the frozen encoder's dense patch tokens (`context`); adaLN (via `c`) is driven
      by timestep alone, not the encoder's pooled token.
    conditioning_mode="cls_cross": the same cross-attention sublayer, but
      `context` is a single pooled-CLS memory token (B, 1, H); adaLN is driven
      by timestep alone (and typically disabled via t_embed_mode="token").

    t_embed_mode="adaln" (default): the above — every block is adaLN-Zero
      modulated by `c`, via its own per-block modulation MLP.
    t_embed_mode="adaln_single" (PixArt-alpha style): same modulation math as
      "adaln", but the hidden->6*hidden MLP is shared across all blocks
      (computed once in CrossAttnJiT.forward as `t0`); each block only owns a
      small (6, hidden) `scale_shift_table` offset added to `t0` before
      chunking into shift/scale/gate.
    t_embed_mode="token" (U-ViT style): no adaLN at all; the timestep instead
      enters as a prepended sequence token (see CrossAttnJiT.forward) and
      reaches every block purely through self-/cross-attention. Blocks become
      plain pre-norm residual blocks.
    """
    def __init__(self, hidden_size, num_heads, conditioning_mode,
                 mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, t_embed_mode="adaln"):
        super().__init__()
        assert conditioning_mode in ("cls_only", "patch_only", "cls_cross")
        assert t_embed_mode in ("adaln", "token", "adaln_single")
        self.conditioning_mode = conditioning_mode
        self.t_embed_mode = t_embed_mode

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)

        if conditioning_mode in ("patch_only", "cls_cross"):
            self.norm3 = RMSNorm(hidden_size, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        if t_embed_mode == "adaln":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        elif t_embed_mode == "adaln_single":
            self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))

    def forward(self, x, c, t0, context, feat_rope):
        if self.t_embed_mode == "adaln":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(6, dim=-1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        elif self.t_embed_mode == "adaln_single":
            B = x.shape[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                (self.scale_shift_table[None] + t0.reshape(B, 6, -1)).chunk(6, dim=1)
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa.squeeze(1), scale_msa.squeeze(1)), rope=feat_rope)
        else:
            x = x + self.attn(self.norm1(x), rope=feat_rope)

        if self.conditioning_mode in ("patch_only", "cls_cross"):
            cross_out, _ = self.cross_attn(self.norm3(x), context, context)
            x = x + cross_out

        if self.t_embed_mode == "adaln":
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        elif self.t_embed_mode == "adaln_single":
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp.squeeze(1), scale_mlp.squeeze(1)))
        else:
            x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnJiT(nn.Module):
    """
    Flow-matching image transformer with a required `conditioning_mode`
    switch: `"cls_only"` drives adaLN-Zero with pooled (t + encoder pooled
    token) conditioning (no cross-attention sublayer), while `"patch_only"` adds
    per-block cross-attention to the frozen encoder's dense patch tokens
    (PixArt-alpha style) with adaLN-Zero driven by timestep alone.

    `t_embed_mode` controls how the timestep reaches the network:
    `"adaln"` (default, DiT-style): every block/final-layer is adaLN-Zero
      modulated by the timestep embedding, each block with its own modulation MLP.
    `"adaln_single"` (PixArt-alpha style): same modulation as "adaln", but the
      hidden->6*hidden MLP is shared across all blocks (computed once as `t0`).
    `"token"` (U-ViT style): no adaLN anywhere; the timestep embedding is
      prepended as an extra sequence token and reaches every block through
      plain self-/cross-attention instead. The token is dropped again before
      the final projection back to pixel space.
    Only tested/intended for conditioning_mode="patch_only" so far.
    """
    def __init__(
        self,
        conditioning_mode: str,
        input_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 64,
        cond_dim: int = 768,
        long_skip_connection: bool = False,
        use_final_conv: bool = True,
        t_embed_mode: str = "adaln",
        in_context_len: int = 0,
        in_context_start: int = 4,
    ):
        super().__init__()
        assert conditioning_mode in ("cls_only", "patch_only", "cls_cross")
        assert t_embed_mode in ("adaln", "token", "adaln_single")
        if long_skip_connection and depth % 2 == 0:
            # Even depth has no bottleneck block, so the last encoder block's
            # output would be both the decoder's input `x` and its first skip
            # -- a degenerate self-concat that carries no earlier-layer info.
            raise ValueError(
                f"long_skip_connection requires odd depth (got depth={depth}); "
                "even depth skips the encoder/decoder bottleneck and makes the "
                "first decoder skip a no-op self-concat."
            )
        # In-context conditioning tokens (JiT-style) are only meaningful when a
        # pooled cond_emb exists to build them from -> cls_only. Not combined
        # with the U-ViT time token (that would put in-context tokens ahead of
        # the time token and desync the token-drop / RoPE offsets).
        self.in_context_len = in_context_len if conditioning_mode == "cls_only" else 0
        self.in_context_start = in_context_start
        if self.in_context_len > 0:
            if t_embed_mode == "token":
                raise ValueError("in_context_len>0 is not supported with t_embed_mode='token'.")
            upper = depth // 2 if long_skip_connection else depth
            if not (0 <= in_context_start < upper):
                raise ValueError(
                    f"in_context_start must be in [0, {upper}) "
                    f"(long_skip_connection={long_skip_connection}, depth={depth}); got {in_context_start}."
                )

        self.conditioning_mode = conditioning_mode
        self.t_embed_mode = t_embed_mode
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.long_skip_connection = long_skip_connection
        self.use_final_conv = use_final_conv

        self.t_embedder = TimestepEmbedder(hidden_size)
        if t_embed_mode == "adaln_single":
            self.shared_adaLN = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.ctx_patch_proj = (
            nn.Linear(cond_dim, hidden_size) if conditioning_mode == "patch_only" else None
        )

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        base_cls = 1 if t_embed_mode == "token" else 0
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=base_cls,
        )
        if self.in_context_len > 0:
            # Blocks at/after in_context_start see in_context_len extra leading
            # (non-spatial) tokens, which must be excluded from the rotary grid.
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size)
            )
            self.feat_rope_incontext = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=hw_seq_len,
                num_cls_token=base_cls + self.in_context_len,
            )

        self.blocks = nn.ModuleList([
            CrossAttnJiTBlock(
                hidden_size, num_heads, conditioning_mode, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                t_embed_mode=t_embed_mode,
            )
            for i in range(depth)
        ])

        if long_skip_connection:
            n_skips = depth // 2
            self.skip_linears = nn.ModuleList([
                nn.Linear(2 * hidden_size, hidden_size) for _ in range(n_skips)
            ])

        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels,
                                      use_adaln=(t_embed_mode in ("adaln", "adaln_single")))
        self.final_conv = (
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            if use_final_conv else None
        )
        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj2.bias)

        if self.conditioning_mode == "patch_only":
            nn.init.xavier_uniform_(self.ctx_patch_proj.weight)
            nn.init.zeros_(self.ctx_patch_proj.bias)

        if self.long_skip_connection:
            for skip_linear in self.skip_linears:
                nn.init.xavier_uniform_(skip_linear.weight)
                nn.init.zeros_(skip_linear.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.in_context_len > 0:
            nn.init.normal_(self.in_context_posemb, std=0.02)

        for block in self.blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            if hasattr(block, "cross_attn"):
                # zero-init cross-attention output projection: sublayer starts as a no-op
                nn.init.zeros_(block.cross_attn.out_proj.weight)
                nn.init.zeros_(block.cross_attn.out_proj.bias)

        if self.t_embed_mode == "adaln_single":
            nn.init.zeros_(self.shared_adaLN[-1].weight)
            nn.init.zeros_(self.shared_adaLN[-1].bias)
            for block in self.blocks:
                nn.init.zeros_(block.scale_shift_table)

        if hasattr(self.final_layer, "adaLN_modulation"):
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

        if self.use_final_conv:
            # zero-init: final_conv starts as a no-op residual, learns to smooth patchify artifacts
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.conditioning_mode == "patch_only":
            context = self.ctx_patch_proj(patch_tokens)      # (B, N_cond, H)
        elif self.conditioning_mode == "cls_cross":
            context = cond_emb.unsqueeze(1)                   # (B, 1, H) single memory token
        else:
            context = None

        t_emb = self.t_embedder(t)
        c = t_emb + cond_emb if self.conditioning_mode == "cls_only" else t_emb
        t0 = self.shared_adaLN(c) if self.t_embed_mode == "adaln_single" else None

        x = self.x_embedder(x) + self.pos_embed

        if self.t_embed_mode == "token":
            # Prepend the timestep as a sequence token (U-ViT style); feat_rope
            # was built with num_cls_token=1 so it leaves this leading token
            # unrotated. `c` goes unused by blocks/final_layer in this mode.
            x = torch.cat([t_emb.unsqueeze(1), x], dim=1)

        # In-context conditioning tokens (JiT-style): built from the pooled
        # cond_emb, injected at block in_context_start, carried through the rest
        # of the network (still updated by self-attn/MLP), and dropped before the
        # final layer. Excluded from the long-skip concat, which carries image
        # patches only. Gated to cls_only (see __init__), so t_embed_mode != token
        # here and the tokens sit at sequence position 0.
        L = self.in_context_len
        in_ctx = None
        if L > 0:
            in_ctx = cond_emb.unsqueeze(1).repeat(1, L, 1) + self.in_context_posemb  # (B, L, H)

        def rope_for(i):
            if L > 0 and i >= self.in_context_start:
                return self.feat_rope_incontext
            return self.feat_rope

        if self.long_skip_connection:
            n_encoder = self.depth // 2
            skips = []
            for i in range(n_encoder):
                if L > 0 and i == self.in_context_start:
                    x = torch.cat([in_ctx, x], dim=1)
                x = self.blocks[i](x, c, t0, context, rope_for(i))
                # Store only the image-patch slice so encoder/decoder skips match.
                skips.append(x[:, L:] if (L > 0 and i >= self.in_context_start) else x)

            decoder_start = n_encoder
            if self.depth % 2 == 1:
                # Odd depth: the middle block is a bottleneck — no skip in, no skip out.
                x = self.blocks[n_encoder](x, c, t0, context, rope_for(n_encoder))
                decoder_start = n_encoder + 1

            for skip_linear, block, i in zip(
                self.skip_linears, self.blocks[decoder_start:], range(decoder_start, self.depth)
            ):
                if L > 0:
                    patch = skip_linear(torch.cat([x[:, L:], skips.pop()], dim=-1))
                    x = torch.cat([x[:, :L], patch], dim=1)  # in-context tokens bypass the skip
                else:
                    x = skip_linear(torch.cat([x, skips.pop()], dim=-1))
                x = block(x, c, t0, context, rope_for(i))
        else:
            for i, block in enumerate(self.blocks):
                if L > 0 and i == self.in_context_start:
                    x = torch.cat([in_ctx, x], dim=1)
                x = block(x, c, t0, context, rope_for(i))

        if L > 0:
            x = x[:, L:]  # drop in-context tokens before projecting back to pixel space
        if self.t_embed_mode == "token":
            x = x[:, 1:]  # drop the time token before projecting back to pixel space

        x = self.final_layer(x, c)
        img = self.unpatchify(x)
        return (img + self.final_conv(img)) if self.use_final_conv else img


def CrossAttnJiT_S_16(**kwargs) -> CrossAttnJiT:
    kwargs.setdefault("depth", 13)
    kwargs.setdefault("hidden_size", 384)
    kwargs.setdefault("num_heads", 6)
    kwargs.setdefault("bottleneck_dim", 64)
    kwargs.setdefault("patch_size", 16)
    kwargs.setdefault("cond_dim", 768)
    return CrossAttnJiT(**kwargs)


def CrossAttnJiT_B_16(**kwargs) -> CrossAttnJiT:
    kwargs.setdefault("depth", 13)
    kwargs.setdefault("hidden_size", 768)
    kwargs.setdefault("num_heads", 12)
    kwargs.setdefault("bottleneck_dim", 128)
    kwargs.setdefault("patch_size", 16)
    kwargs.setdefault("cond_dim", 768)
    return CrossAttnJiT(**kwargs)


_MODEL_REGISTRY = {
    "S_16": CrossAttnJiT_S_16,
    "B_16": CrossAttnJiT_B_16,
}
