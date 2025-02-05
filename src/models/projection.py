import torch


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds, batch_size=1):
        bsz, max_num_ips, _ = image_embeds.shape
        image_embeds = image_embeds.view(bsz * max_num_ips, *image_embeds.shape[2:])
        image_embeds = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        image_embeds = self.norm(image_embeds)
        return image_embeds

    def dtype(self):
        return next(self.parameters()).dtype


class ImageProjDummyModel(torch.nn.Module):
    """Projection Model with dummy tokens"""

    def __init__(
        self,
        cross_attention_dim=2048,
        clip_embeddings_dim=1024,
        magi_embeddings_dim=512,
        clip_extra_context_tokens=4,
        num_dummy_tokens=4,
        use_magi=False,
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.use_magi = use_magi

        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        if use_magi:
            self.proj_magi = torch.nn.Linear(magi_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        self.dummy_tokens = torch.nn.Parameter(torch.randn(num_dummy_tokens, self.cross_attention_dim))

    def forward(self, image_embeds, magi_image_embeds=None):
        bsz, max_num_ips, _ = image_embeds.shape
        image_embeds = image_embeds.view(bsz * max_num_ips, *image_embeds.shape[2:])
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(bsz, max_num_ips * self.clip_extra_context_tokens, self.cross_attention_dim)
        image_embeds = self.norm(image_embeds)
        
        if self.use_magi and magi_image_embeds is not None:
            magi_image_embeds = self.proj_magi(magi_image_embeds)
            magi_image_embeds = magi_image_embeds.view(bsz, max_num_ips * self.clip_extra_context_tokens, self.cross_attention_dim)
            magi_image_embeds = self.norm(magi_image_embeds)
            image_embeds = image_embeds + magi_image_embeds
        
        dummy_tokens = self.dummy_tokens.unsqueeze(0).repeat(bsz, 1, 1)
        image_embeds = torch.cat([dummy_tokens, image_embeds], dim=1)

        return image_embeds

    def dtype(self):
        return next(self.parameters()).dtype