import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


class SpatialIdentityAttnProcessor(AttnProcessor2_0):
    def __init__(self, identity_token_indices, base_masks):
        """
        identity_token_indices: List[List[int]]
            token indices for each identity
        base_masks: Tensor [num_id, 128, 128]
            binary masks in full latent resolution
        """
        super().__init__()
        self.identity_token_indices = identity_token_indices
        self.base_masks = base_masks  # [N, H, W]

        # Flatten identity token list
        self.all_identity_tokens = sorted(
            list({t for group in identity_token_indices for t in group})
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):

        # -------------------------------
        # 1️⃣ Self-attention → fallback
        # -------------------------------
        if encoder_hidden_states is None:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
            )

        # -------------------------------
        # 2️⃣ Standard QKV
        # -------------------------------
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_scores = torch.bmm(query, key.transpose(-1, -2))
        attn_scores = attn_scores * attn.scale

        B_heads, spatial_tokens, text_tokens = attn_scores.shape
        heads = attn.heads
        batch_size = B_heads // heads

        # Reshape → [B, heads, spatial_tokens, text_tokens]
        attn_scores = attn_scores.view(batch_size, heads, spatial_tokens, text_tokens)

        # -------------------------------
        # 3️⃣ Downsample spatial masks
        # -------------------------------
        spatial_size = int(spatial_tokens ** 0.5)

        masks = F.interpolate(
            self.base_masks.unsqueeze(1),  # [N,1,H,W]
            size=(spatial_size, spatial_size),
            mode="nearest"
        ).squeeze(1)  # [N, h, w]

        masks = masks.reshape(masks.shape[0], -1)  # [N, spatial_tokens]
        masks = masks.to(attn_scores.device)

        # -------------------------------
        # 4️⃣ Build Region → Identity map
        # -------------------------------
        # region_id: [spatial_tokens]
        # -1 = background
        region_id = torch.full(
            (spatial_tokens,),
            fill_value=-1,
            device=attn_scores.device,
            dtype=torch.long
        )

        for i in range(masks.shape[0]):
            region_id[masks[i] > 0] = i

        # -------------------------------
        # 5️⃣ HARD GATING
        # -------------------------------
        NEG_INF = -1e9

        for identity_i, token_indices in enumerate(self.identity_token_indices):

            # mask of spatial locations belonging to identity_i
            region_mask = (region_id == identity_i)  # [spatial_tokens]

            # spatial positions NOT belonging to this identity
            not_region_mask = ~region_mask

            for token_idx in token_indices:
                # Block this identity's tokens everywhere
                # except its own region
                attn_scores[:, :, not_region_mask, token_idx] = NEG_INF

        # -------------------------------
        # 6️⃣ Outside all regions → block ALL identity tokens
        # -------------------------------
        background_mask = (region_id == -1)

        for token_idx in self.all_identity_tokens:
            attn_scores[:, :, background_mask, token_idx] = NEG_INF

        # -------------------------------
        # 7️⃣ Softmax + output
        # -------------------------------
        attn_scores = attn_scores.view(B_heads, spatial_tokens, text_tokens)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        hidden_states = torch.bmm(attn_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
