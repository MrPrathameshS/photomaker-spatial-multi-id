import torch
import torch.nn as nn


class IdentitySlotUNet(nn.Module):
    """
    Competitive Multi-Identity Injection UNet Wrapper
    """

    def __init__(
        self,
        unet,
        down_strength=0.4,
        mid_strength=0.7,
        up_strength=0.8,
        temperature=1.0,
        
    ):
        super().__init__()

        self.unet = unet
        self.active_slots = None
        self.down_strength = down_strength
        self.mid_strength = mid_strength
        self.up_strength = up_strength
        self.temperature = temperature

        latent_channels = unet.config.block_out_channels[-1]
        self.proj = nn.Linear(512, latent_channels)

        self.config = unet.config

        # Identity state
        self.identity_embeddings = None       # [B, N, 512]
        self.identity_bboxes = None           # [N, 4]
        self.slot_active_mask = None          # [N]

        self._register_hooks()

    # -------------------------------------------------
    # Required properties for diffusers
    # -------------------------------------------------

    @property
    def device(self):
        return next(self.unet.parameters()).device

    @property
    def dtype(self):
        return next(self.unet.parameters()).dtype

    # -------------------------------------------------
    # Identity setter
    # -------------------------------------------------
    def set_active_slots(self, active_slots):
        self.active_slots = active_slots

    def set_identity_data(self, embeddings, bboxes):
        """
        embeddings: [B, N, 512]
        bboxes:     [N, 4] (normalized)
        """
        self.identity_embeddings = embeddings
        self.identity_bboxes = bboxes

        if embeddings is not None:
            _, N, _ = embeddings.shape
            self.slot_active_mask = torch.ones(
                N,
                dtype=torch.bool,
                device=embeddings.device
            )

    def set_active_slots(self, active_indices):
        """
        active_indices: list of slot indices to activate
        """
        if self.identity_embeddings is None:
            return

        _, N, _ = self.identity_embeddings.shape
        mask = torch.zeros(
            N,
            dtype=torch.bool,
            device=self.identity_embeddings.device
        )

        for idx in active_indices:
            if 0 <= idx < N:
                mask[idx] = True

        self.slot_active_mask = mask

    def clear_identity_data(self):
        self.identity_embeddings = None
        self.identity_bboxes = None
        self.slot_active_mask = None

    # -------------------------------------------------
    # Hook registration
    # -------------------------------------------------

    def _register_hooks(self):

        if len(self.unet.down_blocks) > 0:
            self.unet.down_blocks[0].register_forward_hook(
                self._down_block_hook
            )

        self.unet.mid_block.register_forward_hook(
            self._mid_block_hook
        )

        for i in range(min(2, len(self.unet.up_blocks))):
            self.unet.up_blocks[i].register_forward_hook(
                self._up_block_hook
            )

    # -------------------------------------------------
    # Core Competitive Injection
    # -------------------------------------------------

    def _inject(self, hidden_states, strength):
        """
        Competitive spatial identity injection with:
        • active slot gating
        • safe masking
        • stable softmax competition

        hidden_states: [B, C, H, W]
        """

        if self.identity_embeddings is None:
            return hidden_states

        if self.identity_bboxes is None:
            return hidden_states

        if not isinstance(hidden_states, torch.Tensor):
            return hidden_states

        B, C, H, W = hidden_states.shape
        B_id, N_total, D = self.identity_embeddings.shape

        if B != B_id or N_total == 0:
            return hidden_states

        device = hidden_states.device
        dtype = hidden_states.dtype

        # -------------------------------------------------
        # 1️⃣ Determine active identities
        # -------------------------------------------------

        if self.active_slots is not None:
            active_indices = self.active_slots
        else:
            active_indices = list(range(N_total))

        if len(active_indices) == 0:
            return hidden_states

        # Select only active embeddings + bboxes
        identity_embeddings = self.identity_embeddings[:, active_indices]   # [B, N, D]
        identity_bboxes = self.identity_bboxes[active_indices]              # [N, 4]

        B, N, D = identity_embeddings.shape

        # -------------------------------------------------
        # 2️⃣ Project identity embeddings
        # -------------------------------------------------

        projected = self.proj(
            identity_embeddings.reshape(B * N, D)
        )  # [B*N, C]

        projected = projected.view(B, N, C)

        # Normalize for stability
        projected = projected / (
            projected.norm(dim=2, keepdim=True) + 1e-6
        )

        # -------------------------------------------------
        # 3️⃣ Build spatial masks
        # -------------------------------------------------

        masks = torch.zeros(
            (B, N, 1, H, W),
            device=device,
            dtype=dtype
        )

        for i in range(N):

            x1, y1, x2, y2 = identity_bboxes[i]

            x1_i = int(max(0, min(W, x1.item() * W)))
            x2_i = int(max(0, min(W, x2.item() * W)))
            y1_i = int(max(0, min(H, y1.item() * H)))
            y2_i = int(max(0, min(H, y2.item() * H)))

            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            masks[:, i, :, y1_i:y2_i, x1_i:x2_i] = 1.0

            # area normalization (prevents large bbox dominance)
            area = max((x2_i - x1_i) * (y2_i - y1_i), 1)
            masks[:, i] /= float(area)

        # If no mask pixels active, skip
        if masks.sum() == 0:
            return hidden_states

        # -------------------------------------------------
        # 4️⃣ Compute similarity scores
        # -------------------------------------------------

        hidden_flat = hidden_states.view(B, C, H * W)

        scores = torch.einsum(
            "bnc,bch->bnh",
            projected,
            hidden_flat
        )

        scores = scores.view(B, N, H, W)

        scores = scores * masks.squeeze(2)

        # -------------------------------------------------
        # 5️⃣ Competitive softmax
        # -------------------------------------------------

        temperature = getattr(self, "temperature", 1.0)

        weights = torch.softmax(
            scores / temperature,
            dim=1
        )

        # Re-apply spatial mask
        weights = weights * masks.squeeze(2)

        # -------------------------------------------------
        # 6️⃣ Weighted identity injection
        # -------------------------------------------------

        projected = projected.view(B, N, C, 1, 1)
        weights = weights.unsqueeze(2)

        total_injection = (weights * projected).sum(dim=1)

        # -------------------------------------------------
        # 7️⃣ Apply injection
        # -------------------------------------------------

        hidden_states = hidden_states + strength * total_injection

        return hidden_states


    # -------------------------------------------------
    # Hooks
    # -------------------------------------------------

    def _down_block_hook(self, module, input, output):
        return self._inject(output, self.down_strength)

    def _mid_block_hook(self, module, input, output):
        return self._inject(output, self.mid_strength)

    def _up_block_hook(self, module, input, output):
        return self._inject(output, self.up_strength)

    # -------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)

    # -------------------------------------------------
    # Delegate missing attributes
    # -------------------------------------------------

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)
