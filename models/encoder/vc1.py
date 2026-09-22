import torch
import torch.nn as nn
from vc_models.models.vit import model_utils

class VC1Encoder(nn.Module):
    """
    Wraps Facebook’s VC1 vision model as a Hydra‐configurable encoder.
    Returns embeddings of shape (B, 1, emb_dim).
    """
    def __init__(
        self,
        model_name: str = model_utils.VC1_BASE_NAME,
        trainable: bool = False,
        **kwargs
    ):
        super().__init__()
        # load_model returns: (model, embd_size, transforms, info)
        self.model, self.embd_size, self.transforms, _ = model_utils.load_model(model_name)
        self.name = "vc1"
        self.latent_ndim = 1
        self.emb_dim = self.embd_size

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) raw images (e.g. 3×250×250 or whatever your data provides)
        returns: (B, 1, emb_dim)
        """
        # apply the VC1 preprocessing (resizing, norm, etc.)
        x = self.transforms(x)              # now B×3×224×224
        emb = self.model(x)                 # B×emb_dim
        return emb.unsqueeze(1)             # B×1×emb_dim
