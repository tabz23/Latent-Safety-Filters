import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    # def forward(self, x):
    #     emb = self.base_model.forward_features(x)[self.feature_key]
    #     if self.latent_ndim == 1:
    #         emb = emb.unsqueeze(1) # dummy patch dim
    #     return emb
    def forward(self, x):
        # Prevent NaNs by adding small noise to near-constant inputs
        if torch.isnan(x).any() or torch.std(x) < 1e-5:
            print("⚠️ Input is too flat or contains NaNs — adding noise", flush=True)
            x = x + torch.randn_like(x) * 1e-3

        features = self.base_model.forward_features(x)
        emb = features[self.feature_key]

        # Add dummy patch dimension for CLS token
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)

        # Optional check for NaNs in the output
        if torch.isnan(emb).any():
            print("❌ NaNs detected in DINO embedding!", flush=True)

        return emb


# import os
# import torch
# import torch.nn as nn

# # Bypass torch.hub fork validation
# torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

# class DinoV2Encoder(nn.Module):
#     def __init__(self, name, feature_key):
#         super().__init__()
#         self.name = name
#         self.feature_key = feature_key

#         # Try loading from torch.hub first
#         try:
#             print(f"Loading DINO model from torch.hub: {name}", flush=True)
#             self.base_model = torch.hub.load("facebookresearch/dinov2", name)
#         except Exception as e:
#             print("Failed online DINO load:", e, flush=True)
#             print("Falling back to local checkpoint...", flush=True)
#             local_dir = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints"))
#             os.environ["TORCH_HOME"] = os.path.dirname(local_dir)
#             self.base_model = torch.hub.load("facebookresearch/dinov2", name, source="local")

#         # Optional sanity check: check if head has weights
#         if hasattr(self.base_model, 'head') and hasattr(self.base_model.head, 'weight'):
#             print("DINO head weight norm:", self.base_model.head.weight.norm().item(), flush=True)
#         else:
#             print("DINO head has no weights (Identity or non-parametric).", flush=True)


#         self.emb_dim = self.base_model.num_features

#         if feature_key == "x_norm_patchtokens":
#             self.latent_ndim = 2
#         elif feature_key == "x_norm_clstoken":
#             self.latent_ndim = 1
#         else:
#             raise ValueError(f"Invalid feature key: {feature_key}")

#         self.patch_size = self.base_model.patch_size

#     def forward(self, x):
#         print("Input shape:", x.shape, "Device:", x.device, flush=True)
#         features = self.base_model.forward_features(x)
#         print("Feature keys:", features.keys(), flush=True)

#         emb = features[self.feature_key]
#         print("Embedding shape:", emb.shape,
#               "| Contains NaNs:", torch.isnan(emb).any().item(), flush=True)

#         if self.latent_ndim == 1:
#             emb = emb.unsqueeze(1)  # add dummy patch dim

#         return emb
