# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import torch
import torch.nn as nn
from torchvision import models as torchvision_models

class Wrapper(nn.Module):
    """
    Projector that is used when training Dino. This projector is added on top of a traditional resnet.
    """
    def __init__(self, model, head, use_head=False):
        super().__init__()
        self.head = head
        self.model = model
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.use_head = use_head

    def forward(self, x):
        x = self.model(x)
        if x.ndim > 2:
            x = self.pooling(x).view(x.size(0), -1)
        if self.use_head:
            x = self.head(x)
        return x

class SimCLRHead(nn.Module):
    """
    Projector that is used when training Dino. This projector is added on top of a traditional resnet.
    """
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x

class DINOHead(nn.Module):
    """
    Projector that is used when training Dino. This projector is added on top of a traditional resnet.
    """
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

    def forward(self, x):
        x = self.mlp(x)
        return x

def Projector(emb=8192):
    mlp_spec = f"2048-{8192}-{8192}-{emb}"

    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

class Normalize(nn.Module):
    """
    Projector that is used when training Dino. This projector is added on top of a traditional resnet.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.nn.functional.normalize(x, dim=-1, p=2).detach()
        return x

def get_model(model="dino", use_head=False, pool=True):
    '''
    Select a model that will be used to compute the embeddings needed by RCDM.
    You can use any kind of model, ConvNets/MLPs, or VITs.
    '''

    if model == "supervised":
        embedding_model = torchvision_models.resnet50(pretrained=True)
        embedding_model.fc = nn.Identity()
        return embedding_model.eval()

    elif model == "dino":
        embedding_model = torchvision_models.resnet50()
        embedding_model.fc = nn.Identity()
        embedding_model.head = DINOHead(
                2048,
                60000,
                nlayers=2,
                use_bn=True,
        )
        pretrained_model = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth", map_location="cpu") 
        pretrained_model = pretrained_model["teacher"]
        if "state_dict" in pretrained_model:
            pretrained_model = pretrained_model["state_dict"]
        # remove prefixe "module."
        pretrained_model = {k.replace("module.", ""): v for k, v in pretrained_model.items()}
        pretrained_model = {k.replace("backbone.", ""): v for k, v in pretrained_model.items()}
        for k, v in embedding_model.state_dict().items():
            if k not in list(pretrained_model):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif pretrained_model[k].shape != v.shape:
                print(pretrained_model[k].shape, "/", v.shape)
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                pretrained_model[k] = v
        msg = embedding_model.load_state_dict(pretrained_model, strict=True)
        print(msg)
        return Wrapper(embedding_model, head=embedding_model.head, use_head=use_head)

    elif model == "simclr":
        embedding_model = torchvision_models.resnet50()
        embedding_model.fc = nn.Identity()
        pretrained_model_base = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch", map_location="cpu")
        # Load trunk
        pretrained_model = pretrained_model_base["classy_state_dict"]["base_model"]["model"]["trunk"]
        pretrained_model = {k.replace("_feature_blocks.", ""): v for k, v in pretrained_model.items()}
        msg = embedding_model.load_state_dict(pretrained_model, strict=True)
        print("Trunk:", msg)
        # Load head
        embedding_model_head = SimCLRHead(2048, hidden_dim=2048, bottleneck_dim=128)
        pretrained_model_head = pretrained_model_base["classy_state_dict"]["base_model"]["model"]["heads"]
        pretrained_model_head = {k.replace("0.clf.0", "mlp.0").replace("1.clf.0", "mlp.1"): v for k, v in pretrained_model_head.items()}
        msg = embedding_model_head.load_state_dict(pretrained_model_head, strict=True)
        embedding_model_head.mlp = nn.Sequential(embedding_model_head.mlp[0], nn.ReLU(), embedding_model_head.mlp[1])
        embedding_model.eval()
        print("Head:", msg)
        return Wrapper(embedding_model, embedding_model_head, use_head)

    elif model == "barlow":
        embedding_model = torchvision_models.resnet50()
        embedding_model.fc = nn.Identity()
        pretrained_model_base = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch", map_location="cpu")
        # Load trunk        
        pretrained_model = pretrained_model_base["classy_state_dict"]["base_model"]["model"]["trunk"]
        pretrained_model = {k.replace("_feature_blocks.", ""): v for k, v in pretrained_model.items()}
        msg = embedding_model.load_state_dict(pretrained_model, strict=True)
        print("Trunk:", msg)
        # Load head
        head1 = nn.Sequential(nn.Linear(2048, 8192, bias=False), nn.BatchNorm1d(8192), nn.ReLU())
        head2 = nn.Sequential(nn.Linear(8192, 8192, bias=False), nn.BatchNorm1d(8192), nn.ReLU())
        embedding_model.head = nn.Sequential(head1, head2, nn.Sequential(nn.Linear(8192, 8192, bias=False)))
        pretrained_model_head = pretrained_model_base["classy_state_dict"]["base_model"]["model"]["heads"]
        pretrained_model_head = {k.replace("clf.", ""): v for k, v in pretrained_model_head.items()}
        msg = embedding_model.head.load_state_dict(pretrained_model_head, strict=True)
        print("Head:", msg)
        embedding_model.eval()
        return Wrapper(embedding_model, embedding_model.head, use_head)

    elif model == "vicreg":
        embedding_model = torchvision_models.resnet50()
        embedding_model.fc = nn.Identity()
        pretrained_model_base = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/vicreg/resnet50_fullckpt.pth", map_location="cpu")
        embedding_model.classifier = nn.Identity()
        embedding_model.projector = Projector(emb=8192)
        pretrained = "resnet50_fullckpt.pth"
        pretrained_model_base["model"] = {k.replace("backbone.", ""): v for k, v in pretrained_model_base["model"].items()}
        pretrained_model_base["model"] = {k.replace("module.", ""): v for k, v in pretrained_model_base["model"].items()}
        msg = embedding_model.load_state_dict(pretrained_model_base["model"], strict=True)
        embedding_model.eval()
        print(msg)
        return Wrapper(embedding_model, embedding_model.projector, use_head)

    elif model == "chexworld":
        import argparse
        import os
        import torch.serialization

        from .chexworld_backbone.jepa_vit import vit_base  # see chexworld_backbone/__init__.py for why this is a copy, not a reference

        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)   # CheXWorld's own norm_type="default"
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)    # (data_utils/xray_transform.py get_mean_std)
        CHEXWORLD_CKPT = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "backbone", "weights", "chexworld_pretrained.tar"
        )

        class CheXWorldWrapper(nn.Module):
            """Wraps the vendored CheXWorld ViT-Base/16 (target_encoder, no CLS
            token). pool=True -> (B,768) mean-pooled (RCDM/RCT contract);
            pool=False -> (B,N,768) patch tokens (FeatInv spatial hint)."""
            def __init__(self, pool=True):
                super().__init__()
                self.pool = pool
                self.encoder = vit_base(img_size=224, patch_size=16)
                torch.serialization.add_safe_globals([argparse.Namespace])
                ckpt = torch.load(CHEXWORLD_CKPT, map_location="cpu")
                sd = ckpt["model"]
                encoder_sd = {
                    k[len("target_encoder."):]: v
                    for k, v in sd.items() if k.startswith("target_encoder.")
                }
                self.encoder.load_state_dict(encoder_sd, strict=True)

            def forward(self, x):
                img = (x * 0.5 + 0.5).clamp(0, 1)   # [-1,1] -> [0,1]
                mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_DEFAULT_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
                img = (img - mean) / std
                tokens = self.encoder(img)          # (B, N, 768)
                return tokens.mean(dim=1) if self.pool else tokens

        return CheXWorldWrapper(pool=pool).eval()

    elif model == "xrv_densenet121":
        # `torch` is rebound as a local by the chexworld branch's `import torch.serialization`,
        # so re-import it here rather than relying on the module-level name.
        import torch
        import torchxrayvision as xrv

        class XRVDenseNetWrapper(nn.Module):
            """Wraps TorchXRayVision DenseNet-121 (densenet121-res224-all).
            pool=True -> (B,1024) global-average-pooled embedding, which is exactly
            what xrv's own features2() returns; pool=False -> (B,49,1024) tokens from
            the last dense block. Mirrors rct.denoiser's conditioning encoder so the
            metrics measure in the same space the model was conditioned on."""
            def __init__(self, pool=True):
                super().__init__()
                self.pool = pool
                self.model = xrv.models.DenseNet(weights="densenet121-res224-all")

            def forward(self, x):
                # xrv wants one channel scaled to roughly [-1024, 1024]
                # (xrv.utils.normalize: (2*u - 1) * 1024 for u in [0,1]).
                img = (x * 0.5 + 0.5).clamp(0, 1).mean(dim=1, keepdim=True)
                img = (img * 2.0 - 1.0) * 1024.0
                feat = torch.relu(self.model.features(img))     # (B, 1024, 7, 7)
                tokens = feat.flatten(2).transpose(1, 2)        # (B, 49, 1024)
                return tokens.mean(dim=1) if self.pool else tokens

        return XRVDenseNetWrapper(pool=pool).eval()

    elif model == "raddino":
        import torch
        from transformers import AutoModel

        RADDINO_HF = "microsoft/rad-dino"
        RADDINO_MEAN = (0.485, 0.456, 0.406)
        RADDINO_STD  = (0.229, 0.224, 0.225)

        class RadDinoWrapper(nn.Module):
            """Wraps HF rad-dino (microsoft/rad-dino).
            pool=True -> (B,768) CLS token; pool=False -> (B,N,768) patch tokens.
            Input: [-1,1] float32 RGB tensors."""
            def __init__(self, pool=True):
                super().__init__()
                self.pool = pool
                self.model = AutoModel.from_pretrained(RADDINO_HF)

            def forward(self, x):
                img = (x * 0.5 + 0.5).clamp(0, 1)
                mean = torch.tensor(RADDINO_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
                std  = torch.tensor(RADDINO_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
                img = (img - mean) / std
                out = self.model(pixel_values=img).last_hidden_state  # (B, 1+N, 768)
                return out[:, 0] if self.pool else out[:, 1:]

        return RadDinoWrapper(pool=pool).eval()

    else:
        print("No model found")
        exit(1)
