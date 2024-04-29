import torch
import torch.nn as nn
import timm
import logging

log = logging.getLogger(__name__)


class GenericBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained, model_path="", num_classes=-1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=(pretrained and len(model_path) == 0), num_classes=num_classes)

        if model_path is not None and len(model_path) != 0:
            log.info(f"loading checkpoint from path {model_path}")
            self.load_state_dict(torch.load(model_path))

    def forward(self, x):
        x = self.backbone(x)
        return x