from networks import register_network
from networks.utils import BaseNetwork
import torchvision.models.vision_transformer as timm_vit
import timm
from torch import nn


@register_network("vit")
class VisionTransformer(BaseNetwork):

    def __init__(
        self, model_name: str = "vit_base_patch16_224.augreg_in21k", num_classes: int = 100, pretrained: bool = True
    ):
        super().__init__()
        print(f"Using ViT: {model_name}\tpretrained: {pretrained}\tnum_classes: {num_classes}")
        # self.model = timm_vit.__dict__[model_name](pretrained=pretrained, num_classes=num_classes)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        return self.model(x)
