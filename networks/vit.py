from networks import register_network
from networks.utils import BaseNetwork
import torchvision.models.vision_transformer as timm_vit


@register_network("vit")
class VisionTransformer(BaseNetwork):
    def __init__(self, model_name, num_classes=1000, pretrained=False):
        super().__init__()
        self.model = timm_vit.__dict__[model_name](pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)