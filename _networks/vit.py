from _networks import register_network
from _networks._utils import BaseNetwork
import torchvision.models.vision_transformer as timm_vit
import timm
from torch import nn


@register_network("vit")
class VisionTransformer(BaseNetwork):

    def __init__(
        self, model_name: str = "vit_base_patch16_224.augreg_in21k", num_classes: int = 100, pretrained: bool = True
    ):
        super().__init__()
        if model_name == "dino":
            model_name = "vit_base_patch16_224.dino"
        print(f"Using ViT: {model_name}\tpretrained: {pretrained}\tnum_classes: {num_classes}")
        # self.model = timm_vit.__dict__[model_name](pretrained=pretrained, num_classes=num_classes)
        self.model : timm.models.vision_transformer.VisionTransformer = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x, penultimate=False, prelogits=False, block=None):
        """
        penultimate returns the classification token after the forward_features,
        prelogits returns the prelogits using the vit argument pre_logits=True, which involves more stuff
        than the previous
        """
        if block is not None:
            if type(block) == int:
                if block != 0:
                    return self.model.blocks[block](x)
                elif block == 0:
                    if x.shape[-1] != 224:
                        x = nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
                    x = self.model.patch_embed(x)
                    x = self.model._pos_embed(x)
                    x = self.model.patch_drop(x)
                    x = self.model.norm_pre(x)
                    return self.model.blocks[block](x)
            elif type(block) == str and block == "head":
                x = self.model.norm(x)
                #x = self.pool(x)
                #x = self.fc_norm(x)
                #x = self.head_drop(x)
                #x = self.head(x)
                return self.model.forward_head(x)
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        if prelogits:
            feats = self.model.forward_features(x)
            pre = self.model.forward_head(feats, pre_logits=True)
            return pre
        if penultimate:
            feats = self.model.forward_features(x)
            #pre = self.model.forward_head(feats, pre_logits=True)
            pre = feats[:, 0]
            out = self.model.forward_head(feats)
            return pre, out
        return self.model(x)