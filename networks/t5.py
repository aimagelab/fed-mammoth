from networks.utils import BaseNetwork
from networks import register_network
import timm
from torch import nn
from transformers import T5Tokenizer, T5ForSequenceClassification

# TODO non é un todo ma solo per attirare l'attenzione
# da transformer si possono importare diverse versioni di T5. T5ForSequenceClassification ha una classification head,
# T5ForConditionalGeneration ha una language modeling head. Facendo intent classification direi di usare questa


@register_network("t5")
class T5(BaseNetwork):

    def __init__(self, model_name: str = "t5-small", num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        print(f"Using {model_name}\tpretrained: {pretrained}\tnum_classes: {num_classes}")

        self.model = T5ForSequenceClassification.from_pretrained(model_name)

    def forward(self, x, penultimate=False, prelogits=False):
        """
        penultimate returns the classification token after the forward_features,
        prelogits returns the prelogits using the vit argument pre_logits=True, which involves more stuff
        than the previous

        Ho lasciato perché non saccio cosa faciano tutte shte cuuse
        """

        print("c")

        outputs = self.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        # TODO sistema tokenizer nel dataloader
