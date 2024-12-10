from _networks._utils import BaseNetwork
from _networks import register_network
import timm
import torch
from torch import nn
from transformers import T5Tokenizer, AutoModelForSequenceClassification, T5ForSequenceClassification, T5Model

# TODO non é un todo ma solo per attirare l'attenzione
# da transformer si possono importare diverse versioni di T5. T5ForSequenceClassification ha una classification head,
# T5ForConditionalGeneration ha una language modeling head. Facendo intent classification direi di usare questa


@register_network("t5")
class T5(BaseNetwork):

    def __init__(self, model_name: str = "t5-small", num_classes: int = 150, pretrained: bool = True):
        super().__init__()
        print(f"Using {model_name}\tpretrained: {pretrained}\tnum_classes: {num_classes}")

        self.model = T5Model.from_pretrained(model_name)
        self.embed_dim = self.model.config.d_model
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, penultimate=False, prelogits=False):
        """
        penultimate returns the classification token after the forward_features,
        prelogits returns the prelogits using the vit argument pre_logits=True, which involves more stuff
        than the previous

        Ho lasciato perché non saccio cosa faciano tutte shte cuuse
        """

        decoder_input_ids = torch.ones((x["input_ids"].shape[0], 1), dtype=torch.long).to(x["input_ids"].device)

        outputs = self.model(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )
        decoder_hidden_states = outputs.last_hidden_state
        pooled_output = decoder_hidden_states.mean(dim=1)
        if prelogits:
            return outputs
        if penultimate:
            pre_logits = pooled_output
            return pre_logits, self.head(pooled_output)
        return self.head(pooled_output)
