from argparse import Namespace
import copy
from functools import partial
import math
import torch
from torch import nn
import torch.nn.functional as F
#from backbone import MammothBackbone
from _networks import register_network
from _networks._utils import BaseNetwork
#from backbone.vit import vit_base_patch16_224_prompt_prototype



def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot



# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------


# ranpac_utils/vit.py


import math
from timm.layers import DropPath
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from collections import OrderedDict
import torch


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if config.ffn_adapt:
            self.adaptmlp = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                    )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
            adapt_x = self.adaptmlp(x, add_residual=False)

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt:
            if self.config.ffn_option == 'sequential':
                x = self.adaptmlp(x)
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + x
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        # print("I'm using ViT with adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x,)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# ranpac_utils/inc_net.py

class RanPACNet(nn.Module):

    def __init__(self, backbone: nn.Module, device='cuda'):
        super(RanPACNet, self).__init__()

        self.fc = None
        #self.device = backbone.device
        self.device = device

        tuning_config = Namespace(ffn_adapt=True,
                                  ffn_option="parallel",
                                  ffn_adapter_layernorm_option="none",
                                  ffn_adapter_init_option="lora",
                                  ffn_adapter_scalar="0.1",
                                  ffn_num=64,
                                  d_model=768,
                                  vpt_on=False,
                                  vpt_num=0,
                                  )

        self.convnet = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                         norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)

        backbone.head = nn.Identity()
        state_dict = backbone.state_dict()

        for key in list(state_dict.keys()):
            if 'qkv.weight' in key:
                qkv_weight = state_dict.pop(key)
                q_weight = qkv_weight[:768]
                k_weight = qkv_weight[768:768 * 2]
                v_weight = qkv_weight[768 * 2:]
                state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
                state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
                state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
            elif 'qkv.bias' in key:
                qkv_bias = state_dict.pop(key)
                q_bias = qkv_bias[:768]
                k_bias = qkv_bias[768:768 * 2]
                v_bias = qkv_bias[768 * 2:]
                state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
                state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
                state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
        # second, modify the mlp.fc.weight to match fc.weight
        for key in list(state_dict.keys()):
            if 'mlp.fc' in key:
                fc_weight = state_dict.pop(key)
                state_dict[key.replace('mlp.', '')] = fc_weight

        missing, unexpected = self.convnet.load_state_dict(state_dict, strict=False)
        assert len([m for m in missing if 'adaptmlp' not in m]) == 0, f"Missing keys: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
        self.params_to_optimize = missing

        # freeze all but the adapter
        for name, p in self.convnet.named_parameters():
            if name in missing:
                p.requires_grad = True
                #self.params_to_optimize.append(p)
            else:
                p.requires_grad = False

        self.convnet.eval()

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.convnet.embed_dim, nb_classes).to(self.device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.convnet.embed_dim).to(self.device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x)
        return out


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP = False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn = input
                # inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn, self.weight)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
    




# ranpac_utils/ranpac.py



import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
import sys


@register_network('vit_ranpac')
class RanPAC_Model(BaseNetwork):
    def __init__(self, backbone = None, device = 'cuda'):
        super(RanPAC_Model, self).__init__()
        if backbone is None:
            from _networks.vit import VisionTransformer as ViT
            backbone = ViT().model
        self._network = RanPACNet(backbone, device)

    #@property
    #def training(self):
    #    return self._network.training

    def to(self, device):
        self._network.to(device)

    def train(self, *args):
        self._network.train(*args)

    def eval(self):
        self._network.eval()
    
    def forward(self, x):
        return self._network(x)

    def replace_fc(self, trainloader):
        self._network = self._network.eval()

        #if self.args['use_RP']:
        #    # these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
        #    self._network.fc.use_RP = True
        #    if self.args['M'] > 0:
        #        self._network.fc.W_rand = self.W_rand
        #    else:
        #        self._network.fc.W_rand = None
        self._network.fc.use_RP = True
        self._network.fc.W_rand = self.W_rand
        self._network.fc.W_rand = None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._network.device)
                label = label.to(self._network.device)
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = target2onehot(label_list, self.total_classnum)
        if self.args['use_RP']:
            # print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args['M'] > 0:
                Features_h = torch.nn.functional.relu(Features_f @ self._network.fc.W_rand.cpu())
            else:
                Features_h = Features_f
            self.Q = self.Q + Features_h.T @ Y
            self.G = self.G + Features_h.T @ Features_h
            ridge = self.optimise_ridge_parameter(Features_h, Y)
            Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T  # better nmerical stability than .inv
            self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0], :].to(self._network.device)
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype = Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index] += class_prototype.to(self._network.device)  # for dil, we update all classes in all tasks
                else:
                    # original cosine similarity approach of Zhou et al (2023)
                    class_prototype = Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index] = class_prototype  # for cil, only new classes get updated

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0**np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T  # better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: " + str(ridge))
        return ridge