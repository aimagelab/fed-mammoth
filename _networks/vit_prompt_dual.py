import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.models import named_apply
from timm.layers import trunc_normal_, lecun_normal_, PatchEmbed, Mlp as TimmMlp, DropPath
from _networks import register_network
from _networks._utils import BaseNetwork

from timm.models._builder import build_model_with_cfg
from functools import partial
from utils.tools import str_to_bool





def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()






def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm






class Mlp(TimmMlp):
    def forward(self, x, **kwargs):
        return super().forward(x)


class Attention(nn.Module):
    """
    Attention layer as used in Vision Transformer.

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        qkv_bias: If True, add a learnable bias to q, k, v
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate after the final projection
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, **kwargs):
        """
        Forward pass of the attention layer.

        Args:
            x: Input tensor
        """

        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # NOTE: flash attention is less debuggable than the original. Use the commented code below if in trouble.
        # check torch version
        if torch.__version__ >= '2.1.0':
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop.p)
        else:
            print("Torch verison < 2.1.0 detected. Using the original attention code.")
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            attn_layer=Attention,
            mlp_layer=Mlp
    ):
        super().__init__()
        self.embed_dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, **kwargs):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), **kwargs)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), **kwargs)))
        return x









#attention.py

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()  # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0]  # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1]  # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (B, self.num_heads, C // self.num_heads)

            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#prompt.py

class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            similarity = similarity.t()  # B, pool_size

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
            out['similarity'] = similarity

            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                major_prompt_id = prompt_id[major_idx]  # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # B, top_k

            if prompt_mask is not None:
                idx = prompt_mask  # B, top_k

            out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, B, top_k, length, C
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:, idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            batched_key_norm = prompt_key_norm[idx]  # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['batched_prompt'] = batched_prompt

        return out




class VisionTransformerBackbone(nn.Module):
    """ Vision Transformer.
    This implementation supports LoRA (Layer-wise Relevance Adaptation) parameters if `use_lora=True`.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
            attn_layer=None,
            mlp_layer=None,
            use_lora=False,
            args=None
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
            block_fn: (nn.Module): transformer block
            attn_layer: (nn.Module): attention layer
            args: (Namespace): optional command-line arguments
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU

        attn_layer = attn_layer if attn_layer is not None else Attention
        mlp_layer = mlp_layer if mlp_layer is not None else Mlp
        self.attn_layer = attn_layer
        self.norm_layer = norm_layer
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.weight_init = weight_init
        self.class_token = class_token
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.feature_dim = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.mlp_ratio = mlp_ratio
        self.args = args
        self.init_values = init_values
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.depth = depth
        self.drop_rate = drop_rate
        self.mlp_layer = mlp_layer

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=self.dpr[i],
                norm_layer=norm_layer,
                act_layer=self.act_layer,
                attn_layer=attn_layer,
                mlp_layer=mlp_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

        self.embed_dim = embed_dim

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor, AB={}, return_all=False):
        """
        Compute the forward pass of ViT (features only).
        Can take in an additional argument `AB`, which is a dictionary containing LoRA-style parameters for each block.

        Args:
            x: input tensor
            AB: dictionary containing LoRA-style parameters for each block
            return_all: whether to return all intermediate features

        Returns:
            features for each patch
        """
        int_features = []
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        # NOTE: grad checkpointing was removed from the original timm impl
        for idx, blk in enumerate(self.blocks):
            AB_blk = AB.get(idx)
            if AB_blk is not None:
                x = blk(x, AB_blk)
            else:
                x = blk(x)
            if return_all:
                int_features.append(x.clone())
        x = self.norm(x)

        if return_all:
            int_features.append(x.clone())
            return int_features
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        """
        Compute the forward pass of ViT (head only).
        Expects input of shape [batch_size, num_patches, embed_dim].

        Args:
            x: input tensor
            pre_logits: whether to return the pre-logits (pooled features) or the final class scores

        Returns:
            output tensor with shape [batch_size, num_classes] if `pre_logits` is False, else [batch_size, embed_dim]
        """
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, AB: dict = {}, returnt='out'):
        """
        Compute the forward pass of ViT.
        Can take in an additional argument `AB`, which is a dictionary containing LoRA-style parameters for each block.

        `AB` can contain
        - a single value for each block (e.g. `AB = {0: {"qkv": torch.Tensor(...)}, 1: {"qkv": torch.Tensor(...)}, ...}`)
        - a dictionary for each block with a single key `B` (e.g. `AB = {0: {"qkv": {"B": torch.Tensor(...)}}}`)
        - a dictionary for each block with both `A` and `B` keys of LoRA parameters (e.g. `AB = {0: {"qkv": {"A": torch.Tensor(...), "B": torch.Tensor(...)}}}`)

        Supported keys for each block are `qkv`, `proj`, `fc1`, `fc2`.

        NOTE: The values of `AB` are **summed** with the weights of the corresponding block.

        Args:
            x: input tensor
            AB: dictionary containing LoRA-style parameters for each block
            returnt: return type (a string among `out`, `features`, `both`, or `full`)

        Returns:
            output tensor
        """
        assert returnt in ('out', 'features', 'both', 'full')

        x = self.forward_features(x, AB, return_all=returnt == 'full')
        if returnt == 'full':
            all_features = x
            x = x[-1]
        feats = self.forward_head(x, pre_logits=True)

        if returnt == 'features':
            return feats

        out = self.head(feats)

        if returnt == 'both':
            return out, feats
        elif returnt == 'full':
            return out, all_features
        return out

    def get_params(self, discard_classifier=False) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.

        Returns:
            parameters tensor
        """
        params = []
        for kk, pp in list(self.named_parameters()):
            if not discard_classifier or not 'head' in kk:
                params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self, discard_classifier=False) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.

        Returns:
            gradients tensor
        """
        grads = []
        for kk, pp in list(self.named_parameters()):
            if not discard_classifier or not 'head' in kk:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)
    



class VisionTransformer(VisionTransformerBackbone):

    def __init__(
            self, prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
            top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False,
            use_g_prompt=False, g_prompt_length=None, g_prompt_layer_idx=None, use_prefix_tune_for_g_prompt=False,
            use_e_prompt=False, e_prompt_layer_idx=None, use_prefix_tune_for_e_prompt=False, same_key_value=False, args=None, **kwargs):

        attn_layer = PreT_Attention

        super().__init__(args=args, attn_layer=attn_layer, **kwargs)

        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        self.use_g_prompt = use_g_prompt
        self.g_prompt_layer_idx = g_prompt_layer_idx

        # num_g_prompt : The actual number of layers to which g-prompt is attached.
        # In official code, create as many layers as the total number of layers and select them based on the index
        num_g_prompt = len(self.g_prompt_layer_idx) if self.g_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_g_prompt = use_prefix_tune_for_g_prompt

        self.use_e_prompt = use_e_prompt
        self.e_prompt_layer_idx = e_prompt_layer_idx
        num_e_prompt = len(self.e_prompt_layer_idx) if self.e_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt

        if not self.use_prefix_tune_for_g_prompt and not self.use_prefix_tune_for_g_prompt:
            self.use_g_prompt = False
            self.g_prompt_layer_idx = []

        if use_g_prompt and g_prompt_length is not None and len(g_prompt_layer_idx) != 0:
            if not use_prefix_tune_for_g_prompt:
                g_prompt_shape = (num_g_prompt, g_prompt_length, self.embed_dim)
                if prompt_init == 'zero':
                    self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                elif prompt_init == 'uniform':
                    self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                    nn.init.uniform_(self.g_prompt, -1, 1)
            else:
                if same_key_value:
                    g_prompt_shape = (num_g_prompt, 1, g_prompt_length, self.num_heads, self.embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                    elif prompt_init == 'uniform':
                        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                        nn.init.uniform_(self.g_prompt, -1, 1)
                    self.g_prompt = self.g_prompt.repeat(1, 2, 1, 1, 1)
                else:
                    g_prompt_shape = (num_g_prompt, 2, g_prompt_length, self.num_heads, self.embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                    elif prompt_init == 'uniform':
                        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                        nn.init.uniform_(self.g_prompt, -1, 1)
        else:
            self.g_prompt = None

        if use_e_prompt and e_prompt_layer_idx is not None:
            self.e_prompt = EPrompt(length=prompt_length, embed_dim=self.embed_dim, embedding_key=embedding_key, prompt_init=prompt_init,
                                    prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                                    prompt_key_init=prompt_key_init, num_layers=num_e_prompt, use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
                                    num_heads=self.num_heads, same_key_value=same_key_value)

        self.total_prompt_len = 0
        if self.prompt_pool:
            if not self.use_prefix_tune_for_g_prompt:
                self.total_prompt_len += g_prompt_length * len(self.g_prompt_layer_idx)
            if not self.use_prefix_tune_for_e_prompt:
                self.total_prompt_len += prompt_length * top_k * len(self.e_prompt_layer_idx)

        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        if self.weight_init != 'skip':
            self.init_weights(self.weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        if self.use_g_prompt or self.use_e_prompt:
            if self.use_prompt_mask and train:
                start = task_id * self.e_prompt.top_k
                end = (task_id + 1) * self.e_prompt.top_k
                single_prompt_mask = torch.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.e_prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None

            g_prompt_counter = -1
            e_prompt_counter = -1

            res = self.e_prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            e_prompt = res['batched_prompt']

            for i, block in enumerate(self.blocks):
                if i in self.g_prompt_layer_idx:
                    if self.use_prefix_tune_for_g_prompt:
                        g_prompt_counter += 1
                        # Prefix tunning, [B, 2, g_prompt_length, num_heads, embed_dim // num_heads]
                        idx = torch.tensor([g_prompt_counter] * x.shape[0]).to(x.device)
                        g_prompt = self.g_prompt[idx]
                    else:
                        g_prompt = None
                    x = block(x, prompt=g_prompt)

                elif i in self.e_prompt_layer_idx:
                    e_prompt_counter += 1
                    if self.use_prefix_tune_for_e_prompt:
                        # Prefix tunning, [B, 2, top_k * e_prompt_length, num_heads, embed_dim // num_heads]
                        x = block(x, prompt=e_prompt[e_prompt_counter])
                    else:
                        # Pommpt tunning, [B, top_k * e_prompt_length, embed_dim]
                        prompt = e_prompt[e_prompt_counter]
                        x = torch.cat([prompt, x], dim=1)
                        x = block(x)
                else:
                    x = block(x)
        else:
            x = self.blocks(x)

            res = dict()

        x = self.norm(x)
        res['x'] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res['x']
        if self.class_token and self.head_type == 'token':
            if self.prompt_pool:
                x = x[:, self.total_prompt_len]
            else:
                x = x[:, 0]
        elif self.head_type == 'gap' and self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool and self.class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')

        res['pre_logits'] = x

        x = self.fc_norm(x)

        res['logits'] = self.head(x)

        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res)
        return res


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # modify
    logging.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        # ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if ntok_new > gs_old ** 2:
        ntok_new -= gs_old ** 2
        # expand cls's pos embedding for prompt tokens
        posemb_prefix = posemb_prefix.expand(-1, ntok_new, -1)
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    logging.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                model.patch_embed.grid_size
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict









@register_network("vit_prompt_dual")
class VitDual(BaseNetwork):
    def __init__(self, 
                 model_name: str = "vit_base_patch16_224.augreg_in21k", 
                 pretrained: str_to_bool = True, 
                 num_tasks: int = 10, 
                 num_classes: int = 100, 
                 pool_size: int = 100, 
                 prompt_length: int = 20):
        super().__init__()
        self.n_classes = num_classes
        drop = 0.0
        drop_path = 0.0

        self.original_model = vit_base_patch16_224_dualprompt(
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
        )
        self.original_model.eval()
        top_k = 1
        length = prompt_length
        prompt_pool = True
        prompt_key = True
        initializer = "uniform"
        prompt_key_init = "uniform"
        global_pool = "token"
        head_type = "token"
        same_key_value = False
        size = 10
        batchwise_prompt = False
        embedding_key = "cls"
        use_prompt_mask = True
        #e_prompt hyperparameters
        use_e_prompt = True
        e_prompt_layer_idx = [2, 3, 4]
        use_prefix_tune_for_e_prompt = True
        #g_prompt hyperparameters
        use_g_prompt = True
        g_prompt_length = 5
        g_prompt_layer_idx = [0, 1]
        use_prefix_tune_for_g_prompt = True
        #embed_dim = 768

        self.model = vit_base_patch16_224_dualprompt(
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            prompt_length=length,
            embedding_key=embedding_key,
            prompt_init=prompt_key_init,
            prompt_pool=prompt_pool,
            prompt_key=prompt_key,
            pool_size=size,
            top_k=top_k,
            batchwise_prompt=batchwise_prompt,
            prompt_key_init=prompt_key_init,
            head_type=head_type,
            use_prompt_mask=use_prompt_mask,
            use_g_prompt=use_g_prompt,
            g_prompt_length=g_prompt_length,
            g_prompt_layer_idx=g_prompt_layer_idx,
            use_prefix_tune_for_g_prompt=use_prefix_tune_for_g_prompt,
            use_e_prompt=use_e_prompt,
            e_prompt_layer_idx=e_prompt_layer_idx,
            use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
            same_key_value=same_key_value,
        )

        freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']

        if freeze:
            for p in self.original_model.parameters():
                p.requires_grad = False

            for n, p in self.model.named_parameters():
                if n.startswith(tuple(freeze)):
                    p.requires_grad = False

    def forward(self, x, task_id, train=False, return_outputs=False):

        with torch.no_grad():
            if self.original_model is not None:
                original_model_output = self.original_model(x)
                cls_features = original_model_output['pre_logits']
            else:
                cls_features = None

        outputs = self.model(x, task_id=task_id, cls_features=cls_features, train=train)

        if return_outputs:
            return outputs
        else:
            return outputs['logits']
        
    def get_params(self) -> torch.Tensor:
        param_class = torch.cat([param.reshape(-1) for param in self.model.head.parameters()])
        param_e_prompt = torch.cat([param.reshape(-1) for param in self.model.e_prompt.parameters()])
        param_g_prompt = self.model.g_prompt.reshape(-1)
        return torch.cat([param_class, param_e_prompt, param_g_prompt])
    
    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.model.head.parameters()) + list(self.model.e_prompt.parameters()):
            cand_params = new_params[progress : progress + torch.tensor(pp.size()).prod()].view(pp.size()).detach().clone()
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params
        self.model.g_prompt = nn.Parameter(new_params[progress:].view(self.model.g_prompt.size()).detach().clone())



def create_vision_transformer(variant, base_class=VisionTransformer, pretrained=False, filter_fn=checkpoint_filter_fn, **kwargs):

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = filter_fn

    if variant == 'vit_base_patch16_224_in21k_fn_in1k_old':
        from timm.models import resolve_pretrained_cfg

        pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwpop('pretrained_cfg', None))
        pretrained_cfg.custom_load = True

        return build_model_with_cfg(
            base_class,
            variant,
            pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_filter_fn=_filter_fn,
            pretrained_strict=True,
            **kwargs,
        )
    else:
        return build_model_with_cfg(
            base_class, variant, pretrained,
            pretrained_filter_fn=_filter_fn,
            **kwargs,
        )


def vit_base_patch16_224_dualprompt(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = create_vision_transformer('vit_base_patch16_224.augreg_in21k', base_class=VisionTransformer, filter_fn=checkpoint_filter_fn, pretrained=pretrained, **model_kwargs)
    return model