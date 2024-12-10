import torch
import torch.nn as nn
from .vit import VisionTransformer
import copy
import timm
from _networks import register_network
from _networks._utils import BaseNetwork
import timm.models.vision_transformer as timm_vit
from functools import partial
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_, DropPath
from utils.tools import str_to_bool


class HGPPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, prompt_pool=False):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.prompt_pool = prompt_pool

        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4]

    def process_task_count(self):
        if not self.prompt_pool:
            return
        self.task_count += 1
        for e in self.e_layers:
            K = getattr(self, f"e_k_{e}")
            P = getattr(self, f"e_p_{e}")
            k = self.gram_schmidt(K)
            p = self.gram_schmidt(P)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            redo_counter = 0
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print(f"restarting!!! {redo_counter}")
                            redo_counter += 1
                            if redo_counter > 100:
                                print("failed to init prompt")
                                raise ValueError("failed to init prompt")
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            # B, C = x_querry.shape

            K = getattr(self, f"e_k_{l}")
            p = getattr(self, f"e_p_{l}")
            # pt = int(self.e_pool_size / (self.n_tasks))
            # s = int(self.task_count * pt) if not self.prompt_pool else 0
            # f = int((self.task_count + 1) * pt) if not self.prompt_pool else self.e_pool_size

            # # freeze/control past tasks
            # if train:
            #     if self.task_count > 0:
            #         K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
            #         p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
            #     else:
            #         K = K[s:f]
            #         p = p[s:f]
            # else:
            #     K = K[0:f]
            #     p = p[0:f]

            # # with attention and cosine sim
            # n_K = nn.functional.normalize(K, dim=1)
            # q = nn.functional.normalize(x_querry, dim=1)
            # aq_k = torch.einsum("bd,kd->bk", q, n_K)
            # P_ = torch.einsum("bk,kld->bld", aq_k, p)

            P_ = p.repeat(x_block.shape[0], 1, 1)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).to(t.device)) ** 2).mean()


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False, prompt=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False, prompt=None):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        ckpt_layer=0,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, task_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):

            if prompt is not None:
                if train:
                    p_list, loss, x = prompt.forward(q, i, x, train=True, task_id=task_id)
                    prompt_loss += loss
                else:
                    p_list, _, x = prompt.forward(q, i, x, train=False, task_id=task_id)
            else:
                p_list = None

            x = blk(x, register_blk == i, prompt=p_list)

        x = self.norm(x)

        return x, prompt_loss


@register_network("vit_hgp")
class VitHGP(BaseNetwork):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.augreg_in21k",
        num_classes: int = 100,
        pretrained: str_to_bool = True,
        n_prompts: int = 1,
        prompt_length: int = 200,
        prompt_pool: str_to_bool = False,
        num_tasks: int = 10,
    ):
        super().__init__()
        # vit_model = timm_vit.__dict__["vit_base_patch16_224"](pretrained=pretrained, num_classes=num_classes)
        vit_model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            ckpt_layer=0,
            drop_path_rate=0,
        )
        # from timm.models import vit_base_patch16_224
        load_dict = timm.create_model(model_name, pretrained=True).state_dict()
        if not "dino" in model_name:
            del load_dict["head.weight"]
            del load_dict["head.bias"]
        vit_model.load_state_dict(load_dict, strict=False)
        self.last = nn.Linear(768, num_classes)
        prompt_param = [n_prompts, prompt_length]
        self.prompt = HGPPrompt(768, num_tasks, prompt_param, prompt_pool=prompt_pool)
        self.feat = vit_model

    def forward(self, x, pen=False, train=True):
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        if self.prompt is not None:
            # with torch.no_grad():
            #     q, _ = self.feat(x)
            #     q = q[:, 0, :]
            out, prompt_loss = self.feat(x, prompt=self.prompt, q=None, train=train, task_id=-1)
            out = out[:, 0, :]
        else:
            out, _ = self.feat(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out  # , prompt_loss
        else:
            return out

    def get_params(self, only_trainable=False) -> torch.Tensor:
        if not only_trainable:
            return super().get_params()
        return torch.cat([param.reshape(-1) for param in list(self.last.parameters()) + list(self.prompt.parameters())])
