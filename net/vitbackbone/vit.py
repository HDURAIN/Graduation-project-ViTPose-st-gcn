#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_


# In[2]:


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


# In[3]:


class PatchEmbed(nn.Module):
    """
    Split image into patches and then embed them.
    
    Parameters
    ----------
    img_size : int
        Size of image
        
    patch_size : int
        Size of patches
        
    in_chans : int
        Number of input channels
    
    embed_dim : int
        Number of embeding dimension
    
    ratio : int
        Split ratio of every patch on height and width. If ratio == 1, patch_shape == origin_patch_shape
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))
        
    def forward(self, x, **kwargs):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape '(batch_size, in_chans, img_size_height, img_size_width)'.
        
        Returns
        -------
        torch.Tensor
            Shape '(batch_size, n_patches, embed_dim)'.
        """
        B, C, H, W = x.shape
        x = self.proj(x) # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2) # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2) # (batch_size, n_patches, embed_dim)
        
        return x, (Hp, Wp)


# In[4]:


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# In[5]:


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self attention 多头自注意力 论文中的MHSA
    
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    
    num_heads : int
        Number of attension heads.
            
    qkv_bias : bool
        If true then we include bias to the query, key and value projections.
            
    attn_drop : float
        Dropout probability applied to the query, key and value tensors.
        
    proj_drop : float
        Dropout probability applied to the output tensor.
        
    Attributes
    ----------
    qk_scale : float
        Normalizing constant for the dot product.
    
    attn_head_dim : int
        The dimension of attension heads.
        
    qkv : nn.Linear
        Linear projection for the query, key and value
        
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attension
        heads and maps it into a new space.
        
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5 # scale comes from <<Attension is all you need>>
        
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape '(batch_size, n_patches + 1, dim)'.
        +1 beacause we need a class token

        Returns
        -------
        torch.Tensor
            Shape '(batch_size, n_patches + 1, dim)'.
        """
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (batch_size, n_patches + 1, 3*dim or 3*all_head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1) # (batch_size, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        attn = (q @ k_t) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)

        attn = attn.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = (attn @ v) # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg) # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x) # (n_samples, n_patches + 1, dim)

        return x


# In[6]:


class Mlp(nn.Module):
    """Multilayer perception.
    Feed forward network 前馈神经网络，论文中的FFN
    
    Parameters
    ----------
    in_features : int
        Number of input features.
    
    hidden_features : int
        Number of nodes in the hidden layer.
    
    out_features : int
        Number of output features.
    
    drop : float
        Dropout probability.
    
    Attribute
    ---------
    fc : nn.Linear
        The Firsst linear layer.
    
    act : nn.GELU
        GELU activation function.
    
    fc2 : nn.Linear
        The second linear layer.
    
    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,drop=0.):
        super().__init__()
        # 如果out_features和hidden_features初始为None,赋in_features的值
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        """Run forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape '(batch_size, n_patches + 1, in_features)'.
        
        Returns
        -------
        torch.Tensor
            Shape '(batch_size, n_patches + 1, in_features)'.
        """
        x = self.fc1(x) # (batch_size, n_patches + 1, hidden_features)
        x = self.act(x) # (batch_size, n_patches + 1, hidden_features)
        x = self.drop(x) # (batch_size, n_patches + 1, hidden_features)
        x = self.fc2(x) # (batch_size, n_patches + 1, hidden_features)
        x = self.drop(x) # (batch_size, n_patches + 1, hidden_features)
        
        return x


# In[7]:


class Block(nn.Module):
    """
    Parameters
    ----------
    dim : int 
        Embedding dimension.
    
    n_heads : int
        Number of attension heads.
    
    mlp_ratio : float
        Determines the hidden dimension size of the 'MLP' module with respect to 'dim'.
    
    qkv_bias : bool
        If true then we include bias to the query, key and value projecitons.
    
    p, attn_p : float
        Dropout probability.
    
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization
    
    attn : MultiHeadSelfAttention
        MultiHeadAttention module
    
    mlp : MLP
        MLP module
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0.,act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
                    dim, 
                    num_heads=num_heads, 
                    qkv_bias=qkv_bias, 
                    qk_scale=qk_scale, 
                    attn_drop=attn_drop, 
                    proj_drop=drop, 
                    attn_head_dim=attn_head_dim
        )
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio) # hidden features
        self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim,  
                act_layer=act_layer, 
                drop = drop
        )
    
    def forward(self, x):
        """Run forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape '(batch_size, n_patches + 1, dim)'.
        
        Returns
        -------
        torch.Tensor
            Shape '(batch_size, n_patches + 1, dim)'.
        """
        
        x = x + self.drop_path(self.attn(self.norm1(x))) # with residual
        x = x + self.drop_path(self.mlp(self.norm2(x))) # with residual
        
        return x


# In[ ]:


# @BACKBONES.register_module()
class ViT(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True, 
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        """Encoder 的depth个block
        """
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02) # 截断正态分布

        self._freeze_stages()
        
    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()

