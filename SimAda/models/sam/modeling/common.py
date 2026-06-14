# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class Adapter_Text(nn.Module):
    def __init__(self, D_features, T_features=512, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.text_projection = nn.Linear(T_features, D_features)

        #self.scale = torch.nn.parameter.Parameter(torch.tensor(0.5))
        
    def forward(self, x, text_embedding):
        # x is (BT, HW+1, D)
        # x shape = torch.Size([1, 64, 64, 768])
        #print('x shape: ',x.shape) 
        # text embedding shape = torch.Size([1, 512])
        #print('text shape: ',text_embedding.shape)

        projected_text = self.text_projection(text_embedding).unsqueeze(1).unsqueeze(1)
        #print('projected text shape: ',projected_text.shape)
        # projected text shape: torch.Size([1, 1, 1, 768])

        projected_text = self.act(projected_text)

        #x_plus_t = x + self.scale * projected_text
        x_plus_t = x + projected_text
        #x_times_t = x * projected_text

        xs = self.D_fc1(x_plus_t)
        #xs = self.D_fc1(x_times_t)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class Adapter_Text_Vis(nn.Module):
    def __init__(self, D_features, T_features=512, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.text_projection = nn.Linear(T_features, D_features)
        self.vis_projection = nn.Linear(T_features, D_features)
        
    def forward(self, x, text_embedding, clip_vis_embeddings):
        # x is (BT, HW+1, D)
        projected_text = self.text_projection(text_embedding).unsqueeze(1).unsqueeze(1)
        projected_text = self.act(projected_text)
        #print('projected text:',projected_text.shape) #(1,1,1,768)

        projected_vis = self.vis_projection(clip_vis_embeddings)

        #print('projected_vis',projected_vis.shape) #(1,1025,768)
        cls_token = projected_vis[:, 0:1, :]                  # (B,1,D)
        patches = projected_vis[:, 1:, :]                      # (B, num_patches, D)
        #patches = patches + cls_token                          # add cls token to every patch (broadcast)
        num_patches = patches.shape[1]
        side = int(num_patches ** 0.5)
        assert side * side == num_patches, "num_patches is not a perfect square"
        projected_vis = patches.view(projected_vis.shape[0], side, side, projected_vis.shape[-1])
        #projected_vis = projected_vis[:,:,:].view(projected_vis.shape[0],32,32,projected_vis.shape[-1]) # testing without cls token

        #projected_vis = projected_vis[:,1:,:].view(projected_vis.shape[0],14,14,projected_vis.shape[-1])
        projected_vis = torch.nn.functional.interpolate(projected_vis.permute(0,3,1,2), (64,64), mode='bilinear').permute(0,2,3,1)
        #print(projected_vis.shape) #(1,64,64,768)

        x_plus_t = x + projected_text
        x_plus_t_plus_v = x_plus_t + projected_vis
        #x_times_t = x * projected_text

        xs = self.D_fc1(x_plus_t_plus_v)
        #xs = self.D_fc1(x_times_t)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs + projected_vis
        else:
            x = xs
        return x
    

class Adapter_Vis(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.vis_projection = nn.Linear(512, D_features)
        
    def forward(self, x, clip_vis_embeddings):
        # x is (BT, HW+1, D)
        projected_vis = self.vis_projection(clip_vis_embeddings)

        #print(projected_vis.shape) #(1,1025,768)
        projected_vis = projected_vis[:,1:,:].view(projected_vis.shape[0],32,32,projected_vis.shape[-1])
        #projected_vis = projected_vis[:,1:,:].view(projected_vis.shape[0],14,14,projected_vis.shape[-1])
        projected_vis = torch.nn.functional.interpolate(projected_vis.permute(0,3,1,2), (64,64), mode='bilinear').permute(0,2,3,1)
        #print(projected_vis.shape) #(1,64,64,768)

        x_plus_t_plus_v = x + projected_vis
        #x_times_t = x * projected_text

        xs = self.D_fc1(x_plus_t_plus_v)
        #xs = self.D_fc1(x_times_t)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs + projected_vis
        else:
            x = xs
        return x

    
class NoAdptr_Text_Vis(nn.Module):
    def __init__(self, D_features, T_features=512, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        #self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        #self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.text_projection = nn.Linear(T_features, D_features)
        self.vis_projection = nn.Linear(T_features, D_features)
        
    def forward(self, x, text_embedding, clip_vis_embeddings):
        # x is (BT, HW+1, D)
        projected_text = self.text_projection(text_embedding).unsqueeze(1).unsqueeze(1)
        #projected_text = self.act(projected_text)

        projected_vis = self.vis_projection(clip_vis_embeddings)

        #print(projected_vis.shape) #(1,1025,768)
        projected_vis = projected_vis[:,1:,:].view(projected_vis.shape[0],32,32,projected_vis.shape[-1])
        #projected_vis = projected_vis[:,1:,:].view(projected_vis.shape[0],14,14,projected_vis.shape[-1])
        projected_vis = torch.nn.functional.interpolate(projected_vis.permute(0,3,1,2), (64,64), mode='bilinear').permute(0,2,3,1)
        #print(projected_vis.shape) #(1,64,64,768)

        x_plus_t = x + projected_text
        x_plus_t_plus_v = x_plus_t + projected_vis
        #x_times_t = x * projected_text

        #xs = self.D_fc1(x_plus_t_plus_v)
        ##xs = self.D_fc1(x_times_t)
        #xs = self.act(xs)
        #xs = self.D_fc2(xs)
        xs = x_plus_t_plus_v
        if self.skip_connect:
            x = x + xs + projected_vis
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x_image, x_text):
        B, N, D = x_image.shape  # image features: [batch, tokens, dim]
        _, M, _ = x_text.shape   # text features: [batch, tokens, dim]

        q = self.to_q(x_image)  # [B, N, D]
        k, v = self.to_kv(x_text).chunk(2, dim=-1)  # [B, M, D], [B, M, D]

        q = q.reshape(B, N, self.heads, D // self.heads).transpose(1, 2)  # [B, heads, N, d]
        k = k.reshape(B, M, self.heads, D // self.heads).transpose(1, 2)  # [B, heads, M, d]
        v = v.reshape(B, M, self.heads, D // self.heads).transpose(1, 2)  # [B, heads, M, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, M]
        attn = attn.softmax(dim=-1)

        out = attn @ v  # [B, heads, N, d]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        return self.to_out(out)

class CrossAttentionWrapper(nn.Module):
    def __init__(self, text_dim=512, vision_dim=768, heads=8):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, vision_dim)
        self.cross_attn = CrossAttention(vision_dim, heads)
        self.norm = nn.LayerNorm(vision_dim)

    def forward(self, x_img, x_text):
        B, H, W, D = x_img.shape  # [B, 64, 64, 768]
        x_img = x_img.view(B, H * W, D)     # [B, 4096, 768]

        if x_text.ndim == 2:  # [B, 512]
            x_text = x_text.unsqueeze(1)   # [B, 1, 512]
        x_text = self.text_proj(x_text)    # [B, T, 768]

        x_out = self.cross_attn(self.norm(x_img), x_text)
        x_img = x_img + x_out  # residual
        return x_img.view(B, H, W, D)


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



def to_c(x):
    return x.permute(0,3,1,2)

def to_f(x):
    return x.permute(0,2,3,1)