# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type


class ConvBlock(nn.Module):
    def __init__(self, D_features, r_features, kernel_size=3, act_layer=nn.GELU):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(D_features,r_features,kernel_size = 1, padding = 'same', bias = True)

        
        self.conv_l_1 = nn.Conv2d(r_features,
                              r_features,
                              kernel_size = kernel_size,
                              padding = 'same',
                              bias = True)

        self.conv2 = nn.Conv2d(r_features,D_features,kernel_size = 1, padding = 'same', bias = True)
        self.act = act_layer()
        self.scale = torch.nn.parameter.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_orig = x.permute(0,3,1,2).clone()
        # x : (B, window_size, window_size, embed_dim)
        x = self.conv1(x.permute(0,3,1,2)) # x shape (B,r,H,W)
        xe = x.clone()
        x = self.conv_l_1(x)
        x = self.act(x)
        x = x + self.scale*xe
        x = self.conv2(x)
        x = x + self.scale*x_orig
        return x.permute(0,2,3,1)




# class ConvBlock(nn.Module):
#     def __init__(self, D_features, r_features, kernel_size=3, act_layer=nn.GELU):
#         super().__init__()
#         self.kernel_size = kernel_size
#         #self.fc1 = nn.Linear(D_features, r_features)
#         #self.fc2 = nn.Linear(r_features,D_features)
#         self.conv1 = nn.Conv2d(D_features,r_features,kernel_size = 1, padding = 'same', bias = True)
#         #self.act = act_layer()
#         self.conv_l_1 = nn.Conv2d(r_features,
#                               r_features,
#                               kernel_size = kernel_size,
#                               padding = 'same',
#                               bias = True)
#         self.conv_l_2 = nn.Conv2d(r_features,
#                       r_features,
#                       kernel_size = kernel_size,
#                       padding = 'same',
#                       bias = True)
        
#         self.conv2 = nn.Conv2d(r_features,D_features,kernel_size = 1, padding = 'same', bias = True)


#     def forward(self, x):
#         # x : (B, window_size, window_size, embed_dim)
#         #x = self.fc1(x)
#         x = self.conv1(x.permute(0,3,1,2))
#         #x = self.act(x)
#         x = nn.functional.interpolate(x,scale_factor = 2,mode='bilinear')
#         x = self.conv_l_1(x)
#         x = self.conv_l_2(x)
#         x = nn.functional.interpolate(x,scale_factor = 0.5, mode = 'bilinear')
#         #x = self.conv(x.permute(0,3,1,2))
#         #x = self.fc2(x.permute(0,2,3,1))
#         x = self.conv2(x)
#         return x.permute(0,2,3,1)

class Conv_Scale_Block(nn.Module):
    def __init__(self, D_features, r_features, kernel_size=3, act_layer=nn.GELU, num_experts = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.conv1 = nn.Conv2d(D_features,r_features,kernel_size = 1, padding = 'same', bias = True)
        self.conv_l_1 = nn.Conv2d(r_features,
                              r_features,
                              kernel_size = kernel_size,
                              padding = 'same',
                              bias = True)
        

        self.conv2 = nn.Conv2d(r_features,D_features,kernel_size = 1, padding = 'same', bias = True)
        self.avg_pool = nn.AvgPool2d(kernel_size=64)
        self.H = nn.Linear(16,self.num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x : (B, window_size, window_size, embed_dim)
        x = self.conv1(x.permute(0,3,1,2)) # x shape (B,r,H,W)
        pooled_x = self.avg_pool(x) # average pooling x, shape (B,r,1,1)
        reshaped_pooled_x = torch.reshape(pooled_x,(pooled_x.shape[:2])) # reshaped pooled_x, shape (B,r)
        H_out = self.H(reshaped_pooled_x) # H_out = H(x) shape (B,num_experts)
        _,indices = torch.sort(H_out,dim=1,descending=True)
        H_filtered = torch.clone(H_out)
        
        for b in range(indices.shape[0]):
            for ind in indices[b][2:]:
                H_filtered[b][ind] = -torch.inf
                
        G_out = self.softmax(H_filtered)  # G_out = G(x) shaped (B,num_experts)
        E = []
        
        for i in range(self.num_experts):
            
            xs = nn.functional.interpolate(x,scale_factor = 2**i, mode='bilinear')
            xs = self.conv_l_1(xs)
            xs = nn.functional.interpolate(xs,scale_factor = 0.5**i, mode = 'bilinear')
            E.append(xs)

        E = torch.stack(E)
        G_out = torch.reshape(G_out,(G_out.shape[1],G_out.shape[0],1,1,1))
        x = (E*G_out).sum(dim=0)

        x = self.conv2(x)
        return x.permute(0,2,3,1)


class Conv_All_Scale_Block(nn.Module):
    def __init__(self, D_features, r_features, kernel_size=3, act_layer=nn.GELU, num_experts = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.conv1 = nn.Conv2d(D_features,r_features,kernel_size = 1, padding = 'same', bias = True)
        self.convlist1 = nn.ModuleList()
        # self.conv_l2 = nn.Conv2d(r_features,
        #                       r_features,
        #                       kernel_size = kernel_size,
        #                       padding = 'same',
        #                       bias = True)
        
        for i in range(self.num_experts):
            conv_l = nn.Conv2d(r_features,
                              r_features,
                              kernel_size = kernel_size,
                              padding = 'same',
                              bias = True)
            self.convlist1.append(conv_l)

        
        # self.conv_l_1 = nn.Conv2d(r_features,
        #                       r_features,
        #                       kernel_size = kernel_size,
        #                       padding = 'same',
        #                       bias = True)
        # self.conv_l_2 = nn.Conv2d(r_features,
        #                       r_features,
        #                       kernel_size = kernel_size,
        #                       padding = 'same',
        #                       bias = True)

        self.conv2 = nn.Conv2d(r_features,D_features,kernel_size = 1, padding = 'same', bias = True)
        ## V2

        #initial_weights = torch.rand((self.num_experts,1,1,1,1),requires_grad = True)
        initial_weights = torch.ones((self.num_experts,1,1,1,1),requires_grad = True)
        # Normalize the initial weights so they sum to 1
        normalized_weights = initial_weights / initial_weights.sum()
        # Convert the normalized tensor to a learnable parameter
        self.w_a = torch.nn.parameter.Parameter(normalized_weights)

        #self.w_a = torch.nn.parameter.Parameter(torch.rand((self.num_experts,1,1,1,1),requires_grad = True))
        self.act = act_layer()
        #self.scale = torch.nn.parameter.Parameter(torch.tensor(1))
        self.scale = torch.nn.parameter.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_orig = x.permute(0,3,1,2).clone()
        # x : (B, window_size, window_size, embed_dim)
        x = self.conv1(x.permute(0,3,1,2)) # x shape (B,r,H,W)
        xe = x.clone()
        E = []
        for i in range(self.num_experts):
            
            xs = nn.functional.interpolate(x,scale_factor = 2**i, mode='bilinear')
            xs = self.convlist1[i](xs)
            xs = self.act(xs)
            xs = nn.functional.interpolate(xs,scale_factor = 0.5**i, mode = 'bilinear')
            E.append(xs)

        E = torch.stack(E)
        #x = E.sum(dim=0)
        # V2
        norm_w_a = F.softmax(self.w_a,dim=0)
        #self.w_a = self.w_a.softmax(0) #normalizing
        #x = (E*self.w_a).sum(dim=0)/self.w_a.sum(dim=0)
        x = (E*norm_w_a).sum(dim=0)/norm_w_a.sum(dim=0)
        x = x + self.scale*xe
        #x = self.conv_l2(x)
        #x = x + xe
        x = self.conv2(x)
        x = x + self.scale*x_orig
        return x.permute(0,2,3,1)


class Conv_Adapter_All_Scale_Block(nn.Module):
    def __init__(self, D_features, r_features, kernel_size=3, act_layer=nn.GELU, num_experts = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.fc1 = nn.Linear(D_features,D_features//4)
        self.conv1 = nn.Conv2d(D_features//4,r_features,kernel_size = 1, padding = 'same', bias = True)
        self.convl = nn.Conv2d(r_features,
                              r_features,
                              kernel_size = kernel_size,
                              padding = 'same',
                              bias = True)
        
        self.fc2 = nn.Linear(r_features,D_features//4)
        self.conv2 = nn.Conv2d(D_features//4,D_features,kernel_size = 1, padding = 'same', bias = True)
        ## V2
        initial_weights = torch.ones((self.num_experts,1,1,1,1),requires_grad = True)
        # Normalize the initial weights so they sum to 1
        normalized_weights = initial_weights / initial_weights.sum()
        # Convert the normalized tensor to a learnable parameter
        self.w_a = torch.nn.parameter.Parameter(normalized_weights)
        self.act = act_layer()
        self.scale = torch.nn.parameter.Parameter(torch.tensor(1.0))
        self.layer_attention = LayerAttentionAdapter(num_experts, embed_dim=r_features, hidden_dim=2*r_features)

    def forward(self, x):
        x_orig = x.permute(0,3,1,2).clone()
        # x : (B, window_size, window_size, embed_dim)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv1(x.permute(0,3,1,2)) # x shape (B,r,H,W)
        x = self.act(x)
        xe = x.clone()
        E = []
        for i in range(self.num_experts):
            xs = nn.functional.interpolate(x,scale_factor = 2**i, mode='nearest')
            xs = self.convl(xs)
            xs = self.act(xs)
            xs = nn.functional.interpolate(xs,scale_factor = 0.5**i, mode = 'nearest')
            E.append(xs)

        E = torch.stack(E)
        # V2
        # norm_w_a = F.softmax(self.w_a,dim=0)
        # x = (E*norm_w_a).sum(dim=0)/norm_w_a.sum(dim=0)
        
        # Apply layer attention to get the weighted expert outputs
        x = self.layer_attention(E)  # Output weighted expert result

        
        x = x + self.scale*xe
        x = self.fc2(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x + self.scale*x_orig
        return x.permute(0,2,3,1)

class Adapter_Block(nn.Module):
    def __init__(self, D_features, r_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(D_features,r_features)       
        self.fc2 = nn.Linear(r_features,D_features)
        self.act = act_layer()
        self.scale = torch.nn.parameter.Parameter(torch.tensor(0.5))
        

    def forward(self, x):
        x_orig = x.clone()
        # x : (B, window_size, window_size, embed_dim)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = x + self.scale*x_orig
        return x

class LayerAttentionAdapter(nn.Module):
    def __init__(self, num_experts, embed_dim, hidden_dim):
        super(LayerAttentionAdapter, self).__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim

        # Define a small attention network
        self.attention_fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),  # Projection to hidden dimension
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)  # Output size is num_experts (one attention score per expert)
        )

    def forward(self, expert_outputs):
        """
        expert_outputs: tensor of shape (num_experts, B, C, H, W)
        """
        num_experts, B, C, H, W = expert_outputs.shape

        # Flatten the spatial dimensions and mean pool over (H, W) to create a summary of each expert's output
        expert_outputs_flat = expert_outputs.view(num_experts, B, C, -1)  # Shape: (num_experts, B, C, H*W)
        pooled_x = expert_outputs_flat.mean(dim=-1)  # Shape: (num_experts, B, C)

        # Now, compute a feature summary for attention from the pooled expert outputs
        pooled_summary = pooled_x.mean(dim=0)  # Shape: (B, C), averaged across experts

        # Use the pooled summary to generate attention weights for each expert
        attention_weights = self.attention_fc(pooled_summary)  # Shape: (B, num_experts)
        attention_weights = F.softmax(attention_weights, dim=-1)  # Normalize weights over experts

        # Reshape attention weights for broadcasting over experts' outputs
        attention_weights = attention_weights.view(B, num_experts, 1, 1, 1)  # Shape: (B, num_experts, 1, 1, 1)

        # Apply the attention weights to expert outputs and combine them
        expert_outputs_weighted = (expert_outputs.permute(1, 0, 2, 3, 4) * attention_weights).sum(dim=1)  # Shape: (B, C, H, W)

        return expert_outputs_weighted



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

class Adapter_Conv(nn.Module):
    def __init__(self, D_features, mlp_ratio, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.conv1 = nn.Conv2d(D_features,
                              D_hidden_features,
                              kernel_size = 1,
                              padding = 'same',
                              bias = True)
        # self.convl = nn.Conv2d(D_hidden_features,D_hidden_features,
        #                        kernel_size=3,
        #                        padding = 'same',
        #                        bias=True)
        self.conv2 = nn.Conv2d(D_hidden_features,
                              D_features,
                              kernel_size = 1,
                              padding = 'same',
                              bias = True)
        # self.num_experts = 2
        # initial_weights = torch.ones((self.num_experts,1,1,1,1),requires_grad = True)
        # # Normalize the initial weights so they sum to 1
        # normalized_weights = initial_weights / initial_weights.sum()
        # # Convert the normalized tensor to a learnable parameter
        # self.w_a = torch.nn.parameter.Parameter(normalized_weights)
        
        #self.scale1 = torch.nn.parameter.Parameter(torch.tensor(0.5))
        # if self.skip_connect:
            
        #     self.scale2 = torch.nn.parameter.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        x = x.permute(0,3,1,2)
        xs = self.conv1(x)
        xs = self.act(xs)
        #xs = self.act(xs)
        #xs_clone = xs.clone()
        #xs = self.convl(xs)
        #xs = self.act(xs)
        # E = []
        # for i in range(self.num_experts):
        #     xs_scaled = nn.functional.interpolate(xs,scale_factor = 2**i, mode='nearest')
        #     xs_scaled = self.convl(xs_scaled)
        #     xs_scaled = self.act(xs_scaled)
        #     xs_scaled = nn.functional.interpolate(xs_scaled,scale_factor = 0.5**i, mode = 'nearest')
        #     E.append(xs_scaled)
        # E = torch.stack(E)
        # norm_w_a = F.softmax(self.w_a,dim=0)
        # xs = (E*norm_w_a).sum(dim=0)/norm_w_a.sum(dim=0)
            
        #xs = self.scale1*xs_clone + xs
        xs = self.conv2(xs)
        xs = self.act(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x.permute(0,2,3,1)


class Adapter_Conv_v0(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.convl = nn.Conv2d(D_hidden_features,
                              D_hidden_features,
                              kernel_size = 3,
                              padding = 'same',
                              bias = True)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = xs + to_f(self.convl(to_c(xs)))
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Adapter_Conv_Text(nn.Module):
    def __init__(self, D_features, T_features=512, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.text_projection = nn.Linear(T_features, D_features)
        self.convl = nn.Conv2d(D_hidden_features,
                              D_hidden_features,
                              kernel_size = 3,
                              padding = 'same',
                              bias = True)
        
    def forward(self, x, text_embedding):
        #print('Inside conv text block: text embedding shape: ',text_embedding.shape)
        #projected_text = self.text_projection(text_embedding).unsqueeze(1)  # Expand for broadcasting
        projected_text = self.text_projection(text_embedding).unsqueeze(1).unsqueeze(1)
        
        #print('text_projection shape: ',projected_text.shape)
        #print('x shape: ',x.shape)
        x = x + projected_text  # Fuse text embedding
        #print('after adding, x shape: ',x.shape)
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = xs + to_f(self.convl(to_c(xs)))
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter_Text_Gated_0(nn.Module):
    """
    Text-image similarity gating on top of your FC-Act-FC adapter.

    Shapes supported:
      - Image features: (B, H, W, D)  [recommended here]
      - Or tokens:      (B, N, D)     (will behave as H*W=N)
    """
    def __init__(
        self,
        D_features: int,
        #T_features: int = 512,
        mlp_ratio: float = 0.25,
        act_layer=nn.GELU,
        skip_connect: bool = True,
        use_conv_refiner: bool = True,   # set False to remove the tiny conv head
        gate_on: str = "pre",            # "pre" | "post" | "both"
        zero_init: bool = True,
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.use_conv_refiner = use_conv_refiner
        self.gate_on = gate_on
        self.act = act_layer()

        D_hidden = max(1, int(D_features * mlp_ratio))

        # --- your adapter MLP (FC-Act-FC), unchanged ---
        self.D_fc1 = nn.Linear(D_features, D_hidden)
        self.D_fc2 = nn.Linear(D_hidden, D_features)

        # --- project text to D ---
        #self.text_projection = nn.Linear(T_features, D_features)

        # --- FiLM-style per-channel modulation derived from text (optional but helps) ---
        self.gamma = nn.Linear(D_features, D_features)  # scale
        self.beta  = nn.Linear(D_features, D_features)  # bias

        # --- learnable temperature/offset for the similarity gate ---
        self.gate_temp = nn.Parameter(torch.tensor(20.0))  # higher => sharper gate
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

        # --- lightweight conv refiner over [x_gated ⊕ sim] ---
        if use_conv_refiner:
            # self.refine_dw = nn.Conv2d(D_features + 1, D_features, 3, padding=1, groups=D_features)
            # self.refine_pw = nn.Conv2d(D_features, D_features, 1)
            self.refine_dw = nn.Conv2d(D_features, D_features, 3, padding=1, groups=D_features)
            self.sim_proj  = nn.Conv2d(1, D_features, 1)   # project sim → D channels
            self.refine_pw = nn.Conv2d(D_features, D_features, 1)

        # --- residual safety: a tiny layer scale on the adapter delta ---
        #self.layer_scale = nn.Parameter(torch.zeros(1)) if zero_init else nn.Parameter(torch.ones(1))
        #self.layer_scale = nn.Parameter(torch.tensor(1e-3))
        self.layer_scale = nn.Parameter(torch.ones(1))

        # zero-init for stable start (near-identity behavior)
        if zero_init:
            nn.init.zeros_(self.D_fc2.weight); nn.init.zeros_(self.D_fc2.bias)
            nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta .weight); nn.init.zeros_(self.beta .bias)
            if use_conv_refiner:
                nn.init.zeros_(self.refine_pw.weight); nn.init.zeros_(self.refine_pw.bias)

    @staticmethod
    def _as_bhwc(x):
        # Accept (B,H,W,D) or (B,N,D). If (B,N,D), map to (B,H=1,W=N,D) just to reuse code.
        if x.dim() == 4:
            return x, x.shape[1], x.shape[2], True  # (B,H,W,D), is_grid=True
        elif x.dim() == 3:
            B, N, D = x.shape
            return x.view(B, 1, N, D), 1, N, False   # fake H=1
        else:
            raise ValueError("x must be (B,H,W,D) or (B,N,D)")

    @staticmethod
    def _from_bhwc(y, is_grid):
        # Return to original shape
        if is_grid:
            return y
        else:
            B, H, W, D = y.shape
            return y.view(B, W, D)  # since H==1

    def forward(self, x, text_embedding):
        """
        x: (B,H,W,D) or (B,N,D)
        text_embedding: (B,T_features)
        """
        B = x.shape[0]
        D = x.shape[-1]

        # --- project + normalize text for cosine sim ---
        #t = self.text_projection(text_embedding)            # (B,D)
        t = text_embedding
        t_dir = F.normalize(t, dim=-1).view(B, 1, 1, D)     # (B,1,1,D)

        # --- get BHWC view of image features ---
        x_bhwc, H, W, is_grid = self._as_bhwc(x)
        # cosine similarity per spatial location
        x_dir = F.normalize(x_bhwc, dim=-1)
        sim = (x_dir * t_dir).sum(dim=-1, keepdim=True)     # (B,H,W,1)

        # --- build the gate ---
        gate = torch.sigmoid(self.gate_temp * sim + self.gate_bias)  # (B,H,W,1)

        # --- optional FiLM from text (per-channel)
        gamma = self.gamma(t).view(B, 1, 1, D)
        beta  = self.beta (t).view(B, 1, 1, D)

        # ----- Pre-gating path: modulate adapter input -----
        if self.gate_on in ("pre", "both"):
            # multiplicative (1 + gate*γ) and additive gate*β
            x_mod = (1 + gate * gamma) * x_bhwc + gate * beta
        else:
            x_mod = x_bhwc

        # ----- Your adapter MLP (pointwise over last dim) -----
        y = self.D_fc1(x_mod)
        y = self.act(y)
        y = self.D_fc2(y)  # (B,H,W,D)

        # ----- Post-gating path: modulate adapter output -----
        if self.gate_on in ("post", "both"):
            y = y * gate

        # ----- Optional conv refiner on [gated features ⊕ sim] -----
        if self.use_conv_refiner:
            y_bchw = y.permute(0, 3, 1, 2)                  # (B,D,H,W)
            sim_bchw = sim.permute(0, 3, 1, 2)              # (B,1,H,W)
            z = self.refine_dw(y_bchw)              # depthwise over features
            z = z + self.sim_proj(sim_bchw)         # inject similarity signal
            z = F.gelu(z)
            z = self.refine_pw(z)                   # pointwise mix
            y = z.permute(0, 2, 3, 1)
            # z = torch.cat([y_bchw, sim_bchw], dim=1)        # (B,D+1,H,W)
            # z = self.refine_dw(z)
            # z = F.gelu(z)
            # z = self.refine_pw(z)
            # y = z.permute(0, 2, 3, 1)                       # back to (B,H,W,D)

        # ----- Residual add with layer scale -----
        out = x_bhwc + self.layer_scale * y if self.skip_connect else y

        # return to original shape and also expose the heatmap if caller wants it
        return self._from_bhwc(out, is_grid), sim.squeeze(-1)  # (B,H,W) heatmap

import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter_Text_Gated(nn.Module):
    def __init__(
        self,
        D_features: int,
        mlp_ratio: float = 0.33,        # a touch wider
        act_layer=nn.GELU,
        skip_connect: bool = True,
        use_conv_refiner: bool = True,
        gate_on: str = "both",          # pre + post by default
        zero_init: bool = True,
        K_gates: int = 4,               # multi-head gating
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.use_conv_refiner = use_conv_refiner
        self.gate_on = gate_on
        self.act = act_layer()
        D = D_features
        Dh = max(1, int(D * mlp_ratio))

        # -------- GLU MLP (stronger than vanilla GELU MLP)
        self.fc_in  = nn.Linear(D, 2 * Dh)      # for GLU: splits into (u, v)
        self.fc_out = nn.Linear(Dh, D)

        # -------- Rank-Reduced FiLM + learnable scale
        r = max(4, D // 4)
        self.film_scale = nn.Parameter(torch.tensor(0.25))
        self.gamma = nn.Sequential(nn.Linear(D, r), nn.GELU(), nn.Linear(r, D))
        self.beta  = nn.Sequential(nn.Linear(D, r), nn.GELU(), nn.Linear(r, D))

        # -------- Multi-head gates with softplus temps + centering
        self.raw_temp = nn.Parameter(torch.ones(K_gates) * 1.5)  # temp ≈ 2–2.5
        self.gate_mix = nn.Parameter(torch.full((K_gates,), 1.0 / K_gates))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))
        self.sim_eps = 1e-6

        # -------- Multi-branch DW refiner with dilations + learned mixing
        if use_conv_refiner:
            self.dw3 = nn.Conv2d(D, D, 3, padding=1, groups=D)
            self.dw5 = nn.Conv2d(D, D, 3, padding=2, dilation=2, groups=D)
            self.dw7 = nn.Conv2d(D, D, 3, padding=3, dilation=3, groups=D)
            self.branch_mix = nn.Parameter(torch.ones(3))   # learned weights
            mid = max(1, D // 2)
            self.pw_reduce = nn.Conv2d(D, mid, 1)
            self.sim_proj  = nn.Conv2d(1, mid, 1)
            self.sim_gain  = nn.Parameter(torch.tensor(0.1))
            self.pw_expand = nn.Conv2d(mid, D, 1)

        # -------- Residual scale (small but nonzero)
        self.layer_scale = nn.Parameter(torch.tensor(5e-3))

        # -------- Stable zero-inits where helpful
        if zero_init:
            nn.init.zeros_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)
            if use_conv_refiner:
                nn.init.zeros_(self.pw_expand.weight); nn.init.zeros_(self.pw_expand.bias)

    @staticmethod
    def _as_bhwc(x):
        if x.dim() == 4:  # (B,H,W,D)
            return x, x.shape[1], x.shape[2], True
        elif x.dim() == 3:  # (B,N,D) -> pretend (1,N,D)
            B,N,D = x.shape
            return x.view(B,1,N,D), 1, N, False
        else:
            raise ValueError("x must be (B,H,W,D) or (B,N,D)")

    @staticmethod
    def _from_bhwc(y, is_grid):
        if is_grid: return y
        B,H,W,D = y.shape
        return y.view(B, W, D)  # H==1

    def _make_gate(self, sim):
        # per-sample centering/whitening
        mu  = sim.mean((1,2,3), keepdim=True)
        std = sim.std ((1,2,3), keepdim=True) + self.sim_eps
        sim_n = (sim - mu) / std
        # K gates, softplus(temp_k) > 0
        gates = []
        for k in range(self.raw_temp.numel()):
            temp_k = F.softplus(self.raw_temp[k])
            gates.append(torch.sigmoid(temp_k * sim_n + self.gate_bias))
        w = F.softmax(self.gate_mix, dim=0)
        gate = torch.stack(gates, dim=-1) @ w   # (B,H,W,1)
        return gate

    def forward(self, x, t):
        # x: (B,H,W,D) or (B,N,D), t: (B,D) (already projected to D outside)
        B = x.shape[0]; D = x.shape[-1]
        x_bhwc, H, W, is_grid = self._as_bhwc(x)

        # cosine similarity map
        x_dir = F.normalize(x_bhwc, dim=-1)
        t_dir = F.normalize(t, dim=-1).view(B,1,1,D)
        sim   = (x_dir * t_dir).sum(dim=-1, keepdim=True)     # (B,H,W,1)
        gate  = self._make_gate(sim)                          # (B,H,W,1)

        # pre-FiLM (rank-reduced) + gate
        if self.gate_on in ("pre","both"):
            g = self.gamma(t).view(B,1,1,D)
            b = self.beta (t).view(B,1,1,D)
            x_mod = (1 + self.film_scale * gate * g) * x_bhwc + self.film_scale * gate * b
        else:
            x_mod = x_bhwc

        # GLU MLP
        uv = self.fc_in(x_mod)
        u, v = uv.chunk(2, dim=-1)
        y = u * torch.sigmoid(v)            # GLU
        y = self.fc_out(y)                  # (B,H,W,D)

        # post gate
        if self.gate_on in ("post","both"):
            y = y * gate

        # multi-branch DW refiner
        if self.use_conv_refiner:
            y_bchw   = y.permute(0,3,1,2)
            sim_bchw = sim.permute(0,3,1,2)
            branches = [self.dw3(y_bchw), self.dw5(y_bchw), self.dw7(y_bchw)]
            w = F.softmax(self.branch_mix, dim=0)
            z = w[0]*branches[0] + w[1]*branches[1] + w[2]*branches[2]
            z = self.pw_reduce(z) + self.sim_gain * self.sim_proj(sim_bchw)
            z = F.gelu(z)
            z = self.pw_expand(z)
            y = z.permute(0,2,3,1)

        out = x_bhwc + self.layer_scale * y if self.skip_connect else y
        return self._from_bhwc(out, is_grid), sim.squeeze(-1)


class Adapter_Text_2(nn.Module):
    """
    Simple text-conditioned adapter:
      - compute cosine similarity per location
      - concat sim to features → (B,H,W,D+1)
      - project back to D before MLP
    """
    def __init__(self, D_features: int, mlp_ratio: float = 0.25,
                 act_layer=nn.GELU, skip_connect: bool = True):
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        D = D_features
        Dh = max(1, int(D * mlp_ratio))

        # projection from D+1 → D
        self.sim_proj = nn.Linear(D+1, D)
        #self.sim_proj = nn.Linear(D*2, D)

        # simple MLP adapter
        self.fc1 = nn.Linear(D, Dh)
        self.fc2 = nn.Linear(Dh, D)

        # residual scaling
        self.layer_scale = nn.Parameter(torch.tensor(5e-3))
        #self.layer_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _as_bhwc(x):
        if x.dim() == 4:
            return x, True   # (B,H,W,D)
        elif x.dim() == 3:
            B,N,D = x.shape
            return x.view(B,1,N,D), False
        else:
            raise ValueError("x must be (B,H,W,D) or (B,N,D)")

    @staticmethod
    def _from_bhwc(y, was_grid):
        if was_grid: return y
        B,H,W,D = y.shape
        return y.view(B,W,D)

    def forward(self, x, t_proj):
        """
        x: (B,H,W,D) or (B,N,D)
        t_proj: (B,D)  # text already projected to D
        """
        B,D = x.shape[0], x.shape[-1]
        x_bhwc, was_grid = self._as_bhwc(x)

        # cosine similarity (B,H,W,1)
        x_dir = F.normalize(x_bhwc, dim=-1)
        t_dir = F.normalize(t_proj, dim=-1).view(B,1,1,D)
        #print((x_dir * t_dir).shape) # B,64,64,768
        sim   = (x_dir * t_dir).sum(dim=-1, keepdim=True)
        #sim   = (x_dir * t_dir)

        # concat sim → project back to D
        x_cat = torch.cat([x_bhwc, sim], dim=-1)   # (B,H,W,D+1)
        x_proj = self.sim_proj(x_cat)
        #x_proj = x_bhwc + sim 

        # adapter MLP
        y = self.fc2(self.act(self.fc1(x_proj)))

        out = x_bhwc + self.layer_scale * y if self.skip_connect else y
        return self._from_bhwc(out, was_grid), sim.squeeze(-1)

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