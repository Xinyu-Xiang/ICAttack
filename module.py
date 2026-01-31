import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers

import cfg



class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(cfg.num_parallel):
            setattr(self, 'lrnorm_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        if len(x_parallel) == 1:
            return [getattr(self, 'lrnorm_' + str(2))(x_parallel[0])]
        else:
            return [getattr(self, 'lrnorm_' + str(i))(x) for i, x in enumerate(x_parallel)]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Linear(in_features, hidden_features))
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(nn.GELU())
        self.fc2 = ModuleParallel(nn.Linear(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(drop))

        self.exchange = Exchanger()
        
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask):
        # x: [B, N, C], mask: [B, N]
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        if(cfg.use_exchange):
            if mask is not None:
                fused = x[2]
                x = [x[0], x[1]]
                x = [x_ * mask_.unsqueeze(2) for (x_, mask_) in zip(x, mask)]
                x.append(fused)
                # print(x)
                x = self.exchange(x, mask, mask_threshold_theta = 0.02, mask_threshold_miu = 0.7) 
                

        return x


class CE_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Linear(in_features, hidden_features))
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(nn.GELU())
        self.fc2 = ModuleParallel(nn.Linear(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(drop))

        self.exchange = Exchanger()
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # x: [B, N, C], mask: [B, N]
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = ModuleParallel(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = ModuleParallel(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(attn_drop))
        self.proj = ModuleParallel(nn.Linear(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(proj_drop))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ModuleParallel(nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio))
            self.norm = LayerNormParallel(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x[0].shape
        q = self.q(x)
        q = [q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) for q_ in q]

        if self.sr_ratio > 1:
            tmp = [x_.permute(0, 2, 1).reshape(B, C, H, W) for x_ in x]
            tmp = self.sr(tmp)
            tmp = [tmp_.reshape(B, C, -1).permute(0, 2, 1) for tmp_ in tmp]
            kv = self.kv(self.norm(tmp))
        else:
            kv = self.kv(x)
        kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]
        k, v = [kv_[0] for kv_ in kv], [kv_[1] for kv_ in kv]

        attn = [(q_ @ k_.transpose(-2, -1)) * self.scale for (q_, k_) in zip(q, k)]
        attn = [attn_.softmax(dim=-1) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [(attn_ @ v_).transpose(1, 2).reshape(B, N, C) for (attn_, v_) in zip(attn, v)]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = ModuleParallel(nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim))

    def forward(self, x, H, W):
        B, N, C = x[0].shape
        x = [x_.transpose(1, 2).view(B, C, H, W) for x_ in x]
        x = self.dwconv(x)
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = LayerNormParallel(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = ModuleParallel(DropPath(drop_path)) if drop_path > 0. else ModuleParallel(nn.Identity())
        self.norm2 = LayerNormParallel(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask=None):
        out = self.drop_path(self.attn(self.norm1(x), H, W))  # NORM + ATTENTION
        x = [x_ + out_ for (x_, out_) in zip(x, out)]
        out = self.drop_path(self.mlp(self.norm2(x), H, W, mask=mask))  # NORM + MLP
        x = [x_ + out_ for (x_, out_) in zip(x, out)]
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, patch_size=3, stride=2, in_chans=3, embed_dim=64):
        super().__init__()
        
        self.proj = ModuleParallel(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                   padding=(patch_size // 2 , patch_size // 2 )))
        self.norm = LayerNormParallel(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # _, _, H, W = x[0].shape
        # # dynamic padding
        # pad_h = (H % self.proj.module.stride[0]) // 2
        # pad_w = (W % self.proj.module.stride[1]) // 2

        # # right and bottom padding
        # x = [F.pad(x_, (0, pad_w, 0, pad_h)) for x_ in x]
        x = self.proj(x)

        _, _, H, W = x[0].shape
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]
        x = self.norm(x)

        return x, H, W


class PatchUpsample(nn.Module):
    def __init__(self, in_chans, embed_dim):
        super().__init__()

        self.proj = ModuleParallel(nn.Conv2d(in_chans // 4, embed_dim, kernel_size=1))
        self.norm = LayerNormParallel(embed_dim)
        self.pixelshuffle = ModuleParallel(nn.PixelShuffle(2))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target_size):
        x = self.pixelshuffle(x)
        # _, _, H, W = x[0].shape
        # # dynamic padding
        # pad_h = (H % self.proj.module.stride[0]) // 2
        # pad_w = (W % self.proj.module.stride[1]) // 2

        # # right and bottom padding
        # x = [F.pad(x_, (0, pad_w, 0, pad_h)) for x_ in x]
        x = self.proj(x)
        B, C, H, W = x[0].shape
        target_H, target_W = target_size

        # 调整到目标尺寸
        if H != target_H or W != target_W:
            x = [F.interpolate(x_, size=(target_H, target_W), mode='bilinear', align_corners=False) for x_ in x]

        x = [x_.flatten(2).transpose(1, 2) for x_ in x]

        return x, target_H, target_W


class Predictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.GELU()
        ) for _ in range(cfg.predictor_num_parallel)])

    def forward(self, x):
        x = [self.score_nets[i](x[i]) for i in range(cfg.predictor_num_parallel)]
        return x



class Exchanger(nn.Module):
    def __init__(self):
        super(Exchanger, self).__init__()

    def forward(self, x, mask, mask_threshold_theta, mask_threshold_miu):
        # x: [B, N, C], mask: [B, N]
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])

        # mask = [mask_.repeat(x[0].shape[2], 1).transpose(1,0).unsqueeze(0) for mask_ in mask]
        mask = [mask_.unsqueeze(-1).repeat(1, 1, x[0].shape[2]) for mask_ in mask]
        # print(mask)
        mask0greater = (mask[0] >= mask_threshold_theta)
        x0.masked_scatter_(mask0greater, x[0][mask0greater])
        mask0lesser = (mask[0] < mask_threshold_theta)
        x0.masked_scatter_(mask0lesser, x[1][mask0lesser])
        
        mask1greater = (mask[1] >= mask_threshold_theta)
        x1.masked_scatter_(mask1greater, x[1][mask1greater])
        mask1lesser = (mask[1] < mask_threshold_theta)
        x1.masked_scatter_(mask1lesser, x[0][mask1lesser])
        
        
        noise = x[2]
        # mask_miu0 = (mask[0] >= mask_threshold_miu)
        # fused.masked_scatter_(mask_miu0, x[0][mask_miu0])
        # mask_miu1 = (mask[1] >= mask_threshold_miu)
        # fused.masked_scatter_(mask_miu1, x[1][mask_miu1])
        
        return [x0, x1, noise]


def candidate_elimination(attn: torch.Tensor, keep_ratio: float, H: int, W: int, mask: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        keep_ratio (float): keep ratio of useful region tokens (candidates)
    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """

    bs, hn, _, _ = attn[0].shape
    attn_one = attn[0] 

    H_lens_keep = math.ceil(keep_ratio * H)  # keep ratio
    W_lens_keep = math.ceil(keep_ratio * W)
    lens_keep = H_lens_keep * W_lens_keep
    attn_t = attn_one.mean(dim=2).mean(dim=1)  # B, H, L, L --> B, L

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)  # sort index

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = topk_idx
    removed_index = non_topk_idx
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))
    mask[:, keep_index, :] = 1.0


    return keep_index, removed_index, mask


class CE_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = ModuleParallel(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = ModuleParallel(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(attn_drop))
        self.proj = ModuleParallel(nn.Linear(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(proj_drop))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ModuleParallel(nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio))
            self.norm = LayerNormParallel(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x[0].shape  # [1, 29*40=1160, c=512]
        q = self.q(x)  # [1, 29*40=1160, c=512]
        q = [q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) for q_ in q]  # [1, 1, 29*40=1160, c=512]

        kv = self.kv(x)

        kv = [kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for kv_ in kv]
        k, v = [kv_[0] for kv_ in kv], [kv_[1] for kv_ in kv]  # [1, 1, 15, 512]

        attn = [(q_ @ k_.transpose(-2, -1)) * self.scale for (q_, k_) in zip(q, k)]
        attn = [attn_.softmax(dim=-1) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [(attn_ @ v_).transpose(1, 2).reshape(B, N, C) for (attn_, v_) in zip(attn, v)]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn



class CEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1, ce_keep_ratio=None):
        super().__init__()
        self.norm1 = LayerNormParallel(dim)
        self.attn = CE_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = ModuleParallel(DropPath(drop_path)) if drop_path > 0. else ModuleParallel(nn.Identity())
        self.norm2 = LayerNormParallel(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CE_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.ce_keep_ratio = ce_keep_ratio

        self.stay_index, self.remove_index = None, None
        self.state = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, keep_ratio=None):
        x_out, attn = self.attn(self.norm1(x), H, W)
        # x = x + self.drop_path(x_out)  # list cat not list add
        x = [x_tensor + y_tensor for x_tensor, y_tensor in zip(x, self.drop_path(x_out))]
        
        if self.state is not None:
            self.stay_index, self.remove_index, self.state = candidate_elimination(attn, keep_ratio, H, W, self.state)  # [b, n*keep_ratio, c] remove some of tokens, result is not list here!!!
        elif self.state == None:
            mask = torch.zeros_like(x[0])  # only tensor
            self.stay_index, self.remove_index, mask = candidate_elimination(attn, keep_ratio, H, W, mask)
            self.state = mask
        x_mlp = self.mlp(self.norm2(x), H, W)
        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = [tensor + mlp_tensor for tensor, mlp_tensor in zip(x, self.drop_path(x_mlp))]
        return x, self.stay_index, self.remove_index, self.state


class Noise2VI(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                                  nn.BatchNorm2d(embed_dim),
                                  nn.ReLU())
    def forward(self, x):
        x = self.proj(x)
        return x


class Noise2IR(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
                                  nn.ReLU(),
                                  nn.Linear(embed_dim // 2, embed_dim // 4),
                                  nn.ReLU(),
                                  nn.Linear(embed_dim // 4, 2))

    def forward(self, x, noise):
        batch, c = x.shape[0], x.shape[1]
        noise = noise.permute(0, 2, 3, 1).reshape(batch, -1, c)
        noise = noise.permute(0, 2, 1) # [b, n, c] -> [b, c, n=h*w] 
        alpha, beta = self.proj(noise).chunk(2, dim=2)  # [b,c,n=h*w] -> [b, c, n=2] -> [b, c, 1&1]
        x = alpha.unsqueeze(-1) * x + beta.unsqueeze(-1)
        return x


class MFIAttack(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 128, 64],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1], ce_loc=[0, 1], ce_keep_ratio=[1.0, 0.5],
                 ce_embed_dims=[64, 64], fea_scale=[128, 64, 32, 16]):  # 记得ce是1.0和0.5
        super().__init__()
        self.depths = depths

        # transformer downsampling(Encoder)
        self.patch_embed_enc = nn.ModuleList()
        for i in range(4):
            patch_embed = OverlapPatchEmbed(patch_size=3, stride=2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            self.patch_embed_enc.append(patch_embed)

        ln2 = nn.Linear(embed_dims[1], embed_dims[0])
        ln3 = nn.Linear(embed_dims[2], embed_dims[1])
        ln4 = nn.Linear(embed_dims[3], embed_dims[2])
        
        ln_b1 = nn.Sequential(nn.Identity())
        ln_b2 = nn.Sequential(ln2)
        ln_b3 = nn.Sequential(ln3, ln2)
        ln_b4 = nn.Sequential(ln4, ln3, ln2)
        
        ln_list = [ln_b1, ln_b2, ln_b3, ln_b4]
        self.mlp_before_predictor = nn.ModuleList(ln_list)
        self.score_predictor = Predictor(embed_dims[0])
        
        # noise 2 vi proj and ir 
        self.n_proj_vi = nn.ModuleList()
        self.n_proj_ir = nn.ModuleList()

        for i in range(4):
            n_proj_vi = Noise2VI(embed_dims[i])
            self.n_proj_vi.append(n_proj_vi)
            n_proj_ir = Noise2IR(fea_scale[i]*fea_scale[i])
            self.n_proj_ir.append(n_proj_ir)



        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_enc = 0
        self.block_enc, self.norm_enc = nn.ModuleList(), nn.ModuleList()
        for idx in [0, 1, 2, 3]:
            block_enc = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_enc + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_enc.append(block_enc)
            self.norm_enc.append(LayerNormParallel(embed_dims[idx]))
            cur_enc += depths[idx]
        
        
        # detail: CE+detail
        self.detail_attack, self.norm_detail = nn.ModuleList(), nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        ce_index = 0  # init ce index
        self.ce_loc = ce_loc
        self.ce_keep_ratio = ce_keep_ratio
        cur_ce = 0
        for idx in [0, 1]:
            self.ce_keep_ratio_i = 1.0
            if ce_loc is not None and idx in ce_loc:
                self.ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            block_detail = nn.ModuleList([CEBlock(
                dim=ce_embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_ce + i],
                sr_ratio=sr_ratios[idx], ce_keep_ratio=self.ce_keep_ratio_i)
                for i in range(depths[idx])])
            self.detail_attack.append(block_detail)
            self.norm_detail.append(LayerNormParallel(ce_embed_dims[idx]))
            cur_ce += depths[idx]


        # contrast: CE+Contrast
        self.contrast_attack, self.norm_contrast = nn.ModuleList(), nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        ce_index = 0  # init ce index
        self.ce_loc = ce_loc
        self.ce_keep_ratio = ce_keep_ratio
        cur_ce = 0
        for idx in [0, 1]:
            self.ce_keep_ratio_i = 1.0
            if ce_loc is not None and idx in ce_loc:
                self.ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            block_contrast = nn.ModuleList([CEBlock(
                dim=ce_embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_ce + i],
                sr_ratio=sr_ratios[idx], ce_keep_ratio=self.ce_keep_ratio_i)
                for i in range(depths[idx])])
            self.contrast_attack.append(block_contrast)
            self.norm_contrast.append(LayerNormParallel(ce_embed_dims[idx]))
            cur_ce += depths[idx]
        

        # transformer upsampling——vi
        self.patch_embed_dec = nn.ModuleList()
        for i in range(4)[::-1]:
            patch_embed = PatchUpsample(in_chans=embed_dims[i], embed_dim=embed_dims[max(0, i - 1)])
            self.patch_embed_dec.append(patch_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_dec = 0
        self.block_dec, self.norm_dec = nn.ModuleList(), nn.ModuleList()
        for idx in [2, 1, 0]:
            block_dec = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_dec + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_dec.append(block_dec)
            self.norm_dec.append(LayerNormParallel(embed_dims[idx]))
            cur_dec += depths[idx]

        self.project = ModuleParallel(nn.Conv2d(embed_dims[idx], 3, 1, 1, 0))
        self.tanh = ModuleParallel(nn.Tanh())

        # transformer upsampling——ir
        self.patch_embed_dec2 = nn.ModuleList()
        for i in range(4)[::-1]:
            patch_embed2 = PatchUpsample(in_chans=embed_dims[i], embed_dim=embed_dims[max(0, i - 1)])
            self.patch_embed_dec2.append(patch_embed2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_dec = 0
        self.block_dec2, self.norm_dec2 = nn.ModuleList(), nn.ModuleList()
        for idx in [2, 1, 0]:
            block_dec2 = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_dec + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_dec2.append(block_dec2)
            self.norm_dec2.append(LayerNormParallel(embed_dims[idx]))
            cur_dec += depths[idx]

        self.project2 = ModuleParallel(nn.Conv2d(embed_dims[idx], 3, 1, 1, 0))
        self.tanh2 = ModuleParallel(nn.Tanh())

        # init weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x0, x1, noise, save_heatmap=False, epoch=None, batch_idx=None, frame_idx=None):
        # print('x0:', x0.shape)
        # print('x1:', x1.shape)
        x_stack = torch.stack((x0, x1), dim=0)
        # fused = torch.mean(x_stack, 0)  # 
        x = [x0, x1, noise]
        
        B = x[0].shape[0]
        outs = []
        sizes = [] 
        # x: torch.Size([1, 3, 480, 512])

        count = 0
        # masks = []
        # encoder + exchange
        for i in range(len(self.block_enc)):
            x, H, W = self.patch_embed_enc[i](x)  # they are separation in x and embeding
            # print('embed_enc %d:' % i, x[0].shape)
            # print('H %d:' % i, H)
            # print('W %d:' % i, W)
            # print(self.block_enc)

            sizes.append((H, W))
            for idx, blk in enumerate(self.block_enc[i]):  # into transformer encoder block(att + ffn)
                mask = None
                if idx == len(self.block_enc[i]) - 1:

                    # noise proj2 vi and ir
                    x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
                    x[0] = x[0] + self.n_proj_vi[i](x[2])
                    x[1] = self.n_proj_ir[i](x[1], x[2])
                    x = [x_.permute(0, 2, 3, 1).reshape(B, -1, x_.shape[1]) for x_ in x]

                    # last block to exchange
                    x_to_pred = [x[0], x[1]]
                    feature_before_pred = [self.mlp_before_predictor[count](x_) for x_ in x_to_pred]
                    score = self.score_predictor(feature_before_pred)
                    mask = [score_.reshape(B, -1, 1)[:, :, 0] for score_ in score]  # mask_: [B, N]
                    mask = torch.stack(mask, dim=-1)
                    mask = F.softmax(mask, dim=2)
                    mask = [mask[:,:,0], mask[:,:,1]]
                    # print(mask)
                    count += 1
                x = blk(x, H, W, mask)  # only last mask
            # print('block_enc %d:' % i, x[0].shape)87556304
            x = self.norm_enc[i](x)  # 3 parts [b, N, C]
            outs.append(x)
            # orginal
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]  # list[x0(vi), x1(ir),noise] 3part each of them[1, 512, 29, 40]
        
        
        x_enc = x  # image feature
        x = [x_.permute(0, 2, 3, 1).reshape(B, -1, x_.shape[1]) for x_ in x]  # return back [b, n, c]
        # print(f"x shape before: {[x_.shape for x_ in x]}")
        # detail 
        x_detail = [x[0] + x[2]]  # vi+noise    
        # print(f"x_detail shape before: {[detail.shape for detail in x_detail]}")
        for i in range(len(self.detail_attack)):
            for idx, blk in enumerate(self.detail_attack[i]):  # into transformer encoder block(att + ffn)
                
                x_detail, stay_id_detail, remove_id_detail, msk_detail = blk(x_detail, H, W, self.ce_keep_ratio[i])  # only last mask
                # msk_detail = msk_detail.detach()
                # print(f"x_detail shape: {[detail.shape for detail in x_detail]}")
                # print(f"msk_detail shape: {msk_detail.shape}")
                # zero_positions = (msk_detail == 0).nonzero(as_tuple=True)
                # print("值为 0 的位置：", zero_positions)
                x_detail = [detail.clone() * msk_detail.clone() for detail in x_detail]

                '''IMPERCEPTABLE NOISE STRUCTURE'''
                modified_mask = torch.where(msk_detail.clone() == 1, 
                                            torch.ones_like(msk_detail) * 1, 
                                            torch.ones_like(msk_detail) * 0.5)
                # 将处理后的mask与噪声x[2]相乘，再叠加到x[0]
                noise_component = x[2] * modified_mask  # 噪声按mask加权
                x_detail = [x_detail[0] + noise_component]  

        x_detail_fea = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x_detail]
        # msk_detail_fea = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in msk_detail]

        # contrast
        x_contrast = [x[1] + x[2]]  # ir+noise
        for i in range(len(self.contrast_attack)):
            for idx, blk in enumerate(self.contrast_attack[i]):  # into transformer encoder block(att + ffn)
                x_contrast, stay_id_contrast, remove_id_contrast, msk_contrast = blk(x_contrast, H, W, self.ce_keep_ratio[i])  # only last mask
                # msk_contrast = msk_contrast.detach()
                x_contrast = [contrast.clone() * msk_contrast.clone() for contrast in x_contrast]

                '''IMPERCEPTABLE NOISE STRUCTURE'''
                # OISE_RATE=0.5
                modified_mask_cont = torch.where(msk_contrast.clone() == 1, 
                                            torch.ones_like(msk_contrast), 
                                            torch.ones_like(msk_contrast) * 0.5)
                # 将处理后的mask与噪声x[2]相乘，再叠加到x[0]
                noise_component_cont = x[2] * modified_mask_cont 
                x_contrast = [x_contrast[0] + noise_component_cont]

        x_contrast_fea = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x_contrast]
        # msk_contrast_fea = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in msk_contrast]

        # add detail and contrast
        # x = [x_enc[0] + x_detail_fea[0], x_enc[1] + x_contrast_fea[0]]
        x = [x_detail_fea[0], x_contrast_fea[0]]

        vi = [x[0]]
        ir = [x[1]]
    
        # decoder_vi
        for i in range(len(self.block_dec)):
            vi, H, W = self.patch_embed_dec[i](vi, target_size=sizes[-(i + 2)])
            # print('embed_dec %d:' % i, x[0].shape)
            vi = [x_ + outs_ for (x_, outs_) in zip(vi, outs[::-1][i + 1])]
            for blk in self.block_dec[i]:
                vi = blk(vi, H, W)
            # print('block_dec %d:' % i, x[0].shape)
            vi = self.norm_dec[i](vi)
            vi = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in vi]


        vi, H, W = self.patch_embed_dec[3](vi, target_size=(x0.shape[2], x0.shape[3]))
        # print('embed_enc 4:', x[0].shape)
        vi = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in vi]

        # print(x[0].shape)
        vi = self.tanh(self.project(vi))
        # project to [-1, 1]


        # decoder_ir
        for i in range(len(self.block_dec2)):
            ir, H, W = self.patch_embed_dec2[i](ir, target_size=sizes[-(i + 2)])
            # print('embed_dec %d:' % i, x[0].shape)
            ir = [x_ + outs_ for (x_, outs_) in zip(ir, outs[::-1][i + 1])]
            for blk in self.block_dec2[i]:
                ir = blk(ir, H, W)
            # print('block_dec %d:' % i, x[0].shape)
            ir = self.norm_dec2[i](ir)
            ir = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in ir]


        ir, H, W = self.patch_embed_dec2[3](ir, target_size=(x0.shape[2], x0.shape[3]))
        # print('embed_enc 4:', x[0].shape)
        ir = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in ir]

        # print(x[0].shape)
        ir = self.tanh2(self.project2(ir))
        
        return vi[0], ir[0]






















































class MFIAttack_old(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 128, 64],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1], ce_loc=[0, 1], ce_keep_ratio=[1.0, 0.5],
                 ce_embed_dims=[64, 64]):
        super().__init__()
        self.depths = depths

        # transformer downsampling(Encoder)
        self.patch_embed_enc = nn.ModuleList()
        for i in range(4):
            patch_embed = OverlapPatchEmbed(patch_size=3, stride=2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            self.patch_embed_enc.append(patch_embed)

        ln2 = nn.Linear(embed_dims[1], embed_dims[0])
        ln3 = nn.Linear(embed_dims[2], embed_dims[1])
        ln4 = nn.Linear(embed_dims[3], embed_dims[2])
        
        ln_b1 = nn.Sequential(nn.Identity())
        ln_b2 = nn.Sequential(ln2)
        ln_b3 = nn.Sequential(ln3, ln2)
        ln_b4 = nn.Sequential(ln4, ln3, ln2)
        
        ln_list = [ln_b1, ln_b2, ln_b3, ln_b4]
        self.mlp_before_predictor = nn.ModuleList(ln_list)
        self.score_predictor = Predictor(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_enc = 0
        self.block_enc, self.norm_enc = nn.ModuleList(), nn.ModuleList()
        for idx in [0, 1, 2, 3]:
            block_enc = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_enc + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_enc.append(block_enc)
            self.norm_enc.append(LayerNormParallel(embed_dims[idx]))
            cur_enc += depths[idx]


        # detail: CE+detail
        self.detail_attack, self.norm_detail = nn.ModuleList(), nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        ce_index = 0  # init ce index
        self.ce_loc = ce_loc
        self.ce_keep_ratio = ce_keep_ratio
        cur_ce = 0
        for idx in [0, 1]:
            self.ce_keep_ratio_i = 1.0
            if ce_loc is not None and idx in ce_loc:
                self.ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            block_detail = nn.ModuleList([CEBlock(
                dim=ce_embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_ce + i],
                sr_ratio=sr_ratios[idx], ce_keep_ratio=self.ce_keep_ratio_i)
                for i in range(depths[idx])])
            self.detail_attack.append(block_detail)
            self.norm_detail.append(LayerNormParallel(ce_embed_dims[idx]))
            cur_ce += depths[idx]


        # contrast: CE+Contrast
        self.contrast_attack, self.norm_contrast = nn.ModuleList(), nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        ce_index = 0  # init ce index
        self.ce_loc = ce_loc
        self.ce_keep_ratio = ce_keep_ratio
        cur_ce = 0
        for idx in [0, 1]:
            self.ce_keep_ratio_i = 1.0
            if ce_loc is not None and idx in ce_loc:
                self.ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            block_contrast = nn.ModuleList([CEBlock(
                dim=ce_embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_ce + i],
                sr_ratio=sr_ratios[idx], ce_keep_ratio=self.ce_keep_ratio_i)
                for i in range(depths[idx])])
            self.contrast_attack.append(block_contrast)
            self.norm_contrast.append(LayerNormParallel(ce_embed_dims[idx]))
            cur_ce += depths[idx]
        

        # transformer upsampling——vi
        self.patch_embed_dec = nn.ModuleList()
        for i in range(4)[::-1]:
            patch_embed = PatchUpsample(in_chans=embed_dims[i], embed_dim=embed_dims[max(0, i - 1)])
            self.patch_embed_dec.append(patch_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_dec = 0
        self.block_dec, self.norm_dec = nn.ModuleList(), nn.ModuleList()
        for idx in [2, 1, 0]:
            block_dec = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_dec + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_dec.append(block_dec)
            self.norm_dec.append(LayerNormParallel(embed_dims[idx]))
            cur_dec += depths[idx]

        self.project = ModuleParallel(nn.Conv2d(embed_dims[idx], 3, 1, 1, 0))
        self.tanh = ModuleParallel(nn.Tanh())

        # transformer upsampling——ir
        self.patch_embed_dec2 = nn.ModuleList()
        for i in range(4)[::-1]:
            patch_embed2 = PatchUpsample(in_chans=embed_dims[i], embed_dim=embed_dims[max(0, i - 1)])
            self.patch_embed_dec2.append(patch_embed2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur_dec = 0
        self.block_dec2, self.norm_dec2 = nn.ModuleList(), nn.ModuleList()
        for idx in [2, 1, 0]:
            block_dec2 = nn.ModuleList([Block(
                dim=embed_dims[idx], num_heads=num_heads[idx], mlp_ratio=mlp_ratios[idx], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur_dec + i],
                sr_ratio=sr_ratios[idx])
                for i in range(depths[idx])])
            self.block_dec2.append(block_dec2)
            self.norm_dec2.append(LayerNormParallel(embed_dims[idx]))
            cur_dec += depths[idx]

        self.project2 = ModuleParallel(nn.Conv2d(embed_dims[idx], 3, 1, 1, 0))
        self.tanh2 = ModuleParallel(nn.Tanh())

        # init weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x0, x1, noise):
        # print('x0:', x0.shape)
        # print('x1:', x1.shape)
        x_stack = torch.stack((x0, x1), dim=0)
        x = [x0, x1, noise]
        
        B = x[0].shape[0]
        outs = []
        sizes = [] 
        # x: torch.Size([1, 3, 480, 512])

        count = 0
        # masks = []
        # encoder + exchange
        for i in range(len(self.block_enc)):
            x, H, W = self.patch_embed_enc[i](x)  # they are separation in x and embeding
            # print('embed_enc %d:' % i, x[0].shape)
            # print('H %d:' % i, H)
            # print('W %d:' % i, W)

            sizes.append((H, W))
            for idx, blk in enumerate(self.block_enc[i]):  # into transformer encoder block(att + ffn)
                mask = None
                if idx == len(self.block_enc[i]) - 1:
                    x_to_pred = [x[0], x[1]]
                    feature_before_pred = [self.mlp_before_predictor[count](x_) for x_ in x_to_pred]
                    score = self.score_predictor(feature_before_pred)
                    mask = [score_.reshape(B, -1, 1)[:, :, 0] for score_ in score]  # mask_: [B, N]
                    mask = torch.stack(mask, dim=-1)
                    mask = F.softmax(mask, dim=2)
                    mask = [mask[:,:,0], mask[:,:,1]]
                    # print(mask)
                    count += 1
                x = blk(x, H, W, mask)  # only last mask
            # print('block_enc %d:' % i, x[0].shape)
            x = self.norm_enc[i](x)  # 3 parts [b, N, C]
            outs.append(x)
            # orginal
            x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]  # list[x0(vi), x1(ir),noise] 3part each of them[1, 512, 29, 40]
        
        
        x_enc = x
        x = [x_.permute(0, 2, 3, 1).reshape(B, -1, x_.shape[1]) for x_ in x]  # return back [b, n, c]
        # print(f"x shape before: {[x_.shape for x_ in x]}")
        # detail 
        x_detail = [x[0] + x[2]]  # vi+noise
        # print(f"x_detail shape before: {[detail.shape for detail in x_detail]}")
        for i in range(len(self.detail_attack)):
            for idx, blk in enumerate(self.detail_attack[i]):  # into transformer encoder block(att + ffn)
                
                x_detail, stay_id_detail, remove_id_detail, msk_detail = blk(x_detail, H, W, self.ce_keep_ratio[i])  # only last mask
                # msk_detail = msk_detail.detach()
                # print(f"x_detail shape: {[detail.shape for detail in x_detail]}")
                # print(f"msk_detail shape: {msk_detail.shape}")
                x_detail = [detail * msk_detail for detail in x_detail]
        x_detail_fea = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x_detail]
        # msk_detail_fea = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in msk_detail]

        # contrast
        x_contrast = [x[1] + x[2]]  # ir+noise
        for i in range(len(self.contrast_attack)):
            for idx, blk in enumerate(self.contrast_attack[i]):  # into transformer encoder block(att + ffn)
                x_contrast, stay_id_contrast, remove_id_contrast, msk_contrast = blk(x_contrast, H, W, self.ce_keep_ratio[i])  # only last mask
                # msk_contrast = msk_contrast.detach()
                x_contrast = [contrast * msk_contrast for contrast in x_contrast]
        x_contrast_fea = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x_contrast]
        # msk_contrast_fea = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in msk_contrast]

        # add detail and contrast
        x = [x_enc[0] + x_detail_fea[0], x_enc[1] + x_contrast_fea[0]]
        vi = [x[0]]
        ir = [x[1]]
    
        # decoder_vi
        for i in range(len(self.block_dec)):
            vi, H, W = self.patch_embed_dec[i](vi, target_size=sizes[-(i + 2)])
            # print('embed_dec %d:' % i, x[0].shape)
            vi = [x_ + outs_ for (x_, outs_) in zip(vi, outs[::-1][i + 1])]
            for blk in self.block_dec[i]:
                vi = blk(vi, H, W)
            # print('block_dec %d:' % i, x[0].shape)
            vi = self.norm_dec[i](vi)
            vi = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in vi]


        vi, H, W = self.patch_embed_dec[3](vi, target_size=(x0.shape[2], x0.shape[3]))
        # print('embed_enc 4:', x[0].shape)
        vi = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in vi]

        # print(x[0].shape)
        vi = self.tanh(self.project(vi))
        # project to [-1, 1]


        # decoder_ir
        for i in range(len(self.block_dec2)):
            ir, H, W = self.patch_embed_dec2[i](ir, target_size=sizes[-(i + 2)])
            # print('embed_dec %d:' % i, x[0].shape)
            ir = [x_ + outs_ for (x_, outs_) in zip(ir, outs[::-1][i + 1])]
            for blk in self.block_dec2[i]:
                ir = blk(ir, H, W)
            # print('block_dec %d:' % i, x[0].shape)
            ir = self.norm_dec2[i](ir)
            ir = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in ir]


        ir, H, W = self.patch_embed_dec2[3](ir, target_size=(x0.shape[2], x0.shape[3]))
        # print('embed_enc 4:', x[0].shape)
        ir = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in ir]

        # print(x[0].shape)
        ir = self.tanh2(self.project2(ir))
        
        return vi[0], ir[0]




if __name__ == '__main__':
    # 选择不同型号卡
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    window_size = 8
    modelE = MFIAttack_ours().cuda()
    img_vi = torch.randn(15, 3, 256, 256).cuda()
    img_ir = torch.randn(15, 3, 256, 256).cuda()
    noise = torch.randn(15, 3, 256, 256).cuda()
    # img_ir = torch.randn(1, 3, 460, 630).cuda()

    print(img_vi.shape)
    x1, x2 = modelE(img_vi, img_ir, noise)
    # x1, x2, x3 = modelE(img_ir)
    pass