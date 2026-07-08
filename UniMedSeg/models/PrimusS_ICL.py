import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional, Callable, Sequence, Type
from torch.nn.modules.conv import _ConvNd

# ==============================================================================
# Dependencies (Assumed available from timm)
# ==============================================================================
try:
    from timm.layers import trunc_normal_,RotaryEmbeddingCat
    from timm.models.eva import EvaBlock
except ImportError:
    from timm.models.layers import trunc_normal_
    from timm.layers import RotaryEmbeddingCat
    from timm.models.eva import EvaBlock

from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Callable, List, Union
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Mlp, SwiGLU
from timm.layers import apply_rot_embed_cat

# ==============================================================================
# Helpers
# ==============================================================================

def convert_dim_to_conv_op(dimension: int) -> Type[_ConvNd]:
    if dimension == 1: return nn.Conv1d
    elif dimension == 2: return nn.Conv2d
    elif dimension == 3: return nn.Conv3d
    else: raise ValueError("Unknown dimension")

class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class LayerNormNd(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        idx = (None, slice(None), *([None] * (x.ndim - 2)))
        x = self.weight[idx] * x + self.bias[idx]
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: Tuple[int, ...], input_channels: int, embed_dim: int):
        super().__init__()
        self.proj = convert_dim_to_conv_op(len(patch_size))(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class PatchDecode(nn.Module):
    def __init__(self, patch_size, embed_dim, out_channels, norm=LayerNormNd, activation=nn.GELU):
        super().__init__()
        def _round_to_8(inp): return int(max(8, np.round((inp + 1e-6) / 8) * 8))
        
        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 1):
            stages.append(nn.Sequential(
                nn.ConvTranspose3d(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                norm(channels[s + 1]),
                activation(),
            ))
        stages.append(nn.ConvTranspose3d(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        return self.decode(x)


# ==============================================================================
# Split EVA Attention
# ==============================================================================

class SplitEvaAttention(nn.Module):
    """
    Decoupled Split Attention for ICL.
    Removes O(N^2) Mask bottleneck by physically splitting calculation paths.
    
    Path A (Target): Q_tgt attends to K_all, V_all.
    Path B (Context): Q_ctx (reshaped as batch) attends to [K_tgt, K_ctx_self].
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = False,
        qkv_bias_separate: bool = False, # Added to match EvaAttention
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
        scale_norm: bool = True,
        scale_attn_inner: bool = True,
        # Unused args kept for compatibility
        num_prefix_tokens: int = 0, 
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        qk_norm: bool = False,
        rotate_half: bool = False,
        device=None,
        dtype=None,
        **kwargs
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads if attn_head_dim is None else attn_head_dim
        self.scale = head_dim ** -0.5
        self.qkv_fused = qkv_fused
        self.qkv_bias_separate = qkv_bias_separate
        self.num_prefix_tokens = num_prefix_tokens # Needed for RoPE slicing if used

        attn_dim = head_dim * num_heads

        # --- QKV Projection (Strictly matching EvaAttention structure) ---
        if qkv_fused:
            # Note: EvaAttention uses bias=False for the Linear layer itself
            self.qkv = nn.Linear(dim, attn_dim * 3, bias=False, **dd)
            self.q_proj = self.k_proj = self.v_proj = None
            
            # Manually registered biases to match checkpoint keys
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.empty(attn_dim, **dd))
                self.register_buffer('k_bias', torch.empty(attn_dim, **dd), persistent=False)
                self.v_bias = nn.Parameter(torch.empty(attn_dim, **dd))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            # Unfused: Separate layers
            self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
            self.k_proj = nn.Linear(dim, attn_dim, bias=False, **dd) # K usually has no bias in Eva
            self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias, **dd)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        # --- 2. Norms & Projections ---
        # QK Norm (Optional in Eva, kept for structure matching)
        self.q_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()

        # Output Norm & Projection
        self.norm = norm_layer(attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, dim, **dd)
        
        # Initialize (Only for biases, weights loaded from checkpoint)
        if not self.proj.weight.is_meta:
            self._init_biases()
            
    def _init_biases(self):
        if self.q_bias is not None:
            nn.init.zeros_(self.q_bias)
            nn.init.zeros_(self.v_bias)
        if self.k_bias is not None:
            self.k_bias.zero_()

    def forward(self, x, rope=None, split_sizes: Optional[Tuple[int, int]] = None):
        """
        x: [B, Total_Len, C]
        split_sizes: (target_len, context_pair_len) 
                     e.g. (N, 2*N) where Total_Len = N + L * 2N
        """
        B, T, C = x.shape
        
        # 1. Global Projection & RoPE
        if self.qkv is not None:
            # Handle the weird Eva bias logic
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            
            qkv = qkv.reshape(B, T, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0) # [B, H, T, D]
        else:
            q = self.q_proj(x).reshape(B, T, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, T, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, T, self.num_heads, -1).transpose(1, 2)

        # Apply QK Norm if enabled
        q, k = self.q_norm(q), self.k_norm(k)
        

        # Apply RoPE globally (Position IDs are absolute: 0..T)
        if rope is not None:
            # apply_rot_embed_cat expects [B, T, H, D] usually, but q is [B, H, T, D]
            # Adjusting for timm's implementation or standard implementation
            # Assuming standard timm-like apply_rot_embed_cat:
            q = apply_rot_embed_cat(q, rope)
            k = apply_rot_embed_cat(k, rope)

        # 2. Split Attention Logic
        if split_sizes is None:
            # Fallback to standard full attention if no split info provided
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            tgt_len, ctx_unit_len = split_sizes
            
            # --- Path A: Target Output ---
            # Target Q sees Everything (Global Context)
            q_tgt = q[:, :, :tgt_len, :]      # [B, H, N, D]
            # k, v are global               # [B, H, T, D]
            
            out_tgt = F.scaled_dot_product_attention(q_tgt, k, v, dropout_p=0.0)
            
            # --- Path B: Context Output ---
            # Optimization: Parallelize Contexts by treating L as Batch dimension
            # Current Context Part: [B, H, L*M, D]
            q_ctx = q[:, :, tgt_len:, :]
            k_ctx = k[:, :, tgt_len:, :]
            v_ctx = v[:, :, tgt_len:, :]
            
            if q_ctx.shape[2] > 0: # If contexts exist
                # Calculate number of contexts L
                # Total context length = T - tgt_len
                # L = Total_Ctx / ctx_unit_len
                total_ctx_len = q_ctx.shape[2]
                num_contexts = total_ctx_len // ctx_unit_len
                
                # Reshape to [B*L, H, M, D]
                # We merge B and L into the batch dimension
                q_ctx_flat = rearrange(q_ctx, 'b h (l m) d -> (b l) h m d', l=num_contexts, m=ctx_unit_len)
                k_ctx_flat = rearrange(k_ctx, 'b h (l m) d -> (b l) h m d', l=num_contexts, m=ctx_unit_len)
                v_ctx_flat = rearrange(v_ctx, 'b h (l m) d -> (b l) h m d', l=num_contexts, m=ctx_unit_len)
                
                # Expand Target K/V to match the new Batch size (B*L)
                # k_tgt: [B, H, N, D] -> [B*L, H, N, D]
                k_tgt = k[:, :, :tgt_len, :]
                v_tgt = v[:, :, :tgt_len, :]
                
                k_tgt_exp = k_tgt.repeat_interleave(num_contexts, dim=0)
                v_tgt_exp = v_tgt.repeat_interleave(num_contexts, dim=0)
                
                # Concatenate: Each context sees [Target, Itself]
                # Key/Value Local: [B*L, H, N+M, D]
                k_local = torch.cat([k_tgt_exp, k_ctx_flat], dim=2)
                v_local = torch.cat([v_tgt_exp, v_ctx_flat], dim=2)
                
                # Efficient Attention
                out_ctx_flat = F.scaled_dot_product_attention(q_ctx_flat, k_local, v_local, dropout_p=0.0)
                
                # Reshape back to [B, H, L*M, D]
                out_ctx = rearrange(out_ctx_flat, '(b l) h m d -> b h (l m) d', b=B)
                
                # Merge paths
                x = torch.cat([out_tgt, out_ctx], dim=2)
            else:
                x = out_tgt

        # 3. Final Projection
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.norm(x)
        x = self.proj(x)
        return x

# ==============================================================================
# Split Eva Block
# ==============================================================================

class SplitEvaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        swiglu_mlp: bool = True,
        scale_mlp: bool = False,
        scale_attn_inner: bool = True,
        norm_layer: Callable = LayerNorm,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        drop_path: float = 0.,
        **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # Use our new Split Attention
        self.attn = SplitEvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            scale_norm=scale_attn_inner,
        )
        
        self.init_values = init_values
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        
        # SwiGLU logic (Preserved from original)
        if swiglu_mlp:
             self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=hidden_features,
                norm_layer=norm_layer if scale_mlp else None,
                drop=0.0, # No dropout
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=0.0,
            )
            
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope=None, split_sizes: Optional[Tuple[int, int]] = None):
        # Attention Block with Split Logic
        if self.gamma_1 is not None:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, split_sizes=split_sizes))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, split_sizes=split_sizes))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
        
# ==============================================================================
# Eva Encoder
# ==============================================================================
import torch.utils.checkpoint as checkpoint
class EvaClean(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4 * 2 / 3,
        norm_layer: Callable = LayerNorm,
        init_values: Optional[float] = None,
        ref_feat_shape: Optional[Tuple[int, ...]] = None,
        rope_impl=RotaryEmbeddingCat, # Assuming this exists from your env
        rope_kwargs=None,
        grad_checkpointing: bool = False, 
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        if rope_kwargs is None: rope_kwargs = {}
        
        # RoPE Setup
        if len(ref_feat_shape) == 3:
            rope_dim = round(embed_dim // num_heads / 1.5)
        else:
            rope_dim = embed_dim // num_heads
            
        self.rope = rope_impl(
            rope_dim, in_pixels=False, feat_shape=ref_feat_shape, ref_feat_shape=ref_feat_shape, **rope_kwargs
        )

        # Use SplitEvaBlock
        self.blocks = nn.ModuleList([
            SplitEvaBlock(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=True,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=True,
                scale_mlp=True,
                scale_attn_inner=True,
                norm_layer=norm_layer,
                init_values=init_values,
                drop_path=0.0,
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.grad_checkpointing = grad_checkpointing

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, rope_override=None, split_sizes: Optional[Tuple[int, int]] = None):
        """
        x: [B, L_total, C]
        rope_override: Optional tensor for concatenated RoPE
        split_sizes: Passed down to Attention to trigger Split Calculation
        """
        if rope_override is None:
            rot_pos_embed = self.rope.get_embed()
        else:
            rot_pos_embed = rope_override

        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                # 使用 checkpoint 包装 block
                # 注意：use_reentrant=False 是 PyTorch 推荐的新方式，
                # 它支持传入非 Tensor 参数 (如 split_sizes)
                x = checkpoint.checkpoint(
                    blk, 
                    x, 
                    rot_pos_embed, 
                    split_sizes, 
                    use_reentrant=False
                )
            else:
                x = blk(x, rope=rot_pos_embed, split_sizes=split_sizes)
            
        x = self.norm(x)
        return x


# ==============================================================================
# New: 2D Specific Components
# ==============================================================================
class PatchDecode2d(nn.Module):
    """2D 版本的 Light Decoder"""
    def __init__(self, patch_size, embed_dim, out_channels, norm=LayerNormNd, activation=nn.GELU):
        super().__init__()
        def _round_to_8(inp): return int(max(8, np.round((inp + 1e-6) / 8) * 8))
        
        # 适配 2D Patch Size (取前两个维度)
        patch_size = patch_size[:2] if len(patch_size) == 3 else patch_size
        
        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 1):
            stages.append(nn.Sequential(
                nn.ConvTranspose2d(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                norm(channels[s + 1]),
                activation(),
            ))
        stages.append(nn.ConvTranspose2d(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        return self.decode(x)

    


# ==============================================================================
# Primus ICL (Updated)
# ==============================================================================
class PrimusICL(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...],
        embed_dim: int = 396,
        eva_depth: int = 12,
        eva_numheads: int = 6,
        mlp_ratio: float = 8/3,
        init_values: float = 0.1,
        grad_checkpointing = False,
    ):
        super().__init__()
        
        self.patch_size_3d = patch_embed_size
        self.patch_size_2d = patch_embed_size[:2] 

        # --- Projections ---
        self.down_projection_img = PatchEmbed(self.patch_size_3d, input_channels, embed_dim)
        self.down_projection_label = PatchEmbed(self.patch_size_3d, input_channels, embed_dim)
        self.up_projection = PatchDecode(self.patch_size_3d, embed_dim, output_channels)

        self.down_projection_img_2d = PatchEmbed(self.patch_size_2d, input_channels, embed_dim)
        self.down_projection_label_2d = PatchEmbed(self.patch_size_2d, input_channels, embed_dim)
        self.up_projection_2d = PatchDecode2d(self.patch_size_2d, embed_dim, output_channels)

        # Type Embeds
        self.type_embed = nn.Parameter(torch.zeros(3, 1, 1, embed_dim))
        trunc_normal_(self.type_embed, std=0.02)

        # --- RoPE Calculation (复制 EvaClean 中的逻辑) ---
        # 计算 Feature Map 的形状
        ref_feat_shape_3d = tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)])
        ref_feat_shape_2d = ref_feat_shape_3d[:2]

        # 计算 RoPE 的维度 (保持与 EvaClean 内部逻辑一致)
        # 3D 情况下通常会减少 rope_dim 以节省计算或适应 head_dim
        rope_dim_3d = round(embed_dim // eva_numheads / 1.5)
        rope_dim_2d = embed_dim//eva_numheads
        
        # 实例化两个独立的 RoPE 模块
        # 注意：in_pixels=False 表示我们传入的是 feature map 的 grid，而不是原始像素 grid
        self.rope_3d = RotaryEmbeddingCat(
            dim=rope_dim_3d, 
            in_pixels=False, 
            feat_shape=ref_feat_shape_3d, 
            ref_feat_shape=ref_feat_shape_3d
        )
        
        self.rope_2d = RotaryEmbeddingCat(
            dim=rope_dim_2d, 
            in_pixels=False, 
            feat_shape=ref_feat_shape_2d, 
            ref_feat_shape=ref_feat_shape_2d
        )
        
        # --- Encoder ---
        self.eva = EvaClean(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            mlp_ratio=mlp_ratio,
            # 这里传 ref_feat_shape_3d 主要是为了让 EvaClean 内部初始化不报错，
            # 实际上 forward 时我们会通过 rope_override 覆盖它
            ref_feat_shape=ref_feat_shape_3d, 
            init_values=init_values,
            grad_checkpointing=grad_checkpointing,
            # 我们将在外部处理 RoPE，这里可以传任意符合要求的参数，或者甚至可以修改 EvaClean 不创建 rope
            # 但为了保持 EvaClean 代码不动，我们让它创建这一份冗余的 rope 也没关系
        )

        # Init Weights
        for m in [self.down_projection_img, self.down_projection_label, self.up_projection,
                  self.down_projection_img_2d, self.down_projection_label_2d, self.up_projection_2d]:
            m.apply(InitWeights_He(1e-2))

    def _encode_input(self, x, is_2d: bool, is_label: bool, type_id: int):
        if is_2d:
            proj_layer = self.down_projection_label_2d if is_label else self.down_projection_img_2d
            x_emb = proj_layer(x) 
            x_flat = rearrange(x_emb, 'b c w h -> b (w h) c')
        else:
            proj_layer = self.down_projection_label if is_label else self.down_projection_img
            x_emb = proj_layer(x)
            x_flat = rearrange(x_emb, 'b c w h d -> b (w h d) c')
        
        x_flat = x_flat + self.type_embed[type_id]
        return x_flat

    def forward(self, target_img, context_img, context_label, language_tokens=None):
        """
        target_img: [B, C, W, H, D] or [B, C, W, H]
        context_img: [B, L, C, W, H, D] or [B, L, C, W, H]
        context_label: [B, L, C, W, H, D] or [B, L, C, W, H]
        language_tokens: Optional[torch.Tensor], shape [B, L_lang, embed_dim]
        """
        B = target_img.shape[0]
        L = context_img.shape[1]
        
        # 1. Detect Mode
        is_2d = (target_img.ndim == 4)
        
        # 2. Encode Target
        x_tgt = self._encode_input(target_img, is_2d=is_2d, is_label=False, type_id=2)
        N = x_tgt.shape[1] 

        # 准备 Language Tokens 信息
        if language_tokens is not None:
            # 确保 language_tokens 维度符合要求 [B, L_lang, C]
            # 这里的 C 必须等于 self.embed_dim
            language_len = language_tokens.shape[1]
        else:
            language_len = 0
        
            
        # 3. Encode Contexts & Concatenate
        if L > 0:
            flat_shape_str = "b l c w h -> (b l) c w h" if is_2d else "b l c w h d -> (b l) c w h d"
            flat_ctx_img = rearrange(context_img, flat_shape_str)
            flat_ctx_lbl = rearrange(context_label, flat_shape_str)

            x_ctx_img = self._encode_input(flat_ctx_img, is_2d=is_2d, is_label=False, type_id=0)
            x_ctx_lbl = self._encode_input(flat_ctx_lbl, is_2d=is_2d, is_label=True, type_id=1)

            x_ctx_img = rearrange(x_ctx_img, "(b l) n c -> b l n c", b=B, l=L)
            x_ctx_lbl = rearrange(x_ctx_lbl, "(b l) n c -> b l n c", b=B, l=L)

            x_ctx_combined = torch.stack([x_ctx_img, x_ctx_lbl], dim=2)
            x_ctx_flat = rearrange(x_ctx_combined, "b l pair n c -> b (l pair n) c")
            
            # 拼接顺序：Target -> Language -> Context
            concat_list = [x_tgt]
            if language_len > 0:
                concat_list.append(language_tokens)
            concat_list.append(x_ctx_flat)
            
            x_concat = torch.cat(concat_list, dim=1)
        else:
            # L=0 的情况
            if language_len > 0:
                x_concat = torch.cat([x_tgt, language_tokens], dim=1)
            else:
                x_concat = x_tgt

        # --- 4. RoPE Preparation (Modified) ---
        if is_2d:
            rot_pos_embed = self.rope_2d.get_embed() # [N, D_head/2] or [1, N, ...]
        else:
            rot_pos_embed = self.rope_3d.get_embed() 
        
        rope_concat = None
        if rot_pos_embed is not None:
            # 确定拼接维度 (cat_dim)
            cat_dim = 1 if rot_pos_embed.ndim == 4 else 0
            
            # --- 构造各个部分的 RoPE ---
            
            # Part A: Target Image (直接复用 rot_pos_embed)
            rope_parts = [rot_pos_embed]
            
            # Part B: Language Tokens (使用全零填充，即恒等变换)
            if language_len > 0:
                # 1. 获取形状
                rope_shape = list(rot_pos_embed.shape)
                rope_shape[cat_dim] = language_len
                
                # 2. 计算 split 的维度 (emb 的最后一维包含了 sin 和 cos，所以要除以 2)
                # rot_pos_embed 的最后一维通常是 head_dim (或者 head_dim/2 * 2)
                total_dim = rope_shape[-1]
                half_dim = total_dim // 2
                
                # 3. 构造 Sin=0, Cos=1
                # 你的代码是 sin_emb, cos_emb = emb.chunk(2, -1) -> 前半是 Sin，后半是 Cos
                zeros_sin = torch.zeros(rope_shape[:-1] + [half_dim], device=rot_pos_embed.device, dtype=rot_pos_embed.dtype)
                ones_cos  = torch.ones(rope_shape[:-1] + [half_dim], device=rot_pos_embed.device, dtype=rot_pos_embed.dtype)
                
                
                # 4. 拼接得到恒等变换的 RoPE
                rope_lang = torch.cat([zeros_sin, ones_cos], dim=-1)
                rope_parts.append(rope_lang)
                
            # Part C: Contexts (如果有 Context，重复 Target 的 RoPE)
            if L > 0:
                repeats_ctx = L * 2
                rope_ctx = torch.cat([rot_pos_embed] * repeats_ctx, dim=cat_dim)
                rope_parts.append(rope_ctx)
            
            # 最终拼接
            rope_concat = torch.cat(rope_parts, dim=cat_dim)

        # 5. Forward EVA
        # 修改 Split Info: Target 部分现在包含了 Language Tokens
        if L > 0:
            split_info = (N + language_len, 2 * N)
        else:
            split_info = None

        x_out = self.eva(x_concat, rope_override=rope_concat, split_sizes=split_info)

        # 6. Decode
        # Target Image 始终位于最前面，所以直接取前 N 个 token
        x_tgt_out = x_out[:, :N, :]
        
        if is_2d:
            w, h = self.patch_size_2d[0], self.patch_size_2d[1] 
            fw, fh = self.rope_2d.feat_shape if self.rope_2d.feat_shape else (int(np.sqrt(N)), int(np.sqrt(N)))
            x_tgt_out = rearrange(x_tgt_out, "b (w h) c -> b c w h", w=fw, h=fh)
            logits = self.up_projection_2d(x_tgt_out)
        else:
            fw, fh, fd = self.rope_3d.feat_shape
            x_tgt_out = rearrange(x_tgt_out, "b (w h d) c -> b c w h d", w=fw, h=fh, d=fd)
            logits = self.up_projection(x_tgt_out)

        return logits

        
# ==============================================================================
# Helper Wrapper to instantiate specific configurations
# ==============================================================================

def primus_icl_s(input_channels=1, 
                 output_channels=1, 
                 patch_size=8, 
                 input_shape=(128, 128, 128),
                 grad_checkpointing = False,
                ):
    return PrimusICL(
        input_channels=input_channels,
        output_channels=output_channels,
        patch_embed_size=(patch_size, patch_size, patch_size),
        input_shape=input_shape,
        embed_dim=432, #Should be divided by both 6*numheads and 8*numheads
        eva_depth=12,
        eva_numheads=6,
        grad_checkpointing = grad_checkpointing,
    )

if __name__ == "__main__":
    import torch
    
    # 指定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 初始化模型 (以3D形状初始化，但这不影响2D运行)

    size_image = 128

    patch_size = 8

    model = primus_icl_s(

        input_channels=1, 

        output_channels=1, 

        patch_size=patch_size, 

        input_shape=(size_image, size_image, size_image)

    ).to(device)
    
    # 开启 eval 模式
    model.eval()

    # ==========================================
    # 准备基础数据
    # ==========================================
    # [B, C, D, H, W]
    tgt_3d = torch.randn(1, 1, size_image, size_image, size_image).to(device)
    # [B, L, C, D, H, W]
    ctx_img_3d = torch.randn(1, 1, 1, size_image, size_image, size_image).to(device)
    ctx_lbl_3d = torch.randn(1, 1, 1, size_image, size_image, size_image).to(device)

    # 2D 数据: [B, C, H, W]
    tgt_2d = torch.randn(1, 1, size_image, size_image).to(device)
    # [B, L, C, H, W]
    ctx_img_2d = torch.randn(1, 1, 1, size_image, size_image).to(device)
    ctx_lbl_2d = torch.randn(1, 1, 1, size_image, size_image).to(device)

    # 构造语言 Token [B, L_lang, embed_dim]
    lang_len = 5
    language_tokens = torch.randn(1, lang_len, 432).to(device)

    # ==========================================
    # Test Case 1: 3D Forward Pass (Original)
    # ==========================================
    print("\n--- Test 1: 3D Input (No Language) ---")
    try:
        with torch.no_grad():
            out_3d = model(tgt_3d, ctx_img_3d, ctx_lbl_3d)
        assert out_3d.shape == tgt_3d.shape
        print(">> 3D Pass (No Lang) Successful ✅")
    except Exception as e:
        print(f">> 3D Pass Failed ❌: {e}")
        import traceback; traceback.print_exc()

    # ==========================================
    # Test Case 2: 2D Forward Pass (Original)
    # ==========================================
    print("\n--- Test 2: 2D Input (No Language) ---")
    try:
        with torch.no_grad():
            out_2d = model(tgt_2d, ctx_img_2d, ctx_lbl_2d)
        assert out_2d.shape == tgt_2d.shape
        print(">> 2D Pass (No Lang) Successful ✅")
    except Exception as e:
        print(f">> 2D Pass Failed ❌: {e}")
        import traceback; traceback.print_exc()

    # ==========================================
    # Test Case 3: 3D Forward Pass (+ Language)
    # ==========================================
    print("\n--- Test 3: 3D Input + Language Tokens ---")
    try:
        with torch.no_grad():
            # 传入 language_tokens
            out_3d_lang = model(tgt_3d, ctx_img_3d, ctx_lbl_3d, language_tokens=language_tokens)
        
        print(f"3D+Lang Output Shape: {out_3d_lang.shape}")
        
        # 验证输出形状
        assert out_3d_lang.shape == tgt_3d.shape
        print(">> 3D + Language Pass Successful ✅")
        
        # 简单验证一下数值没有变成全 NaN (检查 RoPE sin/cos 构造是否有严重的广播错误)
        assert not torch.isnan(out_3d_lang).any(), "Output contains NaN!"
        
    except Exception as e:
        print(f">> 3D + Language Pass Failed ❌: {e}")
        import traceback; traceback.print_exc()

    # ==========================================
    # Test Case 4: 2D Forward Pass (+ Language)
    # ==========================================
    print("\n--- Test 4: 2D Input + Language Tokens ---")
    try:
        with torch.no_grad():
            # 传入 language_tokens
            out_2d_lang = model(tgt_2d, ctx_img_2d, ctx_lbl_2d, language_tokens=language_tokens)
            
        print(f"2D+Lang Output Shape: {out_2d_lang.shape}")
        
        # 验证输出形状
        assert out_2d_lang.shape == tgt_2d.shape
        print(">> 2D + Language Pass Successful ✅")
        
    except Exception as e:
        print(f">> 2D + Language Pass Failed ❌: {e}")
        import traceback; traceback.print_exc()
        
    # ==========================================
    # Test Case 5: 3D Input + Language (No Context)
    # ==========================================
    print("\n--- Test 5: 3D Input + Language (No Context) ---")
    # 测试 L=0 的情况，确保 language token 能单独和 target 拼接
    empty_ctx_img = torch.zeros(1, 0, 1, size_image, size_image, size_image).to(device)
    empty_ctx_lbl = torch.zeros(1, 0, 1, size_image, size_image, size_image).to(device)
    
    try:
        with torch.no_grad():
            out_3d_no_ctx = model(tgt_3d, empty_ctx_img, empty_ctx_lbl, language_tokens=language_tokens)
        assert out_3d_no_ctx.shape == tgt_3d.shape
        print(">> 3D + Language (No Context) Successful ✅")
    except Exception as e:
        print(f">> 3D + Language (No Context) Failed ❌: {e}")
        import traceback; traceback.print_exc()