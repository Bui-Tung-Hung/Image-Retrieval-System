"""
LoRA (Low-Rank Adaptation) layers for OpenCLIP.

This module provides LoRA adapters for PyTorch nn.MultiheadAttention layers,
enabling efficient fine-tuning with minimal trainable parameters.

Implementation note: Instead of wrapping nn.MultiheadAttention (which causes recursion),
we inject LoRA parameters directly into the attention blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class LoRALayer(nn.Module):
    """
    Simple LoRA adapter module that can be added to any layer.
    
    Implements: output = base_output + (lora_B @ lora_A @ input) * (alpha / r)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Control flags
        self.enabled = True
        self.merged = False
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.merged:
            return torch.zeros_like(x)
        
        x = self.lora_dropout(x)
        return (F.linear(F.linear(x, self.lora_A), self.lora_B)) * self.scaling


def add_lora_to_attention_block(
    block: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list = None
) -> bool:
    """
    Add LoRA adapters to an attention block.
    
    This function injects LoRA parameters directly into the block
    without wrapping, to avoid recursion issues.
    
    Args:
        block: The ResidualAttentionBlock
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        target_modules: Which modules to add LoRA to
    
    Returns:
        bool: Success status
    """
    if target_modules is None:
        target_modules = ['in_proj', 'out_proj']
    
    if not hasattr(block, 'attn'):
        return False
    
    attn = block.attn
    
    # Check if it's nn.MultiheadAttention
    if not isinstance(attn, nn.MultiheadAttention):
        return False
    
    embed_dim = attn.embed_dim
    
    # Get device from block parameters
    device = next(block.parameters()).device
    
    # Add LoRA for in_proj (Q, K, V)
    if 'in_proj' in target_modules and attn.in_proj_weight is not None:
        # Create LoRA adapters for Q, K, V
        block.lora_q = LoRALayer(embed_dim, embed_dim, r, lora_alpha, lora_dropout).to(device)
        block.lora_k = LoRALayer(embed_dim, embed_dim, r, lora_alpha, lora_dropout).to(device)
        block.lora_v = LoRALayer(embed_dim, embed_dim, r, lora_alpha, lora_dropout).to(device)
        block.lora_enabled_in_proj = True
    
    # Add LoRA for out_proj
    if 'out_proj' in target_modules:
        block.lora_out = LoRALayer(embed_dim, embed_dim, r, lora_alpha, lora_dropout).to(device)
        block.lora_enabled_out_proj = True
    
    # Wrap the original attention forward to include LoRA
    original_forward = block.forward
    
    def forward_with_lora(q_x, k_x=None, v_x=None, attn_mask=None):
        # For self-attention case
        if k_x is None and v_x is None:
            x = q_x
            
            # Apply LoRA to input if enabled
            if hasattr(block, 'lora_enabled_in_proj') and block.lora_enabled_in_proj:
                # Get Q, K, V with LoRA
                ln_x = block.ln_1(x)
                
                # Base Q, K, V projections
                w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
                if attn.in_proj_bias is not None:
                    b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)
                else:
                    b_q = b_k = b_v = None
                
                # Compute with LoRA
                q = F.linear(ln_x, w_q, b_q) + block.lora_q(ln_x)
                k = F.linear(ln_x, w_k, b_k) + block.lora_k(ln_x)
                v = F.linear(ln_x, w_v, b_v) + block.lora_v(ln_x)
                
                # Reshape for multi-head attention
                batch_size, seq_len, _ = x.shape
                q = q.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                k = k.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                v = v.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                
                # Attention computation
                scale = 1.0 / math.sqrt(attn.head_dim)
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                
                if attn_mask is not None:
                    attn_weights = attn_weights + attn_mask
                
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                
                # Reshape back
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
                
                # Output projection with LoRA
                if hasattr(block, 'lora_enabled_out_proj') and block.lora_enabled_out_proj:
                    attn_output = attn.out_proj(attn_output) + block.lora_out(attn_output)
                else:
                    attn_output = attn.out_proj(attn_output)
                
                # Add residual and MLP
                x = x + block.ls_1(attn_output)
                x = x + block.ls_2(block.mlp(block.ln_2(x)))
                return x
        
        # Fall back to original for non-self-attention
        return original_forward(q_x, k_x, v_x, attn_mask)
    
    # Replace forward method
    block.forward = forward_with_lora
    
    return True
