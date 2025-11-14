"""
Helper utilities for managing LoRA adapters in OpenCLIP models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

from .lora_layers import add_lora_to_attention_block, LoRALayer


def apply_lora_to_text_encoder(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: List[str] = None,
) -> int:
    """
    Apply LoRA adapters to the text encoder of a CLIP model.
    
    Args:
        model: The CLIP model (must have a 'transformer' attribute for text encoder)
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA
        target_modules: List of module types to apply LoRA to.
                       Options: 'in_proj', 'out_proj'
    
    Returns:
        int: Number of blocks that were modified with LoRA
    """
    if target_modules is None:
        target_modules = ['in_proj', 'out_proj']  # Default: all attention projections
    
    modified_count = 0
    
    # Access text encoder transformer
    if hasattr(model, 'transformer'):
        transformer = model.transformer
    else:
        logging.warning("Model does not have 'transformer' attribute. LoRA not applied.")
        return 0
    
    # Apply LoRA to transformer blocks
    if hasattr(transformer, 'resblocks'):
        for i, block in enumerate(transformer.resblocks):
            success = add_lora_to_attention_block(
                block,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            
            if success:
                modified_count += 1
                logging.info(f"Applied LoRA to text encoder block {i}")
    
    logging.info(f"Applied LoRA to {modified_count} transformer blocks")
    return modified_count


def enable_lora(model: nn.Module):
    """Enable all LoRA adapters in the model."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.enabled = True


def disable_lora(model: nn.Module):
    """Disable all LoRA adapters in the model."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.enabled = False


def merge_lora(model: nn.Module):
    """
    Merge LoRA weights into base model weights for efficient inference.
    After merging, LoRA parameters are not used during forward pass.
    """
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if not module.merged:
                module.merged = True
                logging.info(f"Merged LoRA parameters in {module}")


def unmerge_lora(model: nn.Module):
    """
    Unmerge LoRA weights from base model weights.
    After unmerging, LoRA parameters are used again during forward pass.
    """
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if module.merged:
                module.merged = False
                logging.info(f"Unmerged LoRA parameters in {module}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total, trainable, and LoRA parameters in the model.
    
    Returns:
        Dict with keys: 'total', 'trainable', 'lora'
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            
            # Check if it's a LoRA parameter
            if 'lora_' in name:
                lora_params += num_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'lora': lora_params
    }


def save_lora_adapters(model: nn.Module, path: str):
    """
    Save only the LoRA adapter weights (not the full model).
    
    Args:
        model: The model with LoRA adapters
        path: Path to save the LoRA weights
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            # Save LoRA parameters
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
            lora_state_dict[f"{name}.enabled"] = module.enabled
            lora_state_dict[f"{name}.merged"] = module.merged
    
    # Also save LoRA parameters from blocks
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.data
    
    torch.save(lora_state_dict, path)
    logging.info(f"Saved LoRA adapters to {path}")
    logging.info(f"LoRA checkpoint size: {len(lora_state_dict)} parameters")


def load_lora_adapters(model: nn.Module, path: str):
    """
    Load LoRA adapter weights into a model.
    
    Args:
        model: The model to load LoRA adapters into
        path: Path to the saved LoRA weights
    """
    lora_state_dict = torch.load(path, map_location='cpu')
    
    # Load parameters
    model_state_dict = model.state_dict()
    for name, value in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = value
        else:
            logging.warning(f"LoRA parameter {name} not found in model")
    
    model.load_state_dict(model_state_dict, strict=False)
    logging.info(f"Loaded LoRA adapters from {path}")
    logging.info(f"Loaded {len(lora_state_dict)} LoRA parameters")


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA parameters.
    
    Returns:
        Dict mapping parameter names to tensors
    """
    lora_state_dict = {}
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.data
    
    return lora_state_dict
