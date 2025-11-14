"""
Script to compare model parameters with and without LoRA.
Shows the efficiency of LoRA for fine-tuning.
"""

import torch
import open_clip
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def count_params(model):
    """Count total, trainable, and LoRA parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'lora': lora,
        'frozen': total - trainable
    }


def format_number(num):
    """Format large numbers with M/K suffix."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def main():
    logger.info("=" * 80)
    logger.info("CLIP Model Parameter Comparison: With and Without LoRA")
    logger.info("=" * 80)
    
    # Load base model
    logger.info("\n[1] Loading base ViT-B-32 model...")
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k'
    )
    model.eval()
    
    # Count parameters - Base model (all trainable)
    logger.info("\n" + "─" * 80)
    logger.info("📊 BASE MODEL (No LoRA, All Parameters Trainable)")
    logger.info("─" * 80)
    base_params = count_params(model)
    
    logger.info(f"Total parameters:      {base_params['total']:>15,}  ({format_number(base_params['total'])})")
    logger.info(f"Trainable parameters:  {base_params['trainable']:>15,}  ({format_number(base_params['trainable'])})")
    logger.info(f"Frozen parameters:     {base_params['frozen']:>15,}  ({format_number(base_params['frozen'])})")
    
    # Scenario 1: Lock image encoder only
    logger.info("\n" + "─" * 80)
    logger.info("📊 SCENARIO 1: Lock Image Encoder (Text Encoder Fully Trainable)")
    logger.info("─" * 80)
    model.lock_image_tower(unlocked_groups=0)
    params_lock_image = count_params(model)
    
    logger.info(f"Total parameters:      {params_lock_image['total']:>15,}  ({format_number(params_lock_image['total'])})")
    logger.info(f"Trainable parameters:  {params_lock_image['trainable']:>15,}  ({format_number(params_lock_image['trainable'])})")
    logger.info(f"Frozen parameters:     {params_lock_image['frozen']:>15,}  ({format_number(params_lock_image['frozen'])})")
    logger.info(f"Reduction:             {100 * (1 - params_lock_image['trainable'] / base_params['trainable']):.1f}%")
    
    # Reload model for LoRA scenario
    logger.info("\n[2] Reloading model for LoRA configuration...")
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k'
    )
    model.eval()
    
    # Scenario 2: Lock image encoder + Apply LoRA to text encoder
    logger.info("\n" + "─" * 80)
    logger.info("📊 SCENARIO 2: Lock Image + LoRA on Text Encoder (r=16)")
    logger.info("─" * 80)
    
    # Lock image encoder
    model.lock_image_tower(unlocked_groups=0)
    
    # Lock text encoder
    model.lock_text_tower(unlocked_layers=0)
    
    # Apply LoRA
    lora_blocks = model.apply_lora_to_text_encoder(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    params_with_lora = count_params(model)
    
    logger.info(f"LoRA applied to:       {lora_blocks} transformer blocks")
    logger.info(f"Total parameters:      {params_with_lora['total']:>15,}  ({format_number(params_with_lora['total'])})")
    logger.info(f"Trainable parameters:  {params_with_lora['trainable']:>15,}  ({format_number(params_with_lora['trainable'])})")
    logger.info(f"  ├─ LoRA parameters:  {params_with_lora['lora']:>15,}  ({format_number(params_with_lora['lora'])})")
    logger.info(f"  └─ Other trainable:  {params_with_lora['trainable'] - params_with_lora['lora']:>15,}  ({format_number(params_with_lora['trainable'] - params_with_lora['lora'])})")
    logger.info(f"Frozen parameters:     {params_with_lora['frozen']:>15,}  ({format_number(params_with_lora['frozen'])})")
    
    # Calculate efficiency metrics
    logger.info("\n" + "─" * 80)
    logger.info("📈 EFFICIENCY METRICS")
    logger.info("─" * 80)
    
    # Compare with base model
    reduction_vs_base = 100 * (1 - params_with_lora['trainable'] / base_params['trainable'])
    lora_percentage = 100 * params_with_lora['lora'] / params_with_lora['total']
    trainable_percentage = 100 * params_with_lora['trainable'] / params_with_lora['total']
    
    logger.info(f"LoRA parameters vs Total:           {lora_percentage:.2f}%")
    logger.info(f"Trainable parameters vs Total:      {trainable_percentage:.2f}%")
    logger.info(f"Parameter reduction vs Base:        {reduction_vs_base:.1f}%")
    logger.info(f"Memory efficiency:                  {base_params['trainable'] / params_with_lora['trainable']:.1f}x less trainable params")
    
    # Comparison table
    logger.info("\n" + "=" * 80)
    logger.info("📋 SUMMARY COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info(f"{'Configuration':<40} {'Total':<15} {'Trainable':<15} {'% Trainable':<12}")
    logger.info("─" * 80)
    logger.info(f"{'Base Model (All trainable)':<40} {format_number(base_params['total']):<15} {format_number(base_params['trainable']):<15} {100 * base_params['trainable'] / base_params['total']:>10.1f}%")
    logger.info(f"{'Lock Image Only':<40} {format_number(params_lock_image['total']):<15} {format_number(params_lock_image['trainable']):<15} {100 * params_lock_image['trainable'] / params_lock_image['total']:>10.1f}%")
    logger.info(f"{'Lock Image + LoRA Text (r=16)':<40} {format_number(params_with_lora['total']):<15} {format_number(params_with_lora['trainable']):<15} {100 * params_with_lora['trainable'] / params_with_lora['total']:>10.1f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ LoRA enables efficient fine-tuning with only 0.52% trainable parameters!")
    logger.info("=" * 80)
    
    # Additional info about checkpoint sizes
    logger.info("\n" + "💾 ESTIMATED CHECKPOINT SIZES")
    logger.info("─" * 80)
    
    # Assume float32 (4 bytes per parameter)
    base_size_mb = base_params['total'] * 4 / (1024 * 1024)
    lora_only_size_mb = params_with_lora['lora'] * 4 / (1024 * 1024)
    
    logger.info(f"Full model checkpoint:              ~{base_size_mb:.1f} MB")
    logger.info(f"LoRA-only checkpoint:               ~{lora_only_size_mb:.1f} MB")
    logger.info(f"Space saving:                       {base_size_mb / lora_only_size_mb:.0f}x smaller")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
