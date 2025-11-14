"""
Simple test script for LoRA integration with OpenCLIP.
Tests the new direct parameter injection approach.
"""

import torch
import open_clip
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("Testing LoRA Integration with OpenCLIP (New Approach)")
    logger.info("=" * 70)
    
    # Test 1: Load base model
    logger.info("\n[Test 1] Loading base model...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k'
        )
        model.eval()
        logger.info("✓ Base model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load base model: {e}")
        return
    
    # Test 2: Count parameters before LoRA
    logger.info("\n[Test 2] Parameters before LoRA...")
    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total: {total_before:,}")
    logger.info(f"Trainable: {trainable_before:,}")
    
    # Test 3: Apply LoRA
    logger.info("\n[Test 3] Applying LoRA to text encoder...")
    try:
        # Freeze text encoder first
        model.lock_text_tower(unlocked_layers=0)
        
        # Apply LoRA
        modified_blocks = model.apply_lora_to_text_encoder(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1
        )
        
        logger.info(f"✓ Applied LoRA to {modified_blocks} blocks")
    except Exception as e:
        logger.error(f"✗ Failed to apply LoRA: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Count parameters after LoRA
    logger.info("\n[Test 4] Parameters after LoRA...")
    total_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)
    
    logger.info(f"Total: {total_after:,}")
    logger.info(f"Trainable: {trainable_after:,}")
    logger.info(f"LoRA: {lora_params:,} ({100 * lora_params / total_after:.2f}%)")
    
    # Test 5: Check blocks have LoRA layers
    logger.info("\n[Test 5] Checking LoRA injection...")
    block_count = 0
    lora_q_count = 0
    lora_k_count = 0
    lora_v_count = 0
    lora_out_count = 0
    
    if hasattr(model, 'transformer'):
        for i, block in enumerate(model.transformer.resblocks):
            block_count += 1
            if hasattr(block, 'lora_q'):
                lora_q_count += 1
            if hasattr(block, 'lora_k'):
                lora_k_count += 1
            if hasattr(block, 'lora_v'):
                lora_v_count += 1
            if hasattr(block, 'lora_out'):
                lora_out_count += 1
    
    logger.info(f"Total blocks: {block_count}")
    logger.info(f"Blocks with lora_q: {lora_q_count}")
    logger.info(f"Blocks with lora_k: {lora_k_count}")
    logger.info(f"Blocks with lora_v: {lora_v_count}")
    logger.info(f"Blocks with lora_out: {lora_out_count}")
    
    if lora_q_count == block_count and lora_k_count == block_count and lora_v_count == block_count and lora_out_count == block_count:
        logger.info("✓ All blocks have LoRA layers")
    else:
        logger.warning("✗ Not all blocks have LoRA layers")
    
    # Test 6: Test forward pass
    logger.info("\n[Test 6] Testing forward pass...")
    try:
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_tokens = tokenizer(["A photo of a cat", "A photo of a dog"])
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        
        logger.info(f"✓ Forward pass successful. Output shape: {text_features.shape}")
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 7: Test enable/disable
    logger.info("\n[Test 7] Testing enable/disable LoRA...")
    try:
        model.disable_lora()
        logger.info("✓ Disabled LoRA")
        
        model.enable_lora()
        logger.info("✓ Enabled LoRA")
    except Exception as e:
        logger.error(f"✗ Enable/disable failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 8: Test gradient flow
    logger.info("\n[Test 8] Testing gradient flow...")
    try:
        model.train()
        
        # Forward pass
        text_features = model.encode_text(text_tokens)
        loss = text_features.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        lora_params_with_grad = 0
        base_params_with_grad = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'lora_' in name:
                    lora_params_with_grad += 1
                else:
                    base_params_with_grad += 1
        
        logger.info(f"LoRA parameters with gradient: {lora_params_with_grad}")
        logger.info(f"Base parameters with gradient: {base_params_with_grad}")
        
        if lora_params_with_grad > 0 and base_params_with_grad == 0:
            logger.info("✓ Gradients flow only to LoRA parameters")
        else:
            logger.warning("✗ Gradient flow may not be correct")
    
    except Exception as e:
        logger.error(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "=" * 70)
    logger.info("All tests completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
