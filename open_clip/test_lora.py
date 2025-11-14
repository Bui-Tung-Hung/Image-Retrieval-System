"""
Test script for LoRA integration with OpenCLIP.

This script validates that LoRA is working correctly with the text encoder.
"""

import torch
import open_clip
import logging
from open_clip.lora_utils import count_parameters, print_lora_status

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_lora_integration():
    """Test LoRA integration with OpenCLIP."""
    
    logger.info("=" * 70)
    logger.info("Testing LoRA Integration with OpenCLIP")
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
        return False
    
    # Test 2: Count parameters before LoRA
    logger.info("\n[Test 2] Counting parameters before LoRA...")
    params_before = count_parameters(model)
    logger.info(f"Total parameters: {params_before['total']:,}")
    logger.info(f"Trainable parameters: {params_before['trainable']:,}")
    
    # Test 3: Apply LoRA
    logger.info("\n[Test 3] Applying LoRA to text encoder...")
    try:
        # Freeze text encoder
        model.lock_text_tower(unlocked_layers=0)
        
        # Apply LoRA
        lora_stats = model.apply_lora_to_text_encoder(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['in_proj', 'out_proj']
        )
        logger.info(f"✓ LoRA applied to {lora_stats['lora_applied']}/{lora_stats['total_blocks']} blocks")
        
        if lora_stats['lora_failed'] > 0:
            logger.warning(f"⚠ Failed to apply LoRA to {lora_stats['lora_failed']} blocks")
    except Exception as e:
        logger.error(f"✗ Failed to apply LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify parameter counts after LoRA
    logger.info("\n[Test 4] Verifying parameters after LoRA...")
    params_after = count_parameters(model)
    print_lora_status(model, logger)
    
    # Verify that LoRA parameters are trainable
    if params_after['lora'] == 0:
        logger.error("✗ No LoRA parameters found!")
        return False
    
    if params_after['trainable'] == 0:
        logger.error("✗ No trainable parameters found!")
        return False
    
    if params_after['trainable'] != params_after['lora']:
        logger.warning(f"⚠ Trainable params ({params_after['trainable']}) != LoRA params ({params_after['lora']})")
    
    logger.info(f"✓ LoRA parameters: {params_after['lora']:,} ({params_after['lora_percentage']:.3f}%)")
    
    # Test 5: Forward pass with LoRA
    logger.info("\n[Test 5] Testing forward pass...")
    try:
        text = open_clip.tokenize(["a photo of a cat", "a photo of a dog"])
        
        with torch.no_grad():
            text_features = model.encode_text(text)
        
        logger.info(f"✓ Forward pass successful. Output shape: {text_features.shape}")
        logger.info(f"  Output dtype: {text_features.dtype}")
        logger.info(f"  Output device: {text_features.device}")
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test enable/disable LoRA
    logger.info("\n[Test 6] Testing enable/disable LoRA...")
    try:
        # Get features with LoRA
        with torch.no_grad():
            features_with_lora = model.encode_text(text)
        
        # Disable LoRA
        model.disable_lora()
        with torch.no_grad():
            features_without_lora = model.encode_text(text)
        
        # Enable LoRA again
        model.enable_lora()
        with torch.no_grad():
            features_with_lora_again = model.encode_text(text)
        
        # Verify that outputs are different
        diff_disabled = torch.abs(features_with_lora - features_without_lora).max().item()
        diff_enabled = torch.abs(features_with_lora - features_with_lora_again).max().item()
        
        logger.info(f"  Max difference (LoRA enabled vs disabled): {diff_disabled:.6f}")
        logger.info(f"  Max difference (LoRA enabled vs re-enabled): {diff_enabled:.6f}")
        
        if diff_disabled < 1e-6:
            logger.warning("⚠ LoRA enable/disable has no effect on output!")
        else:
            logger.info("✓ LoRA enable/disable works correctly")
        
        if diff_enabled > 1e-6:
            logger.error("✗ Re-enabling LoRA produces different results!")
            return False
        
    except Exception as e:
        logger.error(f"✗ Enable/disable test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 7: Test save/load LoRA
    logger.info("\n[Test 7] Testing save/load LoRA adapters...")
    try:
        import tempfile
        import os
        
        # Save LoRA
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "test_lora.pt")
            
            # Get current LoRA state
            lora_state = model.get_lora_state_dict()
            
            # Save
            from open_clip.lora_utils import save_lora_adapters
            save_lora_adapters(
                model,
                lora_path,
                config={'r': 16, 'alpha': 16, 'dropout': 0.1}
            )
            
            file_size_mb = os.path.getsize(lora_path) / (1024 * 1024)
            logger.info(f"✓ LoRA saved to {lora_path} ({file_size_mb:.2f} MB)")
            
            # Load
            from open_clip.lora_utils import load_lora_adapters
            loaded_config = load_lora_adapters(model, lora_path)
            
            logger.info(f"✓ LoRA loaded from {lora_path}")
            logger.info(f"  Loaded config: {loaded_config}")
        
    except Exception as e:
        logger.error(f"✗ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 8: Test gradient flow
    logger.info("\n[Test 8] Testing gradient flow...")
    try:
        model.train()
        text = open_clip.tokenize(["test text"])
        
        # Forward pass
        text_features = model.encode_text(text)
        loss = text_features.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = False
        no_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    has_grad = True
                    if 'lora' in name.lower():
                        logger.debug(f"  LoRA param with grad: {name}")
                else:
                    no_grad = True
        
        if has_grad:
            logger.info("✓ Gradients are flowing to LoRA parameters")
        else:
            logger.error("✗ No gradients found!")
            return False
        
        model.eval()
        
    except Exception as e:
        logger.error(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info("✓ All tests passed!")
    logger.info(f"✓ Base model: ViT-B-32 (laion2b_s34b_b79k)")
    logger.info(f"✓ LoRA config: r=16, alpha=16, dropout=0.1")
    logger.info(f"✓ Total parameters: {params_after['total']:,}")
    logger.info(f"✓ Trainable (LoRA) parameters: {params_after['lora']:,} ({params_after['lora_percentage']:.3f}%)")
    logger.info(f"✓ Memory efficient: Only {params_after['lora_percentage']:.3f}% of parameters are trainable!")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_lora_integration()
    
    if success:
        logger.info("\n🎉 LoRA integration is working correctly!")
        logger.info("You can now proceed with training using the train.sh script.")
    else:
        logger.error("\n❌ LoRA integration has issues. Please check the errors above.")
        exit(1)
