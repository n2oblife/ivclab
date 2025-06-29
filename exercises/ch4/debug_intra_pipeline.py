# #!/usr/bin/env python3
# """
# Debug script to test the intra coding pipeline step by step
# and identify where the blocky artifact is introduced.
# """

import numpy as np
import matplotlib.pyplot as plt
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr
import os

def test_intra_pipeline():
    """Test the intra coding pipeline with debug output"""
    
    print("=" * 80)
    print("DEBUGGING INTRA CODING PIPELINE")
    print("=" * 80)
    
    # Load test image
    test_image_path = 'data/foreman20_40_RGB/foreman0020.bmp'
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Load and display original image
    original_image = imread(test_image_path)
    print(f"📸 Original image shape: {original_image.shape}")
    print(f"📸 Original image dtype: {original_image.dtype}")
    print(f"📸 Original image range: [{original_image.min()}, {original_image.max()}]")
    
    # Create codec with lower quantization scale
    codec = IntraCodec(quantization_scale=0.2, block_shape=(8, 8))
    
    # Train Huffman coder on the same image
    print("\n🔧 Training Huffman coder...")
    codec.train_huffman_from_image(original_image, is_source_rgb=True)
    
    # Test encode/decode
    print("\n🔄 Testing encode/decode pipeline...")
    try:
        result = codec.encode_decode(original_image, return_bpp=False, is_source_rgb=True)
        if len(result) == 4:
            reconstructed_image, bitstream, bitsize, bpp = result
        else:
            reconstructed_image, bitstream, bitsize = result
        
        print(f"✅ Encode/decode successful!")
        print(f"📊 Bitsize: {bitsize}")
        print(f"📊 BPP: {bitsize / (original_image.size / 3):.4f}")
        
        # Calculate PSNR
        psnr = calc_psnr(original_image, reconstructed_image)
        print(f"📊 PSNR: {psnr:.2f} dB")
        
        # Display results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed_image)
        axes[1].set_title(f"Reconstructed Image\nPSNR: {psnr:.2f} dB")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Check for artifacts
        if hasattr(reconstructed_image, 'astype'):
            diff = np.abs(original_image.astype(np.float32) - reconstructed_image.astype(np.float32))
            print(f"📊 Max difference: {diff.max():.2f}")
            print(f"📊 Mean difference: {diff.mean():.2f}")
            
            # Look for blocky patterns in difference
            if diff.max() > 50:  # Significant difference
                print("⚠️  Large differences detected - possible artifacts!")
                
                # Show difference image
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(diff, cmap='hot')
                plt.title("Absolute Difference")
                plt.colorbar()
                
                plt.subplot(1, 2, 2)
                plt.imshow(diff[100:200, 100:200], cmap='hot')  # Zoom in on a region
                plt.title("Difference (Zoomed)")
                plt.colorbar()
                
                plt.tight_layout()
                plt.show()
        
    except Exception as e:
        print(f"❌ Error during encode/decode: {e}")
        import traceback
        traceback.print_exc()

def test_step_by_step():
    """Test each step of the pipeline individually with value range debug prints"""
    print("\n" + "=" * 80)
    print("STEP-BY-STEP PIPELINE TESTING")
    print("=" * 80)
    
    # Load test image
    test_image_path = 'data/foreman20_40_RGB/foreman0020.bmp'
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    original_image = imread(test_image_path)
    codec = IntraCodec(quantization_scale=0.2, block_shape=(8, 8))
    
    print("\n🔧 Step 1: Convert to symbols (with debug prints)")
    try:
        # --- image2symbols pipeline ---
        img = original_image
        from ivclab.signal import rgb2ycbcr
        img_ycbcr = rgb2ycbcr(img)
        if img_ycbcr.ndim == 2:
            img_ycbcr = img_ycbcr[:, :, np.newaxis]
        H, W, C = img_ycbcr.shape
        # Use the same patching as in IntraCodec
        patches = codec.patcher.patch(img_ycbcr)
        print(f"[PATCHES] shape: {patches.shape}")
        dct_patches = codec.dct.transform(patches)
        print(f"[DCT] min: {dct_patches.min():.2f}, max: {dct_patches.max():.2f}, mean: {dct_patches.mean():.2f}")
        quantized = codec.quant.quantize(dct_patches)
        print(f"[QUANT] min: {quantized.min()}, max: {quantized.max()}, mean: {quantized.mean():.2f}")
        zz_scanned = codec.zigzag.flatten(quantized)
        print(f"[ZZ] min: {zz_scanned.min()}, max: {zz_scanned.max()}, mean: {zz_scanned.mean():.2f}")
        symbols = codec.zerorun.encode(zz_scanned)
        print(f"[ZERORUN] symbols: {len(symbols)}")
    except Exception as e:
        print(f"❌ Error in image2symbols: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🔧 Step 2: Convert back to image (with debug prints)")
    try:
        # --- symbols2image pipeline ---
        patch_shape = [H // 8, W // 8, C]
        decoded = codec.zerorun.decode(symbols, original_shape=patch_shape)
        print(f"[ZERORUN DECODE] shape: {decoded.shape}, min: {decoded.min()}, max: {decoded.max()}, mean: {decoded.mean():.2f}")
        inv_zz = codec.zigzag.unflatten(decoded)
        print(f"[UNZZ] shape: {inv_zz.shape}, min: {inv_zz.min()}, max: {inv_zz.max()}, mean: {inv_zz.mean():.2f}")
        dequant = codec.quant.dequantize(inv_zz)
        print(f"[DEQUANT] shape: {dequant.shape}, min: {dequant.min():.2f}, max: {dequant.max():.2f}, mean: {dequant.mean():.2f}")
        idct = codec.dct.inverse_transform(dequant)
        print(f"[IDCT] shape: {idct.shape}, min: {idct.min():.2f}, max: {idct.max():.2f}, mean: {idct.mean():.2f}")
        # Rearrange back to image
        from einops import rearrange
        ycbcr_img = rearrange(idct, 'hp wp c h w -> (hp h) (wp w) c')
        print(f"[REARRANGE] shape: {ycbcr_img.shape}, min: {ycbcr_img.min():.2f}, max: {ycbcr_img.max():.2f}, mean: {ycbcr_img.mean():.2f}")
        if C == 3:
            from ivclab.signal import ycbcr2rgb
            print(f"[TO RGB] ycbcr_img shape: {ycbcr_img.shape}, min: {ycbcr_img.min():.2f}, max: {ycbcr_img.max():.2f}, mean: {ycbcr_img.mean():.2f}")
            rgb_img = ycbcr2rgb(ycbcr_img)
            print(f"[TO RGB] rgb_img shape: {rgb_img.shape}, min: {rgb_img.min():.2f}, max: {rgb_img.max():.2f}, mean: {rgb_img.mean():.2f}")
            reconstructed = np.clip(rgb_img, 0, 255).astype(np.uint8)
            print(f"[CLIPPED UINT8] shape: {reconstructed.shape}, min: {reconstructed.min()}, max: {reconstructed.max()}, mean: {reconstructed.mean():.2f}")
        else:
            reconstructed = np.clip(ycbcr_img, 0, 255).astype(np.uint8)
        print(f"✅ Image reconstructed: {reconstructed.shape if hasattr(reconstructed, 'shape') else 'unknown'}")
        # Compare with original
        if hasattr(reconstructed, 'astype'):
            diff = np.abs(original_image.astype(np.float32) - reconstructed.astype(np.float32))
            print(f"[DIFF] max: {diff.max():.2f}, mean: {diff.mean():.2f}")
    except Exception as e:
        print(f"❌ Error in symbols2image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the debugging tests
    test_intra_pipeline()
    test_step_by_step() 