import numpy as np
import matplotlib.pyplot as plt
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.video import MotionCompensator
import os

# Import the SimpleVideoCodec from ex1.py
from ex1 import SimpleVideoCodec

def run_debug_pipeline(image, quantization_scale=0.2):
    codec = IntraCodec(quantization_scale=quantization_scale, block_shape=(8, 8))
    img_ycbcr = rgb2ycbcr(image)
    if img_ycbcr.ndim == 2:
        img_ycbcr = img_ycbcr[:, :, np.newaxis]
    H, W, C = img_ycbcr.shape
    patches = codec.patcher.patch(img_ycbcr)
    print(f"[DEBUG][PATCHES] shape: {patches.shape}")
    dct_patches = codec.dct.transform(patches)
    print(f"[DEBUG][DCT] min: {dct_patches.min():.2f}, max: {dct_patches.max():.2f}, mean: {dct_patches.mean():.2f}")
    quantized = codec.quant.quantize(dct_patches)
    print(f"[DEBUG][QUANT] min: {quantized.min()}, max: {quantized.max()}, mean: {quantized.mean():.2f}")
    zz_scanned = codec.zigzag.flatten(quantized)
    print(f"[DEBUG][ZZ] min: {zz_scanned.min()}, max: {zz_scanned.max()}, mean: {zz_scanned.mean():.2f}")
    symbols = codec.zerorun.encode(zz_scanned)
    print(f"[DEBUG][ZERORUN] symbols: {len(symbols)}")
    # Reverse pipeline
    patch_shape = [H // 8, W // 8, C]
    decoded = codec.zerorun.decode(symbols, original_shape=patch_shape)
    print(f"[DEBUG][ZERORUN DECODE] shape: {decoded.shape}, min: {decoded.min()}, max: {decoded.max()}, mean: {decoded.mean():.2f}")
    inv_zz = codec.zigzag.unflatten(decoded)
    print(f"[DEBUG][UNZZ] shape: {inv_zz.shape}, min: {inv_zz.min()}, max: {inv_zz.max()}, mean: {inv_zz.mean():.2f}")
    dequant = codec.quant.dequantize(inv_zz)
    print(f"[DEBUG][DEQUANT] shape: {dequant.shape}, min: {dequant.min():.2f}, max: {dequant.max():.2f}, mean: {dequant.mean():.2f}")
    idct = codec.dct.inverse_transform(dequant)
    print(f"[DEBUG][IDCT] shape: {idct.shape}, min: {idct.min():.2f}, max: {idct.max():.2f}, mean: {idct.mean():.2f}")
    from einops import rearrange
    ycbcr_img = rearrange(idct, 'hp wp c h w -> (hp h) (wp w) c')
    print(f"[DEBUG][REARRANGE] shape: {ycbcr_img.shape}, min: {ycbcr_img.min():.2f}, max: {ycbcr_img.max():.2f}, mean: {ycbcr_img.mean():.2f}")
    print(f"[DEBUG][TO RGB] ycbcr_img shape: {ycbcr_img.shape}, min: {ycbcr_img.min():.2f}, max: {ycbcr_img.max():.2f}, mean: {ycbcr_img.mean():.2f}")
    rgb_img = ycbcr2rgb(ycbcr_img)
    print(f"[DEBUG][TO RGB] rgb_img shape: {rgb_img.shape}, min: {rgb_img.min():.2f}, max: {rgb_img.max():.2f}, mean: {rgb_img.mean():.2f}")
    reconstructed = np.clip(rgb_img, 0, 255).astype(np.uint8)
    print(f"[DEBUG][CLIPPED UINT8] shape: {reconstructed.shape}, min: {reconstructed.min()}, max: {reconstructed.max()}, mean: {reconstructed.mean():.2f}")
    return reconstructed

def run_video_codec(image, quantization_scale=0.2):
    video_codec = SimpleVideoCodec(
        quantization_scale=quantization_scale, 
        bounds=(-1000, 4000), 
        end_of_block=4000, 
        block_shape=(8, 8), 
        search_range=4
    )
    recon_rgb, bitstream, bitsize = video_codec.encode_decode(image, frame_num=0, is_source_rgb=True)
    return recon_rgb

def main():
    test_image_path = 'data/foreman20_40_RGB/foreman0020.bmp'
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    image = imread(test_image_path)
    print("\n===== DEBUG PIPELINE =====")
    recon_debug = run_debug_pipeline(image, quantization_scale=0.2)
    psnr_debug = calc_psnr(image, recon_debug)
    print(f"[DEBUG PIPELINE] PSNR: {psnr_debug:.2f} dB")
    print("\n===== VIDEO CODEC PIPELINE =====")
    recon_video = run_video_codec(image, quantization_scale=0.2)
    psnr_video = calc_psnr(image, recon_video)
    print(f"[VIDEO CODEC] PSNR: {psnr_video:.2f} dB")
    # Show and save images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(recon_debug)
    axes[1].set_title(f"Debug Pipeline\nPSNR: {psnr_debug:.2f} dB")
    axes[2].imshow(recon_video)
    axes[2].set_title(f"Video Codec\nPSNR: {psnr_video:.2f} dB")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('mashup_comparison.png')
    plt.show()
    # Save difference images
    diff_debug = np.abs(image.astype(np.float32) - recon_debug.astype(np.float32))
    diff_video = np.abs(image.astype(np.float32) - recon_video.astype(np.float32))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(diff_debug.astype(np.uint8), cmap='hot')
    plt.title('Diff Debug')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(diff_video.astype(np.uint8), cmap='hot')
    plt.title('Diff Video Codec')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('mashup_diffs.png')
    plt.show()
    # Save images for inspection
    from imageio import imwrite
    imwrite('mashup_recon_debug.png', recon_debug)
    imwrite('mashup_recon_video.png', recon_video)

if __name__ == "__main__":
    main()
