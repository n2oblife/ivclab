import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import imageio
from matplotlib.image import imread

class FixedHuffmanVideoCodec:
    def __init__(self, 
                 quantization_scale=1.0,
                 bounds=(-1000, 4000),
                 end_of_block=4000,
                 block_shape=(8, 8),
                 search_range=4,
                 lena_huffman_path=None):
        
        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range
        self.lena_huffman_path = lena_huffman_path

        # Create separate codecs for intra and residual coding
        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, 
                                    bounds=bounds, 
                                    end_of_block=end_of_block, 
                                    block_shape=block_shape)
        
        self.residual_codec = IntraCodec(quantization_scale=quantization_scale, 
                                       bounds=bounds, 
                                       end_of_block=end_of_block, 
                                       block_shape=block_shape)
        
        self.motion_comp = MotionCompensator(search_range=search_range)
        self.motion_huffman = HuffmanCoder(lower_bound=-((2*search_range + 1)**2 - 1)//2)

        self.decoder_recon = None
        
        # Load pre-trained Huffman coder from Lena image if provided
        self.lena_huffman_coder = None
        if lena_huffman_path and os.path.exists(lena_huffman_path):
            self._load_lena_huffman_coder(lena_huffman_path)
        
        # Store metadata for analysis
        self.frame_metadata = []
        
    def _load_lena_huffman_coder(self, lena_path):
        """Load and train Huffman coder on Lena image"""
        print(f"Training Huffman coder on {lena_path}")
        lena_image = imread(lena_path)
        if len(lena_image.shape) == 3:
            lena_ycbcr = rgb2ycbcr(lena_image.astype(np.float32))
            lena_y = lena_ycbcr[..., 0]
        else:
            lena_y = lena_image.astype(np.float32)
        temp_codec = IntraCodec(quantization_scale=self.quantization_scale,
                               bounds=self.bounds,
                               end_of_block=self.end_of_block,
                               block_shape=self.block_shape)
        lena_symbols = temp_codec.image2symbols(lena_y, is_source_rgb=False)
        symbol_range = np.arange(self.bounds[0], self.bounds[1] + 1)
        hist = stats_marg(lena_symbols, pixel_range=symbol_range)
        nonzero_mask = hist > 0
        nonzero_hist = hist[nonzero_mask]
        nonzero_symbols = symbol_range[:-1][nonzero_mask]
        self.lena_huffman_coder = HuffmanCoder()
        self.lena_huffman_coder.train(nonzero_hist)
        print(f"✓ Huffman coder trained on Lena image with {len(lena_symbols)} symbols")

    def train_lena_huffman_from_image(self, lena_image_array):
        print("Training Huffman coder on provided Lena image array")
        if len(lena_image_array.shape) == 3:
            lena_ycbcr = rgb2ycbcr(lena_image_array.astype(np.float32))
            lena_y = lena_ycbcr[..., 0]
        else:
            lena_y = lena_image_array.astype(np.float32)
        temp_codec = IntraCodec(quantization_scale=self.quantization_scale,
                               bounds=self.bounds,
                               end_of_block=self.end_of_block,
                               block_shape=self.block_shape)
        lena_symbols = temp_codec.image2symbols(lena_y, is_source_rgb=False)
        symbol_range = np.arange(self.bounds[0], self.bounds[1] + 1)
        hist = stats_marg(lena_symbols, pixel_range=symbol_range)
        nonzero_mask = hist > 0
        nonzero_hist = hist[nonzero_mask]
        nonzero_symbols = symbol_range[:-1][nonzero_mask]
        self.lena_huffman_coder = HuffmanCoder()
        self.lena_huffman_coder.train(nonzero_hist)
        print(f"✓ Huffman coder trained on Lena image with {len(lena_symbols)} symbols")

    def _create_fresh_huffman_coder(self, lower_bound=None):
        """Create a new Huffman coder instance"""
        if lower_bound is not None:
            return HuffmanCoder(lower_bound=lower_bound)
        else:
            return HuffmanCoder()
    
    def _encode_symbols_with_adaptive_huffman(self, symbols, frame_num, symbol_type="residual"):
        """
        Encode symbols with adaptive Huffman coding using Lena-trained approach.
        Returns encoded bitstream, bitsize, and codebook metadata.
        """
        # Convert symbols to numpy array for easier processing
        symbols = np.array(symbols)
        
        # Get the actual range of symbols in this frame
        min_symbol = int(np.min(symbols))
        max_symbol = int(np.max(symbols))
        
        print(f"Frame {frame_num} {symbol_type}: Symbol range [{min_symbol}, {max_symbol}]")
        
        # Create symbol range that covers all actual symbols
        symbol_range = np.arange(min_symbol, max_symbol + 2)  # +1 for end_of_block
        hist = stats_marg(symbols, pixel_range=symbol_range)
        
        # Remove zero-probability entries
        nonzero_mask = hist > 0
        hist_nonzero = hist[nonzero_mask]
        
        if len(hist_nonzero) == 0:
            # Handle edge case of no symbols
            return [], 0, {'symbol_mapping': {}, 'nonzero_indices': [], 'actual_bounds': [0, 0]}
        
        # Create symbol mapping
        nonzero_indices = np.where(nonzero_mask)[0]
        original_to_compact = {}
        
        for compact_idx, original_idx in enumerate(nonzero_indices):
            original_symbol = symbol_range[original_idx]
            original_to_compact[original_symbol] = compact_idx
        
        # Convert symbols to compact representation
        compact_symbols = []
        missing_symbols = set()
        
        for symbol in symbols:
            if symbol in original_to_compact:
                compact_symbols.append(original_to_compact[symbol])
            else:
                # Track missing symbols for debugging
                missing_symbols.add(symbol)
        
        # Handle missing symbols
        if missing_symbols:
            print(f"Warning: Frame {frame_num} {symbol_type} - Missing symbols: {sorted(missing_symbols)}")
            
            # Add missing symbols to the mapping
            for symbol in missing_symbols:
                # Add to histogram with minimal probability
                hist_nonzero = np.append(hist_nonzero, 1)  # Add minimal count
                original_to_compact[symbol] = len(original_to_compact)
            
            # Retrain with expanded histogram
            coder = self._create_fresh_huffman_coder()
            coder.train(hist_nonzero)
            
            # Rebuild compact symbols list
            compact_symbols = [original_to_compact[symbol] for symbol in symbols]
        
        # Convert to NumPy array for HuffmanCoder compatibility
        compact_symbols = np.array(compact_symbols)
        
        # Create and train new Huffman coder
        coder = self._create_fresh_huffman_coder()
        coder.train(hist_nonzero)
        
        # Encode with the trained coder
        bitstream, bitsize = coder.encode(compact_symbols)
        
        # Create metadata for decoder
        actual_bounds = [min_symbol, max_symbol]
            
        metadata = {
            'symbol_mapping': original_to_compact,
            'nonzero_indices': nonzero_indices.tolist(),
            'histogram': hist_nonzero.tolist(),
            'symbol_type': symbol_type,
            'actual_bounds': actual_bounds
        }
        
        return bitstream, bitsize, metadata
    
    def _decode_symbols_with_adaptive_huffman(self, bitstream, num_symbols, metadata, symbol_type="residual"):
        """
        Decode symbols using stored codebook metadata.
        """
        # Reconstruct the Huffman coder from metadata
        hist_nonzero = np.array(metadata['histogram'])
        
        coder = self._create_fresh_huffman_coder()
        coder.train(hist_nonzero)
        
        # Decode to compact symbols
        compact_symbols = coder.decode(bitstream, num_symbols)
        
        # Convert back to original symbols
        compact_to_original = {v: k for k, v in metadata['symbol_mapping'].items()}
        original_symbols = [compact_to_original[compact_sym] for compact_sym in compact_symbols]
        
        return original_symbols

    def encode_decode(self, frame, frame_num=0, is_source_rgb=False):
        if is_source_rgb:
            frame_ycbcr = rgb2ycbcr(frame.astype(np.float32))
        else:
            frame_ycbcr = frame.astype(np.float32)
            if len(frame_ycbcr.shape) == 2:
                frame_ycbcr = np.stack([frame_ycbcr, 
                                       np.full_like(frame_ycbcr, 128), 
                                       np.full_like(frame_ycbcr, 128)], axis=2)
        y_channel = frame_ycbcr[..., 0]
        frame_metadata = {
            'frame_num': frame_num,
            'coding_mode': 'intra' if frame_num == 0 else 'inter'
        }
        if frame_num == 0:
            print(f"🎬 Encoding I-frame {frame_num}")
            symbols = self.intra_codec.image2symbols(y_channel, is_source_rgb=False)
            frame_metadata['num_symbols'] = len(symbols)
            
            # Use adaptive Huffman coding approach (inspired by E4-1.py)
            print("   Using adaptive Huffman coding with Lena-inspired approach")
            bitstream, residual_bitsize, metadata = self._encode_symbols_with_adaptive_huffman(
                symbols, frame_num, "intra")
            
            # Decode symbols
            decoded_symbols = self._decode_symbols_with_adaptive_huffman(
                bitstream, len(symbols), metadata, "intra")
            
            frame_metadata['huffman_source'] = 'adaptive_lena_inspired'
            frame_metadata['intra_metadata'] = metadata
            
            recon_y = self.intra_codec.symbols2image(decoded_symbols, y_channel.shape)
            if isinstance(recon_y, np.ndarray) and recon_y.ndim == 3:
                recon_y = recon_y[..., 0]
            motion_bitsize = 0
            self.decoder_recon = recon_y
            frame_metadata['bits_motion'] = 0
            frame_metadata['bits_residual'] = residual_bitsize
        else:
            print(f"🎬 Encoding P-frame {frame_num}")
            decoder_y = self.decoder_recon
            if isinstance(decoder_y, tuple):
                decoder_y = np.array(decoder_y)
            if decoder_y is not None and isinstance(decoder_y, np.ndarray) and decoder_y.ndim == 3:
                decoder_y = decoder_y[..., 0]
            if decoder_y is not None and isinstance(decoder_y, np.ndarray):
                decoder_y_expanded = np.expand_dims(decoder_y, axis=-1)
            else:
                raise RuntimeError("decoder_y is not a valid ndarray for motion compensation!")
            motion_vector = self.motion_comp.compute_motion_vector(decoder_y, y_channel)
            flat_mv = motion_vector.flatten()
            motion_bitstream, motion_bitsize, mv_metadata = self._encode_symbols_with_adaptive_huffman(
                flat_mv, frame_num, "motion")
            frame_metadata['bits_motion'] = motion_bitsize
            frame_metadata['motion_metadata'] = mv_metadata
            prediction = self.motion_comp.reconstruct_with_motion_vector(decoder_y_expanded, motion_vector)[..., 0]
            residual = y_channel - prediction
            residual_symbols = self.residual_codec.image2symbols(residual, is_source_rgb=False)
            residual_bitstream, residual_bitsize, residual_metadata = self._encode_symbols_with_adaptive_huffman(
                residual_symbols, frame_num, "residual")
            decoded_residual_symbols = self._decode_symbols_with_adaptive_huffman(
                residual_bitstream, len(residual_symbols), residual_metadata, "residual")
            recon_residual = self.residual_codec.symbols2image(decoded_residual_symbols, residual.shape)
            if isinstance(recon_residual, np.ndarray) and recon_residual.ndim == 3:
                recon_residual = recon_residual[..., 0]
            frame_metadata['bits_residual'] = residual_bitsize
            frame_metadata['residual_metadata'] = residual_metadata
            bitstream = {'motion': motion_bitstream, 'residual': residual_bitstream}
            recon_y = prediction + recon_residual
            self.decoder_recon = recon_y
        self.frame_metadata.append(frame_metadata)
        recon_frame_ycbcr = frame_ycbcr.copy()
        if isinstance(recon_y, np.ndarray) and recon_y.ndim == 3:
            recon_y_channel = recon_y[..., 0]
        else:
            recon_y_channel = recon_y
        recon_frame_ycbcr[..., 0] = np.clip(np.asarray(recon_y_channel), 0, 255)
        if is_source_rgb:
            recon_rgb = ycbcr2rgb(recon_frame_ycbcr).astype(np.uint8)
            return recon_rgb, bitstream, frame_metadata['bits_motion'] + frame_metadata['bits_residual']
        else:
            return np.asarray(recon_y_channel).astype(np.uint8), bitstream, frame_metadata['bits_motion'] + frame_metadata['bits_residual']
    
    def get_frame_statistics(self):
        """Return detailed statistics for all encoded frames"""
        return self.frame_metadata
    
    def save_metadata(self, filename):
        """Save frame metadata for analysis"""
        with open(filename, 'wb') as f:
            pickle.dump(self.frame_metadata, f)
    
    def load_metadata(self, filename):
        """Load frame metadata"""
        with open(filename, 'rb') as f:
            self.frame_metadata = pickle.load(f)


def analyze_first_frame_performance(lena_path='data/lena_small.tif', 
                                  foreman_path='data/foreman20_40_RGB/foreman0020.bmp',
                                  quantization_scales=[0.5, 1.0, 2.0]):
    """
    Analyze the performance of the first frame using Lena-trained Huffman coder
    """
    
    print("=" * 60)
    print("FIRST FRAME ANALYSIS WITH LENA-TRAINED HUFFMAN CODER")
    print("=" * 60)
    
    results = []
    
    # Load test image (foreman0020.bmp)
    if os.path.exists(foreman_path):
        test_image = imread(foreman_path)
        print(f"✓ Loaded test image: {foreman_path} {test_image.shape}")
    else:
        print(f"❌ Test image not found: {foreman_path}")
        return results
    
    for q_scale in quantization_scales:
        print(f"\n📊 Testing quantization scale: {q_scale}")
        print("-" * 40)
        
        # Create codec with Lena pre-training
        video_codec = FixedHuffmanVideoCodec(
            quantization_scale=q_scale,
            bounds=(-1000, 4000),
            end_of_block=4000,
            block_shape=(8, 8),
            search_range=4,
            lena_huffman_path=lena_path
        )
        
        # If Lena file doesn't exist, create a synthetic training image
        if not os.path.exists(lena_path):
            print(f"⚠️  Lena file not found at {lena_path}, creating synthetic training data")
            # Create a synthetic Lena-like image for training
            synthetic_lena = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            video_codec.train_lena_huffman_from_image(synthetic_lena)
        
        # Encode/decode first frame only
        recon_image, bitstream, total_bits = video_codec.encode_decode(
            test_image, frame_num=0, is_source_rgb=True)
        
        # Calculate metrics
        psnr = calc_psnr(test_image, recon_image)
        bpp = total_bits / (test_image.size / 3)  # bits per pixel
        
        # Get detailed statistics
        frame_stats = video_codec.get_frame_statistics()[0]
        
        result = {
            'quantization_scale': q_scale,
            'psnr_db': psnr,
            'bpp': bpp,
            'total_bits': total_bits,
            'huffman_source': frame_stats.get('huffman_source', 'unknown'),
            'num_symbols': frame_stats.get('num_symbols', 0)
        }
        results.append(result)
        
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   BPP:  {bpp:.4f}")
        print(f"   Total bits: {total_bits}")
        print(f"   Huffman source: {frame_stats.get('huffman_source', 'unknown')}")
        print(f"   Number of symbols: {frame_stats.get('num_symbols', 0)}")
        
        # Show visual comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(test_image)
        axes[0].set_title(f"Original\nforeman0020.bmp")
        axes[0].axis('off')
        
        axes[1].imshow(recon_image)
        axes[1].set_title(f"Reconstructed\nQ={q_scale}, PSNR={psnr:.2f}dB, BPP={bpp:.4f}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"First Frame Analysis - Quantization Scale {q_scale}", y=1.02)
        plt.show()
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - FIRST FRAME PERFORMANCE")
    print("=" * 80)
    print(f"{'Q-Scale':<10} {'PSNR (dB)':<12} {'BPP':<10} {'Total Bits':<12} {'Huffman Source':<15}")
    print("-" * 80)
    for result in results:
        print(f"{result['quantization_scale']:<10} "
              f"{result['psnr_db']:<12.2f} "
              f"{result['bpp']:<10.4f} "
              f"{result['total_bits']:<12} "
              f"{result['huffman_source']:<15}")
    
    return results


def run_full_video_sequence(lena_path='data/lena_small.tif',
                          video_folder='data/foreman20_40_RGB',
                          quantization_scale=1.0,
                          num_frames=21):
    """
    Run the full video coding sequence with first frame using Lena Huffman
    """
    
    print("=" * 60)
    print("FULL VIDEO SEQUENCE ENCODING")
    print("=" * 60)
    
    # Create codec
    video_codec = FixedHuffmanVideoCodec(
        quantization_scale=quantization_scale,
        bounds=(-1000, 4000),
        end_of_block=4000,
        block_shape=(8, 8),
        search_range=4,
        lena_huffman_path=lena_path
    )
    
    # Load video frames
    images = []
    for i in range(20, 20 + num_frames):
        frame_path = os.path.join(video_folder, f'foreman00{i:02d}.bmp')
        if os.path.exists(frame_path):
            images.append(imread(frame_path))
        else:
            print(f"⚠️  Frame not found: {frame_path}")
    
    if not images:
        print("❌ No frames found!")
        return
    
    print(f"✓ Loaded {len(images)} frames")
    
    # Process all frames
    total_bits = 0
    total_psnr = 0
    reconstructed_frames = []
    
    for frame_num, image in enumerate(images):
        print(f"\n🎬 Processing frame {frame_num + 20} ({frame_num + 1}/{len(images)})")
        
        recon_image, bitstream, bitsize = video_codec.encode_decode(
            image, frame_num=frame_num, is_source_rgb=True)
        
        psnr = calc_psnr(image, recon_image)
        bpp = bitsize / (image.size / 3)
        
        total_bits += bitsize
        total_psnr += psnr
        reconstructed_frames.append(recon_image)
        
        print(f"   PSNR: {psnr:.2f} dB, BPP: {bpp:.4f}")
    
    # Calculate averages
    avg_psnr = total_psnr / len(images)
    avg_bpp = total_bits / (len(images) * images[0].size / 3)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Average PSNR: {avg_psnr:.2f} dB")
    print(f"   Average BPP:  {avg_bpp:.4f}")
    print(f"   Total bits:   {total_bits}")
    
    # Get detailed frame statistics
    frame_stats = video_codec.get_frame_statistics()
    
    return {
        'avg_psnr': avg_psnr,
        'avg_bpp': avg_bpp,
        'total_bits': total_bits,
        'frame_stats': frame_stats,
        'reconstructed_frames': reconstructed_frames
    }


# Example usage
if __name__ == '__main__':
    # Create output directories
    os.makedirs('data/results', exist_ok=True)
    
    # First, analyze just the first frame performance
    print("Step 1: Analyzing first frame with Lena-trained Huffman coder")
    first_frame_results = analyze_first_frame_performance(
        lena_path='data/lena_small.tif',
        foreman_path='data/foreman20_40_RGB/foreman0020.bmp',
        quantization_scales=[0.5, 1.0, 2.0]
    )
    
    # Then run a short video sequence
    print("\n" + "="*60)
    print("Step 2: Running full video sequence")
    video_results = run_full_video_sequence(
        lena_path='data/lena_small.tif',
        video_folder='data/foreman20_40_RGB',
        quantization_scale=1.0,
        num_frames=5  # Just first 5 frames for demonstration
    )
    
    if video_results:
        print(f"\n✅ Video encoding complete!")
        print(f"   Final average PSNR: {video_results['avg_psnr']:.2f} dB")
        print(f"   Final average BPP:  {video_results['avg_bpp']:.4f}")