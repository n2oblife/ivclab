import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt
import pickle


class AdaptiveVideoCodec:
    def __init__(self, 
                 quantization_scale=1.0,
                 bounds=(-1000, 4000),
                 end_of_block=4000,
                 block_shape=(8, 8),
                 search_range=4,
                 adaptive_huffman=True):
        
        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range
        self.adaptive_huffman = adaptive_huffman

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
        
        # Store codebooks and metadata for adaptive coding
        self.frame_metadata = []
        
    def _create_fresh_huffman_coder(self, lower_bound=None):
        """Create a new Huffman coder instance"""
        if lower_bound is not None:
            return HuffmanCoder(lower_bound=lower_bound)
        else:
            return HuffmanCoder()
    
    def _encode_symbols_with_adaptive_huffman(self, symbols, frame_num, symbol_type="residual"):
        """
        Encode symbols with adaptive Huffman coding.
        Returns encoded bitstream, bitsize, and codebook metadata.
        """
        if not self.adaptive_huffman:
            # Use pre-trained Huffman coder
            if symbol_type == "residual":
                coder = self.residual_codec.huffman
            elif symbol_type == "intra":
                coder = self.intra_codec.huffman
            else:  # motion
                coder = self.motion_huffman
            
            bitstream, bitsize = coder.encode(symbols)
            return bitstream, bitsize, None
        
        # Convert symbols to numpy array for easier processing
        symbols = np.array(symbols)
        
        # Adaptive Huffman coding
        # Compute histogram of current symbols
        if symbol_type == "motion":
            # Motion vectors have specific range
            mv_range = np.arange((2 * self.motion_comp.search_range + 1) ** 2 + 1)
            hist = stats_marg(symbols, pixel_range=mv_range)
            original_symbols = mv_range[:-1]  # Exclude the last range boundary
        else:
            # For residual/intra symbols, adapt bounds to actual data range
            # Get the actual range of symbols in this frame
            min_symbol = int(np.min(symbols))
            max_symbol = int(np.max(symbols))
            
            # Expand bounds if necessary to include all symbols
            actual_min = min(self.bounds[0], min_symbol)
            actual_max = max(self.bounds[1], max_symbol)
            
            print(f"Frame {frame_num} {symbol_type}: Symbol range [{min_symbol}, {max_symbol}], "
                  f"Using bounds [{actual_min}, {actual_max}]")
            
            # Create symbol range that covers all actual symbols
            symbol_range = np.arange(actual_min, actual_max + 2)  # +1 for end_of_block
            hist = stats_marg(symbols, pixel_range=symbol_range)
            original_symbols = symbol_range[:-1]
        
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
            original_symbol = original_symbols[original_idx]
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
            coder = self._create_fresh_huffman_coder(lower_bound=0 if symbol_type == "motion" else None)
            coder.train(hist_nonzero)
            
            # Rebuild compact symbols list
            compact_symbols = [original_to_compact[symbol] for symbol in symbols]
        
        # Convert to NumPy array for HuffmanCoder compatibility
        compact_symbols = np.array(compact_symbols)
        
        # Create and train new Huffman coder
        if symbol_type == "motion":
            coder = self._create_fresh_huffman_coder(lower_bound=0)
        else:
            coder = self._create_fresh_huffman_coder()
            
        coder.train(hist_nonzero)
        
        # Encode with the trained coder
        bitstream, bitsize = coder.encode(compact_symbols)
        
        # Create metadata for decoder
        if symbol_type == "motion":
            actual_bounds = [0, (2 * self.motion_comp.search_range + 1) ** 2]
        else:
            actual_bounds = [actual_min, actual_max]
            
        metadata = {
            'symbol_mapping': original_to_compact,
            'nonzero_indices': nonzero_indices.tolist(),
            'histogram': hist_nonzero.tolist(),
            'symbol_type': symbol_type,
            'actual_bounds': actual_bounds  # Store the actual bounds used
        }
        
        return bitstream, bitsize, metadata
    
    def _decode_symbols_with_adaptive_huffman(self, bitstream, num_symbols, metadata, symbol_type="residual"):
        """
        Decode symbols using stored codebook metadata.
        """
        if not self.adaptive_huffman or metadata is None:
            # Use pre-trained Huffman coder
            if symbol_type == "residual":
                coder = self.residual_codec.huffman
            elif symbol_type == "intra":
                coder = self.intra_codec.huffman
            else:  # motion
                coder = self.motion_huffman
            
            return coder.decode(bitstream, num_symbols)
        
        # Reconstruct the Huffman coder from metadata
        hist_nonzero = np.array(metadata['histogram'])
        
        if symbol_type == "motion":
            coder = self._create_fresh_huffman_coder(lower_bound=0)
        else:
            coder = self._create_fresh_huffman_coder()
            
        coder.train(hist_nonzero)
        
        # Decode to compact symbols
        compact_symbols = coder.decode(bitstream, num_symbols)
        
        # Convert back to original symbols
        compact_to_original = {v: k for k, v in metadata['symbol_mapping'].items()}
        original_symbols = [compact_to_original[compact_sym] for compact_sym in compact_symbols]
        
        return original_symbols
    
    def encode_decode(self, frame, frame_num=0, is_source_rgb=False):
        """
        Encode and decode a frame with adaptive Huffman coding.
        """
        # Convert RGB to YCbCr
        frame_ycbcr = rgb2ycbcr(frame.astype(np.float32))
        y_channel = frame_ycbcr[..., 0]
        
        frame_metadata = {}
        
        if frame_num == 0:
            # Intra-frame coding (I-frame)
            if self.adaptive_huffman:
                # Get symbols from intra codec
                symbols = self.intra_codec.image2symbols(y_channel, is_source_rgb=False)
                
                # Encode with adaptive Huffman
                bitstream, residual_bitsize, metadata = self._encode_symbols_with_adaptive_huffman(
                    symbols, frame_num, "intra")
                
                # Decode symbols
                decoded_symbols = self._decode_symbols_with_adaptive_huffman(
                    bitstream, len(symbols), metadata, "intra")
                
                # Reconstruct image from symbols
                recon_y = self.intra_codec.symbols2image(decoded_symbols, y_channel.shape)
                
                # Ensure recon_y is 2D (grayscale)
                if recon_y.ndim == 3:
                    recon_y = recon_y[..., 0]
                
                frame_metadata['intra'] = metadata
            else:
                recon_y, bitstream, residual_bitsize = self.intra_codec.encode_decode(y_channel, is_source_rgb=False)
                
                # Ensure recon_y is 2D (grayscale)
                if recon_y.ndim == 3:
                    recon_y = recon_y[..., 0]
            
            motion_bitsize = 0
            self.decoder_recon = recon_y
            
        else:
            # Inter-frame coding (P-frame)
            if self.decoder_recon.ndim == 3:
                decoder_y = self.decoder_recon[..., 0]
            else:
                decoder_y = self.decoder_recon

            # Motion estimation
            motion_vector = self.motion_comp.compute_motion_vector(decoder_y, y_channel)
            flat_mv = motion_vector.flatten()
            
            # Encode motion vectors with adaptive Huffman
            motion_bitstream, motion_bitsize, mv_metadata = self._encode_symbols_with_adaptive_huffman(
                flat_mv, frame_num, "motion")
            
            frame_metadata['motion'] = mv_metadata

            # Motion compensated prediction
            decoder_y_expanded = decoder_y[..., np.newaxis]
            prediction = self.motion_comp.reconstruct_with_motion_vector(decoder_y_expanded, motion_vector)[..., 0]

            # Compute residual
            residual = y_channel - prediction

            # Encode residual with adaptive Huffman
            if self.adaptive_huffman:
                residual_symbols = self.residual_codec.image2symbols(residual, is_source_rgb=False)
                
                residual_bitstream, residual_bitsize, residual_metadata = self._encode_symbols_with_adaptive_huffman(
                    residual_symbols, frame_num, "residual")
                
                # Decode residual
                decoded_residual_symbols = self._decode_symbols_with_adaptive_huffman(
                    residual_bitstream, len(residual_symbols), residual_metadata, "residual")
                
                recon_residual = self.residual_codec.symbols2image(decoded_residual_symbols, residual.shape)
                
                # Ensure recon_residual is 2D (grayscale) to match prediction
                if recon_residual.ndim == 3:
                    recon_residual = recon_residual[..., 0]
                
                frame_metadata['residual'] = residual_metadata
                
                # For inter-frames, combine motion and residual bitstreams
                bitstream = {'motion': motion_bitstream, 'residual': residual_bitstream}
            else:
                recon_residual, residual_bitstream, residual_bitsize = self.residual_codec.encode_decode(residual, is_source_rgb=False)
                
                # Ensure recon_residual is 2D (grayscale) to match prediction
                if recon_residual.ndim == 3:
                    recon_residual = recon_residual[..., 0]
                
                # For inter-frames, combine motion and residual bitstreams
                bitstream = {'motion': motion_bitstream, 'residual': residual_bitstream}

            # Reconstruct Y channel
            recon_y = prediction + recon_residual
            self.decoder_recon = recon_y

        # Store frame metadata
        self.frame_metadata.append(frame_metadata)

        # Merge reconstructed Y channel with original CbCr
        recon_frame_ycbcr = frame_ycbcr.copy()
        if recon_y.ndim == 3:
            recon_y_channel = recon_y[..., 0]
        else:
            recon_y_channel = recon_y

        recon_frame_ycbcr[..., 0] = np.clip(recon_y_channel, 0, 255)

        # Convert YCbCr back to RGB
        recon_rgb = ycbcr2rgb(recon_frame_ycbcr).astype(np.uint8)

        bitsize = residual_bitsize + motion_bitsize
        return recon_rgb, bitstream, bitsize
    
    def save_metadata(self, filename):
        """Save frame metadata for decoder"""
        with open(filename, 'wb') as f:
            pickle.dump(self.frame_metadata, f)
    
    def load_metadata(self, filename):
        """Load frame metadata for decoder"""
        with open(filename, 'rb') as f:
            self.frame_metadata = pickle.load(f)


# Example usage with adaptive Huffman coding
if __name__ == '__main__':
    import os
    import pickle
    import cv2
    import imageio
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    # Create output folders if they don't exist
    os.makedirs('data/results/metadata', exist_ok=True)
    os.makedirs('data/results/frames', exist_ok=True)
    os.makedirs('data/results/videos', exist_ok=True)

    # Load test images
    images = []
    for i in range(20, 41):
        images.append(imread(f'data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = []
    all_psnrs = []

    for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
        video_codec = AdaptiveVideoCodec(
            quantization_scale=q_scale, 
            bounds=(-1000, 4000), 
            end_of_block=4000, 
            block_shape=(8, 8), 
            search_range=4,
            adaptive_huffman=True
        )
        
        bpps = []
        psnrs = []
        reconstructed_frames = []

        for frame_num, image in enumerate(images):
            reconstructed_image, bitstream, bitsize = video_codec.encode_decode(
                image, frame_num=frame_num, is_source_rgb=True)

            # Save reconstructed frame
            frame_save_path = f"data/results/frames/q{q_scale}_frame{frame_num:02d}.png"
            imageio.imwrite(frame_save_path, reconstructed_image)
            reconstructed_frames.append(reconstructed_image)

            # Show side-by-side comparison for first frame of each run
            if frame_num == 0:
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                axs[0].imshow(image)
                axs[0].set_title("Original")
                axs[0].axis("off")
                axs[1].imshow(reconstructed_image)
                axs[1].set_title("Reconstructed")
                axs[1].axis("off")
                plt.suptitle(f"Q-scale: {q_scale} | Frame 0")
                plt.tight_layout()
                plt.show()

            # Quality metrics
            bpp = bitsize / (image.size / 3)
            psnr = calc_psnr(image, reconstructed_image)
            print(f"Frame:{frame_num} PSNR: {psnr:.2f} dB bpp: {bpp:.2f}")
            bpps.append(bpp)
            psnrs.append(psnr)

        all_bpps.append(np.mean(bpps))
        all_psnrs.append(np.mean(psnrs))
        print(f"Q-Scale {q_scale}, Average PSNR: {np.mean(psnrs):.2f} dB Average bpp: {np.mean(bpps):.2f}")
        print('-' * 12)

        # Save codec metadata only if not already saved
        metadata_path = f'data/results/metadata/codec_metadata_q{q_scale}.pkl'
        if not os.path.exists(metadata_path):
            video_codec.save_metadata(metadata_path)
            print(f"Metadata saved to {metadata_path}")
        else:
            print(f"Metadata already exists: {metadata_path}, skipping...")

        # Save reconstructed video as .mp4
        video_path = f'data/results/videos/reconstructed_q{q_scale}.mp4'
        height, width, _ = reconstructed_frames[0].shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
        for frame in reconstructed_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"🎞️ Video saved to {video_path}")