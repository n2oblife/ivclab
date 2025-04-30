import numpy as np
import matplotlib.pyplot as plt

def compute_psnr(mse: float, bit_depth: int) -> float:
    MAX = 2**bit_depth - 1
    return 10 * np.log10((MAX ** 2) / mse)

# --- Parameters ---
mse_value = 100.0
bit_depths = list(range(1, 65))  # Now from 1 to 64 bits
highlight_bits = [1, 4, 8, 16, 32, 64]

# --- Compute PSNRs ---
psnr_values = [compute_psnr(mse_value, b) for b in bit_depths]
highlight_psnrs = [compute_psnr(mse_value, b) for b in highlight_bits]

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(bit_depths, psnr_values, label="PSNR", color='teal')
plt.scatter(highlight_bits, highlight_psnrs, color='red', zorder=5, label="Common Bit Depths")

for b, p in zip(highlight_bits, highlight_psnrs):
    plt.text(b, p + 0.5, f"{b}-bit", ha='center', fontsize=8, color='red')

plt.xlabel("Bit Depth")
plt.ylabel("PSNR (dB)")
plt.title(f"PSNR vs Bit Depth (MSE = {mse_value})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
