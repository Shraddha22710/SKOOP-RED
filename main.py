# File: main.py
# Main runner script to compare SKOOP-RED, Vanilla-RED, and Equivariant-RED
# Assumes modular structure: methods/, utils/, pretrained models installed via deepinv

import torch
import time
import matplotlib.pyplot as plt
from utils.data import load_image_and_kernel
from utils.metrics import print_metrics, save_snapshots, plot_grid_comparison, plot_curves
from methods.skoop_red import skoop_red_quadratic
from methods.vanilla_red import vanilla_red
from methods.equivariant_red import equivariant_red
from deepinv.models import DRUNet

# ---------------- CONFIG ----------------
img_path = '/path/to/image.png'
kernel_path = '/path/to/Levin09.mat'
output_dir = './outputs'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

iters = 2500
lam = 0.2
gamma_init = 2.0
gamma_min = 0.02
noise_std = 1 / 255.0
koopman_window = 50
koopman_every = 20

# ---------------- LOAD ----------------
x_true, y_noisy, kernel, otf, otf_conj, img_np = load_image_and_kernel(img_path, kernel_path, device, noise_std)

# ---------------- DENOISER ----------------
denoiser = DRUNet(pretrained='download', in_channels=3, out_channels=3).to(device).eval()
sigma_tensor = torch.tensor([noise_std], device=device)

# ---------------- RUN ----------------
print("Running SKOOP-RED...")
start = time.time()
skoop = skoop_red_quadratic(
    y_noisy, denoiser, lam, gamma_init, gamma_min,
    koopman_window, koopman_every, iters, otf, otf_conj,
    koopman_lookahead=1, beta=4, img_np=img_np
)
print("SKOOP-RED done in %.2fs" % (time.time() - start))

print("Running Vanilla-RED...")
start = time.time()
van_psnr, van_ssim, van_norm, van_img, van_snaps, van_peak = vanilla_red(
    y_noisy, denoiser, lam, gamma_init, iters, otf, otf_conj, img_np
)
print("Vanilla-RED done in %.2fs" % (time.time() - start))

print("Running Equivariant-RED...")
start = time.time()
eqv_psnr, eqv_ssim, eqv_norm, eqv_img, eqv_snaps, eqv_peak = equivariant_red(
    y_noisy, denoiser, lam, gamma_init, iters, otf, otf_conj, img_np
)
print("Equivariant-RED done in %.2fs" % (time.time() - start))

# ---------------- RESULTS ----------------
print_metrics("Vanilla", van_psnr, van_ssim, van_peak)
print_metrics("SKOOP", skoop['psnr'], skoop['ssim'], skoop['best_idx'])
print_metrics("Equiv", eqv_psnr, eqv_ssim, eqv_peak)

save_snapshots(van_snaps, 'vanilla', output_dir)
save_snapshots(skoop['snapshots'], 'skoop', output_dir)
save_snapshots(eqv_snaps, 'equiv', output_dir)

plot_grid_comparison(van_snaps, skoop['snapshots'], eqv_snaps, van_peak, skoop['best_idx'], eqv_peak, output_dir)
plot_curves(van_psnr, skoop['psnr'], eqv_psnr, van_norm, skoop['norm'], eqv_norm, output_dir)

print("Final PSNRs: Vanilla %.2f | SKOOP %.2f | Equiv %.2f" % (van_psnr[-1], skoop['psnr'][-1], eqv_psnr[-1]))
