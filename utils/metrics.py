# File: utils/metrics.py
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def print_metrics(name, psnr_list, ssim_list, idx):
    print(f"[{name}] Peak Iter {idx}: PSNR = {psnr_list[idx]:.2f} dB | SSIM = {ssim_list[idx]:.4f}")

def save_snapshots(snapshots, method, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for k, img in snapshots.items():
        img_out = (img * 255).clip(0, 255).astype('uint8')
        path = os.path.join(output_dir, f"{method}_iter_{k}.png")
        plt.imsave(path, img_out)

def plot_grid_comparison(van_snaps, skoop_snaps, eqv_snaps, v_peak, s_peak, e_peak, output_dir):
    keys = sorted(list(set(van_snaps.keys()) | set(skoop_snaps.keys()) | set(eqv_snaps.keys())))
    rows, cols = 3, len(keys)
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 9))
    methods = [('Vanilla', van_snaps), ('SKOOP', skoop_snaps), ('Equiv', eqv_snaps)]
    for row in range(rows):
        _, snap_dict = methods[row]
        for col, k in enumerate(keys):
            ax = axs[row, col]
            img = snap_dict.get(k)
            if img is not None:
                ax.imshow(img)
                ax.set_title(f"iter={k}", fontsize=10)
            ax.axis('off')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "comparison_grid.png"))
    plt.close()

def plot_curves(van_psnr, skoop_psnr, eqv_psnr, van_norm, skoop_norm, eqv_norm, output_dir):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(van_psnr, label='Vanilla-RED', color='#D55E00')
    axs[0].plot(skoop_psnr, label='SKOOP-RED', color='#0072B2')
    axs[0].plot(eqv_psnr, label='Equiv-RED', color='#009E73')
    axs[0].set_ylabel("PSNR (dB)")
    axs[0].set_xlabel("Iteration")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(van_norm[1:], label='Vanilla-RED', color='#D55E00')
    axs[1].plot(skoop_norm[1:], label='SKOOP-RED', color='#0072B2')
    axs[1].plot(eqv_norm[1:], label='Equiv-RED', color='#009E73')
    axs[1].set_ylabel(r"$\lVert x_t - x_{t-1} \rVert$")
    axs[1].set_xlabel("Iteration")
    axs[1].set_yscale("log")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "curves.png"))
    plt.close()
