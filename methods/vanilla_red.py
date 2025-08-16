import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

@torch.no_grad()
def vanilla_red(y_noisy, denoiser, blur, blurT, lam, gamma, max_iters, img_np):
    x = y_noisy.clone()
    x_prev = x.clone()
    psnr_list, ssim_list, norm_list, snapshot_dict = [], [], [], {}
    best_psnr, best_idx = -np.inf, -1

    for k in range(max_iters):
        Ax = blur(x)
        grad_f = blurT(Ax - y_noisy)
        Dx = denoiser(x, sigma=torch.tensor([1/255.0], device=x.device))
        x_new = x - gamma * (grad_f + lam * (x - Dx))
        x_new = torch.clamp(x_new, 0, 1)

        x_np = x_new[0].permute(1, 2, 0).cpu().numpy()
        x_prev_np = x_prev[0].permute(1, 2, 0).cpu().numpy()
        psnr_val = psnr(img_np, x_np, data_range=1.0)
        ssim_val = ssim(img_np, x_np, channel_axis=2, data_range=1.0)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        norm_list.append(np.linalg.norm(x_np - x_prev_np))

        if k in [9, 99, 499, max_iters - 1]:
            snapshot_dict[k] = np.clip(x_np, 0, 1)
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_idx = k

        x_prev = x.clone()
        x = x_new.clone()

    if best_idx not in snapshot_dict:
        snapshot_dict[best_idx] = np.clip(x_np, 0, 1)

    return psnr_list, ssim_list, norm_list, x_np.copy(), snapshot_dict, best_idx
