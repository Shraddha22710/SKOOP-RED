import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ---- EquivariANT RED ----
def denoise_equivariant(denoiser, x, sigma):
    k = np.random.choice([0, 1, 2, 3])
    x_rot = torch.rot90(x, k=k, dims=[-2, -1])
    with torch.no_grad():
        z_rot = denoiser(x_rot, sigma=sigma) if not hasattr(denoiser, 'forward_denoise') \
            else denoiser.forward_denoise(x_rot, sigma)
    return torch.rot90(z_rot, k=-k, dims=[-2, -1])

def equivariant_red(y_noisy, denoiser, blur, blurT, lam, gamma, max_iters, img_np):
    x = y_noisy.clone()
    psnr_list, ssim_list, norm_list = [], [], []
    x_prev = x.clone()
    snapshot_dict = {}
    best_psnr = -np.inf
    best_idx = -1
    for k in range(max_iters):
        Ax = blur(x)
        grad_f = blurT(Ax - y_noisy)
        with torch.no_grad():
            Dx = denoise_equivariant(denoiser, x, sigma=sigma_tensor)
        x_new = x - gamma * (grad_f + lam * (x - Dx))
        x_new = torch.clamp(x_new, 0, 1)
        x_np = np.clip(x_new[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        x_prev_np = np.clip(x_prev[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        psnr_val = psnr(img_np, x_np, data_range=1.0)
        ssim_val = ssim(img_np, x_np, channel_axis=2, data_range=1.0)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        norm_list.append(np.linalg.norm(x_np - x_prev_np))
        if k in [9, 99, 499, max_iters-1]:
            snapshot_dict[k] = x_np
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_idx = k
        x_prev = x.clone()
        x = x_new.clone()
    if best_idx not in snapshot_dict:
        snapshot_dict[best_idx] = x_np
    return psnr_list, ssim_list, norm_list, x_np.copy(), snapshot_dict, best_idx

def denoise_equivariant_ensemble(denoiser, x, sigma):
    outputs = []
    for k in range(4):  # 0, 90, 180, 270
        x_rot = torch.rot90(x, k=k, dims=[-2, -1])
        for flip in [False, True]:
            x_tf = torch.flip(x_rot, dims=[-1]) if flip else x_rot
            with torch.no_grad():
                z_tf = denoiser(x_tf, sigma=sigma) if not hasattr(denoiser, 'forward_denoise') \
                    else denoiser.forward_denoise(x_tf, sigma)
            if flip:
                z_tf = torch.flip(z_tf, dims=[-1])
            z_tf = torch.rot90(z_tf, k=-k, dims=[-2, -1])
            outputs.append(z_tf)
    return torch.stack(outputs, dim=0).mean(dim=0)

def ensemble_equivariant_red(y_noisy, denoiser, blur, blurT, lam, gamma, max_iters, img_np):
    x = y_noisy.clone()
    psnr_list, ssim_list, norm_list = [], [], []
    x_prev = x.clone()
    snapshot_dict = {}
    best_psnr = -np.inf
    best_idx = -1
    for k in range(max_iters):
        Ax = blur(x)
        grad_f = blurT(Ax - y_noisy)
        with torch.no_grad():
            Dx = denoise_equivariant_ensemble(denoiser, x, sigma=sigma_tensor)
        x_new = x - gamma * (grad_f + lam * (x - Dx))
        x_new = torch.clamp(x_new, 0, 1)
        x_np = np.clip(x_new[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        x_prev_np = np.clip(x_prev[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        psnr_val = psnr(img_np, x_np, data_range=1.0)
        ssim_val = ssim(img_np, x_np, channel_axis=2, data_range=1.0)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        norm_list.append(np.linalg.norm(x_np - x_prev_np))
        if k in [9, 99, 499, max_iters-1]:
            snapshot_dict[k] = x_np
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_idx = k
        x_prev = x.clone()
        x = x_new.clone()
    if best_idx not in snapshot_dict:
        snapshot_dict[best_idx] = x_np
    return psnr_list, ssim_list, norm_list, x_np.copy(), snapshot_dict, best_idx

