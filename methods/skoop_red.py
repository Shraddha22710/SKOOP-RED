import numpy as np
import torch
import scipy.fftpack
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from .utils import extract_koopman_features, koopman_dmd

@torch.no_grad()
def skoop_red_quadratic(
    y_noisy, denoiser, blur, blurT, lam, gamma_init, gamma_min, koopman_window,
    koopman_every, max_iters, koopman_lookahead=3, koopman_pred_threshold=1.05,
    beta=4, img_np=None
):
    gamma = gamma_init
    koopman_history = []
    gamma_list, radius_list, psnr_list, ssim_list, norm_list = [], [], [], [], []
    x = y_noisy.clone()
    x_prev = x.clone()
    snapshot_dict = {}
    best_psnr = -np.inf
    best_idx = -1

    for k in range(max_iters):
        Ax = blur(x)
        grad_f = blurT(Ax - y_noisy)
        Dx = denoiser(x, sigma=torch.tensor([1/255.0], device=x.device))
        x_new = x - gamma * (grad_f + lam * (x - Dx))
        x_new = torch.clamp(x_new, 0, 1)

        koopman_feature = extract_koopman_features(x_new, Dx)
        koopman_history.append(koopman_feature)
        if len(koopman_history) > koopman_window:
            koopman_history.pop(0)

        radius = 0
        eta = 1.0
        instability = False

        if k > koopman_window and k % koopman_every == 0:
            Xn = np.stack(koopman_history[:-1], axis=1)
            Yn = np.stack(koopman_history[1:], axis=1)
            K = koopman_dmd(Xn, Yn)
            eigvals, eigvecs = np.linalg.eig(K)
            inv_eigvecs = np.linalg.inv(eigvecs)
            radius = np.max(np.abs(eigvals))
            feature_last = koopman_history[-1]
            for h in range(1, koopman_lookahead + 1):
                Lambda_h = np.diag(eigvals**h)
                K_h = eigvecs @ Lambda_h @ inv_eigvecs
                y_pred = K_h @ feature_last
                pred_radius = np.max(np.abs(eigvals)**h)
                if np.linalg.norm(y_pred) > koopman_pred_threshold or pred_radius > koopman_pred_threshold:
                    instability = True
                    break
            if instability:
                eta = float(np.clip(1 - beta * (radius - 1) ** 2, 0.2, 1.0))
                gamma = max(gamma * eta, gamma_min)
            else:
                gamma = max(gamma * 0.995, gamma_min)
        else:
            gamma = max(gamma * 0.995, gamma_min)

        radius_list.append(radius)
        gamma_list.append(gamma)
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

    return {
        "psnr": psnr_list,
        "ssim": ssim_list,
        "gamma": gamma_list,
        "radius": radius_list,
        "norm": norm_list,
        "snapshots": snapshot_dict,
        "best_idx": best_idx
    }

