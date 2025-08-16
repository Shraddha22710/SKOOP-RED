import numpy as np
import scipy.fftpack
import torch

def extract_koopman_features(x, Dx):
    r_np = (x - Dx)[0].detach().cpu().numpy()
    features = []
    patch_size = 128
    for c in range(r_np.shape[0]):
        for i in range(0, r_np.shape[1], patch_size):
            for j in range(0, r_np.shape[2], patch_size):
                patch = r_np[c, i:i+patch_size, j:j+patch_size]
                features.append(patch.mean())
                features.append(patch.std())
    for c in range(r_np.shape[0]):
        dct = scipy.fftpack.dctn(r_np[c], norm='ortho')
        features.extend(dct[:3, :3].flatten())
    features.append(r_np.mean())
    features.append(r_np.std())
    return np.array(features)

def koopman_dmd(X, Y, tol=1e-6):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag([1/s if s > tol else 0 for s in S])
    return Y @ Vt.T @ S_inv @ U.T

def skoop_red_quadratic(
    y_noisy, denoiser, blur, blurT, lam, gamma_init, gamma_min,
    koopman_window, koopman_every, max_iters,
    koopman_lookahead=3, koopman_pred_threshold=1.05, beta=4,
    sigma_tensor=None, img_np=None
):
    gamma = gamma_init
    koopman_history = []
    gamma_list, radius_list, psnr_list, ssim_list, norm_list = [], [], [], [], []
    x = y_noisy.clone()
    x_prev = x.clone()
    snapshot_dict = {}
    best_psnr = -np.inf
    best_idx = -1

    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    for k in range(max_iters):
        Ax = blur(x)
        grad_f = blurT(Ax - y_noisy)
        with torch.no_grad():
            Dx = denoiser(x, sigma=sigma_tensor)
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
        x_np = np.clip(x_new[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        x_prev_np = np.clip(x_prev[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)

        if img_np is not None:
            psnr_val = psnr(img_np, x_np, data_range=1.0)
            ssim_val = ssim(img_np, x_np, channel_axis=2, data_range=1.0)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
        norm_list.append(np.linalg.norm(x_np - x_prev_np))

        if k in [9, 99, 499, max_iters-1]:
            snapshot_dict[k] = x_np
        if img_np is not None and psnr_val > best_psnr:
            best_psnr = psnr_val
            best_idx = k

        x_prev = x.clone()
        x = x_new.clone()

    if img_np is not None and best_idx not in snapshot_dict:
        snapshot_dict[best_idx] = x_np
    return dict(
        psnr=psnr_list, ssim=ssim_list, gamma=gamma_list, radius=radius_list,
        norm=norm_list, snapshots=snapshot_dict, best_idx=best_idx
    )
