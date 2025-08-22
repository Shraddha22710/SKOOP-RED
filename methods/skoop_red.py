
import numpy as np
import scipy.fftpack

def extract_koopman_features(x):
    """
    Feature construction for SKOOP-RED:
    For each channel:
      - Global mean and std (2)
      - 4x4 grid means (16)
      - 2x2 DCT coefficients (4)
    Concatenate for all channels: 22x3=66 features.
    Input: x [1, 3, H, W] or [3, H, W] torch.Tensor
    Returns: np.array of shape (66,)
    """
    if hasattr(x, 'detach'):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    if x_np.ndim == 4:  # [B, C, H, W]
        x_np = x_np[0]
    C, H, W = x_np.shape
    features = []
    grid_size = 4
    cell_H = H // grid_size
    cell_W = W // grid_size

    for c in range(C):
        ch = x_np[c]
        # (1) Global mean and std
        features.append(ch.mean())
        features.append(ch.std())
        # (2) 4x4 grid means
        for i in range(grid_size):
            for j in range(grid_size):
                patch = ch[
                    i*cell_H:(i+1)*cell_H,
                    j*cell_W:(j+1)*cell_W
                ]
                features.append(patch.mean())
        # (3) 2D DCT, top-left 2x2 block (lowest frequencies)
        dct = scipy.fftpack.dctn(ch, norm='ortho')
        features.extend(dct[:2, :2].flatten())
    return np.array(features)


def koopman_dmd(X, Y, tol=1e-6):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag([1/s if s > tol else 0 for s in S])
    return Y @ Vt.T @ S_inv @ U.T



#SKOOP-RED Main Loop
def skoop_red(
    y_noisy, denoiser, sr_forward, sr_adjoint, lam, gamma_init, gamma_min,
    koopman_window, koopman_every, max_iters,
    koopman_pred_threshold=1.05, beta=4, img_np=None
):
    """
    SKOOP-RED:
    - Uses extract_koopman_features(x) for 66-dim vector per iterate.
    - Shrinks gamma only if spectral radius >= koopman_pred_threshold.
    """
    gamma = gamma_init
    koopman_history = []
    gamma_list, radius_list, psnr_list, ssim_list, norm_list = [], [], [], [], []
    x = upsample_bicubic_aligned(y_noisy, scale_factor=2)
    x_prev = x.clone()
    snapshot_dict = {}
    best_psnr = -np.inf
    best_idx = -1
    for k in range(max_iters):
        Ax = sr_forward(x)
        grad_f = sr_adjoint(Ax - y_noisy, out_shape=x.shape)
        with torch.no_grad():
            Dx = denoiser(x, sigma=sigma_tensor)
        x_new = x - gamma * (grad_f + lam * (x - Dx))
        x_new = torch.clamp(x_new, 0, 1)

        # --- Koopman feature extraction  ---
        koopman_feature = extract_koopman_features(x_new)
        koopman_history.append(koopman_feature)
        if len(koopman_history) > koopman_window:
            koopman_history.pop(0)

        # --- Koopman update only if enough history and at interval ---
        radius = 0
        if k >= koopman_window and k % koopman_every == 0:
            Xn = np.stack(koopman_history[:-1], axis=1)  # shape (66, w-1)
            Yn = np.stack(koopman_history[1:], axis=1)   # shape (66, w-1)
            K = koopman_dmd(Xn, Yn)
            eigvals = np.linalg.eigvals(K)
            radius = np.max(np.abs(eigvals))
            if radius >= koopman_pred_threshold:
                eta = float(np.clip(1 - beta * (radius - 1) ** 2, 0.2, 1.0))
                gamma = max(gamma * eta, gamma_min)
            else:
                gamma = max(gamma * 0.995, gamma_min)
        else:
            gamma = max(gamma * 0.995, gamma_min)
        radius_list.append(radius)
        gamma_list.append(gamma)

        # --- Metrics --- #
        x_np = np.clip(x_new[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        x_prev_np = np.clip(x_prev[0][0].permute(1,2,0).detach().cpu().numpy(), 0, 1) \
            if x_prev.ndim == 4 else np.clip(x_prev[0].permute(1,2,0).detach().cpu().numpy(), 0, 1)
        psnr_val = psnr(img_np, x_np, data_range=1.0) if img_np is not None else 0
        ssim_val = ssim(img_np, x_np, channel_axis=2, data_range=1.0) if img_np is not None else 0
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        norm_list.append(np.linalg.norm(x_np - x_prev_np))
        if k in [9, 19, max_iters-1]:  # For illustration
            snapshot_dict[k] = x_np
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_idx = k
        x_prev = x.clone()
        x = x_new.clone()
    if best_idx not in snapshot_dict:
        snapshot_dict[best_idx] = np.clip(x_np, 0, 1)
    return dict(
        psnr=psnr_list, ssim=ssim_list, gamma=gamma_list, radius=radius_list, norm=norm_list,
        snapshots=snapshot_dict, best_idx=best_idx
    )
