# File: utils/data.py
import torch
import numpy as np
import cv2
import h5py

def load_image_and_kernel(img_path, kernel_path, device, noise_std):
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_np = img.copy()
    x_true = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Load kernel
    with h5py.File(kernel_path, "r") as f:
        kref = f["kernels"][2, 0]
        kernel_np = np.array(f[kref]).astype(np.float32)
        kernel_np /= kernel_np.sum()
    kernel = torch.tensor(kernel_np).unsqueeze(0).unsqueeze(0).to(device)

    # Compute FFT of kernel
    H, W = img.shape[:2]
    def psf2otf(psf, shape):
        otf = torch.zeros(shape, device=psf.device)
        otf[:, :, :psf.shape[-2], :psf.shape[-1]] = psf
        for dim, size in enumerate(psf.shape[-2:]):
            otf = torch.roll(otf, shifts=-size // 2, dims=dim + 2)
        return torch.fft.fft2(otf)

    otf = psf2otf(kernel, (1, 1, H, W))
    otf_conj = torch.conj(otf)

    # Blur and add noise
    def blur_fft(x, otf):
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * otf))

    y = blur_fft(x_true, otf)
    y_noisy = y + noise_std * torch.randn_like(y)
    y_noisy = torch.clamp(y_noisy, 0, 1)

    return x_true, y_noisy, kernel, otf, otf_conj, img_np
