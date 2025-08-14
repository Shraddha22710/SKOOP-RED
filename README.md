# Stabilizing-RED-using-the-Koopman-Operator
Official code for the IEEE SPL paper "Stabilizing RED using the Koopman Operator (SKOOP-RED)." Includes implementations, demos, and scripts to reproduce results and plots.

## 📌 Overview

Regularization by Denoising (RED) is a powerful framework for solving inverse problems using image denoisers. However, stability issues can arise due to mismatch between the RED gradient and fixed-point iterations.  
**SKOOP-RED** introduces a data-driven stabilization mechanism using a learned Koopman operator to improve convergence without modifying the underlying denoiser.

---

## 🔍 Features

- Plug-and-play compatible with deep denoisers (e.g., DnCNN, DRUNet)
- Learned Koopman operator predicts fixed-point update dynamics
- Drop-in replacement for RED iterations
- Stable convergence with minimal overhead
- Reproducible results for multiple inverse tasks (deblurring, inpainting, super-resolution)

---

## 🛠 Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- NumPy, SciPy, Matplotlib
- (Optional) OpenCV for image loading

Use `requirements.txt` to install all dependencies:
```bash
pip install -r requirements.txt
