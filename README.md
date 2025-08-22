# Stabilizing-RED-using-the-Koopman-Operator: SKOOP-RED
Official code for the IEEE SPL paper
"Stabilizing RED using the Koopman Operator"

Submitted to IEEE Signal Processing Letters (IEEE SPL), July 2025

Includes implementations, demos, and scripts to reproduce results and plots.

## 📌 Overview

Regularization by Denoising (RED) is a powerful framework for solving inverse problems using pretrained denoisers as implicit regularizers for
model-based reconstruction. Although RED gives high-fidelity reconstructions, the use of trained black-box denoisers can result in instability. 
**SKOOP-RED** introduces novel data-driven mechanism for stabilizing RED. This is based on the linear Koopman operator, a classical tool for analyzing nonlinear dynamical systems. Specifically, we use the Koopman operator to capture the local dynamics of the RED iterations. The spectral radius of this operator is used to derive an adaptive step size rule that is modelagnostic, introduces reasonable overhead, and does not require retraining. We present reconstructions using different pretrained denoisers to demonstrate the effectiveness of our stabilization mechanism.

---


## 🛠 Requirements

torch
numpy
opencv-python
matplotlib
scikit-image
h5py
scipy
deepinv (https://deepinv.github.io/deepinv/index.html)
tqdm

#Use `requirements.txt` to install all dependencies:
```python
pip install -r requirements.txt
```

🚀 Getting Started

Clone the repository and install dependencies:
```python
git clone https://github.com/YourUsername/SKOOP-RED.git
cd SKOOP-RED
pip install -r requirements.txt
```

📂 Directory Structure
```python
SKOOP-RED/
├── demo_SR.ipynb           
├── main.py             
├── utils/               
├── methods/
├── Set15C/             
├── requirements.txt
└── README.md
```
🖼 Example: Superresolution Demo
```python
jupyter notebook demo_SR.ipynb
```

Inputs:

#Update img_path in demo_SR.py or the notebook to use your own images.
#Pretrained denoisers (DnCNN, DRUNet, etc) are automatically handled by deepinv (deepinv.github.io/deepinv/index.html).


Outputs:
#Reconstructed images and result plots (PSNR, residuals) are saved in the working directory.

📈 Results:
Vanilla RED is often unstable (PSNR drops, residuals explode).
SKOOP-RED provides robust stabilization, consistent improvement, and high-quality reconstructions across tasks.
#See the paper for full quantitative and visual results.
