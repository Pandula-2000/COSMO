# COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations
> **Update:** Accepted at *ICLR 2026*.

[![Paper](https://img.shields.io/badge/ArXiv-Paper-B31B1B.svg)]([https://arxiv.org/](https://arxiv.org/html/2505.11640v3))
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Tested-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Implicit neural representations (INRs) offer a continuous alternative to discrete signal representations, compactly encoding complex signals across computer vision tasks. However, odd and even symmetric activation functions suffer from attenuation in their post-activation spectrum. We propose COSMO-INR to addresses this limitation. By modulating activation functions using a complex sinusoidal term, the network achieves complete spectral support and mitigates spectral bias.

## Theoretical Background
Using harmonic distortion analysis and Chebyshev polynomial approximation, we show that the raised cosine activation offers the least decay for larger coefficients, providing optimal spectral bandwidth. 

To prevent the attenuation of symmetric components, we define the COSMO-RC activation as:
$$g(x)=\phi(x)e^{j\zeta x}$$

Where $\phi(x)$ is the raised cosine function with a learnable bandwidth $T$ and frequency shift $\zeta$:
$$\phi(x)=\text{sinc}\left(\frac{x}{T}\right)\frac{\cos\left(\frac{\pi\beta x}{T}\right)}{1-\left(\frac{2\beta x}{T}\right)^2}$$

The outputs at each layer are complex-valued and normalized to the unit circle on the complex plane to ensure a stable learning curve. To accelerate convergence, we integrate a task-specific prior knowledge embedder (e.g., ResNet-34 or ResNet3D-18) combined with a sigmoid regularizer to dynamically adjust the $T$ and $\zeta$ parameters.

![Architecture Pipeline](readme_images/model.png)
*Figure 1: Complete pipeline of the COSMO-RC model architecture featuring the prior embedding sigmoid regularizer.*

## Tasks Tested
[cite_start]COSMO-RC establishes state-of-the-art performance across diverse signal representation and inverse problems[cite: 156]:
* **Image Representation** (Kodak and DIV2K datasets)
* **Image Denoising** (DIV2K with Poisson photon noise)
* **Image Super-Resolution** (2x, 4x, and 6x upsampling on DIV2K)
* **Image Inpainting** (Celtic spiral knots image, 20% pixel sampling)
* **3D Occupancy Volume** (Lucy dataset, $512^3$ voxel grid)
* **Neural Radiance Fields (NeRF)** (Lego dataset novel-view synthesis)

## Key Results
[cite_start]COSMO-RC consistently outperforms state-of-the-art activations (SIREN, WIRE, INCODE, FINER) in accuracy, stability, and high-frequency structural preservation[cite: 577, 921].

| Task | Metric | COSMO-RC | Nearest SOTA | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| Image Representation (Kodak) | PSNR | [cite_start]**41.24 dB** [cite: 593] | [cite_start]35.57 dB (INCODE) [cite: 593] | [cite_start]+5.67 dB [cite: 71] |
| Image Denoising | PSNR | [cite_start]**30.25 dB** [cite: 688] | [cite_start]29.79 dB (INCODE) [cite: 689] | [cite_start]+0.46 dB [cite: 71, 682] |
| Super-Resolution (6x) | PSNR | [cite_start]**27.66 dB** [cite: 733] | [cite_start]27.02 dB (FINER) [cite: 733] | [cite_start]+0.64 dB [cite: 71] |
| NeRF (Lego) | PSNR | [cite_start]**29.50 dB** [cite: 913] | [cite_start]26.05 dB (INCODE) [cite: 913] | [cite_start]+3.45 dB [cite: 857] |

![Qualitative Results](docs/results_placeholder.png)
*Figure 2: Qualitative comparisons for Image Denoising and Super-resolution.*

## Getting Started

### 1. Environment Setup
Create a virtual environment and install the required dependencies.

```bash
git clone [https://github.com/USERNAME/COSMO-INR.git](https://github.com/USERNAME/COSMO-INR.git)
cd COSMO-INR
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
