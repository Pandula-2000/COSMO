# [cite_start]COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations [cite: 56, 57]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Tested-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/ArXiv-Paper-B31B1B.svg)](https://arxiv.org/)

[cite_start]Official implementation of **"COSMO-INR: COMPLEX SINUSOIDAL MODULATION FOR IMPLICIT NEURAL REPRESENTATIONS"**[cite: 56, 57].

## Overview
[cite_start]Implicit neural representations (INRs) offer a continuous alternative to discrete signal representations, compactly encoding complex signals across computer vision tasks[cite: 60, 61]. [cite_start]However, odd and even symmetric activation functions suffer from attenuation in their post-activation spectrum[cite: 145]. 

COSMO-INR addresses this limitation. [cite_start]By modulating activation functions using a complex sinusoidal term, the network achieves complete spectral support and mitigates spectral bias[cite: 66, 147].

## Theoretical Background
[cite_start]Using harmonic distortion analysis and Chebyshev polynomial approximation, we show that the raised cosine activation offers the least decay for larger coefficients, providing optimal spectral bandwidth[cite: 301, 311]. 

To prevent the attenuation of symmetric components, we define the COSMO-RC activation as:
[cite_start]$$g(x)=\phi(x)e^{j\zeta x}$$ [cite: 395]

Where $\phi(x)$ is the raised cosine function with a learnable bandwidth $T$ and frequency shift $\zeta$:
[cite_start]$$\phi(x)=\text{sinc}\left(\frac{x}{T}\right)\frac{\cos\left(\frac{\pi\beta x}{T}\right)}{1-\left(\frac{2\beta x}{T}\right)^2}$$ [cite: 405, 408, 409, 410]

[cite_start]The outputs at each layer are complex-valued and normalized to the unit circle on the complex plane to ensure a stable learning curve[cite: 388]. [cite_start]To accelerate convergence, we integrate a task-specific prior knowledge embedder (e.g., ResNet-34 or ResNet3D-18) combined with a sigmoid regularizer to dynamically adjust the $T$ and $\zeta$ parameters[cite: 507, 508, 510, 512].

![Architecture Pipeline](docs/architecture_placeholder.png)
[cite_start]*Figure 1: Complete pipeline of the COSMO-RC model architecture featuring the prior embedding sigmoid regularizer[cite: 561].*

## Tasks Tested
[cite_start]COSMO-RC establishes state-of-the-art performance across diverse signal representation and inverse problems[cite: 156]:
* [cite_start]**Image Representation** (Kodak and DIV2K datasets) [cite: 591, 679]
* [cite_start]**Image Denoising** (DIV2K with Poisson photon noise) [cite: 679, 680]
* [cite_start]**Image Super-Resolution** (2x, 4x, and 6x upsampling on DIV2K) [cite: 726, 728]
* [cite_start]**Image Inpainting** (Celtic spiral knots image, 20% pixel sampling) [cite: 751]
* [cite_start]**3D Occupancy Volume** (Lucy dataset, $512^3$ voxel grid) [cite: 804]
* [cite_start]**Neural Radiance Fields (NeRF)** (Lego dataset novel-view synthesis) [cite: 841, 852]

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
