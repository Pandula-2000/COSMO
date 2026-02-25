# COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Tested-EE4C2C?logo=pytorch)](https://pytorch.org/)
[cite_start][![Paper](https://img.shields.io/badge/ArXiv-Paper-B31B1B.svg)](https://arxiv.org/) Official implementation of **"COSMO-INR: COMPLEX SINUSOIDAL MODULATION FOR IMPLICIT NEURAL REPRESENTATIONS"**[cite: 56, 57].

## Overview
[cite_start]Implicit Neural Representations (INRs) compactly encode complex signals but traditionally suffer from spectral bias and signal attenuation[cite: 60, 64]. [cite_start]We prove that odd and even symmetric activation functions exhibit attenuation in their post-activation spectrum due to missing frequency coefficients in their Chebyshev polynomial approximations[cite: 145, 344]. 

[cite_start]COSMO-INR mitigates this attenuation by introducing a complex sinusoidal modulation to the activation function, ensuring complete spectral support throughout the network[cite: 66]. 

## Theoretical Background
[cite_start]Based on harmonic distortion analysis and Chebyshev polynomial approximation, we demonstrate that the Raised Cosine function offers the least coefficient decay, yielding superior spectral bandwidth[cite: 301, 311]. 

To prevent the suppression of odd/even symmetric components, we define the COSMO-RC activation as:
[cite_start]$$g(x) = \phi(x)e^{j\zeta x}$$ [cite: 395]

[cite_start]Where $\phi(x)$ is the Raised Cosine function with a learnable bandwidth $T$ and frequency shift $\zeta$[cite: 404]:
[cite_start]$$\phi(x) = \text{sinc}\left(\frac{x}{T}\right) \frac{\cos\left(\frac{\pi \beta x}{T}\right)}{1 - \left(\frac{2\beta x}{T}\right)^2}$$ [cite: 405, 408, 409]

[cite_start]The outputs at each layer are complex-valued, normalized to the unit circle to maintain training stability[cite: 388]. [cite_start]We employ a task-specific prior knowledge embedder (e.g., ResNet-34) and a sigmoid regularizer to dynamically adjust the activation parameters ($T$ and $\zeta$), significantly accelerating convergence[cite: 507, 515].

![Architecture Pipeline](docs/architecture_placeholder.png)
[cite_start]*Figure 1: Complete pipeline of the COSMO-RC model architecture featuring the prior embedding sigmoid regularizer.* [cite: 561]

## Tasks Tested
[cite_start]COSMO-RC establishes state-of-the-art performance across a diverse set of signal representation and inverse computer vision tasks[cite: 156]:
* [cite_start]**Image Representation** (Kodak, DIV2K datasets) [cite: 591, 1436]
* [cite_start]**Image Denoising** (DIV2K with Poisson photon noise) [cite: 679, 680]
* [cite_start]**Image Super-Resolution** (2x, 4x, and 6x upsampling on DIV2K) [cite: 726, 728]
* [cite_start]**Image Inpainting** (20% pixel sampling) [cite: 751]
* [cite_start]**3D Occupancy Volume** (Lucy dataset, $512^3$ voxel grid) [cite: 804]
* [cite_start]**Neural Radiance Fields (NeRF)** (Lego dataset view synthesis) [cite: 840, 852]

## Key Results
[cite_start]Our activation consistently outperforms existing SOTA activations (SIREN, WIRE, INCODE, FINER) in both accuracy and high-frequency structural preservation[cite: 577, 856].

| Task | Metric | COSMO-RC | Nearest SOTA | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| Image Representation (Kodak) | PSNR | **41.24 dB** | 35.57 dB (INCODE) | [cite_start]+5.67 dB [cite: 71, 593] |
| Image Denoising | PSNR | **30.25 dB** | 29.79 dB (INCODE) | [cite_start]+0.46 dB [cite: 71, 688, 689] |
| Super-Resolution (6x) | PSNR | **27.66 dB** | 27.02 dB (FINER) | [cite_start]+0.64 dB [cite: 71, 733] |
| NeRF (Lego) | PSNR | **29.50 dB** | 26.05 dB (INCODE) | [cite_start]+3.45 dB [cite: 857, 913] |

![Qualitative Results](docs/results_placeholder.png)
[cite_start]*Figure 2: Qualitative comparisons for Image Denoising and Super-resolution.* [cite: 713, 1506]

## Getting Started

### 1. Environment Setup
Create a virtual environment and install the required dependencies. [cite_start]The code is tested on Python 3.8+ and PyTorch[cite: 566].

```bash
git clone [https://github.com/USERNAME/COSMO-INR.git](https://github.com/USERNAME/COSMO-INR.git)
cd COSMO-INR
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
