# CMPM 118 AIEA: AI Explainability & Accountability Lab
**Onboarding Task: Memory Wrap (SVHN)**

## Project Context
This project implements the Memory Wrap architecture to provide "Explanations by Example" for the SVHN dataset using a MobileNet backbone (`2000.pt`). The goal is to evaluate model interpretability by visualizing the training samples that influence specific model predictions.

## System Specifications
- OS: Windows 11
- GPU: RTX 3070 (8GB)
- CPU: Ryzen 7
- RAM: 16GB
- CUDA: 11.8

## Technical Fixes & Environment Setup
The following adjustments were made to the original repository to ensure functionality on a local Windows/GPU environment:
- **Hardware Acceleration:** Configured PyTorch for CUDA 11.8 to utilize the RTX 3070.
- **Dependency Patch:** Downgraded Pillow to 9.0.0 to fix compatibility issues with legacy research code attributes.
- **Namespace Resolution:** Resolved ModuleNotFound errors by implementing module-based execution (`-m`) and patching sys.path.
- **Batch Optimization:** Set batch_size_test=1 to generate discrete images for qualitative analysis and lab deliverables.

## Usage
Run the following from the `/paper/` directory:
`python -m scripts.generate_memory_images --path_model=models/2000.pt --batch_size_test=1`

## Deliverables
The `/results_samples/` directory contains 13 randomized explanation figures. These visualize the relationship between test inputs and the training samples retrieved by the model memory for classification.

## Credits & References
This project is a hardware-specific implementation based on the research presented in:

**"A self-interpretable module for deep image classification on small data"**
*Biagio La Rosa, Roberto Capobianco and Daniele Nardi.*
Applied Intelligence (2022).
[Official DOI: 10.1007/s10489-022-03886-6](https://doi.org/10.1007/s10489-022-03886-6)

Original Repository: [KRLGroup/memory-wrap](https://github.com/KRLGroup/memory-wrap)