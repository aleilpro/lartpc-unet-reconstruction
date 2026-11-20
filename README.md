# 2D U-Net for LArTPC Semantic Segmentation & Event Classification

A lightweight PyTorch implementation of a 2D U-Net pipeline inspired by the **SPINE** (*Scalable Particle Imaging with Neural Embeddings*) reconstruction workflow, designed for pixel-level semantic segmentation and event classification in Liquid Argon Time Projection Chambers (LArTPC).

## Physics Motivation
Liquid Argon TPC detectors (such as SBND, ICARUS, and DUNE) capture high-resolution topological images of neutrino-argon interactions. Automated event reconstruction relies on accurately separating track-like particles (muons, protons, pions) from electromagnetic showers (electrons, photons) and classifying overall interaction topologies directly from sparse detector projections.

## Repository Contents
* **`MLipynb`** — Main end-to-end pipeline: data loading and preprocessing from the Deep Learn Physics LArTPC dataset (`practice_train_2k.root`), lightweight 2D U-Net architecture definition, GPU training loop, quantitative evaluation, and event-display visualizations.
* **`miniSPINE2000.ipynb`** — Exploratory experiments with dataset (`practice_test_2k.root`), feature/data inspection, and baseline model prototyping.
* **`miniSPINE_unet_weights.pth`** — Pre-trained PyTorch model weights ready for immediate inference and evaluation without retraining from scratch.

## Model & Pipeline Overview
* **Framework:** PyTorch (trained with CUDA GPU acceleration)
* **Architecture:** Lightweight symmetric 2D U-Net with encoder-decoder blocks, skip connections, Batch Normalization, and ReLU activations.
* **Dataset:** Deep Learn Physics public LArTPC dataset (`practice_test_2k.root`, 2,000 simulated LArTPC events).
* **Tasks:**
  1. **Pixel-level Semantic Segmentation:** Separating background, track-like hits, and electromagnetic shower hits.
  2. **Event Classification:** Categorizing event topologies from learned representations.

## How to Run
1. Clone the repository
2.Install the required dependencies:
  torch numpy matplotlib scikit-learn uproot awkward h5py tqdm
3.Download the practice_test_2k.root dataset from the Deep Learn Physics open data repository and update the input file path in miniSPINE2000.ipynb.
4.Open miniSPINE2000.ipynb in Jupyter / VS Code to reproduce training or load miniSPINE_unet_weights.pth directly for inference.
