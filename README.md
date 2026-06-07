

# CattleEyeView: Edge-Native Multi-Task Livestock Analytics

This repository contains the official implementation of the unified multi-task learning framework for precision livestock monitoring. Our system enables real-time detection, instance segmentation, and skeletal pose estimation from overhead surveillance, optimized specifically for low-power edge deployment in farm environments.

## Table of Contents

1. [Overview](https://www.google.com/search?q=%23overview)
2. [Key Architectural Contributions](https://www.google.com/search?q=%23key-architectural-contributions)
3. [Getting Started](https://www.google.com/search?q=%23getting-started)
4. [Training and Reproducibility](https://www.google.com/search?q=%23training-and-reproducibility)
5. [Ablation Study Results](https://www.google.com/search?q=%23ablation-study-results)
6. [License & Citation](https://www.google.com/search?q=%23license--citation)

## Overview

CattleEyeView leverages a shared-backbone architecture (CSPDarknet) to extract bovine morphological biomarkers in real-time. By utilizing a dynamic loss balancing engine (GradNorm), our model maintains high performance across three simultaneous task heads—detection, segmentation, and pose regression—within strict hardware constraints.

## Key Architectural Contributions

* **Unified Multi-Task Head:** A single-pass inference pipeline that reduces redundant feature computation.
* **Dynamic Gradient Balancing:** Implementation of a custom `GradNorm` engine to synchronize training velocities between dense mask pixels and sparse keypoints.
* **Edge-Optimized Design:** Validated at $56\text{ FPS}$ with a $12.4\text{ W}$ power draw on NVIDIA Jetson Orin Nano hardware.

## Getting Started

### Prerequisites

* Python 3.10+
* PyTorch 2.1.0+
* CUDA 12.1+

### Installation

```bash
git clone https://github.com/AnimalEyeQ/CattleEyeView
cd CattleEyeView
pip install -r requirements.txt

```

## Training and Reproducibility

To reproduce the **M3** configuration as described in the paper, run the following command:

```bash
python src/train.py --config config/experiments/M3.yaml --epochs 100

```

Training logs are automatically generated in `./logs/`. You can visualize performance using our provided utility:

```bash
python scripts/plot_convergence.py --log_file logs/M3_metrics.csv

```

## Ablation Study Results

The following table summarizes the performance contributions of each architectural module:

| Model | ECA | GradNorm | Body mAP | Mask mAP | Pose mAP |
| --- | --- | --- | --- | --- | --- |
| M0 | -- | -- | 0.802 | 0.524 | 0.662 |
| M1 | \checkmark | -- | 0.819 | 0.538 | 0.681 |
| M2 | -- | \checkmark | 0.835 | 0.569 | 0.720 |
| **M3** | \checkmark | \checkmark | **0.849** | **0.581** | **0.739** |

## License & Citation

