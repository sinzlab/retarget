# Skeleton Invariant Pose Embedding (SkIP) 🤸‍♂️

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://example.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

---

Welcome to **SkIP**, a framework for abstracting skeletal topology into a shared latent space. With SkIP, you can retarget and reconstruct animations regardless of the underlying skeleton structure.

---
## Table of Contents
- [Overview](#overview-)
- [Installation](#installation-)
- [Usage](#usage-)
  - [Command Line Interface](#1-command-line-interface-cli)
  - [Python API](#2-python-api)
- [Reproducing Results](#reproducing-results-)
  - [Data Preparation](#data-preparation)   
  - [Wandb Setup](#wandb-setup)
  - [Docker & Training](#docker--training)
- [Datasets](#datasets-)
- [Contributing](#contributing-)
- [License](#license-)

---

## Overview 🚀

Skeleton Invariant Pose Embedding (SkIP) abstracts complex skeletal animations into a compact latent representation. By employing a **graph embedding** approach, it encodes the relationships between keypoints and decodes them back into their original form. This flexibility enables the training of diffusion models on varied skeleton topologies and generates animations that maintain motion fidelity.

---

## Installation 🔧

Install SkIP in editable mode with pip:

```bash
pip install -e .
```

---

## Usage 📦


## 1. Command Line Interface (CLI)

SkIP’s CLI provides a straightforward way to process animations without writing any additional code.

### 1.1. Retargeting an Animation

Retarget an animation by mapping a source skeleton to a target skeleton using a pre-trained model.

#### Usage

```bash
python scripts/retarget.py --src_path <path/to/source/animation.bvh> --tgt_path <path/to/target/animation.bvh> --model_name <checkpoint-name> --output_dir <path/to/results>
```

#### Parameter Details

- **`--src_path`**: Absolute or relative path to the source animation file (in BVH format).
- **`--tgt_path`**: Path to the target animation file to which the source will be retargeted.
- **`--model_name`**: The checkpoint identifier for the pre-trained model.
- **`--output_dir`**: Destination directory where the retargeted animation will be stored.

### 1.2. Reconstructing an Animation

Reconstruct an animation using the same model to generate a refined version of the original animation.

#### Usage

```bash
python scripts/retarget.py --src_path <path/to/source/animation.bvh> --model_name <checkpoint-name> --output_dir <path/to/results>
```

#### Parameter Details

- **`--src_path`**: Path to the source animation file.
- **`--model_name`**: Checkpoint identifier for the model.
- **`--output_dir`**: Directory for saving the reconstructed animation.

### 1.3. Using Short Flags

For convenience, short flags can be used with the `visualize.py` script.

#### Example

```bash
python scripts/visualize.py -s <source animation> -t <target animation> -m <checkpoint-name> -o <output directory>
```

- **`-s`**: Source animation file.
- **`-t`**: Target animation file.
- **`-m`**: Model checkpoint name.
- **`-o`**: Output directory.

---

## 2. Python API

For developers looking to integrate SkIP functionality into their projects, the Python API offers detailed control over the animation processing pipeline.

### 2.1. Retargeting an Animation via Python

The following example demonstrates how to load animations, retarget them, and save the output.

```python
from retarget import retarget_animation, load, save

# Define the model checkpoint
model_name = 'checkpoint-name'

# Load the source and target animations
source_animation, names, _ = load('path/to/source/animation.bvh')
target_animation, _, _ = load('path/to/target/animation.bvh')

# Perform retargeting
retargeted_animation = retarget_animation(model_name, source_animation, target_animation)

# Save the retargeted animation
save('path/to/retargeted/animation.bvh', retargeted_animation, names=names)
```

### 2.2. Reconstructing an Animation via Python

Reconstruct an animation using a similar process but with a single input file.

```python
from retarget import retarget_animation, load, save

# Specify the model checkpoint
model_name = 'checkpoint-name'

# Load the source animation
source_animation, names, _ = load('path/to/source/animation.bvh')

# Reconstruct the animation
reconstructed_animation = retarget_animation(model_name, source_animation)

# Save the reconstructed animation
save('path/to/reconstructed/animation.bvh', reconstructed_animation, names=names)
```

### 2.3. Extracting Latent Representations

To obtain latent features from a given animation, use the following method:

```python
from retarget import encode_animation

# Extract pose and trajectory latent representations
pose_latent, trajectory_latent = encode_animation(model_name, source_animation)
```

### 2.4. Low-Level Model API

For advanced use cases, interact directly with the underlying model architecture.

```python
from retarget.model import TransformerAutoEncoder
import torch

# Initialize a custom Transformer AutoEncoder
model = TransformerAutoEncoder(d_input=9, d_model=64, nhead=2, num_layers=4)

# Load a pre-trained model from a local path
model = TransformerAutoEncoder.from_pretrained('local/model')
```

---

## Reproducing Results 🎯

### Data Preparation

```bash
data/
└── combined/
    ├── character1/
    │   ├── action1.bvh
    │   ├── action2.bvh
    │   └── ...
    ├── character2/
    │   ├── action1.bvh
    │   ├── action2.bvh
    │   └── ...
    └── ...
```

### Wandb Setup

```bash
WANDB_API_KEY=your_api_key_here
WANDB_USER=your_username_here
```

### Docker & Training

#### Build the Docker Image

```bash
docker compose build base
```

#### Run Training

```bash
docker compose run python train.py --config configs/main.yaml --output_dir ./models/local
```

---

## Datasets 📚

- **ACAAD dataset**
- **LaFAN1 dataset:** [GitHub Repository](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- **CMU Mocap:** [CMU Motion Capture](http://mocap.cs.cmu.edu/)
- **PFNN dataset:** [GitHub Issue](https://github.com/sebastianstarke/AI4Animation/issues/52)
- **TotalCapture dataset:** [TotalCapture Data](https://cvssp.org/data/totalcapture/)
- **SFU dataset:** [SFU Mocap](https://mocap.cs.sfu.ca/)
- **BandaiNamco dataset:** [GitHub Repository](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset)

---

## Contributing 🤝

Check our [contribution guidelines](CONTRIBUTING.md) to see how you can help improve SkIP.

---

## License 📝

This project is licensed under the [MIT License](LICENSE).

---

*Happy animating!* 🎥✨
