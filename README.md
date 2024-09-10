# Skeleton Invariant Pose Embedding (SkIP) 🤸
The first step is to obtain a latent embedding that can abstract away the topology of the skeleton. To achieve this we need to be able to encode and decode the pose into a shared latent space regardless of the topology of skeleton for these keypoints. This is a graph embedding problem. We need an Encoder $\mathcal{E}(X, E) \rightarrow z$ where $X$ are the features of the keypoint, $E$ are the edges in the graph and $z \in \mathbb{R}^N$ is the latent vector. We also need a Decoder $\mathcal{D}(z, E) \rightarrow X$ which given the graph topology can reconstruct the original pose X. This as a result, will allow training the diffusion model on any topology and generate from any topology.

## Getting Started

### Downloading Data
Download the processed data from [here](#) and place it in the `data` directory.
The data is structured as follows:
```bash
data/
    combined/
        character1/
            action1.bvh
            action2.bvh
            ...
        character2/
            action1.bvh
            action2.bvh
            ...
        ...
```

#### Datasets used
- ACAAD dataset
- [LaFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- [CMU Mocap](http://mocap.cs.cmu.edu/)
- [PFNN dataset](https://github.com/sebastianstarke/AI4Animation/issues/52)
- [TotalCapture dataset](https://cvssp.org/data/totalcapture/)
- [SFU dataset](https://mocap.cs.sfu.ca/)
- [BandaiNamco dataset](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset)

### Wandb setup
create a `.env` file in the root directory and add the following
```bash
WANDB_API_KEY=
WANDB_USER=
```

### Build Docker Image
```bash
docker compose build base
```


### Run TASE training
```bash
docker compose run python train_mixamo.py
```

## Using the model
```python
from retarget.model import TransformerAutoEncoder
import torch

# from scratch
model = TransformerAutoEncoder(d_input=9, d_model=64, nhead=2, num_layers=4)

# from pretrained
model = TransformerAutoEncoder.from_pretrained('local/model')
```
