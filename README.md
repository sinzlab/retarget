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


### Run SkIP training
```bash
docker compose run python train.py --config configs/main.yaml --output_dir ./models/local
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

### Reconstructing an animation
```python
from retarget import retarget_animation, load, save

model_name = 'checkpoint-name'

source_animation, names, _ = load('path/to/source/animation.bvh')

reconstructed_animation = retarget_animation(model_name, source_animation)

save('path/to/reconstructed/animation.bvh', reconstructed_animation, names=names)
```

### Retargeting an animation
```python
from retarget import retarget_animation, load, save

model_name = 'checkpoint-name'

source_animation, names, _ = load('path/to/source/animation.bvh')
target_animation, _, _ = load('path/to/target/animation.bvh') # animation with target skeleton

retargeted_animation = retarget_animation(model_name, source_animation, target_animation)

save('path/to/retargeted/animation.bvh', retargeted_animation, names=names)
```

### Getting latent space
```python
from retarget import encode_animation

pose_latent, trajectory_latent = encode_animation(model_name, source_animation)
```

### Running from command line
For reconstructing an animation, run
```bash
python scripts/retarget.py --src_path path/to/source/animation.bvh --model_name checkpoint-name --output_dir ./results
```

For retargeting an animation, run
```bash
python scripts/retarget.py --src_path path/to/source/animation.bvh --tgt_path path/to/target/animation.bvh --model_name checkpoint-name --output_dir ./results
```

Alternatively you can use short flags for the arguments
```bash
python scripts/visualize.py -s path/to/source/animation.bvh -t path/to/target/animation.bvh -m checkpoint-name -o ./results
```

