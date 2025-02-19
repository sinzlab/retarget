import torch
from torch_geometric.loader import DataLoader
import numpy as np

from retarget.model import graph_to_batch, mask_from_batch
from retarget.utils.Quaternions_old import d6_2_quat, Quaternions
from retarget.utils.Animation import Animation
from retarget.model import TransformerAutoEncoder
from retarget.model import TransformerAutoEncoder
from retarget.tokenizer import Tokenizer
from retarget.utils.BVH import load, save
from retarget.utils.visualize import plot_pose

import argparse

if __name__ == "__main__":

    device = torch.device("cpu")

    # Load pretrained model
    print("LOAD PRETRAINED MODEL")
    model_name = "zany-fire-381"
    checkpoint = torch.load(
        f"./models/local/{model_name}_latest_checkpoint.tar", map_location="cpu"
    )
    # model = TransformerAutoEncoder.from_pretrained(f'local/{model_name}_best_model.pt')
    model = TransformerAutoEncoder.from_pretrained(
        checkpoint["epoch"], checkpoint=checkpoint
    )
    model.eval()
    tokenizer = Tokenizer()

    # string = "Capoeira"

    string = "bow"
    # animation, new_names, _ = load('/home/kia/MOTION_ESTIMATION/DATA_BANDAI_NAMCO/test/val_data/bandai-namco/dataset-2_run_masculine_006.bvh',ground_feet=False)

    if string == "Capoeira":
        animation, new_names, _ = load(
            "/user/kyang2/u12303/skip-dataset/train/Amy/Capoeira.bvh", ground_feet=False
        )
    else:
        #animation, new_names, _ = load(
        #    r"/user/kyang2/u12303/skip-dataset/test/Kaya/Getting Up.bvh",
        #    ground_feet=False,
        #)
        # animation, new_names, _ = load('/user/kyang2/u12303/skip-dataset/test/bandai-namco/dataset-1_bow_old_001.bvh',ground_feet=False)
        animation, new_names, _ = load('/user/kyang2/u12303/skip-dataset/test/bandai-namco/dataset-1_run_active_001.bvh',ground_feet=False)

    #
    animation.positions[..., :, 0:1] -= animation.positions[0:1, 0:1, 0:1]
    animation.positions[..., :, 2:3] -= animation.positions[0:1, 0:1, 2:3]
    
    # convert the animation to a graph
    data = animation.as_graph()

    # encode the graph using the tokenizer
    batch, mask = tokenizer.encode(data)

    with torch.inference_mode():
        # encode the animation into the latent space
        latent = model.encoder(batch.x, batch.pos, batch.edge_index, mask=mask)
        root_trajectory_gt = graph_to_batch(batch.root_trajectory, mask)

        # decode the latent space back into the animation
        y_pred = model.decoder(latent, batch.pos, batch.edge_index, mask=mask)

        fk_pose, edge_indexs = tokenizer.decode(batch, y_pred[..., :, :6])
        rotations = Quaternions(np.stack([d6_2_quat(d6) for d6 in y_pred[..., :, :6]]))
        positions = (fk_pose - fk_pose[..., 0:1, :]).detach().numpy()
        root_trajectory = y_pred[...,0:1,6:]
        root_trajectory[:,0,0] = torch.cumsum(root_trajectory[:,0,0],dim=-1)
        root_trajectory[:,0,2] = torch.cumsum(root_trajectory[:,0,2],dim=-1)
        print(torch.norm(root_trajectory - root_trajectory_gt[:,0:1,:],dim=-1).mean())

        positions += root_trajectory.detach().numpy()
    # positions[:,0] = animation.positions[:,0]

    recon_anim = Animation(
        rotations,
        positions * 170,
        animation.orients,
        animation.offsets * 170,
        animation.parents,
    )

    if string == "Capoeira":
        save(f"Capoeira_{model_name}.bvh", recon_anim)
    else:
        save(f"bow_{model_name}.bvh", recon_anim)

    animation.positions *= 170
    animation.offsets *= 170

    save("bow_no_traj_run_truth.bvh", animation)
    # save("Capoeira_no_traj_truth.bvh",animation)
