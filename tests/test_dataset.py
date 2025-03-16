from retarget.dataset import SkiPDataset
from retarget.model import TransformerAutoEncoder
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import torch

from retarget.types import AnimationBatch
from retarget.augment import augment_rest_pose, augment_root_trajectory
from retarget.utils.Animation import forward_rotations_torch
from retarget.utils.Quaternions import d6_2_rotmat, quat_2_d6, d6_2_quat
from retarget.tokenizer import TokenizerNew
from retarget.utils.visualize import plot_pose

def collate_fn(data_list):
    """
    Batch data items from a list of AnimationData objects.
    """
    motions, source_rest_poses, target_rest_poses = zip(*data_list)
    motion_batch, source_rest_pose_batch, source_mask = AnimationBatch.from_data_list(motions)

    target_joint_counts = [b.x.shape[0] for b in target_rest_poses]
    target_mask = torch.zeros(motion_batch.features.shape[0], motion_batch.features.shape[1], max(target_joint_counts), dtype=torch.bool)
    for i, joint_count in enumerate(target_joint_counts):
        target_mask[i, :, :joint_count] = 1

    target_rest_pose_batch = Batch.from_data_list(target_rest_poses)

    return motion_batch, source_rest_pose_batch, target_rest_pose_batch, source_mask, target_mask

def test_skip_dataset():
    tokenizer = TokenizerNew()
    dataset = SkiPDataset(directory="data/train", tokenizer=tokenizer)
    
    assert len(dataset) > 0

    model = TransformerAutoEncoder.build_from_config()

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    for motion, source_rest_pose, target_rest_pose, source_mask, target_mask in dataloader:
        augmented_item = augment_rest_pose(motion)
        augmented_item = augment_root_trajectory(motion)

        # cycle consistency
        z_pose, z_traj = model.encoder(motion.features, source_rest_pose, source_mask)
        out_pose_retargetted, out_traj_retargetted = model.decoder(z_pose, z_traj, target_rest_pose, target_mask)

        feature_vector = dataset.tokenizer.get_feature_vector(out_pose_retargetted, out_traj_retargetted, target_rest_pose)

        z_pose_retargetted, z_traj_retargetted = model.encoder(feature_vector, target_rest_pose, target_mask)
        out_pose, out_traj = model.decoder(z_pose_retargetted, z_traj_retargetted, source_rest_pose, source_mask)
        
        print('out_pose.shape', out_pose.shape, 'out_traj.shape', out_traj.shape)
        print('out_pose_retargetted.shape', out_pose_retargetted.shape, 'out_traj_retargetted.shape', out_traj_retargetted.shape)
        break

    assert False


