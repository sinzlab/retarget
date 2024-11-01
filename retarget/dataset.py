from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

import retarget.utils.AnimationStructure as AnimationStructure
from retarget.utils.Animation import forward_rotations
from retarget.utils.BVH import load
from retarget.utils.Quaternions_old import Quaternions, quat_2_d6


class MixamoDataset(Dataset):
    def __init__(self, directory, mode="train", ground_feet = False):
        super().__init__()

        animations = {}
        frame_times = {}

        characters = list(Path(directory).glob("*"))

        # exclude_characters = ['Remy', 'Amy', 'Mannequin', ]

        # include_characters = ['Aj', 'Amy', 'BigVegas', 'Ely By K.Atienza', 'Exo Gray', 'Goblin_m', 'Kaya', 'Mannequin', 'Maria J J Ong', 'Michelle']

        # characters = [character for character in characters if character.name in exclude_characters]

        # characters = [character for character in characters if character.name not in exclude_characters]
    
        self.ground_feet = ground_feet
        self.mode = mode
        stride = 1 if mode == "train" else 64

        for character in characters:
            animations[character.name] = {}
            frame_times[character.name] = {}
            actions = list(character.glob("*.bvh"))

            for action in actions:
                try:
                    (
                        animations[character.name][action.name],
                        _,
                        (frame_times[character.name][action.name], _, _),
                    ) = load(action, ground_feet=ground_feet)
                except Exception as e:
                    print(f"Error loading {character.name}/{action.name}: {e}")

            animations[character.name] = list(animations[character.name].values())
            frame_times[character.name] = list(frame_times[character.name].values())

        self.animations = list(animations.values())
        self.frame_times = list(frame_times.values())
        # flatten the lists
        self.animations = [
            animation for character in self.animations for animation in character
        ]
        self.frame_time = [
            frame_time for character in self.frame_times for frame_time in character
        ]
        # Create Empty list to fill with frames as graph
        self.data = []
        
        #If mode = Test, then also include the trajectory positions
        if self.ground_feet:
            self.trajectory = []
        
        # Initialize empty list to put in the previous frames of a given frame
        # This is used for the velocity loss
        # Also Initialize empty list with frame times between frames
        if mode == "train":
            self.data_prev = []
            self.data_prev_prev = []
            self.data_prev_prev_prev = []
            self.time = []

        for animation, time in zip(self.animations, self.frame_time):
            animation_graphs = animation.as_graph()
            n_graphs = len(animation_graphs)

            self.data = self.data + animation_graphs
            
            #Save the root trajectory of animation
            if self.ground_feet:
                self.trajectory += list(torch.tensor(animation.positions[...,0:1,:]))
                
            if self.mode == "train":
                self.time = self.time + [time] * n_graphs
                for i in range(n_graphs):
                    if i == 0:
                        self.data_prev.append(animation_graphs[i])
                        self.data_prev_prev.append(animation_graphs[i])
                        self.data_prev_prev_prev.append(animation_graphs[i])
                    elif i == 1:
                        self.data_prev.append(animation_graphs[i - 1])
                        self.data_prev_prev.append(animation_graphs[i - 1])
                        self.data_prev_prev_prev.append(animation_graphs[i - 1])
                    elif i == 2:
                        self.data_prev.append(animation_graphs[i - 1])
                        self.data_prev_prev.append(animation_graphs[i - 2])
                        self.data_prev_prev_prev.append(animation_graphs[i - 2])
                    else:
                        self.data_prev.append(animation_graphs[i - 1])
                        self.data_prev_prev.append(animation_graphs[i - 2])
                        self.data_prev_prev_prev.append(animation_graphs[i - 3])

        print("=== Mixamo Dataset Summary ===")
        print(
            f"Loaded {len(self.animations)} animation clips for {len(animations)} characters"
        )
        # print summary for each character
        for character, animations in animations.items():
            print(f"{character}")
            print(f"    - Animations: {len(animations)}")
        print(f"Total frames: {len(self.data):,}")
        print("===============================")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].clone()
        if self.mode == "train":

            # If the mode is "train", then also define the frame time
            frame_time = self.time[idx]

            # If the mode is "train", then also define the previous frame
            item_prev = self.data_prev[idx].clone()

            # If the mode is "train", then also define the previous previous frame
            item_prev_prev = self.data_prev_prev[idx].clone()

            # If the mode is "train", then also define the previous previous previous frame
            item_prev_prev_prev = self.data_prev_prev_prev[idx].clone()

            # Offsets from (previous-) previous frame and current frame is the same (From same Animation)
            scaled_offsets = item.offsets.numpy().copy()
            if np.random.rand() < 0.5:
                scaled_offsets = scaled_offsets * np.random.uniform(
                    0.5, 1.5, size=scaled_offsets.shape
                )

            # if np.random.rand() < 0.1:
            #     scaled_offsets = scaled_offsets + np.random.uniform(-5, 5, size=scaled_offsets.shape)

            if np.random.rand() < 0.25:
                scaled_offsets = scaled_offsets * np.random.uniform(0.5, 1.5)

            # For original frame
            parents = item.parents.numpy()
            rotation = item.rotation.numpy()
            edges = item.edge_index.numpy().T

            position = forward_rotations(
                parents, scaled_offsets, Quaternions(rotation[None, ...])
            )[0]

            t_pose = AnimationStructure.t_pose(scaled_offsets, edges)

            item.position = torch.Tensor(position)
            item.offsets = torch.Tensor(scaled_offsets)
            item.t_pose = torch.Tensor(t_pose)

            # For previous frame
            parents_prev = item_prev.parents.numpy()
            rotation_prev = item_prev.rotation.numpy()
            edges_prev = item_prev.edge_index.numpy().T
            scaled_offsets_prev = scaled_offsets.copy()

            position_prev = forward_rotations(
                parents_prev, scaled_offsets_prev, Quaternions(rotation_prev[None, ...])
            )[0]

            t_pose_prev = AnimationStructure.t_pose(scaled_offsets_prev, edges_prev)

            item_prev.position = torch.Tensor(position_prev)
            item_prev.offsets = torch.Tensor(scaled_offsets_prev)
            item_prev.t_pose = torch.Tensor(t_pose_prev)

            # For previous previous frame
            parents_prev_prev = item_prev_prev.parents.numpy()
            rotation_prev_prev = item_prev_prev.rotation.numpy()
            edges_prev_prev = item_prev_prev.edge_index.numpy().T
            scaled_offsets_prev_prev = scaled_offsets.copy()

            position_prev_prev = forward_rotations(
                parents_prev_prev,
                scaled_offsets_prev_prev,
                Quaternions(rotation_prev_prev[None, ...]),
            )[0]

            t_pose_prev_prev = AnimationStructure.t_pose(
                scaled_offsets_prev_prev, edges_prev_prev
            )

            item_prev_prev.position = torch.Tensor(position_prev_prev)
            item_prev_prev.offsets = torch.Tensor(scaled_offsets_prev_prev)
            item_prev_prev.t_pose = torch.Tensor(t_pose_prev_prev)

            # For previous previous previous frame
            parents_prev_prev_prev = item_prev_prev_prev.parents.numpy()
            rotation_prev_prev_prev = item_prev_prev_prev.rotation.numpy()
            edges_prev_prev_prev = item_prev_prev_prev.edge_index.numpy().T
            scaled_offsets_prev_prev_prev = scaled_offsets.copy()

            position_prev_prev_prev = forward_rotations(
                parents_prev_prev_prev,
                scaled_offsets_prev_prev_prev,
                Quaternions(rotation_prev_prev_prev[None, ...]),
            )[0]

            t_pose_prev_prev_prev = AnimationStructure.t_pose(
                scaled_offsets_prev_prev_prev, edges_prev_prev_prev
            )

            item_prev_prev_prev.position = torch.Tensor(position_prev_prev_prev)
            item_prev_prev_prev.offsets = torch.Tensor(scaled_offsets_prev_prev_prev)
            item_prev_prev_prev.t_pose = torch.Tensor(t_pose_prev_prev_prev)

            return item, item_prev, item_prev_prev, item_prev_prev_prev, frame_time
        
        if self.ground_feet:
            return item, self.trajectory[idx]
        
        return item
