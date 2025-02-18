from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

import retarget.utils.AnimationStructure as AnimationStructure
from retarget.utils.Animation import forward_rotations
from retarget.utils.BVH import load
from retarget.utils.Quaternions_old import Quaternions, quat_2_d6


class MixamoDataset(Dataset):
    def __init__(self, directory, mode="train", ground_feet=False, cons_q=8):
        super().__init__()

        animations = {}
        frame_times = {}

        self.cons_q = cons_q

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
        self.data_idx = []

        # Initialize empty list to put in the previous frames of a given frame
        # This is used for the velocity loss
        # Also Initialize empty list with frame times between frames
        # self.data_prev[idx] stores the offset frame from the original frame.
        # if mode == "train":
        # self.time = []

        for animation, time in zip(self.animations, self.frame_time):
            if 1 / time > 100:
                animation_graphs = animation.as_graph(stride=4)
            else:
                animation_graphs = animation.as_graph()

            length_animation = len(animation_graphs)

            if length_animation < 8:
                continue

            n_graphs = length_animation // self.cons_q * self.cons_q
            animation_graphs = animation_graphs[:n_graphs]

            total_frames_currently = len(self.data)

            self.data = self.data + animation_graphs
            self.data_idx += list(
                range(total_frames_currently, total_frames_currently + n_graphs)
            )

            # if self.mode == "train":
            #    self.time = self.time + [time] * n_graphs

        print("=== Mixamo Dataset Summary ===")
        print(
            f"Loaded {len(self.animations)} animation clips for {len(animations)} characters"
        )
        # print summary for each character
        for character, animations in animations.items():
            print(f"{character}")
            print(f"    - Animations: {len(animations) // self.cons_q * cons_q}")
        print(f"Total frames: {len(self.data):,}")
        print("===============================")

    def __len__(self):
        return len(self.data) // self.cons_q

    def get_one_item(self, idx, global_skel_scale=None):
        item = self.data[self.data_idx[idx]].clone()

        if self.mode == "train":

            # If the mode is "train", then also define the frame time
            frame_time = 1 / 30  # self.time[idx]

            scaled_offsets = item.offsets.numpy().copy() * global_skel_scale
            scaled_offsets_prev = scaled_offsets.copy()

            # For original frame
            parents = item.parents.numpy()
            rotation = item.rotation.numpy()
            rotation_prev = item.rotation_prev.numpy()
            edges = item.edge_index.numpy().T

            position = forward_rotations(
                parents, scaled_offsets, Quaternions(rotation[None, ...])
            )[0]

            position_prev = forward_rotations(
                parents, scaled_offsets_prev, Quaternions(rotation_prev[None, ...])
            )[0]

            t_pose = AnimationStructure.t_pose(scaled_offsets, edges)

            item.root_trajectory *= global_skel_scale
            item.position = torch.Tensor(position)
            item.x[:, 6:9] = torch.Tensor(position).clone()
            item.offsets = torch.Tensor(scaled_offsets)
            item.pos = torch.Tensor(t_pose)

            # Set the scaled skeletons new velocity and previous frame
            item.x[:, 9:12] = torch.Tensor(position_prev).clone()
            item.x[:, 12:15] = torch.Tensor(position - position_prev).clone()
            item.x[:, 15:] *= global_skel_scale

            return item, frame_time
        return item

    def __getitem__(self, idx):

        item = []

        if self.mode == "train":
            frame_time = []

            # item_example = self.data[self.data_idx[idx*self.cons_q]].clone()
            # scaled_offset = item_example.offsets.numpy().copy()
            # if np.random.rand() < 0.5:
            #    scaled_offset = scaled_offset * np.random.uniform(
            #        0.5, 1.5, size=scaled_offset.shape
            #    )

            if np.random.rand() < 0.25:
                global_skel_scale = np.random.uniform(0.5, 1.5)
            else:
                global_skel_scale = 1.0

            for i in range(self.cons_q):
                it, f_time = self.get_one_item(idx * self.cons_q + i, global_skel_scale)
                item.append(it)
                frame_time.append(f_time)

            return item  # , frame_time

        for i in range(self.cons_q):
            it = self.get_one_item(idx * self.cons_q + i)
            item.append(it)

        return item
