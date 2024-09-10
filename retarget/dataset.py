from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

import retarget.utils.AnimationStructure as AnimationStructure
from retarget.utils.Animation import forward_rotations
from retarget.utils.BVH import load
from retarget.utils.Quaternions_old import Quaternions, quat_2_d6


class MixamoDataset(Dataset):
    def __init__(self, directory, mode="train"):
        super().__init__()

        animations = {}

        characters = list(Path(directory).glob("*"))

        # exclude_characters = ['Remy', 'Amy', 'Mannequin', ]

        # include_characters = ['Aj', 'Amy', 'BigVegas', 'Ely By K.Atienza', 'Exo Gray', 'Goblin_m', 'Kaya', 'Mannequin', 'Maria J J Ong', 'Michelle']

        # characters = [character for character in characters if character.name in exclude_characters]

        # characters = [character for character in characters if character.name not in exclude_characters]

        self.mode = mode
        stride = 1 if mode == "train" else 64

        for character in characters:
            animations[character.name] = {}
            actions = list(character.glob("*.bvh"))

            for action in actions:
                try:
                    animations[character.name][action.name], _, _ = load(action)
                except Exception as e:
                    print(f"Error loading {character.name}/{action.name}: {e}")

            animations[character.name] = list(animations[character.name].values())

        self.animations = list(animations.values())

        # flatten the lists
        self.animations = [
            animation for character in self.animations for animation in character
        ]

        self.data = []
        for animation in self.animations:
            self.data = self.data + animation.as_graph()

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
        item = self.data[idx]

        if self.mode == "train":
            scaled_offsets = item.offsets.numpy().copy()
            if np.random.rand() < 0.5:
                scaled_offsets = scaled_offsets * np.random.uniform(
                    0.5, 1.5, size=scaled_offsets.shape
                )

            # if np.random.rand() < 0.1:
            #     scaled_offsets = scaled_offsets + np.random.uniform(-5, 5, size=scaled_offsets.shape)

            if np.random.rand() < 0.25:
                scaled_offsets = scaled_offsets * np.random.uniform(0.5, 1.5)

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

        return item
