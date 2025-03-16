import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from torch_geometric.data import Data, Dataset

from retarget.augment import (Augmentions, GlobalSkeletonAugmentor,
                              RestPoseAugmentor, XZTranslationAugmentor, augment_global_skeleton, augment_rest_pose, augment_root_trajectory)
from retarget.utils.Animation import Animation, quat_2_d6
from retarget.utils.BVH import load
from retarget.types import AnimationData
from retarget.tokenizer import TokenizerNew

import torch
import numpy as np

class RestPoseRegistry:
    def __init__(self):
        self.rest_pose_registry = []

    def add(self, rest_pose: Data):
        # check if rest_pose is already in the registry
        if rest_pose in self.rest_pose_registry:
            return
            
        # check if rest_pose is similar to any of the rest_poses in the registry
        for rest_pose_in_registry in self.rest_pose_registry:
            if self.is_similar(rest_pose, rest_pose_in_registry):
                return
        
        self.rest_pose_registry.append(rest_pose)

    def sample(self) -> Data:
        return self.rest_pose_registry[np.random.randint(0, len(self.rest_pose_registry))]
    
    def is_similar(self, rest_pose_a: Data, rest_pose_b: Data) -> bool:
        # check if the rest poses are similar
        same_shape = rest_pose_a.x.shape == rest_pose_b.x.shape
        if not same_shape:
            return False

        same_values = np.allclose(rest_pose_a.x, rest_pose_b.x)
        return same_values
    
    def __len__(self) -> int:
        return len(self.rest_pose_registry)

class SkiPDataset(Dataset):
    """
    Dataset class for loading and processing animations.
    """
    def __init__(
            self,
            directory: str,
            mode: str = "train",
            ground_feet: bool = False,
            consequtive_frames: int = 8,
            augmentors: List[dict] = [
                {
                    "name": "augment_rest_pose",
                    "max_large_angle": 10,
                    "max_small_angle": 10,
                    "probability": 0.25
                },
                {
                    "name": "augment_global_skeleton",
                    "max_scale": 1.1,
                    "min_scale": 0.9,
                    "probability": 0.5
                },
                {
                    "name": "augment_root_trajectory",
                    "max_translation": 10,
                    "probability": 1
                }
            ],
            tokenizer: Optional[TokenizerNew] = None
            ) -> None:
        super().__init__()

        self.consequtive_frames = consequtive_frames

        self.augmentors = augmentors

        characters = list(Path(directory).glob("*"))

        self.ground_feet = ground_feet
        self.mode = mode

        self.rest_pose_registry = RestPoseRegistry()

        self.data: List[AnimationData] = []
        self.characters = []

        if tokenizer is None:
            self.tokenizer = TokenizerNew()
        else:
            self.tokenizer = tokenizer

        for character in characters:
            actions = list(character.glob("*.bvh"))

            for action in actions:
                try:
                    animation, _, (frame_time, _, _) = load(action, ground_feet=ground_feet)
                    data_items, rest_pose = self.process_animation(animation, frame_time)
                    self.data.extend(data_items)
                    self.characters += [character.name] * len(data_items)
                    self.rest_pose_registry.add(rest_pose)

                except Exception as e:
                    print(f"Error loading {character.name}/{action.name}: {e}")


    def process_animation(self, animation: Animation, frame_time: float) -> List[AnimationData]:
        if 1 / frame_time > 100:
            stride = 4
        else:
            stride = 1

        animations = animation.split(stride=stride, consequtive_frames=self.consequtive_frames)

        data_items = []
        for anim in animations:
            encoded_item = self.tokenizer.encode(anim, frame_time=frame_time)
            data_items.append(encoded_item.motion)

        return data_items, encoded_item.rest_pose

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> AnimationData:
        motion = self.data[idx]
        character = self.characters[idx]
        # sample a random skeleton from the registry
        target_rest_pose = self.rest_pose_registry.sample()
        source_rest_pose = motion.rest_pose

        if self.mode == "train":
            for augmentor in self.augmentors:
                if augmentor["name"] == "augment_global_skeleton":
                    motion = augment_global_skeleton(motion, augmentor["max_scale"], augmentor["min_scale"], augmentor["probability"])
                elif augmentor["name"] == "augment_rest_pose":
                    motion = augment_rest_pose(motion, augmentor["max_large_angle"], augmentor["max_small_angle"], augmentor["probability"])
                elif augmentor["name"] == "augment_root_trajectory":
                    motion = augment_root_trajectory(motion, augmentor["max_translation"], augmentor["probability"])

        return motion, source_rest_pose, target_rest_pose
