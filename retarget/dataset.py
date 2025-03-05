import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

from torch_geometric.data import Data, Dataset

from retarget.augment import (Augmentions, GlobalSkeletonAugmentor,
                              RestPoseAugmentor, XZTranslationAugmentor)
from retarget.utils.Animation import Animation
from retarget.utils.BVH import load


class SkIPDataset(Dataset):
    """
    Dataset class for loading and processing animations.

    This dataset loads BVH animation files from a directory structure organized by character,
    converts them to graph representations, and provides methods to access and augment the data.
    """

    def __init__(
        self,
        directory: str,
        mode: str = "train",
        ground_feet: bool = False,
        consequtive_frames: int = 8,
        augmentors: List[str] = [RestPoseAugmentor(), GlobalSkeletonAugmentor(), XZTranslationAugmentor()],
        feature_list: List[str] = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        directory : str
            Path to the directory containing the animation files organized by character
        mode : str, default="train"
            Dataset mode, either "train" or "test"
        ground_feet : bool, default=False
            Whether to ground the feet of the character
        consequtive_frames : int, default=8
            Number of consecutive frames to return as a batch
        feature_list : List[str], default=None
            List of features to include in the graph
        """
        super().__init__()

        animations: Dict[str, Dict[str, Animation]] = {}
        frame_times: Dict[str, Dict[str, float]] = {}

        self.consequtive_frames = consequtive_frames
        self.feature_list = feature_list

        augmentor_name = lambda augmentor: re.sub(r'(?<!^)(?=[A-Z])', '_', augmentor.__class__.__name__).lower()
        self.augmentor = Augmentions(**{
            augmentor_name(augmentor): augmentor for augmentor in augmentors
        })

        characters = list(Path(directory).glob("*"))

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

        self.animations: List[Animation] = list(animations.values())
        self.frame_times: List[List[float]] = list(frame_times.values())
        # flatten the lists
        self.animations = [
            animation for character in self.animations for animation in character
        ]
        self.frame_time: List[float] = [
            frame_time for character in self.frame_times for frame_time in character
        ]
        # Create Empty list to fill with frames as graph
        self.data: List[Data] = []
        self.data_idx: List[int] = []

        # Initialize empty list to put in the previous frames of a given frame
        # This is used for the velocity loss
        # Also Initialize empty list with frame times between frames
        # self.data_prev[idx] stores the offset frame from the original frame.
        # if mode == "train":
        # self.time = []

        for animation, time in zip(self.animations, self.frame_time):
            if 1 / time > 100:
                animation_graphs = animation.as_graph(stride=4, feature_list=self.feature_list)
            else:
                animation_graphs = animation.as_graph(feature_list=self.feature_list)

            length_animation = len(animation_graphs)

            if length_animation < 8:
                continue

            n_graphs = length_animation // self.consequtive_frames * self.consequtive_frames
            animation_graphs = animation_graphs[:n_graphs]

            total_frames_currently = len(self.data)

            self.data = self.data + animation_graphs
            self.data_idx += list(
                range(total_frames_currently, total_frames_currently + n_graphs)
            )

            # if self.mode == "train":
            #    self.time = self.time + [time] * n_graphs

        print("=== Dataset Summary ===")
        print(
            f"Loaded {len(self.animations)} animation clips for {len(animations)} characters"
        )
        # print summary for each character
        for character, animations in animations.items():
            print(f"{character}")
            print(f"    - Animations: {len(animations) // self.consequtive_frames * consequtive_frames}")
        print(f"Total frames: {len(self.data):,}")
        print("===============================")

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        return len(self.data) // self.consequtive_frames

    def get_one_item(
        self, idx: int
    ) -> Union[Data, Tuple[Data, Data, Data, float]]:
        """
        Get a single item from the dataset.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve
        augmentor : Optional[Augmentor], default=None
            Augmentor to apply to the item if in train mode

        Returns
        -------
        Union[Data, Tuple[Data, Data, Data, float]]
            If in train mode, returns (encoder_item, decoder_item, encoder_item_translated, frame_time)
            Otherwise, returns the item
        """
        item = self.data[self.data_idx[idx]].clone()

        if self.mode == "train":
            # If the mode is "train", then also define the frame time
            frame_time = 1 / 30  # self.time[idx]

            encoder_item, decoder_item, encoder_item_translated = self.augmentor(item)
            return encoder_item, decoder_item, encoder_item_translated, frame_time
        return item

    def __getitem__(
        self, idx: int
    ) -> Union[List[Data], Tuple[List[Data], List[Data], List[Data]]]:
        """
        Get a batch of consecutive items from the dataset.

        Parameters
        ----------
        idx : int
            Index of the batch to retrieve

        Returns
        -------
        Union[List[Data], Tuple[List[Data], List[Data], List[Data]]]
            If in train mode, returns (encoder_items, decoder_items, encoder_items_translated)
            Otherwise, returns encoder_items
        """
        encoder_items: List[Data] = []
        if self.mode == "train":
            frame_time: List[float] = []
            decoder_items: List[Data] = []
            encoder_items_translated: List[Data] = []

            self.augmentor.reset()

            for i in range(self.consequtive_frames):
                encoder_item, decoder_item, encoder_item_translated, f_time = (
                    self.get_one_item(idx * self.consequtive_frames + i)
                )
                encoder_items.append(encoder_item)
                decoder_items.append(decoder_item)
                encoder_items_translated.append(encoder_item_translated)
                frame_time.append(f_time)

            return (
                encoder_items,
                decoder_items,
                encoder_items_translated,
            )  # , frame_time

        for i in range(self.consequtive_frames):
            it = self.get_one_item(idx * self.consequtive_frames + i)
            encoder_items.append(it)

        return encoder_items
