import pytest
import time

import torch

from retarget.dataset import SkIPDataset
from retarget.augment import RestPoseAugmentor
from retarget.utils.Animation import forward_rotations
from retarget.utils.Quaternions import Quaternions, d6_2_quat, rotmat_2_d6

def test_dataset_loading_performance():
    """
    Test the performance of loading and iterating through the dataset.
    
    This test measures how long it takes to load items from the dataset
    and ensures the loading time is within acceptable limits.
    """
    # Skip this test if the dataset directory doesn't exist
    try:
        dataset = SkIPDataset(directory="./data/testing")
    except:
        pytest.skip("Dataset directory not found, skipping test")
    
    # Measure the time it takes to iterate through the dataset
    start_time = time.time()
    
    # Only process a few items to keep the test fast
    max_items = min(10, len(dataset))
    for i in range(max_items):
        item = dataset[i]
    
    end_time = time.time()
    
    time_taken = end_time - start_time
    time_per_item = time_taken / max_items
    
    # Print the performance information
    print(f"Time taken: {time_taken:.2g} seconds for {max_items} items")
    print(f"Average time per item: {time_per_item:.2g} seconds")
    
    max_time_per_item = 0.01
    # Test passes if we can iterate through the dataset within the time limit
    assert time_per_item < max_time_per_item


def test_restpose_augmentor_transformation():
    """
    Test that the RestPoseAugmentor correctly transforms poses.
    
    This test verifies that:
    1. The root position remains unchanged after augmentation
    2. The rotations are modified by the augmentation
    3. The rest pose is different after augmentation
    """
    dataset = SkIPDataset(directory="./data/testing")
    restpose_augmentor = RestPoseAugmentor(20, 180)
    
    max_items = min(10, len(dataset))
    for i in range(max_items):
        item = dataset[i][0][0]
        item_aug, _ = restpose_augmentor(item)

        fk_pose = forward_rotations(
            item_aug.parents, item_aug.offsets, Quaternions(d6_2_quat(item_aug.d6)), item_aug.position[0]
        )
        item_aug.pos = torch.Tensor(fk_pose)

        assert torch.allclose(item.position, item_aug.position)

        # check that the rotations are different
        assert not torch.allclose(item.d6, item_aug.d6)

        # check that rest pose is not the same
        assert not torch.allclose(item.pos, item_aug.pos)

def test_restpose_augmentor_consistency():
    """
    Test that the RestPoseAugmentor produces consistent results with the same seed.
    
    This test verifies that multiple calls to the augmentor with the same seed
    produce identical transformations, ensuring deterministic behavior.
    """
    dataset = SkIPDataset(directory="./data/testing")
    restpose_augmentor = RestPoseAugmentor(20, 180)
    
    items = dataset[0][0]
    
    aug_d6s = []
    for item in items:
        rotmat = torch.eye(3).unsqueeze(0).repeat(item.d6.shape[0], 1, 1)
        d6 = rotmat_2_d6(rotmat)
        item.d6 = d6

        item_aug, _ = restpose_augmentor(item)
        aug_d6s.append(item_aug.d6)

    # all the augmentations should be the same
    for aug_d6 in aug_d6s:
        assert torch.allclose(aug_d6s[0], aug_d6)

def test_restpose_augmentor_randomization():
    """
    Test that the RestPoseAugmentor produces different results after reset.
    
    This test verifies that calling reset() on the augmentor changes the random seed,
    resulting in different transformations for each item, ensuring proper randomization.
    """
    dataset = SkIPDataset(directory="./data/testing")
    restpose_augmentor = RestPoseAugmentor(20, 180)
    
    items = dataset[0][0]
    
    aug_d6s = []
    for item in items:
        rotmat = torch.eye(3).unsqueeze(0).repeat(item.d6.shape[0], 1, 1)
        d6 = rotmat_2_d6(rotmat)
        item.d6 = d6

        item_aug, _ = restpose_augmentor(item)
        aug_d6s.append(item_aug.d6)

        restpose_augmentor.reset()

    # all the augmentations should be different
    for i in range(len(aug_d6s)):
        for j in range(i + 1, len(aug_d6s)):
            if i == j:
                continue

            if torch.allclose(aug_d6s[i], aug_d6s[j]):
                if torch.allclose(aug_d6s[i], d6) and torch.allclose(aug_d6s[j], d6):
                    continue

            assert not torch.allclose(aug_d6s[i], aug_d6s[j])