
from retarget.dataset import SkIPDataset
import time

def test_augmentor():
    dataset = SkIPDataset(directory="./data/dataset")

    start_time = time.time()
    for item in dataset:
        item

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2g} seconds")
    
    assert False