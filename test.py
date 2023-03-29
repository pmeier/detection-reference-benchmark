import torch
from torchvision.transforms.functional import pil_to_tensor
from utils import (
    dataset_v1,
    dataset_v2,
    transform_v1,
    transform_v2,
)
import sys


def test_datasets(root):
    v1 = dataset_v1(root)
    v2 = dataset_v2(root)

    assert len(v1) == len(v2)

    for idx in [0, 500, 1000, len(v1) - 1]:
        image_v1, target_v1 = v1[idx]
        image_v2, target_v2 = v2[idx]

        torch.testing.assert_close(pil_to_tensor(image_v1), pil_to_tensor(image_v2))

        assert int(target_v1["image_id"]) == target_v2["image_id"]
        assert (target_v1["labels"] == target_v2["labels"]).all()


def test_transforms(root):
    # FIXME
    pass


if __name__ == "__main__":
    root = sys.argv[1]

    test_datasets(root)
    test_transforms(root)
