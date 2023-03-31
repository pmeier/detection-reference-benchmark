import torch

from torchvision.transforms import functional as F_v1


def classification_dataset_builder(*, input_type, api_version, rng, num_samples):
    return [
        F_v1.to_pil_image(
            # average size of images in ImageNet
            torch.randint(0, 256, (3, 469, 387), dtype=torch.uint8, generator=rng)
        )
        for _ in range(num_samples)
    ]
