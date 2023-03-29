import torch

from torchvision import datapoints
from torchvision.transforms import functional as F_v1


def classification_dataset_builder(*, input_type, api_version, rng, num_samples):
    type_converter = {
        "Tensor": lambda image: image,
        "PIL": F_v1.to_pil_image,
        "Datapoint": datapoints.Image,
    }[input_type]
    return [
        type_converter(
            torch.randint(0, 256, (3, height, width), dtype=torch.uint8, generator=rng)
        )
        # FIXME: make this more realistic
        for height, width in torch.randn((num_samples, 2), generator=rng)
        .mul_(100)
        .add_(400)
        .clamp_(100, 2_000)
        .int()
        .tolist()
    ]
