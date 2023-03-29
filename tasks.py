import torch

import torchvision.transforms.v2 as transforms_v2
from torchvision import transforms as transforms_v1

from datasets import classification_dataset_builder
from transforms import (
    MaybeContiguousV1,
    MaybeContiguousV2,
    MaybePILToTensorV1,
    Pipeline,
    RandomResizedCropWithoutResizeV1,
    RandomResizedCropWithoutResizeV2,
)

TASKS = {
    "classification-simple": {
        "v1": (
            Pipeline(
                [
                    MaybeContiguousV1(),
                    RandomResizedCropWithoutResizeV1(224),
                    transforms_v1.Resize(224, antialias=True),
                    transforms_v1.RandomHorizontalFlip(p=0.5),
                    MaybePILToTensorV1(),
                    transforms_v1.ConvertImageDtype(torch.float32),
                    transforms_v1.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ],
                supported_input_types={"Tensor", "PIL"},
            ),
            classification_dataset_builder,
        ),
        "v2": (
            Pipeline(
                [
                    MaybeContiguousV2(),
                    RandomResizedCropWithoutResizeV2(224),
                    transforms_v2.Resize(224, antialias=True),
                    transforms_v2.RandomHorizontalFlip(p=0.5),
                    transforms_v2.PILToTensor(),
                    transforms_v2.ConvertDtype(torch.float32),
                    transforms_v2.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ],
                supported_input_types=("Tensor", "PIL", "Datapoint"),
            ),
            classification_dataset_builder,
        ),
    },
}


def make_task(name, *, input_type, api_version, dataset_rng, num_samples):
    pipeline, dataset_builder = TASKS[name][api_version]
    if input_type not in pipeline.supported_input_types:
        return None

    dataset = dataset_builder(
        input_type=input_type,
        api_version=api_version,
        rng=dataset_rng,
        num_samples=num_samples,
    )

    return pipeline, dataset
