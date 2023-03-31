from time import perf_counter_ns

import torch

import torchvision.transforms.v2 as transforms_v2
from torchvision import transforms as transforms_v1
from torchvision.transforms import functional as F_v1
from torchvision.transforms.v2 import functional as F_v2


class Pipeline:
    def __init__(self, transforms):
        self.transforms = transforms
        self._times = {}
        self.reset_times()

    def __call__(self, sample):
        for transform in self.transforms:
            start = perf_counter_ns()
            sample = (
                transform(*sample) if isinstance(sample, tuple) else transform(sample)
            )
            stop = perf_counter_ns()
            self._times[transform].append(stop - start)
        return sample

    def reset_times(self):
        self._times = {transform: [] for transform in self.transforms}

    def extract_times(self):
        return {
            type(transform)
            .__name__: torch.tensor(self._times[transform], dtype=torch.float64)
            .mul_(1e-9)
            for transform in self.transforms
        }


def classification_simple_pipeline_builder(*, input_type, api_version):
    if input_type == "Datapoint" and api_version == "v1":
        return None

    if api_version == "v1":
        transforms = transforms_v1
        RandomResizedCropWithoutResize = RandomResizedCropWithoutResizeV1
    elif api_version == "v2":
        transforms = transforms_v2
        RandomResizedCropWithoutResize = RandomResizedCropWithoutResizeV2
    else:
        raise RuntimeError(f"Got {api_version=}")

    pipeline = []

    if input_type == "Tensor":
        pipeline.append(transforms.PILToTensor())
    elif input_type == "Datapoint":
        pipeline.append(transforms.ToImageTensor())

    pipeline.extend(
        [
            RandomResizedCropWithoutResize(224),
            transforms.Resize(224, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    if input_type == "PIL":
        pipeline.append(transforms.PILToTensor())

    pipeline.extend(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return Pipeline(pipeline)


def classification_complex_pipeline_builder(*, input_type, api_version):
    if input_type == "Datapoint" and api_version == "v1":
        return None

    if api_version == "v1":
        transforms = transforms_v1
        RandomResizedCropWithoutResize = RandomResizedCropWithoutResizeV1
    elif api_version == "v2":
        transforms = transforms_v2
        RandomResizedCropWithoutResize = RandomResizedCropWithoutResizeV2
    else:
        raise RuntimeError(f"Got {api_version=}")

    pipeline = []

    if input_type == "Tensor":
        pipeline.append(transforms.PILToTensor())
    elif input_type == "Datapoint":
        pipeline.append(transforms.ToImageTensor())

    pipeline.extend(
        [
            RandomResizedCropWithoutResize(224),
            transforms.Resize(224, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        ]
    )

    if input_type == "PIL":
        pipeline.append(transforms.PILToTensor())

    pipeline.extend(
        [
            transforms.RandomErasing(p=0.2),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return Pipeline(pipeline)


class RandomResizedCropWithoutResizeV1(transforms_v1.RandomResizedCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F_v1.crop(img, i, j, h, w)


class RandomResizedCropWithoutResizeV2(transforms_v2.RandomResizedCrop):
    def _transform(self, inpt, params):
        return F_v2.crop(inpt, **params)
