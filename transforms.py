from time import perf_counter_ns

import torch
from torch import nn

import torchvision.transforms.v2 as transforms_v2
from torchvision import datapoints, transforms as transforms_v1
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
        MaybeContiguous = ToContiguousV1
        RandomResizedCropWithoutResize = RandomResizedCropWithoutResizeV1
    elif api_version == "v2":
        transforms = transforms_v2
        MaybeContiguous = ToContiguousV2
        RandomResizedCropWithoutResize = RandomResizedCropWithoutResizeV2
    else:
        raise RuntimeError(f"Got {api_version=}")

    pipeline = []

    if input_type in {"Tensor", "Datapoint"}:
        pipeline.append(MaybeContiguous())

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


class ToContiguousV1(nn.Module):
    def __init__(self, memory_format=torch.contiguous_format):
        super().__init__()
        self.memory_format = memory_format

    def forward(self, image):
        return image.contiguous(memory_format=self.memory_format)


class RandomResizedCropWithoutResizeV1(transforms_v1.RandomResizedCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F_v1.crop(img, i, j, h, w)


class ToContiguousV2(transforms_v2.Transform):
    _transformed_types = (torch.Tensor,)

    def __init__(self, memory_format=torch.contiguous_format):
        super().__init__()
        self.memory_format = memory_format

    def _transform(self, inpt, params):
        output = inpt.contiguous(memory_format=self.memory_format)
        if isinstance(inpt, datapoints._datapoint.Datapoint):
            output = type(inpt).wrap_like(inpt, output)
        return output


class RandomResizedCropWithoutResizeV2(transforms_v2.RandomResizedCrop):
    def _transform(self, inpt, params):
        return F_v2.crop(inpt, **params)
