from time import perf_counter_ns

import PIL.Image
import torch

from torch import nn
from torch.utils._pytree import tree_map_only

import torchvision.transforms.v2 as transforms_v2
from torchvision import datapoints, transforms as transforms_v1
from torchvision.transforms import functional as F_v1
from torchvision.transforms.v2 import functional as F_v2


class Pipeline(nn.Sequential):
    def __init__(self, transforms, *, supported_input_types, splat_input=False):
        super().__init__(*transforms)
        self.supported_input_types = supported_input_types
        self.splat_input = splat_input
        self._times = {}
        self.reset_times()

    def forward(self, sample):
        for transform in self:
            start = perf_counter_ns()
            sample = transform(*sample) if self.splat_input else transform(sample)
            stop = perf_counter_ns()
            self._times[transform].append(stop - start)
        return sample

    def reset_times(self):
        self._times = {transform: [] for transform in self}

    def extract_times(self):
        return {
            type(transform)
            .__name__: torch.tensor(self._times[transform], dtype=torch.float64)
            .mul_(1e-9)
            for transform in self
        }


class MaybeContiguousV1(nn.Module):
    def __init__(self, memory_format=torch.contiguous_format):
        super().__init__()
        self.memory_format = memory_format

    def forward(self, *inputs):
        return tree_map_only(
            torch.Tensor,
            lambda image: image.contiguous(memory_format=self.memory_format),
            inputs if len(inputs) > 1 else inputs[0],
        )


class RandomResizedCropWithoutResizeV1(transforms_v1.RandomResizedCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F_v1.crop(img, i, j, h, w)


class MaybePILToTensorV1(nn.Module):
    def forward(self, *inputs):
        return tree_map_only(
            PIL.Image.Image,
            F_v1.pil_to_tensor,
            inputs if len(inputs) > 1 else inputs[0],
        )


class MaybeContiguousV2(transforms_v2.Transform):
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
