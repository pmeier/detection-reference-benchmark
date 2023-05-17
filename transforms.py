import functools
import re
from time import perf_counter_ns
from types import SimpleNamespace

import torchvision.transforms.v2 as transforms_v2
from torchvision import datasets, transforms as transforms_v1
from torchvision.datapoints._dataset_wrapper import WRAPPER_FACTORIES
from torchvision.transforms import functional as F_v1
from torchvision.transforms.v2 import functional as F_v2

__all__ = [
    "classification_simple_pipeline_builder",
    "classification_complex_pipeline_builder",
    "detection_ssdlite_pipeline_builder",
]


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
        return [
            (
                re.match(r"(?P<name>.*?)(_?[vV][1,2])?$", type(transform).__name__)[
                    "name"
                ],
                torch.tensor(self._times[transform], dtype=torch.float64).mul_(1e-9),
            )
            for transform in self.transforms
        ]


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
            transforms.Resize((224, 224), antialias=True),
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
            transforms.Resize((224, 224), antialias=True),
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


def detection_ssdlite_pipeline_builder(*, input_type, api_version):
    if input_type == "Datapoint" and api_version == "v1":
        return None

    pipeline = []
    if api_version == "v1":
        pipeline.append(ConvertCocoPolysToMaskV1())

        if input_type == "Tensor":
            pipeline.append(PILToTensorV1())

        pipeline.extend(
            [
                RandomIoUCropV1(),
                RandomHorizontalFlipV1(p=0.5),
            ]
        )

        if input_type == "PIL":
            pipeline.append(PILToTensorV1())

        pipeline.append(ConvertImageDtypeV1(torch.float))

    elif api_version == "v2":
        pipeline.extend(
            [
                WrapCocoSampleForTransformsV2(),
                transforms_v2.ClampBoundingBox(),
                transforms_v2.SanitizeBoundingBox(),
            ]
        )

        if input_type == "Tensor":
            pipeline.append(transforms_v2.PILToTensor())
        elif input_type == "Datapoint":
            pipeline.append(transforms_v2.ToImageTensor())

        pipeline.extend(
            [
                transforms_v2.RandomIoUCrop(),
                transforms_v2.RandomHorizontalFlip(p=0.5),
            ]
        )

        if input_type == "PIL":
            pipeline.append(transforms_v2.PILToTensor())

        pipeline.extend(
            [
                transforms_v2.ConvertDtype(torch.float),
                transforms_v2.SanitizeBoundingBox(),
            ]
        )
    else:
        raise ValueError(f"Got {api_version=}")

    return Pipeline(pipeline)


class RandomResizedCropWithoutResizeV1(transforms_v1.RandomResizedCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F_v1.crop(img, i, j, h, w)


class RandomResizedCropWithoutResizeV2(transforms_v2.RandomResizedCrop):
    def _transform(self, inpt, params):
        return F_v2.crop(inpt, **params)


class WrapCocoSampleForTransformsV2:
    def __init__(self):
        wrapper_factory = WRAPPER_FACTORIES[datasets.CocoDetection]
        # The v2 wrapper depends on the `.ids` attribute of a `CocoDetection` dataset.
        # However, this is eliminated above while filtering out images without
        # annotations. Thus, we fake it here
        mock_dataset = SimpleNamespace(ids=["invalid"])
        wrapper = wrapper_factory(mock_dataset, target_keys=None)
        # The wrapper gets passed the index alongside the sample to wrap. The former is
        # only used to retrieve the image ID by accessing the `.ids` attribute. Thus, we
        # need to use any value so `.ids[idx]` works.
        self.wrapper = functools.partial(wrapper, 0)

    def __call__(self, image, target):
        return self.wrapper((image, target))


# everything below is copy-pasted from
# https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py

from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from pycocotools import mask as coco_mask
from torch import nn, Tensor


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMaskV1:
    def __call__(self, image, target):
        w, h = image.size

        try:
            image_id = target["image_id"]
        except:
            raise
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


class RandomHorizontalFlipV1(transforms_v1.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F_v1.hflip(image)
            if target is not None:
                _, _, width = F_v1.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


class PILToTensorV1(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F_v1.pil_to_tensor(image)
        return image, target


class ConvertImageDtypeV1(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F_v1.convert_image_dtype(image, self.dtype)
        return image, target


class RandomIoUCropV1(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(
                    f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions."
                )
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F_v1.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if (
                min_jaccard_overlap >= 1.0
            ):  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (
                    (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                )
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes,
                    torch.tensor(
                        [[left, top, right, bottom]],
                        dtype=boxes.dtype,
                        device=boxes.device,
                    ),
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F_v1.crop(image, top, left, new_h, new_w)

                return image, target
