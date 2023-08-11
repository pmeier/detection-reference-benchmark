import pathlib

from torch.hub import tqdm

from torchvision import datasets
from torchvision.transforms import functional as F_v1

# COCO_ROOT = "~/datasets/coco"
COCO_ROOT = "/data/disk1/MSCoco/"

__all__ = ["classification_dataset_builder", "detection_dataset_builder"]


def classification_dataset_builder(*, api_version, rng, num_samples):
    return [
        F_v1.to_pil_image(
            # average size of images in ImageNet
            torch.randint(0, 256, (3, 469, 387), dtype=torch.uint8, generator=rng),
        )
        for _ in range(num_samples)
    ]


def detection_dataset_builder(*, api_version, rng, num_samples):
    root = pathlib.Path(COCO_ROOT).expanduser().resolve()
    image_folder = str(root / "train2017")
    annotation_file = str(root / "annotations" / "instances_train2017.json")
    if api_version == "v1":
        dataset = CocoDetectionV1(image_folder, annotation_file, transforms=None)
    elif api_version == "v2":
        dataset = datasets.CocoDetection(image_folder, annotation_file)
    else:
        raise ValueError(f"Got {api_version=}")

    dataset = _coco_remove_images_without_annotations(dataset)

    idcs = torch.randperm(len(dataset), generator=rng)[:num_samples].tolist()
    print(f"Caching {num_samples} ({idcs[:3]} ... {idcs[-3:]}) COCO samples")
    return [dataset[idx] for idx in tqdm(idcs)]


# everything below is copy-pasted from
# https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py

import torch
import torchvision


class CocoDetectionV1(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        raise TypeError(
            f"This function expects dataset of type torchvision.datasets.CocoDetection, instead  got {type(dataset)}"
        )
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset
