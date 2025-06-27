from typing import Any, Dict, List

import torch


def coco_collate_fn(batch: List[Dict[str, Any]]):
    """
    Collate for object detection batches:
      - stacks all images into a (B, C, H, W) tensor
      - leaves targets as a list of dicts with 'boxes' & 'labels'
    """
    imgs = [item["img"] if isinstance(item["img"], torch.Tensor)
            else torch.from_numpy(item["img"]) for item in batch]
    imgs = torch.stack(imgs, dim=0)

    targets: List[Dict[str, Any]] = []
    for i, item in enumerate(batch):
        t = {
            "boxes": torch.as_tensor(item["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(item["labels"], dtype=torch.int64),
            "image_id": torch.tensor([i]),
            "orig_size": torch.tensor([item["height"], item["width"]]),
            "size": torch.tensor(item["img_size"]),
        }
        targets.append(t)

    return imgs, targets
