import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import capybara as cb
import cv2
import numpy as np
import torch

from .transform import DetectionImageAug

# Determine current directory of this script
DIR = cb.get_curdir(__file__)

DEBUG = True


class CoCoDataset:
    """
    Lightweight COCO-style dataset loader that supports YOLO-format labels.

    Each annotation line:
        class x1 y1 x2 y2 ...    # Coordinates normalized to [0,1]
    """

    # Mapping of dataset splits to file lists and directory names
    SPLITS = {
        "train": ("train2017.txt",  "train2017"),
        "val":   ("val2017.txt",    "val2017"),
        "test":  ("test-dev2017.txt", "test-dev2017"),
    }

    def __init__(
        self,
        root: Union[str, Path] = None,
        mode: str = 'train',               # One of 'train', 'val', 'test'
        image_size: Tuple[int, int] = (640, 640),
        transform: Callable = None,        # Augmentation callable
        aug_ratio: float = 0.0,            # Probability for default augmentations
        return_tensor: bool = False,       # Return tensor or numpy
        **kwargs
    ) -> None:
        # Validate dataset mode
        if mode not in self.SPLITS:
            raise ValueError(
                f"`mode` must be one of {list(self.SPLITS)}, got '{mode}'")

        # Set root directory to user-specified or default data folder
        self.root = Path(root) if root else DIR.parent / "data"
        self.mode = mode
        self.image_size = image_size
        self.return_tensor = return_tensor

        # Assign transform: use default DetectionImageAug if none provided
        if transform is None:
            self.transform = DetectionImageAug(
                image_size=self.image_size, p=aug_ratio)
        else:
            self._check_transform(transform)
            self.transform = transform

        # Build the dataset list of dict entries
        self.dataset: List[Dict] = self._build_dataset()

    def _check_transform(self, transform: Callable) -> None:
        """
        Validate a user-provided transform callable for compatibility.

        Requirements:
          1. Must be a callable instance, not a class/type.
          2. Must accept keyword args: image, bboxes, labels.
          3. Must return a tuple: (image, bboxes, labels).
        """
        if not callable(transform) or inspect.isclass(transform):
            raise TypeError(
                "`transform` must be an instantiated callable, not a class/type.")
        sig = inspect.signature(transform)
        if not {'image', 'bboxes', 'labels'}.issubset(sig.parameters):
            raise ValueError(
                "`transform` must accept keyword args 'image', 'bboxes', 'labels'.")

        # Test transform with dummy data
        h, w = self.image_size
        dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
        dummy_bboxes = np.zeros((0, 4), dtype=np.float32)
        dummy_labels = np.zeros((0,), dtype=np.int64)
        try:
            out = transform(image=dummy_img,
                            bboxes=dummy_bboxes, labels=dummy_labels)
        except Exception as e:
            raise ValueError(f"`transform` call failed: {e}")
        if not (isinstance(out, tuple) and len(out) == 3):
            raise ValueError(
                "`transform` must return a tuple (image, bboxes, labels).")

    def _parse_line(self, line: str) -> Tuple[int, np.ndarray]:
        """
        Parse a single annotation line into class and bounding box.

        Returns:
            cls (int): Class index.
            box (np.ndarray): [x_min, y_min, x_max, y_max] normalized to [0,1].
        """
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError("Annotation line must have at least 5 elements.")

        # First element is class, remaining are coordinates
        cls = int(parts[0])
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32)

        # If 4 coords: assume (cx, cy, w, h) format
        if coords.size == 4:
            cx, cy, w, h = coords
            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        else:
            # If more coords: assume polygon, compute bounding box
            xs, ys = coords[0::2], coords[1::2]
            box = np.array([min(xs), min(ys), max(xs),
                           max(ys)], dtype=np.float32)

        # Clip coordinates to [0,1]
        return cls, np.clip(box, 0.0, 1.0)

    def _build_dataset(self) -> List[Dict[str, Any]]:
        """
        Scan image list and corresponding label files to build dataset entries.

        Each entry contains:
            img_path, width, height, bboxes, labels
        """
        fs_name, label_dir = self.SPLITS[self.mode]
        txt_list = (self.root / "coco" / fs_name).read_text().splitlines()

        dataset = []
        for rel in cb.Tqdm(txt_list, desc=f"Scanning {self.mode} split"):
            img_path = self.root / "coco" / rel
            label_path = self.root / "coco" / "labels" / \
                label_dir / Path(rel).with_suffix(".txt").name

            img = cb.imread(img_path)
            if img is None:
                print(f"[WARNING] Could not load image: {img_path}")
                continue
            h, w = img.shape[:2]

            # Parse label file if it exists
            if label_path.exists():
                infos = []
                for ln in label_path.read_text().splitlines():
                    try:
                        infos.append(self._parse_line(ln))
                    except Exception as e:
                        print(f"[WARNING] Invalid annotation {ln}: {e}")
                if infos:
                    classes = np.array([c for c, _ in infos], dtype=np.int64)
                    bboxes = np.vstack([b for _, b in infos])
                else:
                    classes = np.empty((0,), dtype=np.int64)
                    bboxes = np.empty((0, 4), dtype=np.float32)
            else:
                classes = np.empty((0,), dtype=np.int64)
                bboxes = np.empty((0, 4), dtype=np.float32)

            dataset.append({
                'img_path': img_path,
                'width': w,
                'height': h,
                'bboxes': bboxes,
                'labels': classes,
            })

            if DEBUG:
                if len(dataset) > 1000:
                    break

        return dataset

    def __len__(self) -> int:
        """Return number of dataset entries."""
        return len(self.dataset)

    def _pad(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad image to square shape and adjust bounding boxes accordingly.

        Returns:
            padded_img   : Image padded to max(height, width).
            padded_boxes : Boxes shifted by padding offsets.
        """
        h, w = img.shape[:2]
        size = max(h, w)
        pad_h, pad_w = size - h, size - w
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)

        padded_img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Update box coordinates
        padded_boxes = boxes.copy()
        padded_boxes[:, [0, 2]] += left
        padded_boxes[:, [1, 3]] += top
        return padded_img, padded_boxes

    def _resize(self, img: np.ndarray, boxes: np.ndarray):
        """
        Resize image and scale bounding boxes to new dimensions.
        """
        if self.image_size is None:
            return img, boxes

        h0, w0 = img.shape[:2]
        new_w, new_h = self.image_size
        if (h0, w0) == (new_h, new_w):
            return img, boxes

        scale_x, scale_y = new_w / w0, new_h / h0
        img = cb.imresize(img, size=(new_h, new_w),
                          interpolation=cb.INTER.BILINEAR)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        return img, boxes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve and process a dataset entry by index.

        Steps:
            1. Load image and raw labels.
            2. Convert normalized boxes to absolute coords.
            3. Pad and resize image/boxes.
            4. Apply transforms.
            5. Optionally normalize and convert to tensors.
        """
        infos = self.dataset[idx]

        img = cb.imread(infos['img_path'])
        labels = infos['labels']
        boxes = infos['bboxes'].copy()

        # Convert normalized boxes to pixel coords
        boxes[:, [0, 2]] *= infos['width']
        boxes[:, [1, 3]] *= infos['height']

        # Pad and resize
        img, boxes = self._pad(img, boxes)
        img, boxes = self._resize(img, boxes)

        # Apply augmentations
        img, boxes, labels = self.transform(
            image=img, bboxes=boxes, labels=labels)

        # Ensure correct shapes
        boxes = boxes.reshape(-1, 4).astype(np.float32)
        labels = labels.astype(np.int64)

        if self.return_tensor:
            # Convert image to CHW float32 tensor [0,1]
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

            # Normalize boxes back to [0,1]
            iw, ih = self.image_size
            if boxes.size:
                boxes[:, [0, 2]] /= iw
                boxes[:, [1, 3]] /= ih

        return {
            'img': img,
            'img_size': self.image_size,
            'img_path': str(infos['img_path']),
            'width': infos['width'],
            'height': infos['height'],
            'boxes': boxes,
            'labels': labels,
        }


def coco_collate_fn(batch: List[Dict[str, Any]]):
    """
    Collate function for DataLoader to batch detection samples.

    Args:
        batch (List[dict]): Each sample dict must contain:
            'img'      : CHW tensor or HWC ndarray
            'boxes'    : (N,4) ndarray of absolute coords
            'labels'   : (N,) ndarray of class IDs
            'width'    : original image width
            'height'   : original image height
            'img_size' : (W,H) after resize

    Returns:
        imgs   : Tensor (B,3,H,W)
        targets: List[dict] with keys:
            'boxes'    : Tensor (N,4)
            'labels'   : Tensor (N,)
            'image_id' : Tensor([i])
            'orig_size': Tensor([H0,W0])
            'size'     : Tensor([H,W])
    """
    # Stack images into a single tensor
    imgs = torch.stack(
        [torch.tensor(sample['img'], dtype=torch.float32) for sample in batch], dim=0)

    targets = []
    for i, sample in enumerate(batch):
        boxes = torch.tensor(sample['boxes'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.int64)
        targets.append({
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([i], dtype=torch.int64),
            'orig_size': torch.tensor([sample['height'], sample['width']], dtype=torch.int64),
            'size': torch.tensor(sample['img_size'], dtype=torch.int64),
        })

    return imgs, targets
