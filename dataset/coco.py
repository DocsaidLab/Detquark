from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import albumentations as A
import capybara as cb
import cv2
import numpy as np
import torch

DIR = cb.get_curdir(__file__)


class DetectionImageAug:

    def __init__(self, image_size: Tuple[int, int], p=0.5):
        h, w = image_size

        self.pixel_transform = A.Compose(
            transforms=[
                A.OneOf([
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ]),
                A.OneOf([
                    A.ISONoise(),
                    A.GaussNoise(),
                ]),
                A.OneOf([
                    A.Emboss(),
                    A.Equalize(),
                    A.RandomSunFlare(src_radius=120),
                ]),
                A.OneOf([
                    A.ToGray(),
                    A.InvertImg(),
                    A.ChannelShuffle(),
                ]),
                A.ColorJitter(),
            ],
            p=p
        )

        self.spacial_transform = A.Compose(
            transforms=[
                A.RandomSizedBBoxSafeCrop(
                    height=h,
                    width=w,
                    p=p
                ),
                A.ShiftScaleRotate(
                    shift_limit=0,
                    rotate_limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=p
                ),
                A.RandomRotate90(),
                A.Perspective(
                    scale=(0.05, 0.5),
                    keep_size=True,
                    fit_output=True,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=p
                ),
            ],
            p=p,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            )
        )

    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
    ):
        image = self.pixel_transform(image=image)['image']
        image, bboxes, labels = self.spacial_transform(
            image=image,
            bboxes=bboxes,
            labels=labels
        ).values()
        return image, bboxes, labels


class CoCoDataset:
    """
    A light weight dataset loader for COCO style splits + YOLO txt labels.

    每行標註格式：
        cls x1 y1 x2 y2 ...   # 4 or 2k floats (k ≥ 2), **皆為 0~1 範圍**
    """

    SPLITS = {
        "train": ("train2017.txt",  "train2017"),
        "val":   ("val2017.txt",    "val2017"),
        "test":  ("test-dev2017.txt", "test-dev2017"),
    }

    def __init__(
        self,
        root: Union[str, Path] = None,
        mode: str = 'train',  # 'train' | 'val' | 'test'
        image_size: Tuple[int, int] = (640, 640),
        transform: Callable = None,
        aug_ratio: float = 0,
        to_tensor: bool = False,
        **kwargs
    ) -> None:

        if mode not in self.SPLITS:
            raise ValueError(
                f"`mode` must be {list(self.SPLITS)}, but get `{mode}`")

        self.root = Path(root) if root else DIR.parent / "data"
        self.mode = mode
        self.image_size = image_size
        self.to_tensor = to_tensor

        self.transform = DetectionImageAug(
            image_size=self.image_size,
            p=aug_ratio
        )

        # Implement the following in the subclass
        self.dataset: List[Dict] = self._build_dataset()

    def _parse_line(self, line: str) -> tuple[int, np.ndarray]:
        """
        將一行文字解析成 (class, bbox)。
        bbox 為 [x_min,y_min,x_max,y_max]，皆為 0~1 浮點。
        """
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError("標註長度不足 5 個元素")

        cls = int(parts[0])
        coords = np.asarray(list(map(float, parts[1:])), dtype=np.float32)

        if coords.size == 4:
            # (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
            cx, cy, w, h = coords
            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        else:
            # 多邊形 (x1,y1, x2,y2, …) -> 外接方框
            xs = coords[0::2]
            ys = coords[1::2]
            box = np.array(
                [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32
            )

        # clip 到 [0,1] 以防極端值
        box = np.clip(box, 0.0, 1.0)

        return cls, box

    def _build_dataset(self) -> List[Dict]:

        # 讀取「檔名清單」
        fs_name, label_dir = self.SPLITS[self.mode]
        txt_list = self.root / "coco" / fs_name
        txt_list = txt_list.read_text().strip().splitlines()

        dataset = []
        for rel in cb.Tqdm(txt_list, desc=f"Scanning {self.mode} set"):
            img_path = self.root / "coco" / rel
            label_path = self.root / "coco" / "labels" / \
                label_dir / Path(rel).with_suffix(".txt").name

            img = cb.imread(img_path)
            if img is None:
                print(f"[WARNING] 讀不到影像：{img_path}")
                continue
            h, w = img.shape[:2]

            # 解析標註
            if label_path.exists():
                lines = label_path.read_text().strip().splitlines()
                infos = []
                for ln in lines:
                    try:
                        cls, box = self._parse_line(ln)
                        infos.append((cls, box))
                    except Exception as e:
                        print(f"[WARNING] {label_path.name} 異常行：{ln} ({e})")
                if infos:
                    classes = np.fromiter(
                        (c for c, _ in infos), dtype=np.int64, count=len(infos))
                    bboxes = np.stack([b for _, b in infos],
                                      axis=0)         # (N,4) 0‑1
                else:
                    classes = np.empty((0,), dtype=np.int64)
                    bboxes = np.empty((0, 4), dtype=np.float32)
            else:
                classes = np.empty((0,), dtype=np.int64)
                bboxes = np.empty((0, 4), dtype=np.float32)

            dataset.append(
                dict(
                    img_path=img_path,
                    width=w,
                    height=h,
                    bboxes=bboxes,
                    labels=classes,
                )
            )

            if len(dataset) > 10:
                break

        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def _pad(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        將圖像補成正方形，並調整 bbox 座標。

        Args:
            img (np.ndarray): 原始圖像 (H, W, C)
            boxes (np.ndarray): (N, 4), xyxy 格式的絕對座標

        Returns:
            padded_img (np.ndarray): 補成正方形後的圖像
            padded_boxes (np.ndarray): 調整後的 boxes[xyxy]
        """
        h, w = img.shape[:2]
        size = max(h, w)

        pad_h = size - h
        pad_w = size - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Padding image
        padded_img = cv2.copyMakeBorder(
            img,
            top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Adjust bbox
        padded_boxes = boxes.copy()
        padded_boxes[:, [0, 2]] += left
        padded_boxes[:, [1, 3]] += top

        return padded_img, padded_boxes

    def _resize(self, img: np.ndarray, boxes: np.ndarray):

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

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:

        infos = self.dataset[idx]

        img = cb.imread(infos['img_path'])
        labels = infos['labels']
        boxes = infos['bboxes']

        boxes[:, [0, 2]] *= infos['width']
        boxes[:, [1, 3]] *= infos['height']

        # pad & resize
        img, boxes = self._pad(img, boxes)
        img, boxes = self._resize(img, boxes)

        # image aug
        img, boxes, labels = self.transform(
            image=img,
            bboxes=boxes,
            labels=labels
        )

        #
        if self.to_tensor:
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32) / 255.

            ih, iw = self.image_size
            boxes[:, [0, 2]] /= iw
            boxes[:, [1, 3]] /= ih
            boxes = boxes.astype(np.float32)

            labels = labels.astype(np.int64)

        return {
            'img': img,
            'img_size': self.image_size,
            'img_path': str(infos['img_path']),
            'width': infos['width'],
            'height': infos['height'],
            'boxes': boxes,
            'labels': labels,
        }


def detection_collate_fn(batch: List[Dict[str, Any]]):
    """
    Collate function for object‑detection batches.

    Args:
        batch: List of samples, each is a dict with keys
            'img'      : HWC np.ndarray or CHW tensor
            'boxes'    : (N,4) np.ndarray of xyxy absolute coords
            'labels'   : (N,)  np.ndarray of category IDs
            'width'    : original image width
            'height'   : original image height
            'img_size' : tuple (W,H) after resize
            (others ignored)

    Returns:
        imgs:   Tensor of shape (B,3,H,W), float32
        targets: List[dict], each with keys
            'boxes'    : Tensor (N,4) float32
            'labels'   : Tensor (N,)  int64
            'image_id' : Tensor([i])     int64
            'orig_size': Tensor([H0,W0]) int64
            'size'     : Tensor([H,W])   int64
    """
    # Stack images
    imgs = []
    for sample in batch:
        img = sample['img']
        if isinstance(img, np.ndarray):
            # assume HWC uint8 or float, convert to float32 tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)

    # Build targets
    targets: List[Dict[str, Any]] = []
    for i, sample in enumerate(batch):
        boxes = torch.from_numpy(sample['boxes']).float()
        labels = torch.from_numpy(sample['labels']).long()
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([i], dtype=torch.int64),
            'orig_size': torch.tensor([sample['height'], sample['width']], dtype=torch.int64),
            'size': torch.tensor(sample['img_size'], dtype=torch.int64),
        }
        targets.append(target)

    return imgs, targets
