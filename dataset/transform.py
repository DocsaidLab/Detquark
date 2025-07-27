
from typing import Tuple

import albumentations as A
import cv2
import numpy as np


class DetectionImageAug:
    """
    Apply pixel-level and spatial augmentations to detection images.

    Args:
        image_size (Tuple[int, int]): Target image height and width.
        p (float): Probability of applying the augmentation pipeline.
    """

    def __init__(self, image_size: Tuple[int, int], p: float = 0.5):
        h, w = image_size

        # Define pixel-based transforms (noise, blur, color jitter)
        self.pixel_transform = A.Compose(
            transforms=[
                A.OneOf([
                    A.GaussianBlur(),
                    A.MotionBlur()
                ]),
                A.OneOf([
                    A.ISONoise(),
                    A.GaussNoise()
                ]),
                A.OneOf([
                    A.Emboss(),
                    A.Equalize(),
                    A.RandomSunFlare(src_radius=120)
                ]),
                A.OneOf([
                    A.ToGray(),
                    A.InvertImg(),
                    A.ChannelShuffle()
                ]),
                A.ColorJitter(),
            ],
            p=p
        )

        # Define spatial transforms that also adjust bounding boxes
        self.spatial_transform = A.Compose(
            transforms=[
                A.RandomSizedBBoxSafeCrop(height=h, width=w, p=p),
                A.ShiftScaleRotate(
                    shift_limit=0,
                    rotate_limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=p
                ),
                A.RandomRotate90(p=p),
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
                format='pascal_voc',    # (x_min, y_min, x_max, y_max)
                label_fields=['labels']
            )
        )

    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentations to the image and its corresponding bounding boxes and labels.

        Returns:
            Tuple containing the augmented image, bounding boxes, and labels.
        """
        # Apply pixel-level transforms
        image = self.pixel_transform(image=image)['image']
        # Apply spatial transforms (affects image, bboxes, labels)
        transformed = self.spatial_transform(
            image=image,
            bboxes=bboxes,
            labels=labels
        )
        return (
            transformed['image'],
            np.array(transformed['bboxes']),
            np.array(transformed['labels'])
        )
