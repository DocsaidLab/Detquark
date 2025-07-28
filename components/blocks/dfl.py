import torch
import torch.nn as nn


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) integral module.

    This module implements the integral operation described in the paper
    "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes
    for Dense Object Detection" (https://arxiv.org/abs/2006.04388).

    The purpose of DFL is to model bounding box regression as a
    discrete probability distribution, allowing the network to predict
    more accurate and stable localization results by learning a distribution
    over possible offsets rather than a single point estimate.

    Design:
        - Uses a fixed convolutional layer with weights representing discrete
          bin indices to compute the expected value (integral) from the
          predicted probability distribution.
        - The convolution weights are fixed and non-trainable, effectively
          performing a weighted sum over the discrete distribution.

    Application:
        - Applied on the output of a bounding box regression head, which predicts
          discrete distributions for each bounding box coordinate.
        - Converts the discrete probability distribution into continuous
          coordinate offsets for accurate bounding box localization.
    """

    def __init__(self, c1: int = 16):
        """
        Initialize the DFL module.

        Args:
            c1 (int): Number of discrete bins in the predicted distribution.
        """
        super().__init__()
        # Define a convolution layer with fixed weights representing bin indices.
        self.conv = nn.Conv2d(c1, 1, kernel_size=1,
                              bias=False, requires_grad=False)
        # Initialize weights as a vector [0, 1, 2, ..., c1-1]
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = x.view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DFL.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 4*c1, num_anchors),
                              representing discrete distributions for 4 box coordinates.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 4, num_anchors),
                          representing continuous bounding box coordinate predictions.
        """
        b, _, a = x.shape  # batch size, channels (4 * c1), anchors
        # Reshape to (batch_size, 4, c1, anchors)
        x = x.view(b, 4, self.c1, a)
        # Apply softmax over the discrete bin dimension (c1)
        prob = x.softmax(dim=2)
        # Perform integral by fixed conv weights to compute expectation
        out = self.conv(prob.transpose(2, 1)).view(b, 4, a)
        return out
