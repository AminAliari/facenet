"""Library implementing convolutional neural network.

Author
 * Mohammadamin Aliari
"""

import torch.nn as nn


class FaceCNN(nn.Module):
    """This function implements FaceCNN to classify real and fake face images.

    Arguments
    ---------
    label_count : int
        Number of labels for the classification task.
    img_dim : int
        Input image dimension meaning width and height.
    base_filter : int
        Starting filter size. The number of output features multiplies by 2 for each convolution layer.

    Example
    -------
    >>> import torch
    >>> inp_tensor = torch.rand([10, 3, 256, 256])
    >>> model = FaceCNN(label_count=5, img_dim=(256, 256), base_filter=12)
    >>> out_tensor = model(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 5])
    """

    def __init__(self, label_count, img_dim, base_filter=12):
        super(FaceCNN, self).__init__()

        filter_sizes = [1 * base_filter, 2 * base_filter,
                        4 * base_filter, 6 * base_filter,
                        8 * base_filter, 10 * base_filter]

        output_size = calc_output_size(img_dim[0], 5, 2, 2)
        output_size = calc_output_size(output_size, 5, 2, 2)
        output_size = calc_output_size(output_size, 3, 2, 1)
        output_size = calc_output_size(output_size, 3, 2, 1)

        final_layer_size = filter_sizes[3] * output_size * output_size

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filter_sizes[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(),

            nn.Conv2d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(filter_sizes[1]),
            nn.ReLU(),

            nn.Conv2d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filter_sizes[2]),
            nn.ReLU(),

            nn.Conv2d(in_channels=filter_sizes[2], out_channels=filter_sizes[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filter_sizes[3]),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(final_layer_size, 512),
            nn.ReLU(),

            nn.Linear(512, label_count)
        )

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch_size, img_channels, img_dim, img_dim)
            input to convolve.

        Returns
        ---------
        out: torch.Tensor
        Tensor containing the predictions (batch_size, label_count).
        """

        x = self.classifier(x)
        return x


def calc_output_size(img_dim, kernel, stride=1, padding=0):
    """Returns the filter output size after 2D-convolution or 2D-pooling.

    Arguments
    ---------
    img_dim : int
        Input image dimension meaning width and height.
    kernel : int
        Kernel width and height
    stride : int
    padding : int

    Example
    -------
    >>> out_size = calc_output_size(img_dim=256, kernel=5, stride=2, padding=2)
    >>> out_size
    128

    Returns
    ---------
    out: int
    Size of output features after applying the kernel.
    """

    return int(((img_dim - kernel + 2 * padding) / stride) + 1)
