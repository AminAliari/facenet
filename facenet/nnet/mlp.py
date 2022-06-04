"""Library implementing multilayer perceptron.

Author
 * Mohammadamin Aliari
"""

import torch.nn as nn


class MLP(nn.Module):
    """This function implements MLP with one hidden layer.

    Arguments
    ---------
    label_count : int
        Number of labels for the classification task.
    hl_size : int
        Hidden layer neuron dimension.
    img_dim : int
        Input image dimension meaning width and height.
    img_channels : int
        Number of image channels. Example: RGB -> 3 channels.

    Example
    -------
    >>> import torch
    >>> inp_tensor = torch.rand([10, 3, 256, 256])
    >>> model = MLP(label_count=5, hl_size=1000, img_dim=(256, 256), img_channels=3)
    >>> out_tensor = model(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 5])
    """

    def __init__(self, label_count, hl_size, img_dim, img_channels):
        super(MLP, self).__init__()

        input_size = img_dim[0] * img_dim[1] * img_channels  # flatten size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hl_size),
            nn.BatchNorm1d(hl_size),
            nn.ReLU(),

            nn.Linear(hl_size, label_count)
        )

    def forward(self, x):
        """Returns the output of the network.

        Arguments
        ---------
        x : torch.Tensor (batch_size, img_channels, img_dim, img_dim)
            input to network.

        Returns
        ---------
        out: torch.Tensor
        Tensor containing the predictions (batch_size, label_count).
        """

        x = self.classifier(x)
        return x
