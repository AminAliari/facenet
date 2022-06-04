"""Library containing core classes.

Author
 * Mohammadamin Aliari
"""


class Config:
    """Contains all constants such as hyperparameters, random seed, split ratio, etc.
    """

    random_seed = 9898

    use_shuffle = True

    batch_size = 200
    img_channels = 3
    img_dim = (256, 256)

    base_filter = 64

    lr = 1e-4
    epochs = 10
    opt_l2 = 0  # 1e-5
    log_interval = 20
