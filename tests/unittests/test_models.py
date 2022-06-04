import torch
import torch.nn


def test_cnn():
    from facenet.nnet.cnn import FaceCNN

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input = torch.rand([4, 3, 256, 256], device=device)
    model = FaceCNN(label_count=5, img_dim=(256, 256), base_filter=12)
    model = model.to(device)

    output = model(input)
    assert output.shape[-1] == 5


def test_mlp():
    from facenet.nnet.mlp import MLP

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input = torch.rand([4, 3, 256, 256], device=device)
    model = MLP(label_count=5, hl_size=50, img_dim=(256, 256), img_channels=3)
    model = model.to(device)

    output = model(input)
    assert output.shape[-1] == 5


def test_sc():
    from facenet.nnet.linear_classifier import SoftmaxClassifier

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input = torch.rand([4, 3, 256, 256], device=device)
    model = SoftmaxClassifier(label_count=5, img_dim=(256, 256), img_channels=3)
    model = model.to(device)

    output = model(input)
    assert output.shape[-1] == 5
