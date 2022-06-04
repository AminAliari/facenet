def test_dataset():
    from facenet.dataio.dataset import FaceDataset

    dataset = FaceDataset(root_dir='data/samples', img_dim=(256, 256))
    loader = dataset.get_data_loader(batch_size=1, use_shuffle=False)
    assert(128 == len(loader))
    assert((1, 3, 256, 256) == next(iter(loader))[0].shape)
