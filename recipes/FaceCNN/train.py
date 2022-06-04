#!/usr/bin/env/python3
"""
FaceCNN model recipe.
"""

from facenet.core import Config
from facenet.nnet.cnn import FaceCNN
from facenet.dataio.dataset import FaceDataset

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

ARCH = 'FaceCNN'


def train():
    # set random seeds for reproducibility
    random.seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    torch.manual_seed(Config.random_seed)

    # set cuda device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # create datasets and data loaders
    train_dataset = FaceDataset(root_dir='data/train', img_dim=Config.img_dim)
    valid_dataset = FaceDataset(root_dir='data/valid', img_dim=Config.img_dim)
    train_loader = train_dataset.get_data_loader(batch_size=Config.batch_size, use_shuffle=Config.use_shuffle)
    valid_loader = valid_dataset.get_data_loader(batch_size=1, use_shuffle=False)

    # create the model
    model = FaceCNN(label_count=FaceDataset.label_count, img_dim=Config.img_dim, base_filter=Config.base_filter)
    model = model.to(device)
    model.eval()

    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.opt_l2)

    # tracking states
    avg_loss = 0
    best_vloss = 999999
    loss_history = []
    vloss_history = []

    # training loop
    for epoch in range(Config.epochs):
        # traning
        model.train(True)
        last_loss = 0
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            out_labels = model(images)
            loss = loss_fn(out_labels, labels)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            running_loss += loss.item()
            # print every config.log_interval mini-batches
            if i % Config.log_interval == Config.log_interval - 1:
                avg_loss = running_loss / Config.log_interval
                print(f'[epoch_{epoch + 1}, batch_{i + 1}] train_loss: {avg_loss:.3f}')
                running_loss = 0.0
        loss_history.append(last_loss)

        # validation
        model.train(False)
        vloss_iter = 0
        running_vloss = 0
        with torch.no_grad():
            for i, vdata in enumerate(valid_loader):
                vimages, vlabels = vdata[0].to(device), vdata[1].to(device)
                vout_labels = model(vimages)
                vloss = loss_fn(vout_labels, vlabels)
                running_vloss += vloss.item()
                vloss_iter += 1
        avg_vloss = running_vloss / (vloss_iter + 1)
        vloss_history.append(avg_vloss)

        print(f'[epoch_{epoch + 1} ended] train_loss: {avg_loss:.3f}, valid_loss: {avg_vloss:.3f}\n')

        # save the best model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), f'trained/{ARCH}.pt')

    print('\ndone.')

    # save history
    np.savetxt(f'trained/history{ARCH}_loss.csv', loss_history, delimiter=',')
    np.savetxt(f'trained/history{ARCH}_vloss.csv', vloss_history, delimiter=',')


if __name__ == "__main__":
    train()
