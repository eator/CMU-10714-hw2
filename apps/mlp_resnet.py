import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    #if norm == nn.BatchNorm1d:
    #    print("BatchNorm")
    #    n1 = norm(hidden_dim)
    #    n2 = norm(dim)
    #elif norm == nn.LayerNorm1d:
    #    print("LayerNorm")
    #    n1 = norm(dim)
    #    n2 = norm(hidden_dim)
    #else:
    #    print("unexpected norm")
    seq = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            ) 

    return nn.Sequential(nn.Residual(seq), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    seq = [
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
    ]

    for _ in range(num_blocks):
        seq.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))

    seq.append(nn.Linear(hidden_dim, num_classes))

    return nn.Sequential(*seq)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()

    loss_sum = 0
    true_preds = 0
    data_len = len(dataloader.dataset)
    iter_len = len(dataloader)

    loss_func = nn.SoftmaxLoss()

    for X, y in dataloader:
        if opt:
            opt.reset_grad()

        pred = model(X)
        pred_class = pred.numpy().argmax(axis=1)
        true_preds += np.sum(pred_class == y.numpy())

        curr_loss = loss_func(pred, y)
        if opt:
            curr_loss.backward()
            opt.step()
        loss_sum += curr_loss.numpy()

    return (1 - true_preds/data_len), loss_sum/iter_len
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = ndl.data.MNISTDataset(
            data_dir+"/train-images-idx3-ubyte.gz",
            data_dir+"/train-labels-idx1-ubyte.gz",
            )  
    test_set = ndl.data.MNISTDataset(
            data_dir+"/t10k-images-idx3-ubyte.gz",
            data_dir+"/t10k-labels-idx1-ubyte.gz",
            )  

    train_loader = ndl.data.DataLoader(train_set, batch_size, True)
    test_loader = ndl.data.DataLoader(test_set, batch_size)

    model = MLPResNet(28*28, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # start training
    for _ in range(epochs):
        train_error, train_loss = epoch(train_loader, model, opt)
    test_error, test_loss = epoch(test_loader, model)

    return (train_error, train_loss, test_error, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
