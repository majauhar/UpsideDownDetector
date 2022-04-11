import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms.functional as TF

from tools.dataset import UpsideDownDataset
from train import train
from test import test
from observations import observations

import torchvision.models as models
import torch.optim as optim

import argparse

# setting  up the device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def arg_parser():
    parser = argparse.ArgumentParser(description="Upside Down Image Detector")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pretrained', type=bool, default=True)

    return parser.parse_args()

def main(args):
    # transformations
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # downloading CIFAR10 dataset
    trainset = UpsideDownDataset(root='./data', train=True, download=True, transform=train_transform)
    testset = UpsideDownDataset(root='./data', train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128)


    model = models.resnet18(pretrained=args.pretrained)
    model.fc = torch.nn.Linear(512, 2)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()

    model, criterion = model.to(DEVICE), criterion.to(DEVICE)

    print("training started:")

    loss_values = []
    accuracy_values = []
    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch+1))
        epoch_loss = train(model, trainloader, optimizer, criterion, DEVICE)
        loss_values.append(epoch_loss)
        accuracy = test(model, testloader, criterion, DEVICE)
        accuracy_values.append(accuracy)

    plt.plot(np.array(loss_values), 'r')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig('./viz/loss_curve.png')

    fig, ax = plt.subplots()
    ax.plot(np.array(loss_values), color='red', label='losses')
    ax.tick_params(axis='y', labelcolor='red')

    ax2 = ax.twinx()

    ax2.plot(np.array(accuracy_values), color='green', label='accuracy')
    ax2.tick_params(axis='y', labelcolor='green')

    # plt.xlabel('epochs')
    plt.legend()
    plt.savefig('./viz/final_curve.png')

    print("accuracy: {}".format(accuracy))

    observations(model, testloader)




if __name__ =='__main__':
    args = arg_parser()
    main(args)
