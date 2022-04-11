import torch
import matplotlib.pyplot as plt
import numpy as np


def pred_label(model, img):
    model = model.to('cpu')
    img = img.unsqueeze(0)
    logits = model(img)
    pred_probab = torch.nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)

    return y_pred

def save_image(img, title, count):
    fig, ax = plt.subplots()
    imgplot = ax.imshow(img, interpolation='bicubic')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # imgplot = plt.imshow(img, interpolation='bicubic')
    plt.title(title)
    plt.savefig('./viz/img' + str(count))


def observations(model, testloader):
    for imgs, labels in testloader:
        images = [imgs[0].permute(1, 2, 0),
        imgs[1].permute(1, 2, 0),
        imgs[2].permute(1, 2, 0),
        imgs[3].permute(1, 2, 0),
        imgs[4].permute(1, 2, 0)]

        pred_label1 = pred_label(model, imgs[0]).item()
        pred_label2 = pred_label(model, imgs[1]).item()
        pred_label3 = pred_label(model, imgs[2]).item()
        pred_label4 = pred_label(model, imgs[3]).item()
        pred_label5 = pred_label(model, imgs[4]).item()

        titles = ["Pred: {}, Actual: {}".format(pred_label1, labels[0]), 
        "Pred: {}, Actual: {}".format(pred_label2, labels[1]),
        "Pred: {}, Actual: {}".format(pred_label3, labels[2]),
        "Pred: {}, Actual: {}".format(pred_label4, labels[3]),
        "Pred: {}, Actual: {}".format(pred_label5, labels[4])]

        count = 1
        for image, title in zip(images, titles):
            save_image(image, title, count)
            count += 1

        break