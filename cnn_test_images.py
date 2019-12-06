import numpy as np
import torch


def cnn_test_images(test_X, test_Y, num, cnn, loss_func, device):
    """ Testing test data... """
    """ Return accuracy, prediction class, total loss"""
    cnn.eval()
    test_accuracy = 0
    test_prediction = np.zeros(num)
    tot_loss = 0
    for step, b_x in enumerate(test_X):  # gives batch data, normalize x when iterate train_loader
        chan, row, col = b_x.shape
        b_x = b_x.reshape(1, chan, row, col)
        b_x = b_x.float()
        b_x = b_x.to(device)
        output = cnn(b_x)[0]  # cnn output
        del b_x
        b_y = test_Y[step]
        b_y = b_y.reshape(1)
        loss = loss_func(output, b_y)  # cross entropy loss
        tot_loss = tot_loss + loss
        pred_y = torch.max(output.cpu(), 1)[1].data.numpy()
        test_prediction[step] = pred_y
        if pred_y == b_y.cpu().data.numpy():
            test_accuracy = test_accuracy + 1
        if step == int(num-1):
            test_accuracy = test_accuracy / num
            test_accuracy = test_accuracy * 100

    return test_accuracy, test_prediction, tot_loss
