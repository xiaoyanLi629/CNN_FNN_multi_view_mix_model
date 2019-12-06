import torch
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from load_X_Y_data import load_X_Y_data
from cnn_test_images import cnn_test_images
from fnn_test_images import fnn_test_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

train_x_image, train_x_feature, train_y, train_num, test_x_image, test_x_feature, test_y, test_num = load_X_Y_data(device)
train_dataset = Data.TensorDataset(train_x_image, train_x_feature, train_y)
test_dataset = Data.TensorDataset(test_x_image, test_x_feature, test_y)

BATCH_SIZE = 5

'''
train_loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

test_loader = Data.DataLoader(
    dataset=test_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=24,  # n_filters
                kernel_size=10,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=5),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(24, 48, 10, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(5),  # output shape (32, 7, 7)
        )
        self.linear1 = nn.Linear(192, 512)
        self.linear2 = nn.Linear(512, 32)
        self.out = nn.Linear(32, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.relu(x)
        x = self.linear1(x)
        torch.nn.Dropout(0.2)
        x = self.relu(x)
        x = self.linear2(x)
        torch.nn.Dropout(0.2)
        x = self.relu(x)
        out = self.out(x)
        return out, x  # return x for visualization


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()

        self.linear1 = nn.Linear(20, 1024)
        torch.nn.Dropout(0.2)
        self.linear2 = nn.Linear(1024, 2048)
        torch.nn.Dropout(0.2)
        self.linear3 = nn.Linear(2048, 512)
        torch.nn.Dropout(0.2)
        self.linear4 = nn.Linear(512, 48)
        torch.nn.Dropout(0.2)
        self.out = nn.Linear(48, 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        out = self.out(x)
        return out, x  # return x for visualization

# model_name = input('Please input creating model name(hit enter, a random name(1000) will be generated):')


model_name = 'test_model'
print('Constructing model:', model_name)

# setting device on GPU if available, else CPU

EPOCH = 200  # train the training data n times, to save time, we just train 1 epoch
LR = 0.0001  # learning rate
stable_factor = 0  # control model stability

print('Epoch:', EPOCH)
print('Learning rate:', LR)
print('batch size:', BATCH_SIZE)

cnn = CNN()
cnn.to(device)
optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func_cnn = nn.CrossEntropyLoss()  # the target label is not one-hotted

fnn = FNN()
fnn.to(device)
optimizer_fnn = torch.optim.Adam(fnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func_fnn = nn.CrossEntropyLoss()  # the target label is not one-hotted
epoch_list = []
plt.ion()

# misclassified_images = np.zeros((test_num, EPOCH))

train_cnn_accuracy_list = []
train_fnn_accuracy_list = []
test_cnn_accuracy_list = []
test_fnn_accuracy_list = []

for epoch in range(EPOCH):
    epoch_list.append(epoch)
    # for step, (x_image, x_feature, y) in enumerate(train_loader):
    for i in range(int(train_num / BATCH_SIZE)):
        index = np.random.choice(range(train_num), BATCH_SIZE, replace=False)
        x_image = train_x_image[index, :, :, :]
        x_feature = train_x_feature[index, :]
        y = train_y[index]

        # training CNN model
        x_image = x_image.float()
        x_image = x_image.to(device)
        output = cnn(x_image)[0]               # cnn output
        y = y.reshape(BATCH_SIZE)
        cnn_loss = loss_func_cnn(output, y)   # cross entropy loss
        optimizer_cnn.zero_grad()           # clear gradients for this training step
        cnn_loss.backward()                 # backpropagation, compute gradients
        optimizer_cnn.step()                # apply gradients

        # training FNN model
        x_feature = x_feature.to(device)
        x_feature = x_feature.float()
        output = fnn(x_feature)[0]               # fnn output
        # print(output)
        fnn_loss = loss_func_fnn(output, y)   # cross entropy loss
        optimizer_fnn.zero_grad()           # clear gradients for this training step
        fnn_loss.backward()                 # back propagation, compute gradients
        optimizer_fnn.step()                # apply gradients

    cnn.eval()  # parameters for dropout differ from train mode
    cnn_train_accuracy, cnn_train_prediction, cnn_train_tot_loss = cnn_test_images(train_x_image, train_y, train_num, cnn, loss_func_cnn, device)
    cnn_test_accuracy, cnn_test_prediction, cnn_test_tot_loss = cnn_test_images(test_x_image, test_y, test_num, cnn, loss_func_cnn, device)

    fnn_train_accuracy, fnn_train_prediction, fnn_train_tot_loss = fnn_test_images(train_x_feature, train_y, train_num, fnn, loss_func_fnn, device)
    fnn_test_accuracy, fnn_test_prediction, fnn_test_tot_loss = fnn_test_images(test_x_feature, test_y, test_num, fnn, loss_func_fnn, device)

    train_cnn_accuracy_list.append(cnn_train_accuracy)
    train_fnn_accuracy_list.append(fnn_train_accuracy)

    test_cnn_accuracy_list.append(cnn_test_accuracy)
    test_fnn_accuracy_list.append(fnn_test_accuracy)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, train_fnn_accuracy_list, color='g', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list, color='y', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_fnn_accuracy_list, color='r', label='Train cnn model accuracy')
        plt.title('Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch%10 == 0:
        print('Iteration: ', epoch)
        print('Training image accuracy is', cnn_train_accuracy)
        print('Training feature accuracy is', fnn_train_accuracy)
        print('Testing image accuracy is', cnn_test_accuracy)
        print('Testing feature accuracy is', fnn_test_accuracy)
        print('#############################################################')


plt.close()

plt.plot(np.arange(EPOCH), train_cnn_accuracy_list, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), train_fnn_accuracy_list, color='g', label='Train fnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list, color='y', label='Test cnn model accuracy')
plt.plot(np.arange(EPOCH), test_fnn_accuracy_list, color='r', label='Test fnn model accuracy')
plt.title('Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

