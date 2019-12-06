import torch
import pickle


def load_X_Y_data(device):
    """ Loading training and testing data... """
    """ Return training data, training data labels"""

    # loading training data
    train_x_image_file = 'train/train_x_image.pkl'
    train_x_feature_file = 'train/train_x_feature.pkl'
    train_y_file = 'train/train_y.pkl'

    train_x_image = pickle.load(open(train_x_image_file, 'rb'))
    train_x_feature = pickle.load(open(train_x_feature_file, 'rb'))
    train_y = pickle.load(open(train_y_file, 'rb'))

    train_x_image = torch.from_numpy(train_x_image)
    train_x_feature = torch.from_numpy(train_x_feature)
    train_y = torch.from_numpy(train_y)
    train_num = train_y.shape[0]
    train_y = train_y.to(device)

    # loading testing data
    test_x_image_file = 'test/test_x_image.pkl'
    test_x_feature_file = 'test/test_x_feature.pkl'
    test_y_file = 'test/test_y.pkl'

    test_x_image = pickle.load(open(test_x_image_file, 'rb'))
    test_x_feature = pickle.load(open(test_x_feature_file, 'rb'))
    test_y = pickle.load(open(test_y_file, 'rb'))

    test_x_image = torch.from_numpy(test_x_image)
    test_x_feature = torch.from_numpy(test_x_feature)
    test_y = torch.from_numpy(test_y)
    test_num = test_y.shape[0]
    test_y = test_y.to(device)

    return train_x_image, train_x_feature, train_y, train_num, test_x_image, test_x_feature, test_y, test_num
