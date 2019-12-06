import glob
import pickle
from PIL import Image
import PIL
import numpy as np
import pandas as pd

'''
    Save training images in train folder, save training labels csv file, train_y.csv, in train folder
        train_y.csv file in a column
    Save testing images in test folder, save testing labels csv file, test_y.csv, in test folder
        test_y.csv file in a column
'''

# basewidth = input('Resiezed images horisontal dimension (100):')
basewidth = 100

# hsize = input('Resiezed images vertical dimension (100):')
hsize = 100

# Reading train Y data

y_file_name = 'train/train_y.csv'
y = pd.read_csv(y_file_name, header=None)
y = y.values
train_num = y.shape[0]

save_file = 'train/train_y.pkl'
output = open(save_file, 'wb')
pickle.dump(y, output)
output.close()

# Reading train X feature data

x_feature_file_name = 'train/train_x_feature.csv'
x_feature = pd.read_csv(x_feature_file_name, header=None)
x_feature = x_feature.values

save_file = 'train/train_x_feature.pkl'
output = open(save_file, 'wb')
pickle.dump(x_feature, output)
output.close()

# Reading train X image data
save_file = 'train/train_x_image.pkl'
image_data = np.zeros((train_num, 3, basewidth, hsize))

img_index = 0
for filename in sorted(glob.glob('train/*.jpg')):
    img = Image.open(filename)
    img_resize = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img_resize_np = np.array(img_resize)
    img_reshape = np.zeros((1, 3, basewidth, hsize))
    img_reshape[:, 0, :, :] = img_resize_np[:, :, 0]
    img_reshape[:, 1, :, :] = img_resize_np[:, :, 1]
    img_reshape[:, 2, :, :] = img_resize_np[:, :, 2]
    image_data[img_index, :, :, :] = img_reshape
    img_index = img_index + 1

output = open(save_file, 'wb')
pickle.dump(image_data, output)
output.close()

####################################


#testing data


####################################

# Reading test Y data

y_file_name = 'test/test_y.csv'
y = pd.read_csv(y_file_name, header=None)
y = y.values
test_num = y.shape[0]

save_file = 'test/test_y.pkl'
output = open(save_file, 'wb')
pickle.dump(y, output)
output.close()

# Reading test X feature data

x_feature_file_name = 'test/test_x_feature.csv'
x_feature = pd.read_csv(x_feature_file_name, header=None)
x_feature = x_feature.values

save_file = 'test/test_x_feature.pkl'
output = open(save_file, 'wb')
pickle.dump(x_feature, output)
output.close()

# Reading test X data
save_file = 'test/test_x_image.pkl'
image_data = np.zeros((test_num, 3, basewidth, basewidth))

img_index = 0
for filename in sorted(glob.glob('test/*.jpg')):
    img = Image.open(filename)
    img_resize = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img_resize_np = np.array(img_resize)
    img_reshape = np.zeros((1, 3, basewidth, hsize))
    img_reshape[:, 0, :, :] = img_resize_np[:, :, 0]
    img_reshape[:, 1, :, :] = img_resize_np[:, :, 1]
    img_reshape[:, 2, :, :] = img_resize_np[:, :, 2]
    image_data[img_index, :, :, :] = img_reshape
    img_index = img_index + 1

output = open(save_file, 'wb')
pickle.dump(image_data, output)
output.close()
