import pandas as pd
import numpy as np


x_feature_file_name = 'train_x_feature.csv'
x_feature = pd.read_csv(x_feature_file_name, header=None)
x_feature = x_feature.values


x = np.zeros((90, 13))

for i in range(30):
	x[3 * i + 0] = x_feature[i]
	x[3 * i + 1] = x_feature[i]
	x[3 * i + 2] = x_feature[i]

x = np.concatenate((x, x), axis=0)
np.savetxt("x_feature.csv", x, delimiter=",")