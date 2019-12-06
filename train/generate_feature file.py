import numpy as np
import pickle


u_0 = np.random.randint(50, size=20)
sigma_0 = np.random.randint(3, size=20)



u_1 = np.random.randint(50, size=20)
u_1 = u_1 + 100
sigma_1 = np.random.randint(3, size=20)

u_2 = np.random.randint(50, size=20)
u_2 = u_2/10 + 200
sigma_2 = np.random.randint(3, size=20)

u_3 = np.random.randint(50, size=20)
u_3 = u_3/10 + 250
sigma_3 = np.random.randint(3, size=20)

u_4 = np.random.randint(50, size=20)
u_4 = u_4/10 + 300
sigma_4 = np.random.randint(3, size=20)

u_5 = np.random.randint(50, size=20)
u_5 = u_5/10 + 350
sigma_5 = np.random.randint(3, size=20)

u_list = [u_0, u_1, u_2,  u_3, u_4, u_5]
sigma_list = [sigma_0, sigma_1, sigma_2, sigma_3, sigma_4, sigma_5]

cov_list = []
for num in range(6):
    cov = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            if i == j:
                cov[i, j] = sigma_list[num][i]
    cov_list.append(cov)

y = pickle.load(open('train_y.pkl', 'rb'))

train_x_feature = np.zeros((len(y), 20))
for i in range(len(y)):
    a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20 = np.random.multivariate_normal(u_list[int(y[i])], cov_list[int(y[i])], 1).T
    a_list = [a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20]
    for j in range(20):
        train_x_feature[i, j] = a_list[j]

np.savetxt("train_x_feature.csv", train_x_feature, delimiter=",")
