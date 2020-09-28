
from Logistic_regression import Train_and_Test_Logistic_Regression
from SVC import Train_and_Test_SVC
from MLPC import train_and_test_mlpc

from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

from sklearn.datasets import load_digits

'''load mnist'''
X, Y= load_digits(n_class=10, return_X_y=True)

num_restarts= 5

'''logistic regression'''
alpha= 0
num_epoch= 1000
lr= 0.01

score_log, score_std_log= Train_and_Test_Logistic_Regression(X, Y, alpha, num_epoch, lr, num_restarts)

print('score_log:{}'.format(score_log))

'''mlpc'''
hidden_layer_sizes= [50,20, 30]
activation= 'relu'
batch_size= 64
learning_rate_init= 0.01
num_epoch_mlpc= 100
score_mlpc, score_std_mlpc= train_and_test_mlpc(X, Y, hidden_layer_sizes, activation, alpha, batch_size, learning_rate_init, num_epoch_mlpc,
                       num_restarts)

print('score_mlp:{}'.format(score_mlpc))

'''SVC'''
num_epoch_svc= 1000
alpha_svc= 0.001
lr_svc= 0.001
score_svc= Train_and_Test_SVC(X, Y, alpha_svc, num_epoch, lr_svc, num_restarts)

print('score_svc:{}'.format(score_svc))