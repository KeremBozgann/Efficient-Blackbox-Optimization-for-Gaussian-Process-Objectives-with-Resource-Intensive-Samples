
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score

def train_and_test_mlpc(X, Y, hidden_layer_sizes, activation, alpha, batch_size, learning_rate_init, num_epoch,
                        num_restarts):

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)

    score_avg = 0
    score_std_avg = 0

    for i in range(num_restarts):

        mlpc= MLPClassifier(hidden_layer_sizes= hidden_layer_sizes, activation= activation, alpha= alpha, batch_size= batch_size,
                      learning_rate_init= learning_rate_init, max_iter= num_epoch, random_state= i)

        # mlpc.fit(X_train, y_train)
        # score_avg+= mlpc.score(X_test, y_test)

        scores= cross_val_score(mlpc, X, Y, cv= 5)
        score_avg+= scores.mean(); score_std_avg+= scores.std()

    score_avg /= num_restarts; score_std_avg /= num_restarts

    return score_avg, score_std_avg

def test_train_test_split():

    X= np.random.rand(10,2)
    y= np.random.rand(10,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1)




