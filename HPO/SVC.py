


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def Train_and_Test_SVC(X, Y, alpha, num_epoch, lr):

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    score_avg = 0
    score_std_avg = 0
    num_restarts= 1
    for i in range(num_restarts):

        sgdc= make_pipeline(StandardScaler(), SGDClassifier(loss= 'hinge', alpha=alpha, max_iter= num_epoch, learning_rate = 'constant',
                        eta0= lr, validation_fraction= 0.1, early_stopping= False, random_state= i))
        #
        # sgdc.fit(X_train, y_train)
        # score_avg+= sgdc.score(X_test, y_test)

        scores = cross_val_score(sgdc, X, Y, cv=5)
        score_avg += scores.mean();
        score_std_avg += scores.std()

    score_avg /= num_restarts;
    score_std_avg /= num_restarts

    return score_avg, score_std_avg


    score_avg/= num_restarts

    return score_avg

def test_SVC():
