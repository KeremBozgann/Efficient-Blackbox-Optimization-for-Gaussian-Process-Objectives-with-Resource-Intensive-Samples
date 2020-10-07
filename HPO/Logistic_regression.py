

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

class Keras_model_fashion():

    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train= X_train
        self.Y_train= Y_train
        self.X_test= X_test
        self.Y_test= Y_test

    def evaluate_error_and_cost(self, layer_sizes, alpha, l2_regul, num_epoch):
        layer_sizes = np.ndarray.astype(np.rint(np.power(2, layer_sizes)), dtype= int)
        alpha= np.power(10.0, alpha)
        l2_regul= np.power(10.0, l2_regul)
        num_epoch= np.int(np.rint(np.power(10, num_epoch)))
        print('layer_sizes:{}\nalpha:{}\nl2_regul:{}\nnum_epoch:{}'.format(layer_sizes, alpha, l2_regul, num_epoch))
        '''define model'''
        t1= time.clock()
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(layer_sizes[0], activation='relu',  kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[1], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[2], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(layer_sizes[3], activation='relu', kernel_initializer='he_uniform', kernel_regularizer= l2(l2_regul)),
            keras.layers.Dense(10)
        ])
        opt = keras.optimizers.Adam(learning_rate= alpha)
        '''compile model'''
        model.compile(optimizer= opt,  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        '''fit'''
        model.fit(self.train_images, self.train_labels, epochs= num_epoch)

        '''test set accuracy'''
        test_loss, test_acc = model.evaluate(self.test_images,  self.test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        t2= time.clock()

        return np.atleast_2d(-test_acc), np.atleast_2d(t2- t1)

class Logistic():

    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train= X_train
        self.Y_train= Y_train
        self.X_test= X_test
        self.Y_test= Y_test

    def evaluate


def Train_and_Test_Logistic_Regression(X, Y, alpha, num_epoch, lr):

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    num_restarts=1
    score_avg = 0
    score_std_avg= 0
    for i in range(num_restarts):

        sgdc= make_pipeline(StandardScaler(), SGDClassifier(loss= 'log', alpha=alpha, max_iter= num_epoch, learning_rate = 'constant',
                        eta0= lr, validation_fraction= 0.1, early_stopping= False, random_state= i))

        # sgdc.fit(X_train, y_train)

        scores= cross_val_score(sgdc, X, Y, cv= 5)
        score_avg+= scores.mean(); score_std_avg+= scores.std()

    score_avg /= num_restarts; score_std_avg /= num_restarts

    return score_avg, score_std_avg