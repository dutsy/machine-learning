"""
Qussai Firon
Machine Learning EX 3_1
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------------------------
def nudge_dataset(X, Y):
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


# ------------------------------------------------------------------------------
X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, 'float32')
X, Y = nudge_dataset(X, y)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
rbm = BernoulliRBM(random_state=0, verbose=False)

rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])
rbm.learning_rate = 0.06
rbm.n_iter = 10
logistic.C = 6000
elem = []
tim = []
avg = []

# ------------------------------------------------------------------------------
def q3(X_train,X_test):
        print('______________________________________________________________')
        print('X_train:', X_train.shape)
        print('X_test:', X_test.shape)
        print('transofrm X_train:', rbm.transform(X_train).shape)
        print('rbm.intercept_hidden_:', rbm.intercept_hidden_.shape)
        print('______________________________________________________________')
    

# ------------------------------------------------------------------------------
def q1():
    for j in range(2, 21):
        k = j * j
        elem.append(k)
        rbm.n_components = k
        ta = time.perf_counter()
        rbm_features_classifier.fit(X_train, Y_train)
        tb = time.perf_counter()
        tim.append(tb - ta)
        Y_pred = rbm_features_classifier.predict(X_test)
        ps = metrics.precision_score(Y_test, Y_pred, average=None)
        avg.append(np.average(ps))

        plt.figure(figsize=(4.2, 4))
        for i, comp in enumerate(rbm.components_):
            plt.subplot(j, j
                        , i + 1)
            plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
            plt.suptitle('{0} components extracted by RBM'.format(k), fontsize=16)
            plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.show()
        q3(X_train, X_test)


# ------------------------------------------------------------------------------
def q2():
    plt.title("Avg Precsion vs Components")
    plt.xlabel("Components")
    plt.ylabel("Avg Precision")
    plt.plot(elem, avg, 'g', [2, 400], [0.76, 0.76], 'b')
    plt.show()

    plt.title("Avg Time & Components")
    plt.xlabel("Components")
    plt.ylabel("Time difference")
    plt.plot(elem, tim, 'r')
    plt.show()


q1()
q2()