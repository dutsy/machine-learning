"""
Qussai Firon
Machine Learning EX 2
How to use libraries of sklearn, numpy & classification algorithms
"""

# -------------- Imports & Libraries -----------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import svm, metrics
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn import datasets

# -------------------------------------------------------------------------------
# The digits dataset
digits = datasets.load_digits()
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
classifier = svm.SVC(gamma=0.001)
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))

# ------------------------- Q20 ---------------------------------------------------
wrong = []
d_num = []
expect = []


def show_data():
    for i in range((n_samples // 2)):

        if expected[i] != predicted[i]:
            expect.append(expected[i])
            wrong.append(predicted[i])
            d_num.append(i)
            print("Digit:", expected[i], ", Comp says:", predicted[i], ", Place at the array:", i)


def pre_and_exp(wrong_pres):
    plt.suptitle("Test. mis-classification: expected - predicted")
    for index, (j, i, l) in enumerate(wrong_pres):
        plt.subplot(3, 10, index + 1)
        plt.axis('off')
        plt.imshow(digits.images[(n_samples // 2) + l], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(repr(j) + " " + repr(i))
    plt.show()


def q20():
    show_data()
    pre_and_exp(list(zip(expect, wrong, d_num)))


q20()


# ------------------------- Q21 ---------------------------------------------------
# ------------------------- Q21 Functions -----------------------------------------
# -------------------------Functions for Question 21-------------------------------
def lower_rows(mat):
    return [np.sum(mat[len(mat) // 2:]), np.average(mat[len(mat) // 2:]), np.var(mat[len(mat) // 2:])]


# ----------------------------------------------------------------------------------
def upper_rows(mat):
    return [np.sum(mat[:(len(mat) // 2)]), np.average(mat[:(len(mat) // 2)]), np.var(mat[:(len(mat) // 2)])]


# ----------------------------------------------------------------------------------
def first_cols(mat):
    return [np.sum(mat[:, 0:len(mat) // 2]), np.average(mat[:, 0:len(mat) // 2]), np.var(mat[:, 0:len(mat) // 2])]


# -----------------------------------------------------------------------------
def last_cols(mat):
    return [np.sum(mat[:, len(mat) // 2:]), np.average(mat[:, len(mat) // 2:]), np.var(mat[:, len(mat) // 2:])]


# ----------------------------------------------------------------------------------
def max_f(mat):
    length = len(mat)
    return max(max(mat[:, 1]), max(mat[:, length - 2]))


# ----------------------------------------------------------------------------------
def sum_f(mat):
    length = len(mat)
    return np.sum(mat[:, 1]) + np.sum(mat[:, length - 2])


# ------------------------------------------------------------------------------
def diff_first_last(mat):
    return first_cols(mat)[0] - last_cols(mat)[0]


# ---------------------------------------------------------------------------
def diff_upper_lower(mat):
    return upper_rows(mat)[0] - lower_rows(mat)[0]


# ----------------------------------------------------------------------------------
def zero_in_center(mat):
    values = mat[3:5, 3:5]
    return len(values[np.where(values <= 2)])


# ----------------------------------------------------------------------------------
def var_m_and_tm(mat):
    return np.var(mat + np.transpose(mat))


# ----------------------------------------------------------------------------------
def sum_of_m_and_tm(mat):
    return np.sum(mat + np.transpose(mat))


# ----------------------------------------------------------------------------------
def big_in_center(mat):
    return len((mat[3:5, 3:5])[np.where((mat[3:5, 3:5]) >= 13)])


indices_0_1 = np.where(np.logical_and(digits.target >= 0, digits.target <= 1))


# ----------------------------------------------------------------------------------
def var_sum_zero_show(var_arr, sum_arr, zic_arr, title, xl, yl, zl):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig.suptitle(title)
    ax.scatter(var_arr, sum_arr, zic_arr, c=digits.target[indices_0_1])
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_zlabel(zl)
    plt.show()


# -------------------------- Q21 -------------------------------------------------
def Q21():
    diff_f_v = []
    diff_u_v = []
    sum_f_v = []
    z_in_center = []
    num_of_numbers = []
    var_val = []
    sum_val = []
    s_m_tm = []
    v_m_tm = []
    max_val = []
    for i, k in zip(indices_0_1[0], digits.target[indices_0_1]):
        val = digits.images[i]
        z_in_center.append(zero_in_center(val))
        num_of_numbers.append(big_in_center(val))
        diff_f_v.append(diff_first_last(val))
        diff_u_v.append(diff_upper_lower(val))
        var_val.append(np.var(val))
        sum_val.append(sum_f(val))
        max_val.append(max_f(val))
        sum_f_v.append(np.sum(val))
        s_m_tm.append(sum_of_m_and_tm(val))
        v_m_tm.append(var_m_and_tm(val))
    # -------------------------- Q21 G -------------------------------------------------
    X = np.column_stack((sum_f_v, var_val, z_in_center, sum_val, max_val, num_of_numbers, v_m_tm, s_m_tm))
    X_scaled = preprocessing.scale(X)
    Y = digits.target[indices_0_1]
    logistic_classifier = linear_model.LogisticRegression(solver='lbfgs')
    logistic_classifier.fit(X_scaled, Y)
    expected_val = Y
    predicted_val = logistic_classifier.predict(X_scaled)
    print(
        "Logistic regression using [sum_f_v, var_val, z_in_center, sum_val, max_val, num_of_numbers, v_m_tm, s_m_tm] features:\n%s\n" % (
            metrics.classification_report(
                expected_val,
                predicted_val)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_val, predicted_val))
    predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=8)
    print(
        "Logistic regression using [sum_f_v, var_val, z_in_center, sum_val, max_val, num_of_numbers, v_m_tm, s_m_tm] features cross validation:\n%s\n" % (
            metrics.classification_report(expected_val, predicted2)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_val, predicted2))

    # -------------------------- Q21 E -------------------------------------------------
    var_sum_zero_show(var_val, sum_f_v, z_in_center, "Variance, Sum & Zero in Centers", 'Variance', 'Sum', 'Zeros')
    var_sum_zero_show(var_val, sum_f_v, num_of_numbers, "Variance, Sum, Big Numbers in Centers", 'Variance', 'Sum',
                      'Big')
    var_sum_zero_show(v_m_tm, s_m_tm, z_in_center, "Variance of m+m^t, Sum of m+m^t & Zero in Centers", 'Variance',
                      'Sum',
                      'Zeros')
    var_sum_zero_show(v_m_tm, s_m_tm, num_of_numbers, "Var of m+m^t, Sum of m+m^t Big Numbers in Centers",
                      'Variance', 'Sum',
                      'Big')
    var_sum_zero_show(z_in_center, num_of_numbers, v_m_tm, "Zeros, Big Numbers & Var of t&tm", 'Zeros', 'Big',
                      'Var of m+m^t')


Q21()
