import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from scipy.ndimage import shift

# load dataset
mnist = fetch_openml('mnist_784', version=1)

# split into features and labels
X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

# split into training and testing data sets
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=10000, random_state=42)

def binary_classification(digit):
    y_train_digit = (y_tr == digit)
    y_test_digit = (y_ts == digit)
    some_digit = np.array(X)[4]
    some_digit_image = some_digit.reshape(1, 784)

    # training and prediction
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_tr, y_train_digit)
    y_train_digit_predict = sgd_clf.predict(some_digit_image)
    y_train_predict = sgd_clf.predict(X_tr)

    # precision and recall scores
    prec_score = precision_score(y_train_digit, y_train_predict)
    rec_score = recall_score(y_train_digit, y_train_predict)

    # precision-recall vs. decision threshold
    y_scores = cross_val_predict(sgd_clf, X_tr, y_train_digit, cv=3, method='decision_function')
    precisions, recalls, thresholds = precision_recall_curve(y_train_digit, y_scores)
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.show()

    # print results
    print("Binary Classification")
    print("Digit: " + str(digit))
    print("Label: " + str(y[4]))
    print("Prediction: " + str(y_train_digit_predict))
    print("Precision Score: " + str(prec_score))
    print("Recall Score: " + str(rec_score))
    print()

def multiclass_classification():
    # training and prediction
    svm_clf = SVC()
    svm_clf.fit(X_tr, y_tr)
    y_train_predict = svm_clf.predict(X_tr)

    # error analysis
    conf_mx = confusion_matrix(y_tr, y_train_predict)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()

    # print results
    print("Multiclass Classification")
    print("Confusion Matrix: ")
    print(conf_mx)
    print()

def KNN_training():
    # params
    grid_params = {
        'n_neighbors': np.arange(1, 5),
        'weights': ['uniform', 'distance'],
    }

    # search to find optimal params
    gs = GridSearchCV(
        KNeighborsClassifier(),
        grid_params,
        verbose = 1,
        cv = 3,
        scoring = 'accuracy',
        n_jobs = -1
    )

    # training and prediction
    grid_search = gs.fit(X_tr, y_tr)
    best_params = grid_search.best_params_
    best_result = grid_search.best_score_

    # print results
    print("KNN Training")
    print("Best Params: ", best_params)
    print("Best Score: ", best_result)
    print()

def KNN_test():
    # training and prediction
    knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
    knn.fit(X_tr, y_tr)
    y_test_pred = knn.predict(X_ts)
    test_accuracy = accuracy_score(y_ts, y_test_pred)

    # print results
    print("KNN Test")
    print("Classification Report:")
    print(classification_report(y_ts, y_test_pred, digits=4))
    print()

def shift_image(image, dir_x, dir_y):
    image = image.reshape(28, 28) # make multi-dimensional
    shift_image = shift(image, [dir_y, dir_x], cval=0, mode='constant') # shift image one of eight ways
    return shift_image.reshape([-1]) # make one-dimensional

def data_augmentation():
    global X_tr
    global y_tr
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)
    X_train_expanded = [image for image in X_tr]
    Y_train_expanded = [label for label in y_tr]

    # cycle through all images and shift pixels to generate 8 times amount of training data
    for dir_x, dir_y in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)):
        for image, label in zip(X_tr, y_tr):
            X_train_expanded.append(shift_image(np.array(image), dir_x, dir_y)) # shift image
            Y_train_expanded.append(label) # add label

    X_train_expanded = np.array(X_train_expanded)
    Y_train_expanded = np.array(Y_train_expanded)

    X_tr = X_train_expanded
    y_tr = Y_train_expanded

if __name__ == '__main__':
    binary_classification(9)
    multiclass_classification()

    KNN_training()
    KNN_test()

    data_augmentation()
    KNN_test()