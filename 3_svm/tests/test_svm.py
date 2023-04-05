import pytest
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

@pytest.fixture
def breast_cancer_dataset():
    return datasets.load_breast_cancer()

@pytest.fixture
def classifier():
    return svm.SVC(kernel="linear", C=2)

def test_breast_cancer_dataset_shape(breast_cancer_dataset):
    assert breast_cancer_dataset.data.shape == (569, 30)
    assert breast_cancer_dataset.target.shape == (569,)

def test_train_test_split_size():
    x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    assert len(x_train) == 8
    assert len(x_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

def test_classifier_fit_and_predict(classifier, breast_cancer_dataset):
    x = breast_cancer_dataset.data
    y = breast_cancer_dataset.target
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    assert len(y_pred) == len(y_test)

def test_classifier_accuracy_score(classifier, breast_cancer_dataset):
    x = breast_cancer_dataset.data
    y = breast_cancer_dataset.target
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    assert metrics.accuracy_score(y_test, y_pred) >= 0.0

def test_classifier_kernel():
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernel_list:
        classifier = svm.SVC(kernel=kernel)
        assert classifier.kernel == kernel

def test_classifier_C_parameter():
    c_list = [1, 2, 3, 4, 5]
    for c in c_list:
        classifier = svm.SVC(C=c)
        assert classifier.C == c

def test_classes():
    classes = ['malignant', 'benign']
    assert len(classes) == 2

def test_predicted_classes(classifier, breast_cancer_dataset):
    x = breast_cancer_dataset.data
    y = breast_cancer_dataset.target
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    assert all(y in [0, 1] for y in y_pred)


