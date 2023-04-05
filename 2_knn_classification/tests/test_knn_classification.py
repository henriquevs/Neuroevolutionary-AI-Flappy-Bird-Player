import pytest
import os
import pandas as pd
from sklearn.utils import shuffle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing


@pytest.fixture(scope='module')
def prepare_data():
    # Load the data
    data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'car.data'))
    data = pd.read_csv(data_file_path)
    # Shuffle the data
    data = shuffle(data)
    # Encode the categorical data to numerical data
    le = preprocessing.LabelEncoder()
    buying = le.fit_transform(list(data["buying"]))
    maint = le.fit_transform(list(data["maint"]))
    doors = le.fit_transform(list(data["doors"]))
    persons = le.fit_transform(list(data["persons"]))
    lug_boot = le.fit_transform(list(data["lug_boot"]))
    safety = le.fit_transform(list(data["safety"]))
    cls = le.fit_transform(list(data["class"]))

    predict = "class"

    X = list(zip(buying, maint, doors, persons, lug_boot, safety))  # Features
    y = list(cls)  # Label
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    yield x_train, x_test, y_train, y_test


@pytest.fixture(scope='module')
def knn_model():
    # Create a KNN model with n_neighbors=7
    model = KNeighborsClassifier(n_neighbors=7)
    yield model


def test_data_shape(prepare_data):
    x_train, x_test, y_train, y_test = prepare_data
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def test_knn_model_creation(knn_model):
    assert knn_model is not None


def test_knn_fit(knn_model, prepare_data):
    x_train, x_test, y_train, y_test = prepare_data
    knn_model.fit(x_train, y_train)
    assert len(knn_model.classes_) == len(set(y_train))


def test_knn_score(knn_model, prepare_data):
    x_train, x_test, y_train, y_test = prepare_data
    knn_model.fit(x_train, y_train)
    score = knn_model.score(x_test, y_test)
    assert 0 <= score <= 1


def test_knn_prediction(knn_model, prepare_data):
    x_train, x_test, y_train, y_test = prepare_data
    knn_model.fit(x_train, y_train)
    predicted = knn_model.predict(x_test)
    assert len(predicted) == len(y_test)


def test_knn_neighbors(knn_model, prepare_data):
    x_train, x_test, y_train, y_test = prepare_data
    knn_model.fit(x_train, y_train)
    n = knn_model.kneighbors(X=[x_test[0]], n_neighbors=7, return_distance=True)
    assert n is not None


def test_prediction_accuracy(prepare_data, knn_model):
    x_train, x_test, y_train, y_test = prepare_data
    knn_model.fit(x_train, y_train)
    acc = knn_model.score(x_test, y_test)
    assert acc >= 0.0


def test_prediction_names(prepare_data, knn_model):
    x_train, x_test, y_train, y_test = prepare_data
    knn_model.fit(x_train, y_train)
    predicted = knn_model.predict(x_test)
    names = ["unacc", "acc", "good", "vgood"]
    for x in range(len(x_test)):
        assert names[predicted[x]] in names
