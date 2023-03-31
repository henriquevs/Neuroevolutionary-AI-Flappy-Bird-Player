import pytest
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as pyplot
import os
from matplotlib import style


@pytest.fixture
def data():
    # Load the dataset
    data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'student', 'student-mat.csv'))
    data = pd.read_csv(data_file_path, sep=";")
    # Extract relevant columns
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
    return data


@pytest.fixture
def trained_model(data):
    # Prepare the data
    predict = "G3"
    X = np.array(data.drop(labels=[predict], axis=1))
    y = np.array(data[predict])
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # Train the model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    return linear


def test_data_shape(data):
    # Ensure that the dataset has the expected shape
    assert data.shape == (395, 6)


def test_trained_model_accuracy(trained_model, data):
    # Ensure that the trained model has a high accuracy on the testing set
    acc = trained_model.score(data.drop("G3", axis=1), data["G3"])
    assert acc >= 0.5


def test_pickle(trained_model):
    # Ensure that the trained model can be serialized and deserialized using pickle
    data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'result', 'student_model.pickle'))
    with open(data_file_path, "wb") as f:
        pickle.dump(trained_model, f)
    with open(data_file_path, "rb") as f:
        loaded_model = pickle.load(f)
    assert type(loaded_model) == type(trained_model)


def test_plot(data):
    # Ensure that the scatter plot of the data can be generated without errors
    p = "absences"
    style.use("ggplot")
    pyplot.scatter(data[p], data["G3"])
    pyplot.xlabel(p)
    pyplot.ylabel("Final Grade")
    pyplot.close()
