import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# ------------------- DATA PREP --------------------- #
data = pd.read_csv("student/student-mat.csv", sep=";")
print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"

X = np.array(data.drop(labels=[predict], axis=1))  # features or attributes
y = np.array(data[predict])  # outputs

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# print(f"\nx_train={x_train}\n\ny_train={y_train}\n\nx_test={x_test}\n\ny_test={y_test}\n")

# ------------------- MODEL TRAINING --------------------- #
# linear = linear_model.LinearRegression()  # our model
#
# linear.fit(x_train, y_train)  # train the model
# acc = linear.score(x_test, y_test)  # accuracy
# print(f"\nacc={acc}\n")
# print(f"\nCoeffs: {linear.coef_}\n")
# print(f"\nIntercept={linear.intercept_}\n")
#
# # SAVE THE MODEL FOR FUTURE USAGE
# with open("result/student_model.pickle", "wb") as f:
#     pickle.dump(linear, f)

picke_in = open("result/student_model.pickle", "rb")
linear = pickle.load(picke_in)

# Make some predictions
predictions = linear.predict(x_test)
print(f"\npredictions={predictions}\n")

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


