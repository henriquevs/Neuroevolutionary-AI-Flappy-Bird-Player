################### SUPERVISED LEARNING ###################
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

X = np.array(data.drop(labels=[predict], axis=1))  # Features or attributes
y = np.array(data[predict])  # Outputs

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# print(f"\nx_train={x_train}\n\ny_train={y_train}\n\nx_test={x_test}\n\ny_test={y_test}\n")

# ------------------- MODEL TRAINING --------------------- #
desired_acc = 0.85  # Defines the desired accuracy
acc = 0  # Initial accuracy

# ------ UNCOMMENT THIS BLOCK TO RE-TRAIN THE MODEL ------- #
while acc < desired_acc:
    linear = linear_model.LinearRegression()  # our model

    linear.fit(x_train, y_train)  # train the model
    acc = linear.score(x_test, y_test)  # accuracy
    print(f"\nacc={acc}\n")
    print(f"\nCoeffs: {linear.coef_}\n")
    print(f"\nIntercept={linear.intercept_}\n")

    # SAVE THE MODEL FOR FUTURE USAGE
    if acc >= desired_acc:
        with open("result/student_model.pickle", "wb") as f:
            pickle.dump(linear, f)
        break

pickle_in = open("result/student_model.pickle", "rb")
linear = pickle.load(pickle_in)

# Make some predictions
predictions = linear.predict(x_test)
print(f"\npredictions={predictions}\n")

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# ------------------- MODEL PLOT --------------------- #
p = "absences"
style.use("ggplot")  # Improves the plot grid
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
# pyplot.close()
