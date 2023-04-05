################### SUPERVISED LEARNING ###################
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# ------------------- DATA PREP --------------------- #
data = pd.read_csv("./data/car.data")
print(data.head())

le = preprocessing.LabelEncoder()

# Create an array for each of our columns on car.data, make sure we use only integers
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
# print(f"{buying=}")
# print(f"{maint=}")
# print(f"{lug_boot=}")
# print(f"{safety=}")
print(f"{cls=}")

predict = "class"

X = list(zip(buying, maint, doors, persons, lug_boot, safety))  # Features
y = list(cls)  # Label
# print(f"{X=}")
# print(f"{y=}")
# print("Shape of X:", np.shape(X))
# print("Shape of y:", np.shape(y))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# print(f"\nx_train={x_train}\n\ny_train={y_train}\n\nx_test={x_test}\n\ny_test={y_test}\n")

# --------------------- KNN IMPLEMENTATION --------------------- #
model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(f"{acc=}")

# Compare test data and the prediction
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print(f"Predicted: {names[predicted[x]]}, Data: {x_test[x]}, Actual: {names[y_test[x]]}")
    n = model.kneighbors(X=[x_test[x]], n_neighbors=7, return_distance=True)  # print neighbors distances
    print(f"N: {n}\n")
