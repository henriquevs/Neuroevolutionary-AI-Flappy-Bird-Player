################### SUPERVISED LEARNING ###################
import sklearn
from sklearn import datasets
from sklearn import svm  # Our classifier (attemps to create a Hyperplane)
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier  # also used in 2_knn_classification

# Suppor Vector Machines
# 1. We want to maximize the margin
# 
# 2. We take the 2 closest points (vectors) to the Hyperplane so that
# their distance is the maximum from it (no other points will be farther
# than these)
# 
# 3. For no linear data we use something called KERNEL (a function)
# this is simply a function that takes f(x1, x2) -> x3
# So the kernel is a generator of a tridimensional vectorial space
# Also, the KERNEL basically generates a new space dimension (LINEAR ALGEBRA)
# If adding 1 more dimension doesn't fix the problem of generating a 
# meaningful hyperplane, we can continue generating more dimensions
#
# Kernel examples (basically, non-linear functions)
# i) x^2 + y^2 = y
#
# SOFT MARGIN = accepted margin making the 2 vectors used maybe not the
# closest one to the hyperplane

cancer = datasets.load_breast_cancer()  # Dataset with 30+ features
# print(f"\n{cancer.feature_names=}\n")
# print(f"\n{cancer.target_names=}\n")

x = cancer.data
y = cancer.target

# It's not recommended to increase 'test_size' > 0.3 (30%)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# print(f"\nx_train={x_train}\n\ny_train={y_train}\n\nx_test={x_test}\n\ny_test={y_test}\n")
classes = ['malignant', 'benign']

# Support Vector Classification (SVC)
# classifier = svm.SVC(kernel="poly", degree=10)
classifier = svm.SVC(kernel="linear", C=2)  # soft margin => C=1 (default); C=2 => allow double of the points within the margin
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(f"\n[SVC] {acc=}")

# Compare with knn
classifier = KNeighborsClassifier(n_neighbors=13)  # Usually KNN doesn't work well with datasets with many features
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(f"[KNN] {acc=}\n")
