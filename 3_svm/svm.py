import sklearn
from sklearn import datasets
from sklearn import svm  # Our classifier (attemps to create a Hyperplane)

# Suppor Vector Machines
# 1. We want to maximize the margin
# 
# 2. We take the 2 closest points to the Hyperplane so that
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

cancer = datasets.load_breast_cancer()
print(f"\n{cancer.feature_names=}\n")
print(f"\n{cancer.target_names=}\n")

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
print(f"\nx_train={x_train}\n\ny_train={y_train}\n\nx_test={x_test}\n\ny_test={y_test}\n")
classes = ['malignant', 'benign']
