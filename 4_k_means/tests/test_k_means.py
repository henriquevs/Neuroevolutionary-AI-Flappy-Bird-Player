import pytest
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

# Load the dataset
digits = load_digits()
data = digits.data
y = digits.target

# k = len(np.unique(y))  # amount of centroids
k = 10
samples, features = data.shape

@pytest.fixture
def classifier():
    classifier = KMeans(n_clusters=k, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300,
                    tol=0.0001)
    return classifier

# Test whether the classifier object is created successfully
def test_create_classifier(classifier):
    assert isinstance(classifier, KMeans)

# Test whether the classifier is fit correctly
def test_fit_classifier(classifier):
    classifier.fit(data)
    assert len(classifier.cluster_centers_) == k

# Test whether the classifier is predicting correctly
def test_predict_classifier(classifier):
    classifier.fit(data)
    predicted_labels = classifier.predict(data)
    assert len(predicted_labels) == len(y)

# Test whether the inertia score is calculated correctly
def test_inertia_score(classifier):
    classifier.fit(data)
    inertia_score = classifier.inertia_
    assert isinstance(inertia_score, float)

# Test whether the homogeneity score is calculated correctly
def test_homogeneity_score(classifier):
    classifier.fit(data)
    homogeneity_score = classifier.score(data)
    assert isinstance(homogeneity_score, float)

# Test whether the completeness score is calculated correctly
def test_completeness_score(classifier):
    classifier.fit(data)
    completeness_score = classifier.score(data)
    assert isinstance(completeness_score, float)

# Test whether the v_measure score is calculated correctly
def test_v_measure_score(classifier):
    classifier.fit(data)
    v_measure_score = classifier.score(data)
    assert isinstance(v_measure_score, float)

# Test whether the adjusted_rand_score is calculated correctly
def test_adjusted_rand_score(classifier):
    classifier.fit(data)
    adjusted_rand_score = classifier.score(data)
    assert isinstance(adjusted_rand_score, float)

# Test whether the adjusted_mutual_info_score is calculated correctly
def test_adjusted_mutual_info_score(classifier):
    classifier.fit(data)
    adjusted_mutual_info_score = classifier.score(data)
    assert isinstance(adjusted_mutual_info_score, float)

# Test whether the silhouette_score is calculated correctly
def test_silhouette_score(classifier):
    classifier.fit(data)
    silhouette_score = classifier.score(data)
    assert isinstance(silhouette_score, float)
