import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from util import get_elapsed
from util import Log


log = Log(__name__, enable_console=True, enable_file=False)


class Clustering:
    def __init__(self, data, feature_names, dataset_name=None):
        self.feature_names = feature_names
        self.x = pd.DataFrame(np.c_[data[feature_names]], columns=feature_names)
        self.dataset_name = "" if dataset_name is None else dataset_name
        self.wcss = None
        self.inertia = None
        self.labels = None

    def __preprocessing(self, x, standardize, pca):
        start = time.time()
        if standardize:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)

        if pca:
            variance_cumulated = self.__get_cumulated_variance(x)
            n_components = np.where(variance_cumulated > 0.8)[0][0]
            pca_model = PCA(n_components=n_components)
            pca_model.fit(x)
            x = pca_model.transform(x)

        end = time.time()
        return x, get_elapsed(start, end)

    def __exec(self, x, n_clusters):
        start = time.time()
        km = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        km.fit(x)
        end = time.time()
        return km.inertia_, km.labels_, get_elapsed(start, end)

    def __get_cumulated_variance(self, x):
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        pca_model = PCA()
        pca_model.fit(x)
        variance_cumulated = pca_model.explained_variance_ratio_.cumsum()
        return variance_cumulated

    def exec(self, n_clusters, standardize=False, pca=False):
        log.d("Clustering {} preprocessing".format(self.dataset_name))
        x, elapsed = self.__preprocessing(self.x, standardize, pca)
        log.d("elapsed time: {}".format(elapsed))

        log.d("Clustering {} k-means process".format(self.dataset_name))
        inertia, labels, elapsed = self.__exec(x, n_clusters)
        log.d("elapsed time: {}".format(elapsed))

        self.inertia = inertia
        self.labels = labels
        return self

    def test(self, range_clusters=range(1, 50), standardize=False, pca=False):
        log.d("Clustering {} preprocessing".format(self.dataset_name))
        x, elapsed = self.__preprocessing(self.x, standardize, pca)
        log.d("elapsed time: {}".format(elapsed))

        log.d("Clustering {} k-means process".format(self.dataset_name))
        start = time.time()
        wcss = []
        for c in range_clusters:
            inertia, _, _ = self.__exec(x, c)
            wcss.append(inertia)

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))

        self.wcss = wcss
        return self

    def show_variance(self):
        log.d("Clustering {} show cumulative variance".format(self.dataset_name))
        variance_cumulated = self.__get_cumulated_variance(self.x)

        plt.figure(figsize=(10, 8))
        plt.plot(range(1, variance_cumulated.shape[0] + 1), variance_cumulated, marker="o", linestyle="--")
        plt.title("Explained Variance by Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.show()
        return self

    def show_wcss(self):
        log.d("Clustering {} show wcss".format(self.dataset_name))
        if self.wcss is None:
            log.e("Clustering error: perform test method before show_wcss")
            return self

        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(self.wcss) + 1), self.wcss, marker="o", linestyle="--")
        plt.title("K-Means Clustering")
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")
        plt.show()
        return self
