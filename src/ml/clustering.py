import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from util.util import get_elapsed
from util.log import Log
from util.constant import IMAGE_FOLDER


log = Log(__name__, enable_console=True, enable_file=False)

CLUSTER_ID = "c_id"


class Clustering:
    def __init__(self, data, feature_names, dataset_name=None, save_file=False):
        self.feature_names = feature_names
        self.x = pd.DataFrame(np.c_[data[feature_names]], columns=feature_names)
        self.dataset_name = "" if dataset_name is None else dataset_name
        self.save_file = save_file
        self.wcss = None
        self.inertia = None
        self.labels = None
        self.model = None
        self.method = None
        self.x_preprocessed = None

        self.image_folder = os.path.join(IMAGE_FOLDER, dataset_name) if dataset_name is not None else IMAGE_FOLDER
        if not os.path.exists(self.image_folder) and self.save_file:
            os.makedirs(self.image_folder)

    def __preprocessing(self, x, standardize=False, normalize=False, unit_norm=False, pca=False, components=None):
        start = time.time()
        columns = x.columns
        x = x.to_numpy()
        if standardize:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)

        if normalize:
            mmscaler = MinMaxScaler()
            x = mmscaler.fit_transform(x)

        if unit_norm:
            normalizer = Normalizer()
            x = normalizer.fit_transform(x)

        if pca:
            if type(components) is list:
                ret_x = np.empty((len(components), x.shape[0]))
                i = 0
                for c in components:
                    if len(c) > 1:
                        pca_model = PCA(n_components=1)
                        ret_x[i] = pca_model.fit_transform(x[:, columns.get_loc(c[0]):columns.get_loc(c[-1])]).squeeze()
                    else:
                        ret_x[i] = x[:, columns.get_loc(c[0])]
                    i += 1
                x = ret_x.T
            elif type(components) is int:
                n_components = int(components)
                pca_model = PCA(n_components=n_components)
                x = pca_model.fit_transform(x)
            else:
                if components is not None:
                    log.w("__preprocessing: not expected \"components\", take it as None")
                variance_cumulated = self.__get_cumulated_variance(x)
                n_components = np.where(variance_cumulated > 0.8)[0][0]
                pca_model = PCA(n_components=max(n_components, 1))
                x = pca_model.fit_transform(x)

        end = time.time()
        return x, get_elapsed(start, end)

    def __exec(self, x, method, n_clusters):
        start = time.time()
        if method == "k-means":
            km = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
            km.fit(x)
            inertia = km.inertia_
            labels = km.labels_
            model = km
        elif method == "mean-shift":
            bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=500)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(x)
            inertia = None
            labels = ms.labels_
            model = ms
        elif method == "gaussian-mixture":
            mg = GaussianMixture(n_components=n_clusters)
            mg.fit(x)
            inertia = None
            labels = mg.predict(x)
            model = mg
        elif method == "full-agglomerative":
            fa = AgglomerativeClustering(linkage="complete", n_clusters=n_clusters, compute_distances=True)
            fa.fit(x)
            inertia = None
            labels = fa.labels_
            model = fa
        elif method == "ward-agglomerative":
            fa = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters, compute_distances=True)
            fa.fit(x)
            inertia = None
            labels = fa.labels_
            model = fa
        else:
            log.e("Clustering __exec: method {} not recognised".format(method))
            return None, None, 0
        end = time.time()
        return inertia, labels, model, get_elapsed(start, end)

    def __get_cumulated_variance(self, x):
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        pca_model = PCA()
        pca_model.fit(x)
        variance_cumulated = pca_model.explained_variance_ratio_.cumsum()
        return variance_cumulated

    def exec(self, method, n_clusters, standardize=False, normalize=False, unit_norm=False, pca=False, components=None):
        log.d("Clustering {} preprocessing".format(self.dataset_name))
        x, elapsed = self.__preprocessing(self.x, standardize=standardize, normalize=normalize, unit_norm=unit_norm,
                                          pca=pca, components=components)
        self.x_preprocessed = pd.DataFrame(x)
        log.d("elapsed time: {}".format(elapsed))
        log.d("components: {}".format(x.shape[1]))

        log.d("Clustering {} exec {}".format(self.dataset_name, method))
        inertia, labels, model, elapsed = self.__exec(x, method, n_clusters)
        log.d("elapsed time: {}".format(elapsed))

        self.method = method
        self.inertia = inertia
        self.labels = labels
        self.model = model
        return self

    def test(self, method, range_clusters=range(1, 50), standardize=False, normalize=False, unit_norm=False,
             pca=False, components=None):
        log.d("Clustering {} preprocessing".format(self.dataset_name))
        x, elapsed = self.__preprocessing(self.x, standardize=standardize, normalize=normalize, unit_norm=unit_norm,
                                          pca=pca, components=components)
        log.d("elapsed time: {}".format(elapsed))
        log.d("components: {}".format(x.shape[1]))

        log.d("Clustering {} k-means test in range {}".format(self.dataset_name, range_clusters))
        start = time.time()
        wcss = []
        for c in range_clusters:
            inertia, _, _, _ = self.__exec(x, method, c)
            wcss.append(inertia)
            sys.stdout.write("\r {:.3f} %".format(c * 100 / range_clusters.stop))

        sys.stdout.write("\r")
        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))

        self.wcss = wcss
        return self

    def show_variance(self, title="Explained Variance by Components", save_file=False, prefix=None):
        prefix = "" if prefix is None else "{}_".format(prefix)
        filename = prefix + "variance.png"
        log.d("Clustering {} show cumulative variance".format(self.dataset_name))
        start = time.time()
        variance_cumulated = self.__get_cumulated_variance(self.x)

        plt.figure(figsize=(10, 8))
        plt.plot(range(1, variance_cumulated.shape[0] + 1), variance_cumulated, marker="o", linestyle="--")
        plt.title(title)
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")

        if save_file:
            plt.savefig(os.path.join(self.image_folder, filename))

        plt.show()
        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_wcss(self, title="Within Cluster Sums of Squares", save_file=False, prefix=None):
        prefix = "" if prefix is None else "{}_".format(prefix)
        filename = prefix + "wcss.png"
        log.d("Clustering {} show wcss".format(self.dataset_name))
        if self.wcss is None:
            log.e("Clustering error: perform test method before show_wcss")
            return self

        start = time.time()
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(self.wcss) + 1), self.wcss, marker="o", linestyle="--")
        plt.title(title)
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")

        if save_file:
            plt.savefig(os.path.join(self.image_folder, filename))

        plt.show()
        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_dendrogram(self, title="Hierarchical Dendrogram", save_file=False, prefix=None):
        model = self.model
        method = self.method
        log.d("Clustering {} plot dendrogram".format(self.dataset_name))
        if (model is None) or ((method != "ward-agglomerative") and (method != "full-agglomerative")):
            log.e("Clustering error: method performed not compliant with dendrogram, perform an hierarchical method")
            return self

        start = time.time()
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, truncate_mode='level', p=6)
        plt.title(title)

        if save_file:
            filename = prefix + "_dendrogram.png"
            plt.savefig(os.path.join(self.image_folder, filename))
        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def silhouette(self):
        if (self.labels is None) or (self.x_preprocessed is None):
            log.e("Clustering error: labels None, perform exec before silhouette score")
            return None

        return silhouette_score(self.x_preprocessed, self.labels, sample_size=20000)

    def stats(self):
        log.d("Clustering {} stats".format(self.dataset_name))
        if (self.labels is None) or (self.method is None):
            log.e("Clustering error: impossible to print stats if not exec earlier")
            return self

        x = self.x.copy()
        x[CLUSTER_ID] = self.labels

        group = x.groupby(by=CLUSTER_ID)
        log.i("***************************** Clustering - Method {:23} ***************************".format(self.method))
        log.i("[FEATURES MEAN]: \n{}".format(group.mean()))
        log.i("[FEATURES STD]: \n{}".format(group.std()))
        log.i("[SILHOUETTE SCORE]: {}".format(self.silhouette()))
        log.i("*******************************************************************************************************")

        return self
