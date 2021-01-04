"""Data Analysis

"""
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from .util import get_elapsed
from .log import Log

IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "image")

log = Log(__name__, enable_console=True, enable_file=False)


class DataAnalysis:
    def __init__(self, data, feature_names, dataset_name=None, filename_prefix=None, target=None, save_file=False,
                 log_lvl=None):
        self.feature_names = feature_names
        self.target_name = target
        self.dataset_name = "" if dataset_name is None else dataset_name
        self.filename_prefix = "" if filename_prefix is None else filename_prefix + "_"
        self.x = data[feature_names]
        self.y = None if target is None else data[target]
        self.save_file = save_file

        if log_lvl is not None:
            if not isinstance(log_lvl, int):
                raise ValueError("Invalid log level: {}".format(log_lvl))
            log.set_level(log_lvl)

        self.image_folder = os.path.join(IMAGE_FOLDER, dataset_name) if dataset_name is not None else IMAGE_FOLDER
        if not os.path.exists(self.image_folder) and self.save_file:
            os.makedirs(self.image_folder)

    def show_distributions(self, filename="distributions.png",
                           title="Single Feature Distributions"):
        log.d("DataAnalysis show distributions of {}".format(self.dataset_name))
        start = time.time()

        feature_names = self.feature_names
        fig, axs = plt.subplots(int((len(feature_names) + 1) / 2), 2, figsize=(20, 3*len(feature_names)), squeeze=False)
        for index, feature in enumerate(feature_names):
            sns.histplot(self.x, x=feature, stat="density", kde=True, ax=axs[int(index / 2), index % 2])
            axs[int(index / 2), index % 2].set_title(feature, fontsize=16)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.suptitle(title, fontsize=16)
        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_2d_distributions(self, couples, filename="2D_distributions.png", title="2D Distributions"):
        log.d("DataAnalysis show 2D distribution of {} on couples {}".format(self.dataset_name, couples))
        start = time.time()

        fig, axs = plt.subplots(len(couples), 1, figsize=(20, 12*len(couples)), squeeze=False)
        for index, (x, y) in enumerate(couples):
            sns.kdeplot(data=self.x, x=x, y=y, ax=axs[index, 0])
            axs[index, 0].set_title("{} and {}".format(x, y), fontsize=16)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.suptitle(title, fontsize=16)
        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_joint(self, on, filename="joint.png", title="Joint"):
        log.d("DataAnalysis show joint plot of {} on {}".format(self.dataset_name, on))
        start = time.time()

        if len(on) == 3:
            x, y, z = on
        else:
            x, y = on
            z = None

        g = sns.JointGrid(data=self.x, x=x, y=y, hue=z, palette="Spectral")
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, kde=True, stat="density")
        g.plot_marginals(sns.rugplot, height=-.15, clip_on=False)
        g.fig.suptitle("{} - {} - {} ".format(x, y, z) + title)

        if self.save_file:
            g.savefig(os.path.join(self.image_folder, self.filename_prefix + "{}:{}:{}_".format(x, y, z) + filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_line(self, on, filename="line.png", title="Line"):
        log.d("DataAnalysis show line plot of {} on {}".format(self.dataset_name, on))
        start = time.time()

        x, y, z, w = None, None, None, None
        legend = "auto"
        if len(on) == 3:
            x, y, z = on
        elif len(on) == 4:
            x, y, w, z = on
            legend = "full"

        p = sns.relplot(x=x, y=y, kind="line", hue=z, col=w, col_wrap=None if w is None else 10, estimator=None,
                        data=self.x, legend=legend, sort=False, palette="Spectral")

        if p is not None:
            if self.save_file:
                p.savefig(os.path.join(self.image_folder, self.filename_prefix + "{}:{}:{}:{}_".format(x, y, z, w)
                                       + filename))

            p.fig.suptitle("{}, {}, {} and {} ".format(x, y, z, w) + title)
            plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_target_distribution(self, filename="target_distribution.png",
                                 title="Check normal distribution on target feature"):
        log.d("DataAnalysis show target distribution of {}".format(self.dataset_name))
        start = time.time()

        if self.y is None:
            log.e("DataAnalysis target feature not specified")
            return self

        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.histplot(self.y, bins=30, kde=True)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.suptitle(title, fontsize=16)
        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_relations(self, filename="relations.png"):
        log.d("DataAnalysis show relations between features and target feature of {}".format(self.dataset_name))
        start = time.time()

        if self.y is None:
            log.e("DataAnalysis target feature not specified")
            return

        fig, axs = plt.subplots(len(self.feature_names), 1, figsize=(14, 60), squeeze=False)
        for index, feature in enumerate(self.feature_names):
            axs[index, 0].scatter(x=self.x[feature], y=self.y, marker="|")
            axs[index, 0].set_xlabel(feature)
            axs[index, 0].set_ylabel("Target")

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.suptitle("Scatter of target over features", fontsize=16)
        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_correlation_matrix(self, filename="correlation_matrix.png",
                                title="Correlation matrix of features"):
        log.d("DataAnalysis show correlation matrix of {}".format(self.dataset_name))
        start = time.time()

        correlation_matrix = self.x.corr().round(2)
        # annot = True to print the values inside the square
        sns.heatmap(data=correlation_matrix, annot=True)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.suptitle(title, fontsize=16)
        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self
