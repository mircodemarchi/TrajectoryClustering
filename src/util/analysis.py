"""Data Analysis

"""
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import seaborn as sns
import os
import time
from .util import get_elapsed
from .log import Log
from .constant import IMAGE_FOLDER, HTML_FOLDER

log = Log(__name__, enable_console=True, enable_file=False)


class DataAnalysis:
    def __init__(self, data, feature_names, dataset_name=None, prefix=None, target=None, save_file=False,
                 log_lvl=None):
        self.feature_names = feature_names
        self.target_name = target
        self.dataset_name = "" if dataset_name is None else dataset_name
        self.filename_prefix = "" if prefix is None else prefix + "_"
        self.title_prefix = "" if prefix is None else prefix.capitalize()
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

        self.html_folder = os.path.join(HTML_FOLDER, dataset_name) if dataset_name is not None else HTML_FOLDER
        if not os.path.exists(self.html_folder) and self.save_file:
            os.makedirs(self.html_folder)

    def show_distributions(self, filename="distributions.png",
                           title="Single Feature Distributions"):
        log.d("DataAnalysis show distributions of {}".format(self.dataset_name))
        start = time.time()

        title = "{} {}".format(self.title_prefix, title)
        feature_names = self.feature_names
        fig, axs = plt.subplots(int((len(feature_names) + 1) / 2), 2, figsize=(20, 3*len(feature_names)), squeeze=False)
        for index, feature in enumerate(feature_names):
            sns.histplot(self.x, x=feature, stat="density", kde=True, ax=axs[int(index / 2), index % 2])
            axs[int(index / 2), index % 2].set_title(feature, fontsize=16)
        plt.suptitle(title, fontsize=16)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_2d_distributions(self, couples, filename="2D_distributions.png", title="2D Distributions"):
        log.d("DataAnalysis show 2D distribution of {} on couples {}".format(self.dataset_name, couples))
        start = time.time()

        title = "{} {}".format(self.title_prefix, title)
        fig, axs = plt.subplots(len(couples), 1, figsize=(20, 12*len(couples)), squeeze=False)
        for index, (x, y) in enumerate(couples):
            sns.kdeplot(data=self.x, x=x, y=y, ax=axs[index, 0])
            axs[index, 0].set_title("{} and {}".format(x, y), fontsize=16)
        plt.suptitle(title, fontsize=16)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_joint(self, on, filename="joint.png", title="Joint"):
        log.d("DataAnalysis show joint plot of {} on {}".format(self.dataset_name, on))
        start = time.time()

        title = "{} {}".format(self.title_prefix, title)
        x, y, z = None, None, None
        if len(on) == 3:
            x, y, z = on
            title = "{}, {} and {} ".format(x, y, z) + title
        elif len(on) == 2:
            x, y = on
            title = "{} and {} ".format(x, y) + title
        else:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        g = sns.JointGrid(data=self.x, x=x, y=y, hue=z, palette="Spectral")
        g.plot_joint(sns.scatterplot, marker=".")
        g.plot_marginals(sns.histplot, kde=True, stat="density")
        g.plot_marginals(sns.rugplot, height=-.15, clip_on=False)
        g.fig.suptitle(title)

        if self.save_file:
            if z is None:
                filename = self.filename_prefix + "*{}*{}*_".format(x, y) + filename
            else:
                filename = self.filename_prefix + "*{}*{}*{}*_".format(x, y, z) + filename
            g.savefig(os.path.join(self.image_folder, filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_line(self, on, filename="line.png", title="Line"):
        log.d("DataAnalysis show line plot of {} on {}".format(self.dataset_name, on))
        start = time.time()

        title = "{} {}".format(self.title_prefix, title)
        x, y, z, w = None, None, None, None
        legend = "auto"
        if len(on) == 3:
            x, y, z = on
            title = "{}, {} and {} ".format(x, y, z) + title
        elif len(on) == 4:
            x, y, w, z = on
            legend = "full"
            title = "{}, {}, {} and {} ".format(x, y, z, w) + title
        else:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        p = sns.relplot(x=x, y=y, kind="line", hue=z, col=w, col_wrap=None if w is None else 10, estimator=None,
                        data=self.x, legend=legend, sort=False, palette="Spectral")
        p.fig.suptitle(title)

        if self.save_file:
            if w is None:
                filename = self.filename_prefix + "*{}*{}*{}*_".format(x, y, z) + filename
            else:
                filename = self.filename_prefix + "*{}*{}*{}*{}*_".format(x, y, z, w) + filename
            p.savefig(os.path.join(self.image_folder, filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_line_map(self, on, hover_data=None, filename="line_map.html"):
        log.d("DataAnalysis show line map of {} on {}".format(self.dataset_name, on))
        start = time.time()

        if len(on) != 3:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        x, y, z = on
        fig = px.line_mapbox(self.x, lat=x, lon=y, color=z,
                             hover_name=None if hover_data is None else hover_data[0],
                             hover_data=None if hover_data is None else hover_data[1:])
        fig.update_layout(mapbox_style="open-street-map",
                          mapbox_zoom=8,
                          margin={"r": 0, "t": 0, "l": 0, "b": 0})
        plotly.offline.plot(fig, filename=os.path.join(self.html_folder, self.filename_prefix + filename))

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_scatter_map(self, on, hover_data=None, filename="scatter_map.html"):
        log.d("DataAnalysis show scatter map of {} on {}".format(self.dataset_name, on))
        start = time.time()

        if len(on) != 3:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        x, y, z = on
        fig = px.scatter_mapbox(self.x, lat=x, lon=y, color=z,
                                hover_name=None if hover_data is None else hover_data[0],
                                hover_data=None if hover_data is None else hover_data[1:])
        fig.update_layout(
            mapbox_style="stamen-toner",
            mapbox_zoom=8,
            margin={"r": 0, "t": 0, "l": 0, "b": 0})
        plotly.offline.plot(fig, filename=os.path.join(self.html_folder, self.filename_prefix + filename))

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_target_distribution(self, filename="target_distribution.png",
                                 title="Check normal distribution on target feature"):
        log.d("DataAnalysis show target distribution of {}".format(self.dataset_name))
        start = time.time()

        title = "{} - {}".format(self.title_prefix, title)
        if self.y is None:
            log.e("DataAnalysis target feature not specified")
            return self

        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.histplot(self.y, bins=30, kde=True)
        plt.suptitle(title, fontsize=16)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_relations(self, filename="relations.png"):
        log.d("DataAnalysis show relations between features and target feature of {}".format(self.dataset_name))
        start = time.time()

        if self.y is None:
            log.e("DataAnalysis target feature not specified")
            return self

        fig, axs = plt.subplots(len(self.feature_names), 1, figsize=(14, 60), squeeze=False)
        for index, feature in enumerate(self.feature_names):
            axs[index, 0].scatter(x=self.x[feature], y=self.y, marker="|")
            axs[index, 0].set_xlabel(feature)
            axs[index, 0].set_ylabel("Target")
        plt.suptitle("Scatter of target over features", fontsize=16)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_correlation_matrix(self, filename="correlation_matrix.png",
                                title="Correlation matrix of features"):
        log.d("DataAnalysis show correlation matrix of {}".format(self.dataset_name))
        start = time.time()

        title = "{} - {}".format(self.title_prefix, title)
        correlation_matrix = self.x.corr().round(2)
        # annot = True to print the values inside the square
        sns.heatmap(data=correlation_matrix, annot=True)
        plt.suptitle(title, fontsize=16)

        if self.save_file:
            plt.savefig(os.path.join(self.image_folder, self.filename_prefix + filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self
