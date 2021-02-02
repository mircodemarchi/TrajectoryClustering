"""Data Analysis

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#from mpl_toolkits.basemap import Basemap
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np
import pandas as pd
import os
import time
from .util import get_elapsed
from .log import Log
from .constant import IMAGE_FOLDER, HTML_FOLDER

log = Log(__name__, enable_console=True, enable_file=False)

PALETTE = "Set1"


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
            sns.histplot(self.x, x=feature, stat="density", common_norm=True, kde=True,
                         ax=axs[int(index / 2), index % 2])
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

        g = sns.JointGrid(data=self.x, x=x, y=y, hue=z, palette=PALETTE)
        g.plot_joint(sns.scatterplot, marker=".", linewidth=0, legend=False)
        g.plot_marginals(sns.histplot, kde=False, common_norm=True, stat="density", alpha=0.5, linewidth=1,
                         element="step", fill=False)
        # g.plot_marginals(sns.rugplot, height=-.15, clip_on=False)
        g.fig.suptitle(title)

        if self.save_file:
            if z is None:
                filename = self.filename_prefix + "-{}-{}-_".format(x, y) + filename
            else:
                filename = self.filename_prefix + "-{}-{}-{}-_".format(x, y, z) + filename
            g.savefig(os.path.join(self.image_folder, filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_line(self, on, groupby, filename="line.png", title="Line"):
        log.d("DataAnalysis show line plot of {} on {}".format(self.dataset_name, on))
        start = time.time()

        title = "{} {}".format(self.title_prefix, title)
        fig, ax = plt.subplots()
        x, y, z = None, None, None
        if len(on) == 3:
            x, y, z = on
            title = "{}, {} and {} ".format(x, y, z) + title
            unique = self.x[z].unique()
            palette = dict(zip(unique, sns.color_palette(PALETTE, n_colors=len(unique))))
            for _, g in self.x.groupby(by=groupby):
                sns.lineplot(x=x, y=y, hue=z, estimator=None, data=g, legend=False, sort=False, palette=palette, ax=ax)
        elif len(on) == 2:
            x, y = on
            title = "{} and {} ".format(x, y) + title
            palette = sns.color_palette(n_colors=1)[0]
            for _, g in self.x.groupby(by=groupby):
                sns.lineplot(x=x, y=y, estimator=None, data=g[[x, y]], legend=False, sort=False, color=palette, ax=ax)
        else:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        fig.suptitle(title)

        if self.save_file:
            if z is None:
                filename = self.filename_prefix + "-{}-{}-_".format(x, y) + filename
            else:
                filename = self.filename_prefix + "-{}-{}-{}-_".format(x, y, z) + filename
            fig.savefig(os.path.join(self.image_folder, filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_3d_line(self, on, groupby, filename="3d_line.png", title="3D Line"):
        log.d("DataAnalysis show 3d line plot of {} on {}".format(self.dataset_name, on))
        start = time.time()

        title = "{} {}".format(self.title_prefix, title)
        if len(on) == 3:
            x, y, z = on
            title = "{}, {} and {} ".format(x, y, z) + title
        else:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        unique = self.x[z].unique()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # # Previous version
        # ax.set_zlim(self.x[z].min(), self.x[z].max())
        # ax.set_xlim(self.x[x].min(), self.x[x].max())
        # ax.set_ylim(self.x[y].min(), self.x[y].max())
        # cmap = ListedColormap(sns.color_palette(PALETTE, n_colors=len(unique)).as_hex())
        # norm = BoundaryNorm(unique, cmap.N)
        # for _, g in self.x.groupby(by=groupby):
        #     if g[z].unique().size == 1:
        #         points = g[[x, y, z]].to_numpy().reshape(-1, 1, 3)
        #         segments = np.concatenate([points, np.roll(points, -1, axis=0)], axis=1)
        #         lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidths=0.5, array=g[z])
        #         ax.add_collection3d(lc)
        #     else:
        #         for _, sub_g in g.groupby(by=z):
        #             points = sub_g[[x, y, z]].to_numpy().reshape(-1, 1, 3)
        #             segments = np.concatenate([points, np.roll(points, -1, axis=0)], axis=1)
        #             lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidths=0.5, array=sub_g[z])
        #             ax.add_collection3d(lc)

        palette = pd.DataFrame(sns.color_palette(PALETTE, n_colors=len(unique)).as_hex(), index=unique)
        for _, g in self.x.groupby(by=groupby):
            if g[z].unique().size == 1:
                ax.plot3D(g[x], g[y], g[z], palette.loc[g[z].unique()[0], 0], linewidth=0.5)
            else:
                for sub_g_name, sub_g in g.groupby(by=z):
                    ax.plot3D(sub_g[x], sub_g[y], sub_g[z], palette.loc[sub_g_name, 0], linewidth=0.5)
        plt.suptitle(title)

        if self.save_file:
            filename = self.filename_prefix + "-{}-{}-{}-_".format(x, y, z) + filename
            plt.savefig(os.path.join(self.image_folder, filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_line_table(self, on, filename="line_table.png", title="Line Table"):
        log.d("DataAnalysis show line table plot of {} on {}".format(self.dataset_name, on))
        start = time.time()

        title = "{} {}".format(self.title_prefix, title)
        if len(on) == 4:
            x, y, w, z = on
            title = "{}, {}, {} and {} ".format(x, y, z, w) + title
        else:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        p = sns.relplot(x=x, y=y, kind="line", hue=z, col=w, col_wrap=None if w is None else 10, estimator=None,
                        data=self.x, legend="full", sort=False, palette=PALETTE)
        p.fig.suptitle(title)

        if self.save_file:
            filename = self.filename_prefix + "-{}-{}-{}-{}-_".format(x, y, z, w) + filename
            p.savefig(os.path.join(self.image_folder, filename))

        plt.show()

        end = time.time()
        log.d("elapsed time: {}".format(get_elapsed(start, end)))
        return self

    def show_line_map(self, on, groupby, hover_data=None, filename="line_map.html"):
        log.d("DataAnalysis show line map of {} on {}".format(self.dataset_name, on))
        start = time.time()

        if len(on) != 3:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        x, y, z = on
        unique = self.x[z].unique()
        palette = pd.DataFrame(sns.color_palette(PALETTE, n_colors=len(unique)).as_hex(), index=unique, columns=["p"])
        fig = make_subplots()
        for _, g in self.x.groupby(by=groupby):
            fig.add_scattermapbox(lat=g[x], lon=g[y], mode="lines",
                                  line=go.scattermapbox.Line(color=palette.loc[g[z], "p"].values[0]),
                                  name=None if hover_data is None else str(g[hover_data[0]].values[0]),
                                  text=None if hover_data is None else g[hover_data[1:]].values.tolist())
        # fig = px.line_mapbox(self.x, lat=x, lon=y, color=z,
        #                      hover_name=None if hover_data is None else hover_data[0],
        #                      hover_data=None if hover_data is None else hover_data[1:])
        fig.update_layout(mapbox_style="stamen-toner",
                          mapbox_zoom=7,
                          mapbox_center_lat=self.x[x].mean(),
                          mapbox_center_lon=self.x[y].mean(),
                          showlegend=False,
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

    def show_3d_map(self, on, groupby, hover_data=None, filename="3d_map.html"):
        log.d("DataAnalysis show 3d map of {} on {}".format(self.dataset_name, on))
        start = time.time()

        if len(on) != 3:
            log.e("show_line parameter error: \"on\" size {}".format(len(on)))
            return self

        y, x, z = on
        unique = self.x[z].unique()
        palette = pd.DataFrame(sns.color_palette(PALETTE, n_colors=len(unique)).as_hex(), index=unique, columns=["p"])
        fig = make_subplots()
        for _, g in self.x.groupby(by=groupby):
            fig.add_scatter3d(x=g[x], y=g[y], z=g[z], line=dict(color=palette.loc[g[z], "p"]), mode="lines")
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
