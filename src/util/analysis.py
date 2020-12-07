"""Data Analysis

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DataAnalysis:
    def __init__(self, data, feature_names, target):
        self.feature_names = feature_names
        self.target_name = target
        self.x = pd.DataFrame(np.c_[data[feature_names]], columns=feature_names)
        self.y = data[target]

    def show_target_distribution(self, file=False, filename="target_distribution.png"):
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.histplot(self.y, bins=30, kde=True)

        if file:
            plt.savefig(filename)

        plt.suptitle("Check normal distribution on target feature", fontsize=16)
        plt.show()

    def show_relations(self, file=False, filename="relations.png"):
        fig, axs = plt.subplots(int((len(self.feature_names) + 1) / 2), 2, figsize=(14, 30))
        for index, feature in enumerate(self.feature_names):
            axs[int(index / 2), index % 2].scatter(x=self.x[feature], y=self.y)
            axs[int(index / 2), index % 2].set_xlabel(feature)
            axs[int(index / 2), index % 2].set_ylabel("Target")

        if file:
            plt.savefig(filename)

        plt.suptitle("Scatter of target over features", fontsize=16, y=0.99)
        plt.show()

    def show_correlation_matrix(self, file=False, filename="correlation_matrix.png"):
        correlation_matrix = self.x.corr().round(2)
        # annot = True to print the values inside the square
        sns.heatmap(data=correlation_matrix, annot=True)

        if file:
            plt.savefig(filename)

        plt.suptitle("Correlation matrix of features", fontsize=16)
        plt.show()
