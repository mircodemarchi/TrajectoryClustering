"""Linear Regression

Library implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from util import DataAnalysis


class LinearRegressionWrapper:
    """LinearRegressionWrapper"""

    def __init__(self, data, feature_names, target):
        self.analysis = DataAnalysis(data, feature_names, target)
        self.feature_names = feature_names
        self.target_name = target
        self.x = pd.DataFrame(np.c_[data[feature_names]], columns=feature_names)
        self.y = data[target].to_numpy()
        self.coefficients = None
        self.y_pred = None
        self.residuals = None

    def show_data_analysis(self, file=False):
        self.analysis.show_target_distribution(file)
        self.analysis.show_relations(file)
        self.analysis.show_correlation_matrix(file)

    def show_regression_on_data(self, file=False, filename="show_regression_on_data.png"):
        fig, axs = plt.subplots(int((len(self.feature_names) + 1) / 2), 2, figsize=(14, 35))
        lrcoeff_sk = self.coefficients[:len(self.feature_names)]
        bias_sk = self.coefficients[len(self.feature_names)]
        for index, feature in enumerate(self.feature_names):
            axs[int(index / 2), index % 2].scatter(x=self.x[feature], y=self.y,
                                                   marker='o', c='b', label="Ground Truth")
            axs[int(index / 2), index % 2].scatter(x=self.x[feature], y=self.y_pred,
                                                   marker='o', c='g', label="Predictions")

            axs[int(index / 2), index % 2].plot(self.x[feature],
                                                lrcoeff_sk[index] * self.x[feature] + bias_sk,
                                                c='r', label="Linear Regression Line")

            axs[int(index / 2), index % 2].set_xlabel(feature)
            axs[int(index / 2), index % 2].set_ylabel(self.target_name)

        if file:
            plt.savefig(filename)

        fig.suptitle("Show Regression in relation with features\n\n\n", fontsize=16)
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        plt.show()

    def print_coefficient_importance(self):
        lrcoeff_sk = self.coefficients[:len(self.feature_names)]
        print("Feature number with minor importance on positive side: ",
              np.argmin(np.where(lrcoeff_sk > 0, lrcoeff_sk, np.full(lrcoeff_sk.shape[0], np.Inf))) + 1, " [",
              self.feature_names[np.argmin(np.where(lrcoeff_sk > 0, lrcoeff_sk, np.full(lrcoeff_sk.shape[0], np.Inf)))],
              "]\n")
        print("Feature number with major importance on positive side: ", np.argmax(lrcoeff_sk) + 1,
              " [", self.feature_names[np.argmax(lrcoeff_sk)], "]\n")
        print("Feature number with minor importance: ", np.argmin(np.abs(lrcoeff_sk)) + 1,
              " [", self.feature_names[np.argmin(np.abs(lrcoeff_sk))], "]\n")
        print("Feature number with major importance: ", np.argmax(np.abs(lrcoeff_sk)) + 1,
              " [", self.feature_names[np.argmax(np.abs(lrcoeff_sk))], "]\n")

    def show_regression_analysis(self, file=False, filename="show_regression_analysis.png"):
        # Plot the results of the linear regression
        fig, axs = plt.subplots(4, figsize=(12, 20))

        # Plot predicted result compared with real results.
        axs[0].set_title("Relation between predicted and real results", fontsize=16)
        axs[0].scatter(self.y, self.y_pred)
        axs[0].set_xlabel("Measured")
        axs[0].set_ylabel("Predicted")
        axs[1].set_title("Plot of prediction and ground truth with ordered indexing", fontsize=16)
        axs[1].plot(self.y, "bo", label="Ground Truth")
        axs[1].plot(self.y_pred, "go", label="Predictions")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].legend(loc="best")

        # Plot residuals analysis.
        sns.histplot(self.residuals, kde=True, ax=axs[2])
        axs[2].set_title("Residuals distribution", fontsize=16)
        axs[2].set_xlabel("Residuals")
        axs[2].set_ylabel("Frequency")
        stats.probplot(self.residuals, plot=axs[3], fit=False)
        axs[3].set_title("Residuals in relation with normal probability plot", fontsize=16)

        if file:
            plt.savefig(filename)

        plt.show()

    def exec_expected(self, preprocessing=True):
        x_exec = self.x.copy()

        scaler = StandardScaler()
        scaler.fit(x_exec)

        # Compute preprocessing.
        if preprocessing:
            x_exec = pd.DataFrame(np.c_[scaler.transform(x_exec)], columns=self.feature_names)

        # Perform linear regression model estimation.
        lr_sk = LinearRegression()
        lr_sk.fit(x_exec, self.y)
        self.coefficients = np.append(lr_sk.coef_, lr_sk.intercept_)

        # Perform prediction of linear regression.
        self.y_pred = lr_sk.predict(x_exec)

        # Calculate Residuals.
        self.residuals = self.y - self.y_pred

        # Calculate Mean Square Error.
        mse = mean_squared_error(self.y, self.y_pred)
        return mse

    def exec_exact(self, preprocessing=True):
        x_exec = self.x.copy()

        # Compute preprocessing.
        if preprocessing:
            x_exec = (x_exec - self.x.mean()) / self.x.std()

        # Perform linear regression model estimation.
        x_np = x_exec.to_numpy()
        theta = np.linalg.inv(x_np.T.dot(x_np)).dot(x_np.T).dot(self.y)
        bias = np.mean(self.y) - theta.T.dot(np.mean(x_np, axis=0))
        self.coefficients = np.append(theta, bias)

        # Perform prediction of linear regression.
        self.y_pred = x_np.dot(theta) + bias

        # Calculate Residuals.
        self.residuals = self.y - self.y_pred

        # Calculate Mean Square Error.
        mse = np.mean(np.square(self.residuals))
        return mse

    def exec_gradient_descent(self, preprocessing=True):
        x_exec = self.x.copy()

        # Define learning rate.
        def gradient_learning_rate(i):
            return 1e-5

        # Define gradient descent end condition.
        def gradient_end_condition(coeff1, coeff2, i):
            return (coeff1 == coeff2).all()

        # Compute preprocessing.
        if preprocessing:
            x_exec = (x_exec - self.x.mean()) / self.x.std()

        # Perform linear regression model estimation.
        x_np = x_exec.to_numpy()

        # Gradient descent to estimate theta
        theta = np.random.rand(len(self.feature_names))
        iter = 0
        while True:
            theta_prev = theta.copy()
            theta -= gradient_learning_rate(iter) * (-2) * x_np.T.dot((self.y - x_np.dot(theta)))
            iter += 1

            if gradient_end_condition(theta, theta_prev, iter):
                break

        # Calculate bias.
        bias = np.mean(self.y) - theta.T.dot(np.mean(x_np, axis=0))
        self.coefficients = np.append(theta, bias)

        # Perform prediction of linear regression.
        self.y_pred = x_np.dot(theta) + bias

        # Calculate Residuals.
        self.residuals = self.y - self.y_pred

        # Calculate Mean Square Error.
        mse = np.mean(np.square(self.residuals))
        return mse, iter

    def get_theta(self):
        if self.coefficients:
            return self.coefficients[:len(self.feature_names)]
        return None

    def get_bias(self):
        if self.coefficients:
            return self.coefficients[len(self.feature_names)]
        return None
