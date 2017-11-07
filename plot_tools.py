"""
The :mod:`plot_tools` module contains data visualization tools
for use in the titanic problem
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
DEAD = 0
ALIVE = 1
FEMALE = 0
MALE = 1


class ClassifierTestPlotter:
    def __init__(self, one_hot_data: pd.DataFrame, x_label: str, y_label: str, target_label: str):
        """Plots actual and predicted test set side by side"""
        self.data = one_hot_data
        self.x_label = x_label
        self.y_label = y_label
        self.target_label = target_label

        self.plot = plt.subplot(121, title='Actual')
        self.plot.set_xlabel(self.x_label)
        self.plot.set_ylabel(self.y_label)

        for _, passenger in self.data.iterrows():
            self.plot.scatter(passenger[self.x_label],
                              passenger[self.y_label],
                              c='red' if passenger[self.target_label] == DEAD else 'green',
                              alpha=0.1)

    def decode_one_hot(self, one_hot_data: pd.DataFrame):
        """Recombine one_hot_encoded data columns into one, for easy plotting"""
        pass

    def feature_range(self) -> np.ndarray:
        """Find max values in each dimension"""
        rounded_features = self.data[[self.x_label, self.y_label]].astype(int)
        test_set_dimensions = np.stack((rounded_features.min(axis=0), rounded_features.max(axis=0))).T
        # print('Data range: ', test_set_dimensions)
        return test_set_dimensions

    def plot_prediction(self, classifier):
        """Uses an SKLearn classifier to classify points and plots prediction"""
        side_plot = plt.subplot(122, title='Predicted')

        for _, passenger in self.data.iterrows():
            side_plot.scatter(passenger[self.x_label],
                              passenger[self.y_label],
                              c='red' if classifier.predict(passenger.drop(self.target_label).values.reshape(1, -1)) == DEAD else 'green',
                              alpha=0.1)
        plt.show()
