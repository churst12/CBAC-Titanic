import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
# Constants
DEAD = 0
ALIVE = 1
FEMALE = 0
MALE = 1


class ClassifierTestPlotter:
    def __init__(self, one_hot_data: pd.DataFrame, x_label: str, y_label: str, target_label: str):
        self.data = one_hot_data
        self.x_label = x_label
        self.y_label = y_label
        self.target_label = target_label

        self.plot = plt.subplot(121, title="Actual")
        self.plot.set_xlabel(self.x_label)
        self.plot.set_ylabel(self.y_label)

        for index, passenger in self.data.iterrows():
            self.plot.scatter(passenger[self.x_label],
                              passenger[self.y_label],
                              c='red' if passenger[self.target_label] == DEAD else 'green',
                              alpha=0.1)

    def feature_range(self) -> np.ndarray:
        # Find max values in each dimension
        rounded_features = self.data[[self.x_label, self.y_label]].astype(int)
        test_set_dimensions = np.stack((rounded_features.min(axis=0), rounded_features.max(axis=0))).T
        # print("Data range: ", test_set_dimensions)
        return test_set_dimensions

    # Not currently working well
    #
    # def plot_prediction_field(self, classifier):
    #     test_set_dimensions = self.feature_range()
    #
    #     test_parameters = self.data.mean(axis=0).drop(self.target_label)
    #
    #     for x_coord in range(test_set_dimensions[0, 0], test_set_dimensions[0, 1] + 1):
    #         for y_coord in range(test_set_dimensions[1, 0], test_set_dimensions[1, 1] + 1):
    #             test_parameters[self.x_label] = x_coord
    #             test_parameters[self.y_label] = y_coord
    #
    #             self.plot.scatter(x_coord + 0.1,
    #                               y_coord,
    #                               alpha=0.1,
    #                               c='red' if classifier.predict(test_parameters.values.reshape(1, -1)) == DEAD else 'green')
    #     plt.show()

    def plot_prediction(self, classifier):
        side_plot = plt.subplot(122, title="Predicted")

        for index, passenger in self.data.iterrows():
            side_plot.scatter(passenger[self.x_label] + 0.1,
                              passenger[self.y_label],
                              c='red' if classifier.predict(passenger.drop(self.target_label).values.reshape(1, -1)) == DEAD else 'green',
                              alpha=0.1)
        plt.show()
