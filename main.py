import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors

# Constants
DEAD = 0
ALIVE = 1


def load_data(train_data_path):
    # Loading data from CSV into pandas data frame
    train_data = pd.read_csv(train_data_path, index_col='PassengerId')
    # print(train_data.info())
    # print(train_data.describe())
    return train_data


def prepare_data(train_data):
    # Selecting data we care about and cleaning it up
    observed_data = train_data[["Survived", "Age", "Fare"]]
    non_null_observed_data = observed_data.dropna(axis=0)  # drop rows that contain null value

    features = non_null_observed_data[["Age", "Fare"]].values
    labels = non_null_observed_data["Survived"].values
    return features, labels


def initialize_classifier(features, labels):
    # Make and train a classifier
    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(features, labels)
    return classifier


def initialize_plot(features, labels):
    # Plot how predictor would work for some hypothetical data

    plot = plt.subplot()
    plot.set_xlabel("Age")
    plot.set_ylabel("Ticket Fare")
    plt.ion()  # Allows dynamic plot refresh

    data_to_plot = np.concatenate((labels.reshape(-1, 1), features), axis=1)
    for passenger in data_to_plot:
        plot.scatter(passenger[1],
                     passenger[2],
                     c='red' if passenger[0] == DEAD else 'green',
                     alpha=0.1)
    return plot


def feature_range(features):
    # Find max values in each dimension
    int_rounded_features = features.astype(int)
    test_set_dimensions = [int_rounded_features[:, 0].max(), int_rounded_features[:, 1].max()]
    # print("Data range: ", test_set_dimensions)
    return test_set_dimensions


def plot_prediction_field(classifier, features, plot):
    test_set_dimensions = feature_range(features)

    for x_coord in range(0, test_set_dimensions[0], 5):
        for y_coord in range(0, test_set_dimensions[1], 50):
            plot.scatter(x_coord,
                         y_coord,
                         alpha=0.7,
                         c='red' if classifier.predict([[x_coord, y_coord]]) == DEAD else 'green'
                         )
            plt.pause(1e-9)


# What actually happens

train_data_path = "Data/train.csv"
train_data = load_data(train_data_path)

features, labels = prepare_data(train_data)
classifier = initialize_classifier(features, labels)
print("The model can now classify with accuracy: ", classifier.score(features, labels))

plot = initialize_plot(features, labels)
plot_prediction_field(classifier, features, plot)
