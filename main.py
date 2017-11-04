"""
Titanic
"""
import pandas as pd
from sklearn import naive_bayes

from data_tools import prepare_data
from plot_tools import ClassifierTestPlotter

# Set up train data
train_data_path = 'Data/train.csv'
train_data = pd.read_csv(train_data_path, index_col='PassengerId')
clean_train_data = prepare_data(train_data)

# Choose which classifier you want to use
classifier = naive_bayes.GaussianNB()

# Set up testing data
test_data_path = "Data/test.csv"
test_data = pd.read_csv(test_data_path, index_col='PassengerId')

test_data_labels_path = "Data/gender_submission.csv"
test_data_labels = pd.read_csv(test_data_labels_path, index_col='PassengerId')

clean_test_data = prepare_data(pd.concat([test_data, test_data_labels], axis=1))

# Fit classifier
classifier.fit(clean_train_data.drop('Survived', axis=1),
               clean_train_data['Survived'])

# Output score
print('Model accuracy: ', classifier.score(clean_test_data.drop('Survived', axis=1),
                                           clean_test_data['Survived']))
# Plot results
plot = ClassifierTestPlotter(clean_test_data, 'Age', 'Sex', 'Survived')
plot.plot_prediction(classifier)
