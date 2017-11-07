"""
Titanic
"""
import pandas as pd
from sklearn import ensemble, naive_bayes, svm, tree
from sklearn.model_selection import train_test_split

from data_tools import prepare_data
from plot_tools import ClassifierTestPlotter

# Set up train data
data_path = 'Data/train.csv'
data = pd.read_csv(data_path, index_col='PassengerId')

clean_data = prepare_data(data)
my_train_set, my_test_set = train_test_split(clean_data, test_size=0.2)

# Classifier setup
classifiers = {
    1: ensemble.RandomForestClassifier(),
    2: tree.DecisionTreeClassifier(),
    3: svm.SVC(),
    4: naive_bayes.GaussianNB(),
}
classifier = classifiers.get(1)
classifier.fit(my_train_set.drop('Survived', axis=1),
               my_train_set['Survived'])

# Output score
print('Model accuracy on train data: ', classifier.score(my_train_set.drop('Survived', axis=1),
                                                         my_train_set['Survived']))
print('Model accuracy on test data: ', classifier.score(my_test_set.drop('Survived', axis=1),
                                                        my_test_set['Survived']))
# Plot results
plot = ClassifierTestPlotter(my_test_set, 'Prefix', 'Fare', 'Survived')
plot.plot_prediction(classifier)
