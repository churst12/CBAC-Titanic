"""
Titanic
"""
import pandas as pd
from sklearn import tree

from data_tools import prepare_data
from plot_tools import ClassifierTestPlotter

train_data_path = 'Data/train.csv'
train_data = pd.read_csv(train_data_path, index_col='PassengerId')
clean_train_data = prepare_data(train_data)

classifier = tree.DecisionTreeClassifier()
classifier.fit(clean_train_data.drop('Survived', axis=1),
               clean_train_data['Survived'])
print('Model accuracy: ', classifier.score(clean_train_data.drop('Survived', axis=1),
                                           clean_train_data['Survived']))

plot = ClassifierTestPlotter(clean_train_data, 'Age', 'Sex', 'Survived')
plot.plot_prediction(classifier)
