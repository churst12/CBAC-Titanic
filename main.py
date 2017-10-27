import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm

train_data_path = "Data/train.csv"

train_data = pd.read_csv(train_data_path)


print(train_data.describe())

clf = svm.SVC()
clf.fit(train_data["Age"], train_data["Survived"])


test_x = range(80)

predictions = clf.predict(test_x)

plot = plt.subplot()
plot.plot(test_x, predictions)
plt.show()
