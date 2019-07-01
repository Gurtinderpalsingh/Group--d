import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
from sklearn import metrics

heart = pd.read_csv("D:\heartdisease\heart.csv")

heart.shape

heart.head()

X = heart.drop('target', axis=1)
y = heart['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print(y_pred)

plt.axis([0, 5, 0, 100])
x = [0, 1, 2]
y = [0, metrics.accuracy_score(y_test, y_pred) * 100, 0]
plt.bar(x, y, align='center')

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()
