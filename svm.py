import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
from sklearn import metrics
import pylab as pl

style.use("ggplot")

heart = pd.read_csv("D:\heartdisease\heart.csv")

heart.shape

heart.head()

X = heart.drop('target', axis=1)
y = heart['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
clf2 = svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

# cm=confusion_matrix(y_test,y_pred)
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# print y_test

print(y_pred)

plt.axis([0, 5, 0, 100])
x = [0, 1, 2]
y = [0, metrics.accuracy_score(y_test, y_pred) * 100, 0]
plt.bar(x, y, align='center')

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()
