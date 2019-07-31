import tensorflow as tf
from tensorflow import keras
import numpy
import pandas as pd
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix


data = pd.read_csv("D:\heart.csv")




predictors = data.drop("target",axis=1)
target = data["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.30,random_state=0)
print("Training features have {0} records and Testing features have {1} records.".\
      format(X_train.shape[0], X_test.shape[0]))


def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
   
    """
    Fit the model and print the score.
    
    """
    
    # instantiate model
    model = classifier(**kwargs)
    
    # train model
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    
    return model




logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

y_pred_lr = logreg.predict(X_test)
print(y_pred_lr)



lr =accuracy_score(y_pred_lr,Y_test)

print("The accuracy score achieved using Logistic Regression is: "+str(lr)+" %")






model = train_model(X_train, Y_train, X_test, Y_test, LogisticRegression)


clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(X_test, Y_test)


clf.score(X_test, Y_test)




matrix= confusion_matrix(Y_test, y_pred_lr)



precision = precision_score(Y_test, y_pred_lr)


print("Precision: ",precision)



from sklearn.metrics import recall_score


recall = recall_score(Y_test, y_pred_lr)


print("Recall is: ",recall)


plt.axis([0, 5, 0, 100]) 
x = [0,1,2]
y = [0,precision*100,0]
plt.bar(x, y, align='center')
    
plt.title('Accuracy')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()


pd.crosstab(y_pred_lr,y_pred_lr).plot(kind="bar",figsize=(20,60))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('No. of people Suffering')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()

def plotAge():
    facet_grid = sns.FacetGrid(data, hue='target')
    facet_grid.map(sns.kdeplot, "age", shade=True, ax=axes[0])
    legend_labels = ['disease false', 'disease true']
    for t, l in zip(axes[0].get_legend().texts, legend_labels):
        t.set_text(l)
        axes[0].set(xlabel='age', ylabel='density')

    avg = data[["age", "target"]].groupby(['age'], as_index=False).mean()
    sns.barplot(x='age', y='target', data=avg, ax=axes[1])
    axes[1].set(xlabel='age', ylabel='disease probability')

    plt.clf()
    plt.show()


fig_age, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

plotAge()
file = open("D:\conclusion\logostichdlr.txt", "w")

file.write(str(lr)) 
file.close() 
