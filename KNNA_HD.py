from sklearn import metrics
import pandas as pd


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn.metrics import confusion_matrix

data = pd.read_csv("D:/heart.csv")




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



from sklearn.neighbors import KNeighborsClassifier
knn = train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier, n_neighbors=8)

knn.fit(X_train, Y_train)

y_pred_knn = knn.predict(X_test)
print(y_pred_knn)


score_knn = round(accuracy_score(y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")

from sklearn.neighbors import KNeighborsClassifier
model = train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier)


# Seek optimal 'n_neighbours' parameter
for i in range(1,10):
    print("n_neigbors = "+str(i))
    train_model(X_train, Y_train, X_test, Y_test, KNeighborsClassifier, n_neighbors=i)


from sklearn.metrics import confusion_matrix

matrix= confusion_matrix(Y_test, y_pred_knn)

from sklearn.metrics import precision_score

precision = precision_score(Y_test, y_pred_knn)

print("Precision: ",precision)




from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(Y_test, y_pred_knn)
print(confusion_matrix(Y_test, y_pred_knn))  
print(classification_report(Y_test, y_pred_knn))
print("Accuracy:",metrics.accuracy_score(Y_test,y_pred_knn))
print (Y_test)



print (y_pred_knn)

 
plt.axis([0, 5, 0, 100]) 
x = [0,1,2]
y = [0,metrics.accuracy_score(Y_test,y_pred_knn)*100,0]
plt.bar(x, y, align='center')
    
plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()


pd.crosstab(y_pred_knn,y_pred_knn).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('No. of people Suffering')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()
