#input http://127.0.0.1:5000/predict?s_length=5.7&s_width=5.6&p_length=4.3&p_width=7.8
#ouput 1, 2,0

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


#loading the dataset
iris= load_iris()
X=iris.data
y=iris.target


#split the data
X_train ,X_test ,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.30)

#Build the model
clf= RandomForestClassifier(n_estimators=10)

#train the classifier
clf.fit(X_train,y_train)

#predictions
preicted =clf.predict(X_test)

#check accuracy
print(accuracy_score(preicted,y_test))


#pickle the model
with open('C:/Users/t/PycharmProjects/randomforestclassfier/rf.pkl','wb')as model_pkl:
    pickle.dump(clf,model_pkl)

