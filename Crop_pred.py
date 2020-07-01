#importing the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus

#Reading the csv file
data=pd.read_csv('./cpdata.csv')

label= pd.get_dummies(data.label).iloc[: , 1:]
data= pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
print('The data present in one row of the dataset is')
print(data.head(1))
train=data.iloc[:, 0:4].values
test=data.iloc[: ,4:].values

#Dividing the data into training and test set
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing Decision Tree classifier
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor()

#Fitting the classifier into training set
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

'''
features = ['temperature','humidity','ph','rainfall']


data = tree.export_graphviz(clf, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')
s
'''
from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a = accuracy_score(y_test,pred)
print("The accuracy of this model is: ", a*100)

#Using firebase to import data that is uploaded to the cloud from iot

'''
from firebase import firebase
firebase =firebase.FirebaseApplication('')
tp = firebase.get('/Data',None)

'''
from cloudTestData import data

temp = data['Air Temp']
humidity = data['Soil Humidity']
pH = data['Soil pH']
rain = data['Rainfall']


l=[]
l.append(humidity)
l.append(temp)
l.append(pH)
l.append(rain)
predictcrop=[l]

# Putting the names of crop in a single list
crops=['wheat','mungbean','Tea','millet','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','orange','papaya','watermelon','pomegranate']
cr='rice'


#Predicting the crop
predictions = clf.predict(predictcrop)
count=0
for i in range(0,30):
    if(predictions[0][i]==1):
        c=crops[i]
        count=count+1
        break
    i=i+1
if(count==0):
    print('The predicted crop is %s'%cr)
else:
    print('The predicted crop is %s'%c)

#Sending the predicted crop to cloud to render it in application
'''
cp = firebase.put('/predicted','crop',c)

'''

