# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:, -1].values

# Encoding The Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

#Splitting The Data Into Train and Test Datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Importing The Neural Network Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Dropout is used to stop oversampling
def classifier_build():
    classifier = Sequential()
    classifier.add(Dense(activation='relu',output_dim=8,input_dim=12))
    Dropout(p=0.1)
    classifier.add(Dense(activation='relu',output_dim=8))
    Dropout(p=0.1)
    classifier.add(Dense(activation='sigmoid',output_dim=1))
    classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
    return classifier

#Testing the accuracy of training set
classifier = classifier_build()
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
y_pred = classifier.predict(X_train)
y_pred = (y_pred>0.5)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_pred))


#Performing K_Cross_Validation For Getting Correct Accuracy
def classifier_build():
    classifier = Sequential()
    classifier.add(Dense(activation='relu',output_dim=8,input_dim=12))
    classifier.add(Dense(activation='relu',output_dim=8))
    classifier.add(Dense(activation='sigmoid',output_dim=1))
    classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
    return classifier


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
classifier = KerasClassifier(build_fn=classifier_build,batch_size=10,epochs=30)
accuracies = cross_val_score(estimator=classifier,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             n_jobs=-1)
print(accuracies.mean())


#Performing Hyperparameter tuning using GridSearchCV
def classifier_build(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation='relu',output_dim=8,input_dim=12))
    classifier.add(Dense(activation='relu',output_dim=8))
    classifier.add(Dense(activation='sigmoid',output_dim=1))
    classifier.compile(optimizer=optimizer,metrics=['accuracy'],loss='binary_crossentropy')
    return classifier

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn=classifier_build)
parameters={'batch_size':[25,32],'epochs':[25,40,100],'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

