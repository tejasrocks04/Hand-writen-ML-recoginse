import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils import shuffle

#reading data 
dataframe = pd.read_csv("csv\datasetAlabels.csv")
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

print("reading data")

#print (dataframe)
X = dataframe.drop(['label'],axis=1)
#print(X)
Y = dataframe['label']
#print(Y)


X_train, Y_train =  X, Y
X_test,Y_test = X,Y

print("showing img")
grid_data = X_train.values[13].reshape(28,28)
plt.imshow(grid_data,interpolation=None,cmap="gray")
plt.title(Y_train.values[13])
plt.show()


#model fit traning
model = svm.SVC(kernel="linear",C=1)
print ("Fitting this might take some time .....")
model.fit(X_train,Y_train)

#saving model
joblib.dump(model, "model/_Atozlabel_linear") 

#prediting
print ("predicting .....")
predictions = model.predict(X_test)

print ("Getting Accuracy .....")
print ("Score", metrics.accuracy_score(Y_test, predictions))