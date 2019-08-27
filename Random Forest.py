import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 

s=time.time()
# Importing the dataset
dataset=pd.read_csv('Skin_NonSkin.txt' , sep = '\t')
dataset.columns=['B','G','R','RESULT']
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()*100
accuracies.std()

e=s-time.time()
accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(50,50))
ax = fig.add_subplot(111, projection='3d')

x =X[:50858,0]
y=X[:50858,1]
z=X[:50858,2]


ax.scatter(x, y, z, c='r', marker='.')

x1 =X[50857:,0]
y1=X[50857:,1]
z1=X[50857:,2]

ax.scatter(x1, y1, z1, c='b', marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
