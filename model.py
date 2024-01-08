import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle

data = pd.read_csv('Obesity Classification.csv')

data['Gender'] = data['Gender'].apply(lambda x: 1 if x == "Male" else 0)
X = data.drop(['ID', 'Label'], axis=1)
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mySVC = SVC(C=1.0, kernel='rbf', gamma='auto')  # Example, tune these values
mySVC.fit(X_train_scaled, y_train)  # Correct position for model fitting

y_pred = mySVC.predict(X_test_scaled)

# Now, save the model after fitting and evaluating
pickle.dump(mySVC, open('modelmay.pkl.pkl', 'wb'))
