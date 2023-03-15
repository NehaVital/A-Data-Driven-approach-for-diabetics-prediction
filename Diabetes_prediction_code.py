import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Import your dataset into the files in the case if your using google colaboratory
diabetes_data = pd.read_csv('/content/drive/MyDrive/diabetes .csv')

#Mount your googledrive
from google.colab import drive
drive.mount('/content/drive')

#print first five data from the dataset
diabetes_data.head()

#print the shape of the dataset
diabetes_data.shape

diabetes_data['Outcome'].value_counts()

diabetes_data.groupby('Outcome').mean()

x_tab = diabetes_data.drop(columns = 'Outcome', axis = 1)
y_tab = diabetes_data['Outcome']
print(x_tab)
print(y_tab)

#TRAIN TEST SPLIT
x_tab_train, x_tab_test, y_tab_train, y_tab_test = train_test_split(x_tab, y_tab, test_size = 0.01, stratify = y_tab, random_state = 2)
print(x_tab.shape, x_tab_train.shape, x_tab_test.shape)
classifier = svm.SVC(kernel = 'linear')

#Traing the support vector machine classifier
classifier.fit(x_tab_train,y_tab_train)

#Accuracy score
x_tab_train_predicter = classifier.predict(x_tab_train)
training_data_accuracy = accuracy_score(x_tab_train_predicter,y_tab_train.values)
print(training_data_accuracy)
X_test_predicter = classifier.predict(x_tab_test)
testing_data_accuracy = accuracy_score(X_test_predicter,y_tab_test.values)


#PREDICTION
input_data = (1,85,66,29,0,26.6,0.351,31)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
print(testing_data_accuracy)
