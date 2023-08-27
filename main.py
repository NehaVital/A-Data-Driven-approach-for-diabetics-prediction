import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv('D:/PycharmProjects/DBP/diabetes.csv')

x_tab = diabetes_data.drop(columns = 'Outcome', axis = 1)
y_tab = diabetes_data['Outcome']

x_tab_train, x_tab_test, y_tab_train, y_tab_test = train_test_split(x_tab, y_tab, test_size = 0.01, random_state = 2)

classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_tab_train,y_tab_train)

x_tab_train_predicter = classifier.predict(x_tab_train)
training_data_accuracy = accuracy_score(x_tab_train_predicter,y_tab_train.values)
X_test_predicter = classifier.predict(x_tab_test)
testing_data_accuracy = accuracy_score(X_test_predicter,y_tab_test.values)

st.set_page_config(page_title="A Data-Driven Approach for Diabetes Prediction", page_icon=":pill:", layout="wide")

# creating a function for Prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = classifier.predict(input_data_reshaped)  # Use the classifier to predict

    if prediction[0] == 0:
        return False
    else:
        return True



def main():
    # giving a title
    st.title('A Data-Driven Approach for Diabetes Prediction')

    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # code for Prediction

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        result = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        if result == True:
            with open('diabetic_result.html', 'r') as f:
                html_string = f.read()
            # open the HTML file in a new page
            st.write(html_string, unsafe_allow_html=True)
        else:
            with open('nondiabetic_result.html', 'r') as f:
                html_string = f.read()
            # open the HTML file in a new page
            st.write(html_string, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
