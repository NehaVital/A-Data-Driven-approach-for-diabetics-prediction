import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/preet/OneDrive/Desktop/dbp/train_model.sav', 'rb'))

st.set_page_config(page_title="A Data-Driven Approach for Diabetes Prediction", page_icon=":pill:", layout="wide")

# creating a function for Prediction
def diabetes_prediction(input_data): 
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return 'The person is non-diabetic'
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
        result = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        if result == True:    
            with open('C:/Users/preet/OneDrive/Desktop/diabetic_result.html', 'r') as f:
                html_string = f.read()
            # open the HTML file in a new page
            st.write(html_string, unsafe_allow_html=True)
        else:
            with open('C:/Users/preet/OneDrive/Desktop/nondiabetic_result.html', 'r') as f:
                html_string = f.read()
            # open the HTML file in a new page
            st.write(html_string, unsafe_allow_html=True)
            
if __name__ == '__main__':
    main()
