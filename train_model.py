import pickle

file_name = "trained_model.sav"
pickle.dump(classifier, open(file_name,'wb'))  

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

input_data = (1,103,30,38,83,43.3,0.183,33)

#changing the input_data into numpyarray
 
input_data_np = np.asarray(input_data)
 
#reshaping the array as we are predicting for one instance 
input_data_reshape = input_data_np.reshape(1,-1)

prediction =  classifier.predict(input_data_reshape)
print(prediction)

if(prediction[0] == 0):
  print("The person is non-diabetic")
else:
  print("The person in diabetic")
