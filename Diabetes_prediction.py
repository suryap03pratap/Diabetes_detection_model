from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
import pickle
import streamlit as st


# loading the dataset
diabetes_dataset = pd.read_csv(
    'E:\Machine Learning\Diseaese Prediction\diabetes_prediction_dataset.csv')

# separating the data and labels
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# print(x)
# print(y)


# Separate categorical features
# Assuming these are categorical

# categorical_cols = ['gender', 'smoking_history']
#
# numerical_cols = list(set(x.columns) - set(categorical_cols))
# numerical_cols = list(set(diabetes_dataset.columns) - set(categorical_cols))
####
# one_hot = OneHotEncoder()
# transformer = ColumnTransformer([("one_hot",one_hot,categorical_cols)],remainder="passthrough")
# transformer_x = transformer.fit_transform(x)

# # One-hot encode categorical features
# encoder = OneHotEncoder(handle_unknown='ignore')
# encoded_data = encoder.fit_transform(
#     transformer_x[categorical_cols])  # replace x with diabetes_dataset


# Combine encoded and numerical features (corrected)
# encoded_data_df = pd.DataFrame(encoded_data.toarray())  # Convert to DataFrame
# x = pd.concat([encoded_data_df, transformer_x[numerical_cols]],
#               axis=1)  # replaced x with diabetes_dataset

# Combine encoded and numerical features
# x = pd.concat([encoded_data.toarray(), x[numerical_cols]], axis=1)

# Standardize numerical features
# scaler = StandardScaler()
# scaler.fit(x)
# standardized_data = scaler.transform(x)

# x.columns = x.columns.astype(str)  # Convert column names to strings


categorical_cols = ['gender', 'smoking_history']
numerical_cols = list(set(diabetes_dataset.columns) - set(categorical_cols))

one_hot = OneHotEncoder()
transformer = ColumnTransformer(
    [("one_hot", one_hot, categorical_cols)], remainder="passthrough")
transformer_x = transformer.fit_transform(x)

scaler = StandardScaler()
scaler.fit(transformer_x)
standardized_data = scaler.transform(transformer_x)


# data standarization or data preprocessing transforming the data in same range
# scaler = StandardScaler()
# # scaler.fit(x)

# standarized_data = scaler.transform(x)#replace x with transformer_x
# print(standarized_data)

# the preprocessed data is now stored in x and y variable to train the model
# x represents the data and y represents the model
x = standardized_data
y = diabetes_dataset['Outcome']

# Spliting data into training and test data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2)
# test_size = 0.2 means that 20 % of data is made as test data and rest 80% as training data
# statisfy = y means that the spliting must be done in consideration of y label dataset which means there must be proportion of diabetic as well as non diabetic cases in doth sets of data

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)  # trains the data using svm

# getting the accuracy score of model when compared the prediction of x_train data with y_train the actual label or answer
# accuracy score of training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("The accuracy score of training data is: ")
print(training_data_accuracy)

# accuracy score of test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("The accuracy score of training data is: ")
print(test_data_accuracy)

print("The f1 score of the model is: ")
print(f1_score(x_train_prediction, y_train, average=None))

# it is very important that does the scores comes almost same if training score is more higher than tests data then it leads to overfitting
# vice verse then it's underfitting


# taking input and letting us know weather a person is diabetic or not
# input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
# input_data = ('Female', 80, 0, 1, 'never', 25.19, 6.6, 140)


# input_data = ('Female', 80, 0, 1, 'never', 25.19, 6.6, 140)

# # Use the same categorical_cols for one-hot encoding


# input_df = pd.DataFrame([input_data], columns=['gender', 'age', 'hypertension',
#                         'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
# input_transformed = transformer.transform(input_df)

# std_input_data = scaler.transform(input_transformed)
# input_data_as_numpy_array = np.asarray(std_input_data)


# # changing the input from list to numpy array for faster execution and reshaping ability so that the model does not gets confused
# # input_data_as_numpy_array = np.asarray(input_data)

# # changing the shape as we are predicting for one instances
# input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# # we also need to standazise the data
# std_data = scaler.transform(input_data_reshape)
# print("Input data Before standarization: ", input_data)
# print("Input data After standarization: ", std_data)

# # our ML model classifier returns a list as output not an integer but it only has one value so can be accessed using prediction[0]
# prediction = classifier.predict(std_data)
# print("The prediction of input value")
# print(prediction)

# if (prediction[0] == 1):
#     print("The person is  diabetic")
# else:
#     print("The person is not diabetic")


# pickling it
filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


input_data = ('Female', 80, 0, 1, 'never', 25.19, 6.6, 140)

# Use the same categorical_cols for one-hot encoding


input_df = pd.DataFrame([input_data], columns=['gender', 'age', 'hypertension',
                        'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
input_transformed = transformer.transform(input_df)

std_input_data = scaler.transform(input_transformed)
input_data_as_numpy_array = np.asarray(std_input_data)


# changing the input from list to numpy array for faster execution and reshaping ability so that the model does not gets confused
# input_data_as_numpy_array = np.asarray(input_data)

# changing the shape as we are predicting for one instances
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# we also need to standazise the data
std_data = scaler.transform(input_data_reshape)
print("Input data Before standarization: ", input_data)
print("Input data After standarization: ", std_data)

# our ML model classifier returns a list as output not an integer but it only has one value so can be accessed using prediction[0]
prediction = loaded_model.predict(std_data)
print("The prediction of input value")
print(prediction)

if (prediction[0] == 1):
    print("The person is  diabetic")
else:
    print("The person is not diabetic")


def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    # input_data_as_numpy_array = np.asarray(input_data)

    # # reshape the array as we are predicting for one instance
    # input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # prediction = loaded_model.predict(input_data_reshaped)
    # print(prediction)

    # if (prediction[0] == 0):
    #   return 'The person is not diabetic'
    # else:
    #   return 'The person is diabetic'

    # input_data = ('Female', 80, 0, 1, 'never', 25.19, 6.6, 140)

    # Use the same categorical_cols for one-hot encoding

    input_df = pd.DataFrame([input_data], columns=['gender', 'age', 'hypertension',
                                                   'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
    input_transformed = transformer.transform(input_df)

    std_input_data = scaler.transform(input_transformed)
    input_data_as_numpy_array = np.asarray(std_input_data)


# changing the input from list to numpy array for faster execution and reshaping ability so that the model does not gets confused
# input_data_as_numpy_array = np.asarray(input_data)

# changing the shape as we are predicting for one instances
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# we also need to standazise the data
    std_data = scaler.transform(input_data_reshape)
    print("Input data Before standarization: ", input_data)
    print("Input data After standarization: ", std_data)

# our ML model classifier returns a list as output not an integer but it only has one value so can be accessed using prediction[0]
    prediction = loaded_model.predict(std_data)
    print("The prediction of input value")
    print(prediction)

    if (prediction[0] == 1):
        print("The person is  diabetic")
    else:
        print("The person is not diabetic")


def main():

    # giving a title
    st.title('Diabetes Detection ')

    # getting the input data from the user

    Gender = st.text_input('Gender')
    Age = st.text_input('Age')
    hypertension = st.text_input('Hypertension value')
    Heart_disease = st.text_input('Heart Disease')
    Smoking_history = st.text_input('Smoking history')
    BMI = st.text_input('BMI value')
    HbA1c_level = st.text_input(
        'HbA1c_level value')
    blood_glucose_level = st.text_input('blood_glucose_level')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction(
            [Gender, Age, hypertension, Heart_disease, Smoking_history, BMI, HbA1c_level, blood_glucose_level])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
