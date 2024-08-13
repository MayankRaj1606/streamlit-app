import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('C:\\Machine_Learning\\diabeties_predictor\\diabetes.csv')

# Split the dataset into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, clf.predict(X_test))

# Streamlit app
st.title('Diabetes Prediction App')
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Model Accuracy: {:.2f}</p>'.format(accuracy), unsafe_allow_html=True)

st.header('Input Patient Data')

# Collect user input features
def user_input_features():
    st.sidebar.header('User Input Features')
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
    skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=900, value=79)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=21.0)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=33)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the input features
st.subheader('User Input Data')
st.write(input_df)

# Make predictions
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

st.subheader('Prediction')
diabetes_result = 'Diabetic' if prediction[0] else 'Not Diabetic'
st.markdown(f"<h3 style='color: {'red' if diabetes_result == 'Diabetic' else 'green'};'>{diabetes_result}</h3>", unsafe_allow_html=True)

st.subheader('Prediction Probability')
proba_df = pd.DataFrame(prediction_proba, columns=['Not Diabetic', 'Diabetic'])
st.write(proba_df)

# Adding some footer information
st.markdown("""
<hr>
<p style='text-align: center;'>
Developed by Your Name | <a href="https://www.Mayankwebsite.com" target="_blank">Mayank's Website</a>
</p>
""", unsafe_allow_html=True)
