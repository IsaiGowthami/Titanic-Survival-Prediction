
import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival probability.")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encoding categorical features
sex_encoded = 1 if sex == "Female" else 0
embarked_C, embarked_Q, embarked_S = 0, 0, 0
if embarked == "C":
    embarked_C = 1
elif embarked == "Q":
    embarked_Q = 1
else:
    embarked_S = 1

# Prepare input for model
input_features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_C, embarked_Q, embarked_S]])

# Predict survival
if st.button("Predict Survival"):
    prediction = model.predict(input_features)
    survival_prob = model.predict_proba(input_features)[0][1]

    if prediction[0] == 1:
        st.success(f"The passenger is likely to survive with a probability of {survival_prob:.2f}.")
    else:
        st.error(f"The passenger is unlikely to survive with a probability of {1 - survival_prob:.2f}.")
