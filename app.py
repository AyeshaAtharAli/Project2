import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model_filename = 'tuned_random_forest_model.joblib'
try:
    loaded_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error(f"Error: Model file '{model_filename}' not found. Make sure the model is saved correctly.")
    loaded_model = None # Set loaded_model to None to prevent further execution if file is not found

if loaded_model is not None:
    # Set the title of the Streamlit application
    st.title('Titanic Survival Prediction')

    st.write("""
    Enter the passenger details to predict if they would have survived the Titanic disaster.
    """)

    # Create input fields for each feature
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    age = st.number_input('Age', min_value=0.1, max_value=100.0, value=30.0)
    sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, value=0)
    parch = st.number_input('Number of Parents/Children Aboard', min_value=0, value=0)
    fare = st.number_input('Fare', min_value=0.0, value=10.0)

    # Represent boolean features with select boxes
    sex = st.selectbox('Sex', ['male', 'female'])
    embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

    # Convert categorical inputs to the format used by the model (one-hot encoding)
    sex_male = 1 if sex == 'male' else 0
    embarked_q = 1 if embarked == 'Q' else 0
    embarked_s = 1 if embarked == 'S' else 0

    # Create a button to trigger the prediction
    if st.button('Predict Survival'):
        # Create a pandas DataFrame from the user's input
        # Ensure the column order and names match the training data (X DataFrame)
        input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s]],
                                  columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'])

        # Make a prediction
        prediction = loaded_model.predict(input_data)

        # Display the prediction
        if prediction[0] == 1:
            st.success('Prediction: Survived')
        else:
            st.write('Prediction: Did Not Survive')
