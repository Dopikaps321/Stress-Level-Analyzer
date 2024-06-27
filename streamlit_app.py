import streamlit as st
import joblib
import numpy as np
import os

# Load the trained model and scaler
model_path = 'C:\\Users\\USER\\Desktop\\stress\\stress_level_model.pkl'
scaler_path = 'C:\\Users\\USER\\Desktop\\stress\\scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("Model or scaler file not found. Please ensure the files are in the correct location.")
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    def main():
        st.title('Stress Level Detection App')
        st.write('Enter the features to predict stress levels.')

        # User input fields
        sr = st.number_input('Skin Response (sr)', min_value=0.0)
        rr = st.number_input('Respiratory Rate (rr)', min_value=0.0)
        t = st.number_input('Temperature (t)', min_value=0.0)
        lm = st.number_input('Limb Movement (lm)', min_value=0.0)
        bo = st.number_input('Blood Oxygen (bo)', min_value=0.0)
        rem = st.number_input('Rapid Eye Movement (rem)', min_value=0.0)
        sh = st.number_input('Sleeping Hours (sh)', min_value=0.0)
        hr = st.number_input('Heart Rate (hr)', min_value=0.0)

        if st.button('Predict Stress Level'):
            # Prepare the input data
            input_data = np.array([[sr, rr, t, lm, bo, rem, sh, hr]])

            # Normalize the input data using the loaded scaler
            input_data_scaled = scaler.transform(input_data)

            # Make the prediction using the loaded model
            prediction = model.predict(input_data_scaled)

            # Display the prediction
            st.write("### Predicted Stress Level:")
            st.write(prediction[0])

        # Add download button for model file
        with open(model_path, "rb") as file:
            st.download_button(
                label="Download Model",
                data=file,
                file_name="stress_level_model.pkl",
                mime="application/octet-stream"
            )

        # Add download button for scaler file
        with open(scaler_path, "rb") as file:
            st.download_button(
                label="Download Scaler",
                data=file,
                file_name="scaler.pkl",
                mime="application/octet-stream"
            )

    if __name__ == '__main__':
        main()

