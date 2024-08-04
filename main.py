import streamlit as st
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
def main():
    st.title("Plantation Adviser")

    # Input fields
    st.sidebar.header("Input Parameters")
    nitrogen = st.sidebar.number_input("Nitrogen content of soil")
    phosphorus = st.sidebar.number_input("Phosphorus content of soil")
    potassium = st.sidebar.number_input("Potassium content of soil")
    temperature = st.sidebar.number_input("Average Temperature (°C)")
    humidity = st.sidebar.number_input("Average Humidity (%)")
    ph_level = st.sidebar.number_input("pH level of soil")
    rainfall = st.sidebar.number_input("Amount of rainfall (mm)")
    input_arr = np.array([[nitrogen,phosphorus,potassium,temperature,humidity,ph_level,rainfall]])
    
    

    # Display inputs
    st.write("### Input Parameters:")
    st.write("- Nitrogen content of soil:", nitrogen)
    st.write("- Phosphorus content of soil:", phosphorus)
    st.write("- Potassium content of soil:", potassium)
    st.write("- Average Temperature:", temperature, "°C")
    st.write("- Average Humidity:", humidity, "%")
    st.write("- pH level of soil:", ph_level)
    st.write("- Amount of rainfall received:", rainfall, "mm")

    lr_model = joblib.load('lr_model.joblib')

    scaler = joblib.load('scaler.joblib')
    
    labels = ['Apple','Banana','Blackgram','Chickpea','Coconut','Coffee','Cotton','Grapes','Jute','Kidneybeans','Lentil','Maize','Mango','Mungbean','Mothbeans','Muskmelon','Orange','Papaya','Pigeonpeas','Pomegranate','Rice','Watermelon']

    def predict(input_data):
        input_arr = input_data.reshape(1,-1)
        model_input = scaler.transform(input_arr)
        predictions = lr_model.predict(model_input)
        return predictions
    
    if st.button('Predict'):
        predictions = predict(input_arr)
        indices = np.where(predictions == 1)[1]
        original_labels = [labels[i] for i in indices]
        if not original_labels:
            st.write("Model couldn't get that one. Try again with a different data 😳")
        else:
            st.write("The best crop to grow would be:\n")
            st.markdown(f"<p style='font-size:24px'><b>{original_labels[0]}</b></p>", unsafe_allow_html=True)



    # Add prediction logic here

if __name__ == "__main__":
    main()