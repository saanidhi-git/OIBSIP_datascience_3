# CAR PRICE PREDICTION STREAMLIT WEB APP 

import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


MODEL_PATH = "car_price_model.pkl"
CURRENT_YEAR = datetime.now().year



@st.cache_resource
def load_model():
    """Loads the pre-trained pipeline model."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except FileNotFoundError:
        # NOTE: This error now explicitly tells the user to run the training script
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please run your 'car_prediction.py' script first to generate the model file.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_price(pipeline, input_data):
    """Makes a prediction using the loaded pipeline."""
    try:
        
        prediction = pipeline.predict(input_data)[0]
        return prediction 
    except Exception as e:
        
        st.error(f"Prediction Error: Data input structure mismatch. Details: {e}")
        return None


#  STREAMLIT APP LAYOUT

def main():
    st.title("🚗 Used Car Price Predictor")
    st.markdown("### Estimate the resale value of your car (Prices in Lakhs INR)")

    # Load the model once
    pipeline = load_model()

    if pipeline is None:
        return

    # INPUTS ON MAIN DASHBOARD
    with st.container(border=True): 
        st.header("Enter Car Specifications")
        
      
        col1, col2 = st.columns(2)

       
        with col1:
            
            present_price = st.slider(
                "Showroom Price (Ex-showroom, in Lakhs)",
                min_value=0.5,
                max_value=40.0,
                value=8.0,
                step=0.5
            )

           
            driven_kms = st.number_input(
                "Kilometers Driven (KMs)",
                min_value=100,
                max_value=500000,
                value=15000,
                step=1000
            )

            
            fuel_type = st.selectbox(
                "Fuel Type",
                options=['Petrol', 'Diesel', 'CNG'],
                index=0
            )

            
            selling_type = st.selectbox(
                "Selling Type",
                options=['Dealer', 'Individual'],
                index=0
            )

      
        with col2:
           
            year = st.slider(
                "Manufacturing Year",
                min_value=2003,
                max_value=CURRENT_YEAR,
                value=2017,
                step=1
            )

            
            owner = st.selectbox(
                "Number of Previous Owners",
                options=[0, 1, 2, 3],
                index=0
            )

           
            transmission = st.selectbox(
                "Transmission Type",
                options=['Manual', 'Automatic'],
                index=0
            )
            
            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        
    
    st.markdown("---")

    # Prediction Logic
    if st.button("Calculate Estimated Price", use_container_width=True, type="primary"):
        
       
        
        input_data = pd.DataFrame({
            'Year': [year], 
            'Present_Price': [present_price],
            'Driven_kms': [driven_kms],
            'Fuel_Type': [fuel_type],
            'Selling_type': [selling_type],
            'Transmission': [transmission],
            'Owner': [owner]
        })
        
      
        feature_order = ['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
        input_data = input_data[feature_order]

        
      
        with st.spinner('Calculating best price...'):
            predicted_price = predict_price(pipeline, input_data)

        if predicted_price is not None:
            
            st.success("## 🎯 Predicted Selling Price:")
            st.markdown(f"""
                <div style="background-color: #e0f7fa; padding: 20px; border-radius: 10px; text-align: center; border: 3px solid #00bcd4;">
                    <h1 style="color: #00796b; margin: 0;">₹ {predicted_price:.2f} Lakhs</h1>
                    <p style="margin-top: 5px; color: #555;">(Estimated resale value)</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.info(f"Note: This price is an estimate based on a Random Forest Model, using data up to {CURRENT_YEAR}.")


if __name__ == "__main__":
    main()
