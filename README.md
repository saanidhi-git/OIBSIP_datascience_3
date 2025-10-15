
CAR PRICE PREDICITION WITH MACHINE LEARNING

# 🚗 Car Price Prediction using Random Forest Regression

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Algorithm Used (Random Forest Regressor)](#algorithm-used-random-forest-regressor)
3.  [Dataset and Features](#dataset-and-features)
4.  [Project Structure](#project-structure)
5.  [Setup and Installation](#setup-and-installation)
6.  [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
7.  [Results and Visualization](#results-and-visualization)
8.  [Conclusion](#conclusion)

***

## 1. Project Overview

This project aims to accurately predict the selling price of a used car based on its various features, such as year of manufacture, kilometers driven (KMs), fuel type, and seller type. This is a **regression** problem, where the goal is to predict a continuous numerical value (the price).

We employ the **Random Forest Regressor** model for its ability to handle complex, non-linear feature interactions and high-dimensional data, and deploy the solution using an interactive **Streamlit** web application.

### Key Features:
* **Data Cleaning:** Handling missing values and outliers.
* **Feature Engineering:** Converting categorical data (e.g., fuel type, transmission) into numerical format using techniques like **One-Hot Encoding**.
* **Model Training:** **Random Forest Regression** implemented via `scikit-learn`.
* **Interactive App:** A Streamlit interface for users to input car features and get a predicted price.

***

## 2. Algorithm Used (Random Forest Regressor)

The prediction is driven by the **Random Forest Regressor** algorithm.

* **How it Works:** Random Forest is an **ensemble learning** method that constructs a multitude of decision trees during training and outputs the average of the predictions of the individual trees.
* **Suitability for Price Prediction:**
    * **Robustness:** It is highly resistant to overfitting (when tuned correctly).
    * **Non-Linearity:** It naturally models non-linear relationships between features (e.g., price decay is non-linear with respect to car age).
    * **Feature Importance:** It provides insights into which features (like `Year` or `KMs Driven`) are most influential in determining the car's price.

***

## 3. Dataset and Features

The dataset contains historical information on various used cars. Feature engineering was a critical step, converting the raw data into a format suitable for the model.

| Feature | Type | Example Values |
| :--- | :--- | :--- |
| `Year` | Numerical | 2017, 2012 |
| `Kms_Driven` | Numerical | 45000, 15000 |
| `Fuel_Type` | Categorical (Encoded) | Petrol, Diesel, CNG |
| `Seller_Type` | Categorical (Encoded) | Dealer, Individual |
| `Transmission` | Categorical (Encoded) | Manual, Automatic |
| **`Selling_Price`** | **Target (Numerical)** | **5.5 L, 12.0 L** |

***

## 4. Project Structure

The repository is organized as follows:
car-price-prediction/
├── .gitignore
├── README.md
├── requirements.txt           # Lists all necessary Python libraries
├── car_regressor.py           # Main ML code: loads, processes, trains RF model, saves model and preprocessor.
└── streamlit_app.py           # Streamlit code for the interactive web interface.


***

## 5. Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/car-price-prediction.git](https://github.com/YourUsername/car-price-prediction.git)
    cd car-price-prediction
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the ML Training Script:** This trains the Random Forest model and saves both the model (`.pkl` file) and any required preprocessors (like the encoder/scaler) for the Streamlit app.
    ```bash
    python car_regressor.py
    ```

***

## 6. How to Run the Streamlit App

The project includes an interactive Streamlit application for demonstration.

1.  **Ensure you have completed the Setup steps above.**

2.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

3.  The app will automatically open in your web browser at a local address (usually `http://localhost:8501`).

### 📷 Streamlit Application Preview

<p align="center">
  <img width="1562" height="911" alt="car" src="https://github.com/user-attachments/assets/5a4d1192-9604-451f-9389-2adc535b0ca8" />
  <br>
  <em>Figure 1: Streamlit App Interface allowing users to input car features (Year, KMs, Fuel Type) and get an estimated price.</em>
</p>

***

## 7. Results and Visualization

### Model Performance

The Random Forest Regressor model was evaluated on the test set using standard regression metrics.

* **R-squared ($R^2$):** **[Insert your calculated R-squared Score]** *(Measures how well the model fits the observed data, higher is better.)*
* **Mean Absolute Error (MAE):** **[Insert your calculated MAE Value]** *(Indicates the average magnitude of error in the predictions.)*

### Key Visualization

Visualization is essential for understanding the model's performance and feature importance.

<p align="center">
  <br>
  <em>Figure 2: Plot showing the top N features (e.g., Year and KMs Driven) ranked by their importance in predicting the car's price.</em>
</p>

***

## 8. Conclusion

The **Random Forest Regressor** demonstrated strong performance in predicting car prices, capturing the non-linear relationship between features like car age and the final selling price. This project provides a complete, deployable solution for car price estimation, validated by strong $R^2$ metrics and made accessible via Streamlit.
