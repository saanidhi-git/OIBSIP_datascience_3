# CAR PRICE PREDICTION MODEL 


# 1️. IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 2️. LOAD THE DATASET

df = pd.read_csv("car data.csv")


print("✅ Data Loaded Successfully\n")
print("Shape of dataset:", df.shape)
print("\nFirst 8 rows of data:\n", df.head(8))
print("\nInformation about columns:\n")
print(df.info())
print("\nDescription of data:\n", df.describe(include='all'))
print("\nMissing values in each column:\n", df.isnull().sum())


# 3️. DEFINE FEATURES (X) AND TARGET (y)
# 'Selling_Price' is what we want to predict
# 'Car_Name' is not useful for prediction, so we drop it
X = df.drop(columns=['Car_Name', 'Selling_Price'])
y = df['Selling_Price']
print("\nX shape:", X.shape, "| y shape:", y.shape)


# 4️. SPLIT INTO TRAINING AND TESTING DATA
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")


# 5️. DEFINE CATEGORICAL & NUMERICAL COLUMNS
categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
numeric_cols = ['Year', 'Present_Price', 'Driven_kms', 'Owner']


# 6️. CREATE TRANSFORMERS
# OneHotEncoder for categorical data
# StandardScaler for numerical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
numeric_transformer = StandardScaler()


# 7️. COMBINE THEM USING COLUMNTRANSFORMER

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, categorical_cols),
    ('num', numeric_transformer, numeric_cols)
])


# 8️. DEFINE THE MODEL 
model = RandomForestRegressor(random_state=42)


# 9️. CREATE THE PIPELINE 
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])


# 10. TRAIN (FIT) THE MODEL
pipe.fit(X_train, y_train)
print("\n🚀 Model training complete!")


# 11. MAKE PREDICTIONS
y_pred = pipe.predict(X_test)


# 12. EVALUATE THE MODEL 
# Calculate MSE first, then take the square root to get RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5 # or np.sqrt(mse) if numpy is imported

r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation Results:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 13.  CHECK FEATURE IMPORTANCE
# Extract feature names after one-hot encoding
ohe_features = pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = list(ohe_features) + numeric_cols

importances = pipe.named_steps['model'].feature_importances_
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n📊 Top 10 Important Features Affecting Car Price:")
print(feature_importances.head(10))


# 14.  SAVE TRAINED MODEL
import joblib
joblib.dump(pipe, "car_price_model.pkl")
print("\n💾 Model saved successfully as 'car_price_model.pkl'!")
