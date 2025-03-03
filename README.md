# smart-utility-monitor

Below is a complete Python program that simulates a smart utility monitor. This basic implementation uses a dummy dataset, a simple linear regression model for predictions, and provides recommendations. This program does not interact with real-world data or APIs but instead gives a structure for further development with actual data sources and advanced models.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample utility data generation (Date, Electricity, Water, Gas usage)
def generate_dummy_data():
    np.random.seed(0)
    dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
    electricity_usage = np.random.uniform(5, 30, size=len(dates))  # kWh
    water_usage = np.random.uniform(50, 300, size=len(dates))  # Liters
    gas_usage = np.random.uniform(1, 15, size=len(dates))  # Cubic meters
    data = pd.DataFrame({'Date': dates, 'Electricity': electricity_usage, 
                         'Water': water_usage, 'Gas': gas_usage})
    return data

# Load data
def load_data():
    try:
        data = generate_dummy_data()
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train a simple regression model
def train_model(data, feature_cols, target_col):
    try:
        X = data[feature_cols]
        y = data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Evaluating the model
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Model Evaluation for {target_col} - MSE: {mse}, R2: {r2}")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

# Make predictions and provide recommendations
def predict_and_recommend(model, data, feature_cols, utility_type):
    try:
        tomorrow_features = data[feature_cols].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(tomorrow_features)[0]
        
        recommendation = ""
        if utility_type == 'Electricity':
            if prediction > 25:
                recommendation = "Consider using less heating/cooling equipment."
        elif utility_type == 'Water':
            if prediction > 250:
                recommendation = "Look for leaks or consider shorter showers."
        elif utility_type == 'Gas':
            if prediction > 12:
                recommendation = "Check insulation or reduce heating duration."
        
        print(f"Predicted {utility_type} usage for tomorrow: {prediction:.2f}")
        print(f"Recommendation: {recommendation if recommendation else 'Usage is within normal range.'}")
    except Exception as e:
        print(f"Error in prediction and recommendation: {e}")

# Main function
def main():
    data = load_data()
    if data is not None:
        feature_columns = ['Electricity', 'Water', 'Gas']
        
        # Train and evaluate models for each utility
        for feature in feature_columns:
            print(f"\nTraining model for {feature} usage prediction:")
            model = train_model(data, feature_columns, feature)
            if model is not None:
                predict_and_recommend(model, data, feature_columns, feature)

if __name__ == "__main__":
    main()
```

### Notes:
- **Data Generation:** The generated dataset here is random. In a real-world scenario, you would load actual utility usage data.
- **Model Choice:** A simple linear regression is used for demonstration. Depending on your dataset, more complex models like Gradient Boosting or Neural Networks could be used.
- **Error Handling:** Basic error handling is included around data loading, model training, and prediction sections.
- **Recommendations:** Basic recommendations are printed based on predicted usage levels. This can be improved with domain-specific insights.

This code serves as a framework and can be expanded with additional features such as data visualization, user interface, and real utility data integration.