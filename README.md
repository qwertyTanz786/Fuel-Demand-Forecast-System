# UAE Smart Fuel Demand Forecasting System

## Project Overview

The Smart Fuel Demand Forecasting System is a Machine Learning-powered application that predicts fuel demand for different vehicle categories across the UAE. The system analyzes vehicle specifications, fuel types, location data, and temporal factors to estimate fuel consumption and associated costs.

The project consists of:

- Machine Learning Model (XGBoost)
- Interactive Streamlit Dashboard
- Fuel Demand Analytics & Visualization Module

The system helps users make informed fueling decisions through AI-driven demand forecasting.

---

# Features

## Machine Learning Features

- Fuel demand prediction using XGBoost Regressor
- Feature engineering for improved accuracy
- Vehicle-specific tank capacity mapping
- Traffic impact analysis
- Station popularity analysis
- Fuel cost estimation

## Dashboard Features

- Interactive vehicle configuration panel
- Real-time fuel demand prediction
- Estimated fuel cost calculation
- Tank utilization analysis
- Petrol company comparison charts
- Dynamic alerts and recommendations
- Modern glassmorphism UI design

---

# Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Core Programming |
| Pandas | Data Processing |
| XGBoost | Machine Learning |
| Scikit-Learn | Model Training |
| Joblib | Model Serialization |
| Streamlit | Web Dashboard |
| Plotly | Interactive Visualizations |

---

# Dataset Features

The model uses the following inputs:

## Vehicle Information

- Vehicle Type
- Vehicle Model
- Vehicle Class
- Tank Capacity

## Fuel Information

- Fuel Type
- Fuel Price

## Location Information

- Emirate
- Petrol Station

## Time Information

- Month
- Weekday

## Engineered Features

- Traffic Score
- Station Popularity

---

# Machine Learning Workflow

## 1. Data Preprocessing

The dataset undergoes:

- Date conversion
- Month extraction
- Weekday extraction
- Missing value handling
- Feature engineering
- One-hot encoding

---

## 2. Feature Engineering

Additional features created include:

### Tank Capacity

Vehicle-specific tank capacities are mapped manually.

### Vehicle Class

Vehicles are categorized into:

- Light
- Medium
- Performance
- Commercial
- Heavy

### Traffic Score

Traffic congestion levels are assigned based on emirates:

| Emirate | Score |
|----------|---------|
| Dubai | 5 |
| Abu Dhabi | 4 |
| Sharjah | 3 |
| Ajman | 2 |
| Al Ain | 2 |

### Station Popularity

Popularity scores are calculated from petrol station frequency within the dataset.

---

## 3. Model Training

The system uses XGBoost Regressor.

### Model Parameters

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Target Variable

```text
Fuel_Demand_Liters
```

---

## 4. Model Export

After training:

```python
joblib.dump(model, "fuel_model.pkl")
joblib.dump(training_columns, "training_columns.pkl")
```

This enables deployment without retraining.

---

# Prediction Workflow

The user provides:

1. Emirate
2. Vehicle Model
3. Fuel Type
4. Petrol Station
5. Month
6. Weekday

The system then:

- Generates engineered features
- Applies one-hot encoding
- Aligns data with training columns
- Predicts fuel demand
- Estimates fuel cost

---

# Fuel Cost Calculation

Fuel prices are predefined:

| Fuel Type | Price (AED/L) |
|------------|-------------|
| Diesel | 4.33 |
| E-Plus 91 | 3.76 |
| Special 95 | 3.83 |
| Super 98 | 3.95 |

Estimated cost is calculated as:

```text
Estimated Cost = Predicted Demand × Fuel Price
```

---

# Dashboard Components

## Vehicle Configuration Panel

Users can select:

- Vehicle Model
- Emirate
- Month
- Weekday
- Fuel Type
- Petrol Station

---

## Fuel Demand Metrics

The dashboard displays:

### Fuel Demand

```text
Predicted Fuel Consumption (Liters)
```

### Estimated Cost

```text
Predicted Fuel Expense (AED)
```

### Tank Capacity

```text
Maximum Fuel Capacity
```

---

## Fuel Demand Comparison

The dashboard compares predictions across:

- ENOC
- EPPCO
- ADNOC
- CAFU
- Emarat

Interactive Plotly bar charts are generated for comparison.

---

## Tank Utilization Analysis

The system calculates:

```text
Tank Utilization % =
Predicted Demand / Tank Capacity
```

and displays a visual progress bar.

---

# Intelligent Alerts

## Low Demand Alert

Triggered when:

```text
Demand ≤ 33% of Tank Capacity
```

Recommendation:

> Current fuel level should be sufficient.

---

## High Demand Alert

Triggered when:

```text
Demand > 100 Liters
```

Recommendation:

> Increased driving activity or heavy traffic expected.

---

## Capacity Warning

Triggered when:

```text
Demand > Tank Capacity
```

Recommendation:

> Refueling will likely be required.

---

# User Interface Design

The Streamlit application includes:

- Animated gradient background
- Glassmorphism cards
- Interactive hover effects
- Dynamic metric animations
- Responsive Plotly charts
- Modern sidebar navigation

---

# Project Structure

```text
UAE_Smart_Fuel_Forecasting/
│
├── fuel_model.pkl
├── training_columns.pkl
├── fuel_forecasting_model.py
├── streamlit_dashboard.py
├── Fuel_Dataset_Monthly_Pricing.csv
│
└── README.md
```

---

# Future Enhancements

Potential improvements include:

- Real-time fuel price integration
- Weather-based demand forecasting
- Route-aware fuel consumption estimation
- MLOps deployment pipeline
- Mobile application integration
- Live petrol station traffic analysis
- Historical trend forecasting
- Fuel efficiency recommendations

---

# Conclusion
The UAE Smart Fuel Demand Forecasting System demonstrates how Machine Learning can be applied to transportation and fuel analytics. By combining feature engineering, XGBoost regression, and an interactive Streamlit dashboard, the system provides actionable fuel demand insights and cost estimates for a wide variety of vehicle categories operating across the UAE.
