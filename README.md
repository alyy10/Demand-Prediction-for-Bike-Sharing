# Demand Prediction for Bike Sharing

A comprehensive machine learning project that predicts bike sharing demand using historical data and weather conditions. The project implements separate prediction models for casual and registered users, providing detailed insights through model interpretability tools.

## Project Overview

This project addresses the challenge of predicting bike sharing demand by leveraging machine learning techniques on the UCI Bike Sharing Dataset. The solution employs LightGBM regression models trained separately for casual and registered users, achieving better performance than a single unified model.

Key features:
- **Dual Model Architecture**: Separate models for casual and registered users
- **Advanced Feature Engineering**: Time-based features, rolling statistics, and categorical binning
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) for understanding predictions
- **Web Application**: Flask-based interface for real-time predictions
- **Comprehensive Analysis**: Full ML pipeline from EDA to deployment

## Dataset

The project uses the **UCI Bike Sharing Dataset** which contains bike sharing rental patterns from Capital Bikeshare system in Washington D.C. spanning two years (2011-2012).

### Data Features

**Temporal Features:**
- `season`: Season (1: winter, 2: spring, 3: summer, 4: fall)
- `yr`: Year (0: 2011, 1: 2012)
- `mnth`: Month (1 to 12)
- `hr`: Hour (0 to 23)
- `weekday`: Day of the week (0: Sunday to 6: Saturday)

**Environmental Features:**
- `temp`: Normalized temperature in Celsius (-8°C to 39°C range)
- `atemp`: Normalized "feels like" temperature
- `hum`: Normalized humidity (0-100%)
- `windspeed`: Normalized wind speed (0-67 km/h range)
- `weathersit`: Weather situation (1: Clear, 2: Mist/Cloudy, 3: Light Snow/Rain, 4: Heavy Rain/Snow)

**Calendar Features:**
- `holiday`: Whether the day is a holiday
- `workingday`: Whether the day is a working day

**Target Variables:**
- `casual`: Count of casual users
- `registered`: Count of registered users  
- `cnt`: Total count (casual + registered)

## Methodology

### 1. Data Preprocessing & Feature Engineering

**Weather Data Normalization:**
- Temperature normalization: `(temp + 8) / (39 + 8)`
- Wind speed normalization: `windspeed / 67`
- Humidity normalization: `humidity / 100`

**Categorical Feature Engineering:**
- **Day Type Classification**: 
  - 0: Holiday
  - 1: Weekend  
  - 2: Working day
- **Weather Situation Merging**: Categories 3 and 4 combined due to low frequency

**Temporal Feature Engineering:**
- **Hour Binning for Registered Users**: Strategic time periods based on commuting patterns
  - Bins: [1.5, 5.5, 6.5, 8.5, 16.5, 18.5, 20.5, 22.5]
  - Remapped to capture peak usage patterns
- **Hour Binning for Casual Users**: Leisure-focused time periods  
  - Bins: [7.5, 8.5, 10.5, 17.5, 19.5, 21.5]
  - Optimized for recreational usage patterns

**Time Series Features:**
- **Rolling Mean (12-hour window)**: Captures short-term trends
- **3-Day Historical Sum**: Uses 24h, 48h, and 72h lagged values

### 2. Model Architecture

**Dual Model Approach:**
- **Model 1 (Casual Users)**: Optimized for recreational usage patterns
- **Model 2 (Registered Users)**: Optimized for commuting patterns

**Algorithm:** LightGBM (Light Gradient Boosting Machine)
- Chosen for its efficiency with categorical features
- Superior performance on time series data
- Built-in handling of missing values

**Hyperparameter Optimization:**
- Conducted using Optuna framework
- Cross-validation with time series splits
- RMSE as optimization metric

### 3. Model Interpretability

**SHAP Integration:**
- Waterfall plots for individual predictions
- Feature importance analysis
- Local and global explanation capabilities
- Real-time explanation generation in web app

## Usage Examples

### Model Training

```python
# Run the training script
python train.py

# This will:
# 1. Load and preprocess the bike sharing data
# 2. Apply feature engineering transformations
# 3. Train separate LightGBM models for casual and registered users
# 4. Save trained models as model_1.pkl and model_2.pkl
```

**Training Process:**
1. **Data Loading**: Reads `hour.csv` with date parsing
2. **Feature Engineering**: Applies all preprocessing steps
3. **Model Training**: Trains two separate LightGBM regressors
4. **Model Persistence**: Saves models using pickle format


**Example Prediction Request:**
```python
# Input parameters
data = {
    'temp': 25,        # Temperature in Celsius  
    'hum': 65,         # Humidity percentage
    'windspeed': 15,   # Wind speed in km/h
    'hr': 8,           # Hour (8 AM)
    'season': 2,       # Spring
    'mnth': 4,         # April
    'weekday': 1,      # Monday
    'weathersit': 1,   # Clear weather
    'holiday': 0       # Not a holiday
}

# Returns: Total predicted demand + SHAP explanations
```

### Jupyter Notebook Analysis

The `complete_model_dev.ipynb` notebook provides comprehensive analysis:

**1. Data Exploration:**
```python
# View data structure and basic statistics
data.info()
data.describe()

# Visualize target variable distributions
plt.figure(figsize=(15, 5))
for i, col in enumerate(['cnt', 'casual', 'registered']):
    plt.subplot(1, 3, i+1)
    data[col].hist(bins=50)
    plt.title(f'Distribution of {col}')
```

**2. Feature Engineering Pipeline:**
```python
# Apply comprehensive feature engineering
def prepare_data(data):
    # Weather situation merging
    data['weathersit'].replace(4, 3, inplace=True)
    
    # Day type creation
    data['day_type'] = np.where(data['holiday'] == 1, 0, 2)
    data['day_type'] = np.where((data['weekday'] == 6) | 
                               (data['weekday'] == 0), 1, data['day_type'])
    
    # Hour binning for different user types
    # ... (detailed binning logic)
    
    # Rolling statistics
    data['rolling_mean_12_hours_casual'] = data['casual'].rolling(12).mean()
    data['rolling_mean_12_hours_registered'] = data['registered'].rolling(12).mean()
    
    # Lagged features  
    data['3_days_sum_casual'] = (data['casual'].shift(24) + 
                                data['casual'].shift(48) + 
                                data['casual'].shift(72))
    
    return data
```

**3. Model Training and Evaluation:**
```python
# Feature selection for each model
features_casual = ['temp', 'hum', 'windspeed', 'hr', '3_days_sum_casual',
                  'rolling_mean_12_hours_casual', 'season', 'yr', 'mnth',
                  'day_type', 'weathersit', 'CasualHourBins', 'weekday']

features_registered = ['temp', 'hum', 'windspeed', 'hr', '3_days_sum_registered',
                      'rolling_mean_12_hours_registered', 'season', 'yr', 'mnth', 
                      'day_type', 'weathersit', 'RegisteredHourBins', 'weekday']

# Model training with optimized hyperparameters
model_casual = LGBMRegressor(**params_casual, random_state=100)
model_registered = LGBMRegressor(**params_registered, random_state=100)

model_casual.fit(X_casual, y_casual)
model_registered.fit(X_registered, y_registered)
```

## Results & Performance

### Model Performance Metrics

**Evaluation Approach:**
- Time series cross-validation to prevent data leakage
- RMSE (Root Mean Square Error) as primary metric
- Separate evaluation for casual and registered user models

**Key Findings:**

1. **Dual Model Superiority**: Separate models for casual and registered users significantly outperform a single unified model

2. **Feature Importance Insights:**
   - **Casual Users**: Weather conditions and time of day are primary drivers
   - **Registered Users**: Commuting patterns (hour bins) and day type are most influential
   - **Common Factors**: Temperature and humidity affect both user types

3. **Temporal Patterns:**
   - **Registered Users**: Strong peaks during commuting hours (7-9 AM, 5-7 PM)
   - **Casual Users**: Gradual increase throughout daylight hours, peaks on weekends

4. **Seasonal Impact:**
   - Higher demand in spring and summer months
   - Weather conditions have stronger impact on casual users
   - Registered users show more consistent patterns regardless of weather

### Business Impact

**Operational Benefits:**
- **Resource Optimization**: Better bike distribution planning
- **Maintenance Scheduling**: Predictive maintenance based on usage patterns  
- **Capacity Planning**: Strategic station placement and bike allocation
- **Revenue Optimization**: Dynamic pricing strategies based on demand predictions

**Predictive Accuracy:**
- Real-time demand forecasting with hourly granularity
- Weather-adjusted predictions for operational planning
- User-type specific insights for targeted marketing strategies



## Technical Implementation Details

### Model Hyperparameters

**Casual Users Model (model_1.pkl):**
```python
parameters1 = {
    "objective": "regression",
    "metric": "rmse", 
    "boosting_type": "gbdt",
    "lambda_l1": 0.2019055894080857,
    "lambda_l2": 3.5275169933928286e-07,
    "num_leaves": 110,
    "feature_fraction": 0.92,
    "bagging_fraction": 1.0,
    "n_estimators": 117
}
```

**Registered Users Model (model_2.pkl):**
```python
parameters2 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt", 
    "lambda_l1": 0.010882827930218712,
    "lambda_l2": 0.2708162972907513,
    "num_leaves": 71,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7260429650751228,
    "bagging_freq": 2,
    "n_estimators": 147
}
```

### Web Application Architecture

**Flask Backend:**
- Model loading and caching for performance
- Real-time feature engineering pipeline
- SHAP explanation generation
- RESTful API design for predictions

**Frontend Features:**
- Intuitive parameter input forms
- Real-time prediction display
- Interactive SHAP visualizations
- Responsive design for multiple devices

## Future Enhancements

**Model Improvements:**
- Integration of external weather APIs for real-time data
- Deep learning models for complex temporal patterns
- Ensemble methods combining multiple algorithms
- Online learning for continuous model updates

**Application Features:**
- Real-time dashboard for operational monitoring
- Mobile application for field operations
- Integration with bike sharing management systems
- Advanced analytics and reporting capabilities

**Data Enhancement:**
- Integration of additional data sources (events, traffic, etc.)
- Spatial analysis for location-based predictions
- User behavior modeling for personalized recommendations
- Economic factors integration for demand modeling

---

*This project demonstrates end-to-end machine learning implementation from exploratory data analysis to production deployment, showcasing best practices in feature engineering, model selection, and interpretable AI.*
