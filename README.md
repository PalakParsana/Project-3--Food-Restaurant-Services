# AI Demand Forecasting & Inventory Optimization

## Overview: 
Build an AI-powered demand forecasting model for a restaurant chain that predicts daily sales volume for menu items using historical POS data and engineered features — enabling managers to reduce food waste and optimize supply chain decisions.

**Traditional inventory systems rely on static estimates and intuition, often leading to:**

 Over-ordering → food spoilage and losses
 Under-ordering → stockouts and missed revenue
 This project replaces guesswork with data-driven forecasting.

## Key Metrics: 

 **Model performance is evaluated using:**

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
 
 **The model is designed to capture:**

  * Weekly demand patterns (e.g., weekend spikes)
  * Seasonal trends
  * Long-term growth patterns

  ##  Users & Use Cases:

###  Restaurant Manager

**Primary Need:** Efficient inventory planning and waste reduction

**Use Case:**

* Uses daily/weekly demand forecasts to decide how much raw material to purchase
* Adjusts preparation based on predicted demand trends
* Minimizes food waste while ensuring availability of popular items

---

### Supply Chain Director

**Primary Need:** Optimized procurement and logistics management

**Use Case:**

* Analyzes aggregated forecasts across multiple outlets
* Plans bulk purchasing to reduce costs
* Coordinates supply distribution based on predicted demand

---

### Data Scientist / Analyst

**Primary Need:** Accurate modeling and feature engineering

**Use Case:**

* Builds and improves forecasting models using historical and external data
* Engineers features like lag variables, rolling averages, and seasonal indicators
* Evaluates model performance using MAE and RMSE

---

### Business Owner / Stakeholder

**Primary Need:** Strategic decision-making and profitability

**Use Case:**

* Uses insights from demand forecasts to plan promotions and pricing strategies
* Identifies peak sales periods and growth trends
* Makes data-driven decisions to improve overall business performance

---

## Technology Stack:

* Component                 Technology

* Data Processing           Python, Pandas, NumPy
* Machine Learning          Scikit-Learn, XGBoost
* Time Series               Statsmodels, Prophet
* Visualization             Matplotlib, Seaborn, Plotly
* Development               Jupyter Notebook

## System Architecture:

## 🏗️ System Architecture

```
┌────────────────────┐
│   Raw Sales Data   │
│ (CSV / Database)   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Data Preprocessing │
│ - Clean data       │
│ - Handle missing   │
│ - Format datetime  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Feature Engineering│
│ - Time features    │
│ - Lag variables    │
│ - Rolling stats    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Model Training     │
│ - Linear Regression│
│ - Random Forest    │
│ - XGBoost          │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Model Evaluation   │
│ - MAE              │
│ - RMSE             │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Visualization      │
│ - Trends           │
│ - Predictions      │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Business Insights  │
│ - Demand patterns  │
│ - Inventory plans  │
└────────────────────┘
```

# End-to-End Pipeline

1. **Data Ingestion**

Collect historical sales and external data

2. **Preprocessing**

Clean, format, and prepare time-series data

3. **Feature Engineering**

Create meaningful features like lag values and rolling averages

4. **Model Training**

Train baseline and advanced ML models

5. **Evaluation**

Compare models using MAE & RMSE

6. **Visualization & Insights**

Generate business insights from predictions

##  Project Roadmap:

---

## Week 1: Data Ingestion & Exploratory Data Analysis (EDA)

* Load and inspect the dataset
* Clean and preprocess raw data
* Convert and format datetime index
* Perform time-series EDA:

  * Trend analysis (overall sales pattern)
  * Seasonality detection (weekly/monthly trends)
  * Autocorrelation analysis
* Identify missing values and anomalies

---

## Week 2: Feature Engineering 

* Create time-based features:

  * Day of week, month, year
  * Weekend/holiday indicators
* Generate lag features:

  * Previous sales (`lag_7`, `lag_14`, etc.)
* Build rolling statistics:

  * Moving averages
  * Rolling standard deviation
* Perform **sequential train-test split** (no random split)

---

## Week 3: Model Training & Selection

* Train baseline model:

  * Linear Regression
* Implement advanced models:

  * Random Forest Regressor
  * XGBoost / Prophet
* Perform hyperparameter tuning
* Use **time-series cross-validation**
* Compare models using MAE and RMSE

---

## Week 4: Evaluation & Business Reporting

* Evaluate model performance:

  * MAE, RMSE
* Visualize:

  * Actual vs Predicted sales
* Analyze feature importance
* Extract business insights:

  * Peak demand periods
  * Inventory planning recommendations
* Prepare final report and GitHub documentation

---
