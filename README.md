# Bike Sharing Demand Forecast

## Project Overview

This project builds a machine learning model to predict the hourly demand for a bike-sharing program in Washington, D.C. By analyzing historical rental data along with environmental and temporal features (like weather, temperature, hour, and day), we can forecast the total number of bike rentals (`count`) for a given hour.

This is a classic regression project, and the solution is based on the [Bike Sharing Demand competition on Kaggle](https://www.kaggle.com/c/bike-sharing-demand/data).

## Business Objectives

An accurate demand forecast is crucial for a bike-sharing company to optimize its operations. This model helps to:
* **Manage Inventory:** Ensure enough bikes are available at high-demand locations and times.
* **Optimize Operations:** Schedule maintenance and re-distribution of bikes during low-demand periods.
* **Plan for Growth:** Understand the key drivers of demand to inform business strategy.

## Dataset

* **Source:** [Kaggle: Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data)
* **Files Used:** `train.csv`, `test.csv`

**Key Features:**
* `datetime`: Hourly date & time
* `season`: 1 (spring), 2 (summer), 3 (fall), 4 (winter)
* `holiday`: Whether the day is a holiday (1/0)
* `workingday`: Whether the day is neither a weekend nor holiday (1/0)
* `weather`: 1 (Clear), 2 (Mist), 3 (Light Rain/Snow), 4 (Heavy Rain)
* `temp`: Temperature in Celsius
* `atemp`: "Feels like" temperature in Celsius
* `humidity`: Relative humidity
* `windspeed`: Wind speed
* `count`: Total number of rentals (Target Variable)

## Methodology

### 1. Feature Engineering
The most important step was processing the `datetime` column. I extracted new, more useful features:
* `year`
* `month`
* `hour`
* `weekday`

### 2. Exploratory Data Analysis (EDA)
I analyzed the data to find key patterns:
* **Commuter Pattern:** Demand peaks sharply around 8 AM and 5-6 PM on working days, indicating heavy use for commuting.
* **Weather Impact:** Rentals drop significantly during bad weather (Light Rain/Snow) and high humidity.
* **Seasonal Trend:** Demand is lowest in winter (Jan/Feb) and peaks in summer and fall (June-Sept).
* **Feature Correlation:** `temp` and `atemp` ("feels like" temp) were almost perfectly correlated, so `atemp` was dropped to avoid multicollinearity.

### 3. Data Pre-processing (Target Transformation)
The competition's evaluation metric is the **Root Mean Squared Logarithmic Error (RMSLE)**. To optimize for this, the target variable `count` was transformed using `log(count + 1)`. The model was trained to predict this log-transformed value, and the final predictions were converted back using `expm1()`.

### 4. Modeling
Two models were trained and compared:

* **Baseline Model: Random Forest Regressor**
    * A strong, reliable model that provided a good baseline score and clear feature importances.
    * **Key finding:** `hour` was by far the most important feature, followed by `temp` and `workingday`.

* **Final Model: XGBoost Regressor**
    * A powerful gradient-boosting model known for high performance.
    * This model achieved a significantly better RMSLE score on the validation set.
    * The final submission was generated using this model, trained on the complete dataset.

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Souhardya05/Bike-Sharing-Demand-Forecast.git
    cd Bike-Sharing-Demand-Forecast
    ```
2.  **Download the data:**
    * Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/bike-sharing-demand/data).
    * Place both files in the main project directory.

3.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```

5.  **Run the notebook:**
    * Open the `Bike_Sharing_Demand_Project.ipynb` file in Jupyter Notebook, Jupyter Lab, or VS Code.
    * Run the cells from top to bottom.
    * The final output, `bike_demand_submission.csv`, will be saved in the project directory.

## Key Findings & Conclusion

This project successfully built a model to predict hourly bike demand.

* **Key Drivers:** Demand is primarily driven by **commuter patterns** (time of day, working day) and **environmental factors** (temperature, weather).
* **Model Performance:** The `XGBoost Regressor` proved tobe a highly effective model for this type of time-series/regression task, outperforming the Random Forest baseline.
