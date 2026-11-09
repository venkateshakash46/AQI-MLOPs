# End-to-End MLOps Project: Air Quality Prediction and Monitoring

An end-to-end **Machine Learning Operations (MLOps)** project that predicts Air Quality Index (AQI) classes using XGBoost and implements model lifecycle management, API deployment, performance monitoring, and data drift detection.

## 1. Project Overview

Building a machine learning model is only one part of a complete machine learning system. In a real-world environment, models must also be trained, evaluated, tracked, deployed, and continuously monitored.

This project demonstrates an end-to-end MLOps workflow by developing an Air Quality Prediction system using environmental data and integrating different tools to manage the machine learning lifecycle.

The project uses **XGBoost** for prediction, **MLflow** for experiment tracking and model registry, **FastAPI** for model serving, **Prometheus** for API monitoring, and **Evidently** for data drift detection.

## 2. Problem Statement

Air quality is influenced by several environmental factors, including particulate matter, gases, humidity, temperature, and wind conditions.

The objective of this project is to build a machine learning model that predicts the AQI class based on environmental input features and deploy it through an API.

Along with prediction, the project focuses on monitoring the deployed application and identifying changes in production input data.

## 3. Project Architecture

```text
              Environmental Dataset
                       |
                       v
               Data Preprocessing
                       |
                       v
                Data Validation
                       |
                       v
                 Data Splitting
                       |
                       v
                XGBoost Training
                       |
                       v
               Model Evaluation
                       |
                       v
             MLflow Experiment Tracking
                       |
                       v
                Model Registry
                       |
                       v
                 FastAPI Serving
                       |
              +--------+--------+
              |                 |
              v                 v
        Prometheus          Production
         Monitoring          Data Storage
              |                 |
              v                 v
       API Metrics          Evidently
                              |
                              v
                       Data Drift Report
```

## 4. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Feature Store | Feast |
| API Development | FastAPI |
| API Server | Uvicorn |
| API Monitoring | Prometheus |
| Data Drift Monitoring | Evidently |
| Environment Management | Poetry |
| Development Environment | PyCharm |

## 5. Implementation Details

### 5.1 Data Pipeline

The initial stage focuses on preparing the environmental dataset for machine learning.

Implemented activities:

- Data loading and inspection.
- Data validation.
- Handling missing and duplicate records.
- Data cleaning and preprocessing.
- Preparing features and target variables.
- Splitting the dataset into training and testing data.

The target variable for prediction is `AQI_Class`.

### 5.2 Model Training

The project uses **XGBoost (Extreme Gradient Boosting)**, an ensemble learning algorithm based on gradient-boosted decision trees.

**Why XGBoost?**

- Performs well on structured and tabular datasets.
- Captures nonlinear relationships between environmental variables.
- Learns interactions between different input features.
- Provides useful hyperparameters to control model complexity.
- Integrates with the MLflow ecosystem.

The model uses environmental features such as:

- PM2.5 and PM10
- NO2, SO2, O3, CO, NO and NOx
- Relative humidity
- Temperature
- Wind speed
- Wind direction

### 5.3 Model Evaluation

After training, the model is evaluated using the prepared evaluation dataset.

The evaluation stage helps assess the model's predictive performance before registering it for serving.

### 5.4 MLflow Experiment Tracking

MLflow is integrated to track machine learning experiments and maintain model lifecycle information.

Implemented features:

- Experiment tracking.
- Logging model parameters and evaluation metrics.
- Tracking model artifacts.
- Registering trained models.
- Maintaining model versions through the Model Registry.

The trained XGBoost model is registered under the name `xgboost`.

### 5.5 Feature Store with Feast

Feast is used as the feature store component of the project.

It provides a structured way to manage and retrieve features through feature views and an entity-based feature store.

### 5.6 FastAPI Model Serving

FastAPI is used to expose the registered XGBoost model through a REST API.

The API accepts environmental measurements as input and returns the predicted AQI class.

Main endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Check API availability |
| `/predict` | POST | Generate AQI class predictions |
| `/metrics` | GET | Expose Prometheus metrics |

Interactive API documentation is available through:

`http://127.0.0.1:8000/docs`

### 5.7 Prometheus API Monitoring

Prometheus is integrated to monitor the performance and operational behavior of the FastAPI application.

The following metrics are implemented:

| Metric | Description |
|---|---|
| `prediction_requests_total` | Total number of prediction requests |
| `prediction_errors_total` | Total number of failed prediction requests |
| `prediction_latency_seconds` | Time taken to generate predictions |

Prometheus periodically scrapes metrics from the FastAPI `/metrics` endpoint and stores them as time-series data.

This helps monitor API activity, errors, and prediction latency over time.

### 5.8 Evidently Data Drift Monitoring

Evidently is integrated to compare reference data with current production input data.

**Reference data:** Preprocessed environmental dataset used as the baseline.

**Production data:** Actual input features received through the FastAPI prediction endpoint and stored in `production_data.csv`.

The production data collection mechanism allows incoming prediction inputs to be used for drift analysis.

The `DataDriftPreset` is used to compare the distributions of the reference and production datasets.

The generated HTML report provides information about:

- Dataset-level drift.
- Feature-level drift.
- Changes in feature distributions.
- Statistical comparisons between reference and current data.

**Demonstration:** Four prediction requests were generated through the FastAPI endpoint. Their input values were stored in the production data file and used to test the Evidently reporting workflow.

The four requests demonstrate the integration and execution of the pipeline; they are not intended to establish a statistically reliable production drift conclusion.

## 6. Project Workflow

The complete workflow implemented in this project is:

1. Load and preprocess environmental data.
2. Validate and prepare the dataset.
3. Train the XGBoost model.
4. Evaluate the trained model.
5. Track experiments using MLflow.
6. Register the trained model in MLflow Model Registry.
7. Serve predictions through FastAPI.
8. Monitor API requests, errors, and latency using Prometheus.
9. Collect production input data.
10. Compare reference and production data using Evidently.
11. Generate an HTML data drift report.

## 7. Running the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/venkateshakash46/AQI-MLOPs.git
cd AQI-MLOPs
cd "project 1"
```

### Step 2: Set Up the Environment

Install the required dependencies using the Poetry configuration provided in the project.

```bash
poetry install
```

Activate or use the appropriate Python environment for the required pipeline.

### Step 3: Run the Training Pipeline

Run the training pipeline using the project's package structure:

```bash
python -m project1.Pipelines.training_pipeline.MainTrainPipeline
```

This executes the training, evaluation, experiment tracking, and model registration stages.

### Step 4: Start the MLflow Server

Configure the tracking URI according to your local MLflow database.

```bash
mlflow server --backend-store-uri <YOUR_MLFLOW_DATABASE_URI> --host 127.0.0.1 --port 5000
```

Open:

`http://127.0.0.1:5000`

### Step 5: Start the FastAPI Application

```bash
python -m uvicorn project1.Pipelines.serving_pipeline.app:app --reload
```

Open:

`http://127.0.0.1:8000/docs`

Use the `/predict` endpoint to submit environmental measurements.

### Step 6: Start Prometheus

Start the Prometheus server using your configured `prometheus.yml` file.

```bash
prometheus.exe --config.file=prometheus.yml
```

Open:

`http://127.0.0.1:9090`

### Step 7: Generate the Data Drift Report

After collecting production inputs through the prediction API, execute:

```bash
python -m project1.Pipelines.serving_pipeline.drift
```

The generated HTML report will be saved as:

```text
project1/Data/drift_report.html
```

Open the HTML file in a browser to explore the data drift analysis.

**Note:** Run commands from the appropriate project/environment directory and configure the Python import path as required by your repository structure.

## 8. Key Learnings

Through this project, I gained practical experience in:

- Developing an end-to-end machine learning pipeline.
- Training and evaluating an XGBoost model.
- Tracking experiments and managing model versions using MLflow.
- Understanding feature store integration using Feast.
- Deploying machine learning models through REST APIs.
- Monitoring API performance using Prometheus.
- Collecting production input data.
- Detecting potential data distribution changes using Evidently.
- Understanding the importance of continuous monitoring in production ML systems.

## 9. Future Enhancements

Potential improvements include:

- Automating periodic data drift checks.
- Increasing production data collection for more reliable drift analysis.
- Integrating drift metrics with the monitoring system.
- Adding automated model retraining when appropriate.
- Containerizing the application using Docker.
- Implementing CI/CD for automated testing and deployment.
- Deploying the application to a cloud environment.

## 10. Conclusion

This project demonstrates how individual machine learning components can be connected to build a practical end-to-end MLOps workflow.

It goes beyond model development by incorporating experiment tracking, model versioning, API serving, operational monitoring, and data drift analysis.

The project helped me understand that successful machine learning systems require not only accurate models but also proper deployment, monitoring, and maintenance.

---

**Next Learning Journey:** LLMOps 🚀

Exploring the lifecycle, deployment, evaluation, and monitoring of Large Language Models.

### Author

M.Sc. Data Science | Machine Learning | MLOps | LLMOps

