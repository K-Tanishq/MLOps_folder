from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn


class IrisDataProcessor:
    def __init__(self):
        # Load and initialize the Iris dataset
        self.iris = load_iris()
        self.data = None
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    
    def prepare_data(self):
        # Convert data to DataFrame with column names
        self.data = pd.DataFrame(
            data=np.c_[self.iris['data'], self.iris['target']],
            columns=self.iris['feature_names'] + ['target']
        )
        
        # Feature scaling
        features = self.data.iloc[:, :-1]  # all columns except target
        features_scaled = self.scaler.fit_transform(features)
        self.data.iloc[:, :-1] = features_scaled

        # Train-test split
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_stats(self):
        # Statistical analysis on features
        return self.data.describe()
    

class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = models = {
            'Logistic_Regression': LogisticRegression(),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.results = {}

    def run_experiment(self):
        # Cross-validation and model training with MLflow tracking
        X_train, X_test, y_train, y_test = (
            self.data_processor.X_train, 
            self.data_processor.X_test, 
            self.data_processor.y_train, 
            self.data_processor.y_test
        )

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # Training the model
                model.fit(X_train, y_train)
                
                # Predictions and evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="macro")
                recall = recall_score(y_test, y_pred, average="macro")
                
                # Logging metrics
                self.log_results(model_name, accuracy, precision, recall, cv_scores.mean())
                self.results[model_name] = (accuracy, precision, recall)

    def log_results(self, model_name, accuracy, precision, recall, cv_score):
        # MLflow logging
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("cross_val_score", cv_score)

class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.quantized_coefficients = None
        self.quantized_intercept = None

    def quantize_model(self):
        # Get Logistic Regression model
        logistic_model = self.experiment.models["LogisticRegression"]
        
        # Perform quantization by rounding coefficients and intercept
        self.quantized_coefficients = np.round(logistic_model.coef_, decimals=1)
        self.quantized_intercept = np.round(logistic_model.intercept_, decimals=1)
        
        # Print quantized coefficients and intercept for verification
        print("Quantized Coefficients:", self.quantized_coefficients)
        print("Quantized Intercept:", self.quantized_intercept)
    
    def run_tests(self):
        # Simple test to ensure quantization was successful
        assert self.quantized_coefficients is not None, "Quantized coefficients not available"
        assert self.quantized_intercept is not None, "Quantized intercept not available"
        print("Quantized model has been tested successfully.")

processor = IrisDataProcessor()
X_train, X_test, y_train, y_test = processor.prepare_data()

# Run experiments
experiment = IrisExperiment(processor)
# experiment.run_experiment()

# Optimize and test
optimizer = IrisModelOptimizer(experiment)
# optimizer.quantize_model()
# optimizer.run_tests()


models = {
    'Logistic_Regression': LogisticRegression(),
    'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# MLflow experiment logging
mlflow.set_experiment("Iris_Models")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred, average="macro")
        # recall = recall_score(y_test, y_pred, average="macro")

        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("R-squared", r2)            
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_metric("precision", precision)
        # mlflow.log_metric("recall", recall)
        # mlflow.log_metric("cross_val_score", cv_scores)
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} - MSE: {mse}")
        print(f"{model_name} - R-squared: {r2}")
       
