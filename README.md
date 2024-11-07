# Vehicle Maintenance Prediction 

## Overview
This project leverages machine learning models to predict whether a vehicle requires maintenance based on multiple features such as mileage, age, service history, and accident history.

## Models Used and Their Performance

### 1. **K-Nearest Neighbors (KNN)**
   - **Accuracy**: 78.37%
   - **Precision**: Class 0: 32%, Class 1: 82%
   - **Recall**: Class 0: 11%, Class 1: 94%
   - **F1-Score**: Class 0: 17%, Class 1: 88%
   - **Summary**: KNN struggled with classifying class 0 (no maintenance) accurately. It was heavily biased towards predicting class 1 (maintenance needed), with a high recall but low precision for class 0.

### 2. **Logistic Regression**
   - **Accuracy**: 94.41%
   - **Precision**: Class 0: 86%, Class 1: 96%
   - **Recall**: Class 0: 84%, Class 1: 97%
   - **F1-Score**: Class 0: 85%, Class 1: 97%
   - **Summary**: Logistic Regression performed well, especially for class 1. It showed a balanced performance in both precision and recall, making it a reliable model for predicting vehicle maintenance needs.

### 3. **Random Forest Classifier**
   - **Accuracy**: 100%
   - **Precision**: 100% for both classes
   - **Recall**: 100% for both classes
   - **F1-Score**: 100% for both classes
   - **Summary**: Random Forest achieved perfect accuracy and performance metrics, making it the best-performing model. It was able to classify both classes correctly without any errors, demonstrating the power of ensemble learning.

### 4. **XGBoost**
   - **Accuracy**: 100%
   - **Precision**: 100% for both classes
   - **Recall**: 100% for both classes
   - **F1-Score**: 100% for both classes
   - **Summary**: XGBoost also achieved perfect accuracy, surpassing other models in performance. It demonstrated exceptional predictive power, particularly due to its gradient boosting framework, which optimizes model predictions iteratively.

## Data and Features
The dataset contains multiple features regarding the vehicle’s condition and history:
- Vehicle Model
- Mileage
- Service History
- Accident History
- Vehicle Age
- Fuel Type
- Engine Size
- Transmission Type
- Odometer Reading
- Last Service Date
- Warranty Expiry Date
- Tire Condition
- Battery Status
- Brake Condition
- Insurance Premium
- Fuel Efficiency
- **Target Variable**: `Need Maintenance` (binary: 1 for maintenance needed, 0 for no maintenance)

## Conclusion
- **Best Performing Models**: Both Random Forest and XGBoost achieved perfect scores across all metrics, making them the best models for this problem.
- **Model Comparison**: Logistic Regression was a strong performer with 94% accuracy, while KNN struggled, showing a higher recall for class 1 but poor classification for class 0.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Load the dataset:
   ```python
   import pandas as pd
   data = pd.read_csv('vehicle_maintenance_data.csv')
   ```
3. Train and evaluate models using Jupyter Notebooks or Python scripts.

## Future Improvements
- **Data Preprocessing**: Refining feature engineering, handling missing data, and scaling features could boost performance further.
- **Exploring Other Models**: More complex algorithms such as Support Vector Machines (SVM) or neural networks could be tested for improved results.
- **Hyperparameter Tuning**: Fine-tuning the Random Forest and XGBoost models could further optimize accuracy.

---

