# Income Inference

## Project Description

**Income Inference** is a project developed to infer the income of individuals based on various demographic and socioeconomic characteristics. Using machine learning techniques, this project aims to create a predictive model that can estimate whether or not a person's income exceeds a specific threshold (e.g., $50,000 annually). The project includes everything from data collection and processing, model selection and training, to performance evaluation.

## Project Structure

The project is structured as follows:

- **`datasets/`**: Contains the datasets used to train and test the models.
- **`notebooks/`**: Contains Jupyter notebooks demonstrating exploratory data analysis, data cleaning, and experiments with different models.
- **`src/`**: Source code for data processing, model building, and evaluation.
- **`models/`**: Stores trained models for later reuse or evaluation.
- **`README.md`**: This file provides details about the project.

## Technologies Used

- **Languages**: Python
- **Libraries**:
- `pandas` for data manipulation.
- `numpy` for mathematical calculations.
- `scikit-learn` for creating and evaluating machine learning models.
- `matplotlib` and `seaborn` for data visualization.
- **Tools**:
- Jupyter Notebooks for interactive development and exploratory data analysis.
- Git for version control.

## Implemented Models

The project uses ten different machine learning models, including:

1. Logistic Regression
2. Decision Trees
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)
6. SVM (Support Vector Machine)
7. Naive Bayes
8. Artificial Neural Networks (MLPClassifier)
9. XGBoost
10. LightGBM

Each model has been evaluated using performance metrics such as Accuracy, Precision, Recall, F1-Score, and the ROC curve to compare its effectiveness.
