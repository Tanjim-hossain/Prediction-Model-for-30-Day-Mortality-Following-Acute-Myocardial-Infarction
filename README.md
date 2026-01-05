# Prediction Model for 30-Day Mortality Following Acute Myocardial Infarction

## Overview
This project develops and validates a multivariable logistic regression model to predict 30-day mortality in patients admitted with Acute Myocardial Infarction (AMI). Using a dataset of 785 patients, the model identifies key risk factors to assist in early risk stratification and treatment optimization.

## Project Structure
- **AMI_Mortality_Prediction.ipynb**: The Python source code for data cleaning, exploratory data analysis (EDA), and model development.
- **ami_patient_data.csv**: The dataset containing 785 patient records with features such as Age, Killip Class, and ST Elevation.
- **Project_Report.pdf**: A detailed technical report covering the methodology, statistical analysis, and validation results adhering to TRIPOD standards.

## Key Findings
- **Model Performance**: The model achieved an **AUC of 0.758**, indicating good discrimination, and a **Brier Score of 0.169**, indicating good reliability.
- **Top Predictors**:
  1. ST Elevation
  2. Killip Class
  3. Age

## Methodology
- **Data Handling**: Addressed invalid codes and missing values using median/mode imputation.
- **Modeling**: Weighted Logistic Regression was used to handle class imbalance (6.6% mortality rate).
- **Validation**: Performance was assessed via Repeated Stratified 5-Fold Cross-Validation.

## Author
**Tanjim Hossain** Hasselt University  
*January 2026*
