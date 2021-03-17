# Predicting-Customer-Churn_Neural-Networks_Deep Learning
AI for Decision Modelling_Project which involves to develop a deep learning based engine that predicts customers who are likely to churn.

PROJECT : PREDICTING CUSTOMER CHURN FOR A TELECOM COMPANY

PROBLEM TYPE: CLASSIFICATION

Domain: Telecommunication

(I)Problem Statement :
The major problem of Telecom companies is customer churn (loss of a customer to a competitor). The acquisition of a new customer is very expensive compared to retention of existing customers. Small percentage of reduction in churn improves huge margins to Telecom companies. The companies perform various target marketing activities or reward customer through offers and incentives, to retain a customer if he is identified before he churns. The current challenge, is to develop a deep learning based engine that predicts
customers who are likely to churn.

Metric: Accuracy


(II) Data Description :
Data presented contains attributes related to users’ information, the usage plans, charges billed, payment method etc, and the target column of interest is if the user has churned out or not. The task is to build a predictive model that can predict user ratings with reasonably good accuracy and sensitivity.

 (III) Approach/Strategy :
i. Loaded the data and understood it; You will observe that its predictors belong to three different types, numeric/integer, categorical.

ii. Exploratory analysis to analyse the data.

iii. Did necessary type conversions.

iv. Columns like CustomerID can be removed from the analysis.

v. Split the data into train and validation sets and performing preprocessing appropriately on each of them.
• Dealt with missing values
• On numeric data : applied  standardisation technique, preferably using standard scaler.
• On categorical data: Applied one-hot encoding/label encoding as appropriate.

vi. Built deep neural net model, compiled and fitted the model. Tuned it to improve validation accuracy/recall. Observed the performance

vii. Using auto encoders, got deep features for the same input, and using the deep features, built and tuned it to a good model and observed the performance

viii.Also, as there is class imbalance in the data, and recall, being an important metric for this problem is highly effected by the imbalance,
tried to work on mitigating the effect of class imbalance. Explored parameters like class weight while fitting the model, and analysed the performance.
