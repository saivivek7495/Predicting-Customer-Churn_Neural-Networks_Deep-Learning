# Predicting-Customer-Churn_Neural-Networks_Deep Learning
AI for Decision Modelling_Project which involves to develop a deep learning based engine that predicts customers who are likely to churn.

PROJECT : PREDICTING CUSTOMER CHURN FOR A TELECOM COMPANY

PROBLEM TYPE: CLASSIFICATION

Domain: Telecommunication

Problem Statement
The major problem of Telecom companies is customer churn (loss of a customer to a competitor). The acquisition of a new customer is very expensive compared to retention of existing customers. Small percentage of reduction in churn improves huge margins to Telecom companies. The companies perform various target marketing activities or reward customer through offers and incentives, to retain a customer if he is identified before he churns. The current challenge, is to develop a deep learning based engine that predicts
customers who are likely to churn.

Metric: Accuracy


Data Description
Data presented contains attributes related to users’ information, the
usage plans, charges billed, payment method etc, and the target column of interest is if the user has churned out or not. The task is to build a predictive model that can predict user ratings with reasonably good accuracy and sensitivity.

 Approach/Strategy -
i. Load the data and understand it; You will observe that its predictors
belong to three different types, numeric/integer, categorical.
ii. Exploratory analysis to analyse the data.
iii. Do necessary type conversions
iv. Columns like CustomerID can be removed from the analysis
v. Split the data into train and validation sets and performing preprocessing appropriately on each of them.
• Deal with missing values if any
• On numeric data : apply a standardisation technique, preferably
using standard scaler
• On categorical data: Apply one-hot encoding/label encoding as
appropriate
vi. Build deep neural net model, compile and fit the model. Tune it to
improve validation accuracy/recall. Observe the performance
vii. Using auto encoders, get deep features for the same input, and using
the deep features, build and tune to a good model and observe the
performance
viii.Also, as there is class imbalance in the data, and recall, being an important metric for this problem is highly effected by the imbalance,
try to work on mitigating the effect of class imbalance. Explore parameters like class weight while fitting the model, and analyse the
performan
