#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DEEP LEARNING_ANN_MODULE END PROJECT : PREDICTING CUSTOMER CHURN FOR A TELECOM COMPANY  

# The current challenge, is to develop a Deep learning based engine that predicts customers who are likely to churn.


# In[2]:


# Importing Neccessary Libraries

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc

import tensorflow as tf

from keras.regularizers import l2

from keras.models import Sequential # Sequential model is a linear stack of layers
from keras.layers import Dense, Dropout, BatchNormalization

from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import Callback
from keras import backend as K

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[3]:


#fix random seed for reproducibility 
#Seed function is used to save the state of a random function, so that it can generate same random numbers on multiple executions of the code

np.random.seed(123)

tf.set_random_seed(123)   #tf.random.set_seed(123)


# In[4]:


#Get current working directory

PATH = os.getcwd()


# In[5]:


# Set the Working directory
 
os.chdir(PATH)


# In[6]:


# Loading the data

Telcochurn = pd.read_csv("Telcochurn.csv",header=0)


# In[7]:


# Exploratory Data Analysis and Preprocessing

# 1) Identification of Varibales and Datatypes


# In[8]:


Telcochurn.dtypes


# In[9]:


print(Telcochurn.shape)


# In[10]:


#Display the columns

Telcochurn.columns


# In[11]:


#Display the index
Telcochurn.index


# In[12]:


Telcochurn.head()


# In[13]:


# Summary Statistics and Distribution of the Columns
Telcochurn.describe()


# In[14]:


Telcochurn.describe(include = 'object' )


# In[15]:


# 2) Non - Graphical Univariate Analysis
#Distribution of dependent variable

Telcochurn.Churn.value_counts()


# In[16]:


Telcochurn.Churn.value_counts(normalize = True)*100


# In[17]:


Telcochurn.SeniorCitizen.value_counts()


# In[18]:


Telcochurn[Telcochurn.Churn == 'Yes'].InternetService.value_counts(normalize = True)*100


# In[19]:


Telcochurn[Telcochurn.Churn == 'Yes'].Contract.value_counts(normalize = True)*100


# In[20]:


# People having Month-to-month Contract are about 88 % who are churning out.
 #Customer is impacted by the Monthly Charges you offer 


# In[21]:


Telcochurn[Telcochurn.Churn == 'Yes'].PaymentMethod.value_counts(normalize = True)*100


# In[22]:


Telcochurn[Telcochurn.Churn == 'Yes'].tenure.value_counts(normalize = True)*100


# In[23]:


Telcochurn.isnull().sum()


# In[24]:


#Convert all the attributes to appropriate type- Data type conversion


# In[25]:


Telcochurn['SeniorCitizen'] = Telcochurn['SeniorCitizen'].map({1:'yes', 0:'no'})


# In[26]:


Telcochurn.dtypes


# In[27]:


Telcochurn.TotalCharges


# In[28]:


Telcochurn.TotalCharges.values


# In[29]:


Telcochurn.MonthlyCharges.values


# In[30]:


# Convert pandas.Series from dtype object to float, and errors to nans/ invalid parsing will be set as na
pd.to_numeric(Telcochurn.TotalCharges, errors = 'coerce')


# In[31]:


pd.to_numeric(Telcochurn.TotalCharges, errors = 'coerce').isnull()


# In[32]:


Telcochurn[pd.to_numeric(Telcochurn.TotalCharges, errors = 'coerce').isnull()]


# In[33]:


Telcochurn = Telcochurn[Telcochurn.TotalCharges != ' ']


# In[34]:


pd.to_numeric(Telcochurn.TotalCharges)


# In[35]:


Telcochurn['TotalCharges'] = pd.to_numeric(Telcochurn.TotalCharges)


# In[36]:


Telcochurn.dtypes


# In[37]:


#Telcochurn[Telcochurn['TotalCharges'].isnull()] # Wherever the value is set to True it will return those rows # Drop These rows


# In[38]:


Telcochurn.shape


# In[39]:


Telcochurn.iloc[488]


# In[40]:


Telcochurn.isnull().sum()


# In[41]:


Telcochurn.drop('customerID',axis = 1,inplace = True)


# In[42]:


Telcochurn.reset_index()


# In[43]:


#Telcochurn.dropna(inplace = True)


# In[44]:


Telcochurn.dtypes


# In[45]:


Telcochurn.nunique()


# In[46]:


for column in Telcochurn:
        print(f'{column} : {Telcochurn[column].unique()}')


# In[47]:


for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','Churn']:
    Telcochurn[col] = Telcochurn[col].astype('category')


# In[48]:


Telcochurn.dtypes


# In[49]:


#Telcochurn['Churn'] = Telcochurn['Churn'].map({'Yes': 1,'No':0})


# In[50]:


#Telcochurn['Churn'].value_counts()


# In[51]:


# YOUR CODE HERE
#Telco_X_train = Telcochurn_new.drop(['Churn'], axis = 1)
#Telco_y_train = Telcochurn_new["Churn"]


# In[52]:


#type(Telco_y_train)


# In[53]:


cat_columns = list(Telcochurn.select_dtypes('category').columns)
num_columns = list(Telcochurn.columns.difference(cat_columns))


# In[54]:


cat_columns


# In[55]:


num_columns


# In[56]:


# Convert categorical columns in dummies. Using the 'pd.get_dummies' method.
Telcochurn_cat_dummy = pd.get_dummies(columns = cat_columns,
                                      data = Telcochurn[cat_columns],
                                      prefix = cat_columns,
                                      prefix_sep = "_",
                                      drop_first = True)


# In[57]:


Telcochurn_cat_dummy.dtypes


# In[58]:


Telcochurn_cat_dummy.head()


# In[59]:


Telcochurn_cat_dummy.shape


# In[60]:


Telcochurn_cat_dummy.dtypes


# In[61]:


# YOUR CODE HERE
# Impute numerical columns

# Apply label encoder to each column with categorical data
 #num_imputer = SimpleImputer()
#imputed_X_train = pd.DataFrame(num_imputer.fit_transform(X_train[numerical_columns]),
                              # columns = numerical_columns)
#imputed_X_test  = pd.DataFrame(num_imputer.transform(X_test[numerical_columns]),
                               #columns = numerical_columns)


# In[62]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#Telcochurn_num = pd.DataFrame(scaler.fit_transform(Telcochurn[num_columns]),
                              #columns = num_columns)


# In[63]:


#scale = StandardScaler()


# In[64]:


#scale.fit(Telcochurn[num_columns])


# In[65]:


#Telcochurn_std = pd.DataFrame(scale.transform(Telcochurn[num_columns]),
                              #columns = num_columns)


# In[66]:


#Telcochurn_std.dtypes


# In[67]:


Telcochurn_num = Telcochurn[num_columns]


# In[68]:


Telcochurn_num.dtypes


# In[69]:


pd.concat([Telcochurn_num,Telcochurn_cat_dummy], axis=1)


# In[70]:


Final_Telcochurn = pd.concat([Telcochurn_cat_dummy,Telcochurn_num], axis = 1)


# In[71]:


Final_Telcochurn.shape


# In[72]:


Final_Telcochurn.dtypes


# In[73]:


Final_Telcochurn.isnull().sum().sum()


# In[74]:


Final_Telcochurn.dropna(axis = 0)


# In[75]:


Final_Telcochurn.isnull().sum().sum()


# In[76]:


# Train and Validation Split & performing preprocessing appropriately on each of them

X, y = Final_Telcochurn.loc[:,Final_Telcochurn.columns!='Churn_Yes'].values, Final_Telcochurn.loc[:,'Churn_Yes'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=123)


# In[77]:


print(X_train.shape)
print(X_valid.shape)


# In[78]:


print(pd.value_counts(y_train))
print(pd.value_counts(y_valid))


# In[79]:


print(pd.value_counts(y_train)/y_train.size * 100)
print(pd.value_counts(y_valid)/y_valid.size * 100)


# In[80]:


Final_Telcochurn.dtypes


# In[81]:


scale = StandardScaler()
scale.fit(X_train)
X_train_std = scale.transform(X_train)
X_valid_std = scale.transform(X_valid)


# In[82]:


np.random.seed(123)
tf.set_random_seed(123)


# In[83]:


from keras import models
from keras import layers


# In[84]:


Telco_NN = models.Sequential()
#
Telco_NN.add(layers.Dense(25, 
                            input_shape=(X_train_std.shape[1], ), 
                            activation='relu', 
                            kernel_initializer='glorot_normal'))
#
Telco_NN.add(layers.Dense(20, 
                            activation='relu', 
                            kernel_initializer='glorot_normal'))
#
Telco_NN.add(layers.Dense(1, 
                            activation='sigmoid', 
                            kernel_initializer='glorot_normal'))


# In[85]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[86]:


from keras import optimizers


# In[87]:


get_ipython().system('pip install PyPi')


# In[88]:


get_ipython().system('pip install helper')


# In[89]:


#from helper import accuracy_plot, loss_plot, compute_metrics


# In[90]:


Telco_NN.summary()


# In[91]:


Telco_NN.compile(loss='binary_crossentropy', 
                   optimizer='adam', 
                   metrics=['accuracy'])


# In[92]:


get_ipython().run_cell_magic('time', '', 'Telco_NN_Model = Telco_NN.fit(X_train_std, \n                                        y_train, \n                                        epochs=100,\n                                        batch_size= 32, \n                                        validation_split=0.2, \n                                        shuffle=True)')


# In[96]:


print(Telco_NN_Model.history.keys())


# In[97]:


accuracy_plot(Telco_NN_Model)


# In[98]:


loss_plot(Telco_NN_Model)


# In[99]:


Telco_NN_Model_Train__pred = Telco_NN.predict_classes(X_train_std)
                                               
Telco_NN_Model_Test_pred = Telco_NN.predict_classes(X_valid_std)


# In[100]:


import helper


# In[101]:


os.getcwd()


# In[102]:


#from helper.py import accuracy_plot, loss_plot, compute_metrics


# In[103]:


#!wget -c https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-to-pytorch/helper.py 
#import helper


# In[104]:


from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Function to compute the metrics
def compute_metrics(train_pred, y_train, test_pred, y_test):
    # Confusion matrix for train predictions
    confmat = confusion_matrix(y_train, train_pred)
    
    print('Train metrics')
    print('Confusion matrix')
    print(confmat)
    print("----------------------------------------------")    
    TP = confmat[0,0]
    TN = confmat[1,1]
    FN = confmat[0,1]
    FP = confmat[1,0]
    Total = TP + TN + FP + FN

    # Accuracy: Overall, how often is the classifier correct?
    Accuracy = (TP+TN)/Total 
    # Misclassification Rate: Overall, how often is it wrong?
    # equivalent to 1 minus Accuracy also known as "Error Rate"
    Misclassification_Rate = (FP+FN)/Total

    # True Positive Rate: When it's actually yes, how often does it predict yes?
    # also known as "Sensitivity" or "Recall"
    Actual_Yes = TP + FN
    Recall = TP/Actual_Yes

    # False Positive Rate: When it's actually no, how often does it predict yes?
    Actual_No = TN + FP
    FPR = FP/Actual_No

    # True Negative Rate: When it's actually no, how often does it predict no?
    # equivalent to 1 minus False Positive Rate, also known as "Specificity"
    TNR = TN/Actual_No

    # Precision: When it predicts yes, how often is it correct?
    Predicted_Yes = TP + FP
    Precission = TP/Predicted_Yes

    # Prevalence: How often does the yes condition actually occur in our sample?
    Prevalance = Actual_Yes / Total
    
    # F1 Score
    f1 = 2 * (Precission * Recall) / (Precission + Recall)

    print('Accuracy: ', Accuracy)
    print('Precission: ', Precission)
    print('Recall: ', Recall)
    print('F1 Score: ', f1)
    print("")
    print("==============================================")
    print("")
    # Confusion matrix for train predictions
    confmat = confusion_matrix(y_test, test_pred)
    print('Test metrics')
    print('Confusion matrix')
    print(confmat)
    print("----------------------------------------------")  
    TP = confmat[0,0]
    TN = confmat[1,1]
    FN = confmat[0,1]
    FP = confmat[1,0]
    Total = TP + TN + FP + FN

    # Accuracy: Overall, how often is the classifier correct?
    Accuracy = (TP+TN)/Total 
    # Misclassification Rate: Overall, how often is it wrong?
    # equivalent to 1 minus Accuracy also known as "Error Rate"
    Misclassification_Rate = (FP+FN)/Total

    # True Positive Rate: When it's actually yes, how often does it predict yes?
    # also known as "Sensitivity" or "Recall"
    Actual_Yes = TP + FN
    Recall = TP/Actual_Yes

    # False Positive Rate: When it's actually no, how often does it predict yes?
    Actual_No = TN + FP
    FPR = FP/Actual_No

    # True Negative Rate: When it's actually no, how often does it predict no?
    # equivalent to 1 minus False Positive Rate, also known as "Specificity"
    TNR = TN/Actual_No

    # Precision: When it predicts yes, how often is it correct?
    Predicted_Yes = TP + FP
    Precission = TP/Predicted_Yes

    # Prevalence: How often does the yes condition actually occur in our sample?
    Prevalance = Actual_Yes / Total
    
    # F1 Score
    f1 = 2 * (Precission * Recall) / (Precission + Recall)

    print('Accuracy: ', Accuracy)
    print('Precission: ', Precission)
    print('Recall: ', Recall)
    print('F1 Score: ', f1)

# Function to draw plot for the train and validation accuracies
def accuracy_plot(history):
    plt.clf() # Clears the figure
    history_dict = history.history
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, label='Training accuracy')
    plt.plot(epochs, val_acc_values, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Function to draw plot for the train and validation loss 
def loss_plot(history):
    plt.clf() # Clears the figure
    history_dict = history.history
    acc_values = history_dict['loss']
    val_acc_values = history_dict['val_loss']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, label='Training loss')
    plt.plot(epochs, val_acc_values, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# In[105]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[106]:


compute_metrics(Telco_NN_Model_Train__pred, y_train, Telco_NN_Model_Test_pred, y_valid)


# In[107]:


# Using auto encoders, get deep features for the same input, and using the deep features, build and tune to a good model
# and observe the performance

# Derive new non-linear features using autoencoder


# In[108]:


from keras.models import Sequential, Model
from keras.layers import Dense, Input


# In[109]:


# The size of encoded and actual representations
encoding_dim = 16   # this is the size of our encoded representations
actual_dim = X_train_std.shape[1]


# In[110]:


# Input placeholder
input_attrs = Input(shape=(actual_dim,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_attrs)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(actual_dim, activation='sigmoid')(encoded)


# In[111]:


# this model maps an input to its reconstruction
autoencoder = Model(input_attrs, decoded)


# In[112]:


print(autoencoder.summary())


# In[113]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[114]:


autoencoder.fit(X_train_std, X_train_std, epochs=100)


# In[115]:


# this model maps an input to its encoded representation
encoder = Model(input_attrs, encoded)


# In[116]:


print(encoder.summary())


# In[117]:


X_train_nonLinear_features = encoder.predict(X_train_std)
X_test_nonLinear_features = encoder.predict(X_valid_std)


# In[118]:


X_train_nonLinear_features[1:2,:]


# In[119]:


encoder.get_weights()


# In[123]:


# Combining new non-linear features to X_train and X_test respectively


# In[124]:


X_train = np.concatenate((X_train_std, X_train_nonLinear_features), axis=1)
X_test = np.concatenate((X_valid_std, X_test_nonLinear_features), axis=1)


# In[125]:


#Perceptron Model Building with both actual and non-linear features


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#type(Telcochurn_num)


# In[ ]:


#Telcochurn_num.head()


# In[ ]:


#cat_attr_label = list(Telcochurn_new[['InternetService', 'Contract', 'PaymentMethod']])


# In[ ]:


#cat_attr_label


# In[ ]:


#cat_attr_onehot = list(Telcochurn_new[['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']])


# In[ ]:


#cat_attr_onehot 


# In[ ]:




