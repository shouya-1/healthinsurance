#!/usr/bin/env python
# coding: utf-8

# In[99]:


#Python Libraries
import pandas as pd #Data Processing and CSV file I/o
import numpy as np #for numeric operations
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

#spliting and scaling the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

#metric


# ## Collecting Data

# In[100]:


insurance_df = pd.read_csv('insurance.csv')


# In[101]:


#changing categorical values into bianry
gender = {'male': 0,'female': 1}
smoker = {'yes':1, 'no':0}
loc ={'northwest': 0, 'southwest':1, 'southeast':2, 'northeast':3}
insurance_df['sex']= [gender[item] for item in insurance_df['sex']]
insurance_df['smoker'] = [smoker[item] for item in insurance_df['smoker']]
insurance_df['region']= [loc[item] for item in insurance_df['region']]

insurance_df.head(15)


# In[102]:


print(f"The Numbers of Rows and Columns in this data set are: {insurance_df.shape[0]} rows and {insurance_df.shape[1]} columns.")


# # Exploratory Data Analysis(EDA)

# In[103]:


insurance_df.info()


# In[144]:


#statistics summary
# insurance_df.describe().T


# In[104]:


#creating correlation matrix
corr=insurance_df.corr()


# In[105]:


#plotting the correlation matrix
plt.figure(figsize=(16,12))
ax = sns.heatmap(corr, annot=True, square=True, fmt='.3f', linecolor='black')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.title('Correlation Heatmap')
plt.show();


# In[106]:


corr_matrix = insurance_df.corr()
corr_matrix['charges'].sort_values(ascending=False)


# _Here we see that charges for smokers are higher than non-smokers. Sex,children,region do not affect the price much_

# In[107]:


#counting the missing values in numerical features
insurance_df.isnull().sum()


# # Feature Scaling

# In[108]:


# segregating the target variable
X = insurance_df.drop(columns='charges')
y = insurance_df['charges']
#splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[125]:


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[126]:


print(f"In X_train dataset there are: {X_train.shape[0]} rows and {X_train.shape[1]} columns.")
print(f"In X_test dataset there are: {X_test.shape[0]} rows and {X_test.shape[1]} columns.")
print(f"The shape of y_train is: {y_train.shape}")
print(f"The shape of y_test is: {y_test.shape}")


# In[127]:


models = [LinearRegression(),
          RandomForestRegressor(), AdaBoostRegressor(),
          Lasso(), Ridge()]
  
for i in range(5):
    models[i].fit(X_train, y_train)
  
    print(f'{models[i]} : ')
    pred_train = models[i].predict(X_train)
    print('Training Error : ', mape(y_train, pred_train))
  
    pred_test = models[i].predict(X_test)
    print('Validation Error : ', mape(y_test, pred_test))
    print()


# _RandomForestRegressor has the least error value_

# In[166]:


# Instantiate and fit the RandomForestClassifier
regressor = RandomForestRegressor(n_estimators=20, random_state=42)
regressor.fit(X_train, y_train)


# In[173]:


from sklearn.metrics import mean_squared_error
prediction = regressor.predict(X_test)
mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
print(mse)
print(rmse)


# In[179]:


#Correlation between predicted and actual results.
plt.scatter(y_test,prediction)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[177]:

st.header('Medical Insurance Predictor')
# Predict charges for new customer : Name- patientzero
name = st.text_input('Enter Your Name', 'Patient')
st.write('Fill in your basic information ', name)
age = st.sidebar.slider('How old are you?', 1,  85, 18)
sex = st.sidebar.radio(
    "What\'s your gender",
    ('Male', 'Female'))

if sex == 'Male':
    st.write('You selected male.')
    sex=0
else:
    st.write("You selected female.")
    sex=1
st.write("You're ", age, 'years old')

region = st.sidebar.radio(
    "What\'s your region",
    ('northwest', 'southwest', 'southeast', 'northeast'))
if region == 'northwest':
    region=0
if region == 'southwest':
    region=1
if region == 'southeast':
    region=2
if region == 'northeast':
    region=3

col1, buff, col2 = st.columns([2,1,2])
with col1:
    BMI = st.number_input("BMI:")
with col2:
    children = st.number_input('children:')
    
smoker = st.sidebar.radio(
    "Do you smoke?",
    ('Yes', 'No'))
if smoker == 'Yes':
    smoker=1
if smoker == 'No':
    smoker=0

data = {'age' : age,
        'sex' : sex,
        'bmi' : BMI,
        'children' : children,
        'smoker' : smoker,
        'region' : region}
index = [1]
patientzero_df = pd.DataFrame(data,index)

agree = st.checkbox('Click to see your data entry')

if agree:
    patientzero_df


# In[178]:


prediction_patientzero = regressor.predict(patientzero_df)
st.write(f"Medical Insurance cost for {name} is : ",prediction_patientzero)


# In[ ]:




