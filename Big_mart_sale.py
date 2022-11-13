#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Read the data
train_data=pd.read_csv(r"C:/Users/Dell/Downloads/train_v9rqX0R.csv")


# In[3]:


#Displays the first five rows of the dataframe by default
train_data.head()


# In[4]:


#Read the data
test_data=pd.read_csv(r"C:/Users/Dell/Downloads/test_AbJTz2l.csv")


# In[5]:


#Displays the first five rows of the dataframe by default
test_data.head()


# In[6]:


#Check the diamension of dataset
train_data.shape


# In[7]:


#Check the diamension of dataset
test_data.shape


# In[8]:


#Prints information about the DataFrame
train_data.info()


# # Data Pre-Processing

# In[9]:


#Prints information about the DataFrame
test_data.info()


# In[10]:


#Returns the number of missing values in each column
train_data.isna().sum()


# In[11]:


#Returns the number of missing values in each column
test_data.isna().sum()


# In[12]:


#Fill the null values
train_data['Item_Weight']=train_data['Item_Weight'].fillna(train_data['Item_Weight'].median())


# In[13]:


#Check the mode of column 'Outlet_Size'
train_data['Outlet_Size'].mode()


# In[14]:


#Fill the null values
train_data['Outlet_Size']=train_data['Outlet_Size'].fillna(train_data['Outlet_Size'].mode()[0])


# In[15]:


#Fill the null values
test_data['Item_Weight']=test_data['Item_Weight'].fillna(test_data['Item_Weight'].median())


# In[16]:


#Check the mode of column 'Outlet_Size'
test_data['Outlet_Size'].mode()


# In[17]:


#Fill the null values
test_data['Outlet_Size']=test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode()[0])


# In[18]:


#checking again for null values
train_data.isna().sum()


# In[19]:


#checking again for null values
test_data.isna().sum()


# # Exploratory Data Analysis using seabon

# In[20]:


#distribution plot of Item_Weight
plt.figure(figsize=(8,6))
sns.distplot(test_data['Item_Weight'])
plt.show()


# In[21]:


#distribution plot of Item_Visibility
plt.figure(figsize=(8,6))
sns.distplot(test_data['Item_Visibility'])
plt.show()


# In[22]:


#distribution plot of Item_MRP
plt.figure(figsize=(8,6))
sns.distplot(test_data['Item_MRP'])
plt.show()


# In[23]:


#count plot of Outlet_Establishment_Year
plt.figure(figsize=(8,6))
sns.countplot(test_data['Outlet_Establishment_Year'])
plt.show()


# In[24]:


#count plot of Item_Fat_Content
plt.figure(figsize=(8,6))
sns.countplot(test_data['Item_Fat_Content'])
plt.show()


# In[25]:


#count plot of Item_Type
plt.figure(figsize=(10,6))
sns.countplot(test_data['Item_Type'])
plt.xticks(rotation=90)
plt.show()


# In[26]:


#count plot of Outlet_Identifier
plt.figure(figsize=(10,6))
sns.countplot(test_data['Outlet_Identifier'])
plt.xticks(rotation=90)
plt.show()


# In[27]:


#count plot of Outlet_Size
plt.figure(figsize=(8,6))
sns.countplot(test_data['Outlet_Size'])
plt.xticks(rotation=90)
plt.show()


# In[28]:


#count plot of Outlet_Location_Type
plt.figure(figsize=(8,6))
sns.countplot(test_data['Outlet_Location_Type'])
plt.xticks(rotation=90)
plt.show()


# In[29]:


#count plot of Outlet_Type
plt.figure(figsize=(8,6))
sns.countplot(test_data['Outlet_Type'])
plt.show()


# In[30]:


#finding the unique values
train_data['Item_Identifier'].nunique()


# In[31]:


#finding the unique values
train_data['Item_Fat_Content'].nunique()


# In[32]:


#finding the unique values
train_data['Item_Type'].nunique()


# In[33]:


#finding the unique values
train_data['Outlet_Identifier'].nunique()


# In[34]:


#finding the unique values
train_data['Outlet_Size'].nunique()


# In[35]:


#finding the unique values
train_data['Outlet_Location_Type'].nunique()


# In[36]:


#finding the unique values
train_data['Outlet_Type'].nunique()


# In[37]:


#drop the two columns 
train_data1=train_data.drop(['Item_Identifier','Outlet_Identifier'],axis=1)
test_data1=test_data.drop(['Item_Identifier','Outlet_Identifier'],axis=1)


# In[38]:


#display train_data1
train_data1


# In[39]:


#display test_data1
test_data1


# # Label Encoding

# In[40]:


#import library
from sklearn.preprocessing import LabelEncoder
label_encode=LabelEncoder()


# In[41]:


train_data1=train_data1.apply(label_encode.fit_transform)


# In[42]:


#display label encoded data
train_data1


# In[43]:


test_data1=test_data1.apply(label_encode.fit_transform)


# In[44]:


#display label encoded data
test_data1


# In[45]:


# Splitting our data into train and test
X=train_data1.drop('Item_Outlet_Sales',axis=1)
y=train_data1['Item_Outlet_Sales']


# In[46]:


#import library
from sklearn.model_selection import train_test_split
#split the data
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


# # Decision Tree Regression

# In[47]:


#import library
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)


# In[48]:


#import library
from sklearn.metrics import mean_squared_error,r2_score


# In[49]:


#print R2 value MSE and RMSE
print("R squared value :", r2_score(y_test,y_pred_dt))
print("Mean squared error :", mean_squared_error(y_test,y_pred_dt))
print("Root Mean Square Error :",np.sqrt(mean_squared_error(y_test,y_pred_dt)))


# # Random Forest Regression

# In[50]:


#import library
from sklearn.ensemble import RandomForestRegressor


# In[51]:


rfr=RandomForestRegressor()


# In[52]:


rfr.fit(X_train,y_train)
y_pred_rfr=rfr.predict(X_test)


# In[53]:


#print R2 value MSE and RMSE
print("R squared value :", r2_score(y_test,y_pred_rfr))
print("Mean squared value :", mean_squared_error(y_test,y_pred_rfr))
print("Root Mean Square Error :",np.sqrt(mean_squared_error(y_test,y_pred_rfr)))


# # Linear Regression

# In[54]:


#import library
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model1=lr.fit(X_train,y_train)
linear_pred=model1.predict(X_test)


# In[55]:


#print R2 value MSE and RMSE
print("R squared value :", r2_score(y_test,linear_pred))
print("Mean squared error :", mean_squared_error(y_test,linear_pred))
print("Root Mean Square Error :",np.sqrt(mean_squared_error(y_test,linear_pred)))


# # Lasso Regreesion

# In[56]:


#import library
from sklearn.linear_model import Lasso


# In[57]:


lasso_reg=Lasso(alpha=1)
model2=lasso_reg.fit(X_train,y_train)
lasso_pred=model2.predict(X_test)


# In[58]:


#print R2 value MSE and RMSE
print("R squared value :", r2_score(y_test,lasso_pred))
print("Mean squared error :", mean_squared_error(y_test,lasso_pred))
print("Root Mean Square Error :",np.sqrt(mean_squared_error(y_test,lasso_pred)))


# # Gradient Boosting Regression

# In[59]:


#import library
from sklearn.ensemble import GradientBoostingRegressor


# In[60]:


gbr=GradientBoostingRegressor(max_depth=2,n_estimators=100,learning_rate=0.2)
gbr.fit(X_train,y_train)
gradient_pred=gbr.predict(X_test)


# In[61]:


#print R2 value MSE and RMSE
print("R squared value :", r2_score(y_test,gradient_pred))
print("Mean squared error :", mean_squared_error(y_test,gradient_pred))
print("Root Mean Square Error :",np.sqrt(mean_squared_error(y_test,gradient_pred)))


# Gradient boosting regression model gets better R2 value.But we take decision tree regression to predict test data.Other models shows an error "Negative values in 'Item_Outlet_Sales' column is not allowed.

# In[62]:


output_data=test_data[['Item_Identifier','Outlet_Identifier']]


# In[63]:


output_data


# In[64]:


#predicting the values for test data
predicted_value=dt.predict(test_data1)


# In[65]:


predicted_value


# In[66]:


output_data['Item_Outlet_Sales']=predicted_value


# In[67]:


output_data


# In[68]:


output_data.to_csv("D:\\test_file.csv", index = False)

