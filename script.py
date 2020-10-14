# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[3]:


data = pd.read_csv("dataset.csv")
df = data.copy()
df.head()


# In[4]:


df.info()


# In[5]:


df["model"].value_counts()


# In[6]:


df["transmission"].value_counts()


# In[7]:


df[df["transmission"] == "Other"]


# In[8]:


# drop row which consist of "Other" to get better predict, because it has just single value

df = df.drop(index = df[df["transmission"] == "Other"].index)
df


# In[9]:


df["fuelType"].value_counts()


# In[10]:


df.describe().T


# In[11]:


df.corr()


# In[12]:


# price range for model names of car

(sns.FacetGrid(df, hue = "model", height = 8)
.map(sns.kdeplot, "price", shade = True)
    .add_legend());


# In[ ]:





# ## Data Preprocessing

# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[14]:


# save and select required values

price = df["price"]
df_columns = df.drop(["price"], axis = 1).select_dtypes(["int64", "float64"]).columns
numeric_data = df.drop(["price"], axis = 1).select_dtypes(["int64", "float64"])


# In[15]:


# scale data

scaled_data = scaler.fit_transform(numeric_data)
scaled_data


# In[16]:


df_new = df.drop(df.select_dtypes(["int64", "float64"]), axis = 1)
df_new


# In[17]:


# combine data with scaled data

df_new[df_columns] = scaled_data
df_new["price"] = price


# In[18]:


df_new


# In[19]:


df_new.info()


# In[20]:


# One Hot Encoder categorical variables

dummy_df = pd.get_dummies(df_new)
dummy_df


# In[21]:


# import statsmodels library to look through some statistic values

import statsmodels.api as sm


# In[22]:


X = dummy_df.drop("price", axis = 1)
y = dummy_df["price"]


# In[23]:


sm_model = sm.OLS(y, X).fit()


# In[24]:


sm_model.summary()


# In[ ]:





# ## Modelling

# In[25]:


from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42)


# In[27]:


X_train.shape


# In[28]:


X_test.shape


# In[ ]:





# In[29]:


def model_test(model_name): # test models with default parameters
    
    model = model_name().fit(X_train, y_train)
    print(model_name.__name__)
    print("Train MSE : ", np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
    print("Test MSE : ", np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
    print("Train R2 Score : ", model.score(X_train, y_train))
    print("Test R2 Score : ", model.score(X_test, y_test))


# In[30]:


def model_tuning(model,parameters): # seach for best parameters for models
    
    tuned_model = GridSearchCV(model, parameters, cv = 10, scoring = "neg_mean_squared_error").fit(X_train, y_train)
    return tuned_model.best_params_


# In[31]:


def model_optimization(model_name, params): # test models with hyperparameters and optimization
    
    model =  model_name().set_params(**params).fit(X_train, y_train)    
    print("Tuned Train MSE : ", np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
    print("Tuned Test MSE : ", np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
    print("Tuned Train R2 Score : ", model.score(X_train, y_train))
    print("Tuned Test R2 Score : ", model.score(X_test, y_test))
    print("--------------------------------")
    return model


# In[ ]:





# # Linear Regression

# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


model_test(LinearRegression)


# In[34]:


# validation score

lr_model = LinearRegression()
np.sqrt(-cross_val_score(lr_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error").mean())


# In[ ]:





# # Ridge Regression

# In[35]:


from sklearn.linear_model import Ridge


# In[36]:


model_test(Ridge)


# In[37]:


# validation score

from sklearn.linear_model import RidgeCV
lambdas = 10**np.linspace(10,-2,100)*0.5 
ridge_cv = RidgeCV(alphas = lambdas, 
                   scoring = "neg_mean_squared_error",
                   normalize = True)
ridge_cv.fit(X_train, y_train)
ridge_cv.alpha_


# In[38]:


ridge_tuned = Ridge(alpha = ridge_cv.alpha_, 
                   normalize = True).fit(X_train,y_train)


# In[39]:


np.sqrt(mean_squared_error(y_train, ridge_tuned.predict(X_train)))


# In[ ]:





# # KNN

# In[40]:


from sklearn.neighbors import KNeighborsRegressor


# In[41]:


model_test(KNeighborsRegressor)


# In[42]:


# model hyperparameter

knn_model = KNeighborsRegressor()
knn_params = {"n_neighbors" : range(2,30)}
best_parameters = model_tuning(knn_model, knn_params)
best_parameters


# In[43]:


# tuned model results

knn_tuned = model_optimization(KNeighborsRegressor, best_parameters)
knn_tuned


# In[ ]:





# # SVR

# In[44]:


from sklearn.svm import SVR


# In[45]:


model_test(SVR)


# In[46]:


svr_model = SVR(kernel = "linear")
svr_params = {"C" : [100,1000],
             "tol" : [0.1,0.01]}
best_parameters = model_tuning(svr_model, svr_params)
best_parameters


# In[47]:


svr_model = model_optimization(SVR, best_parameters)
svr_model


# In[ ]:





# # Decision Tree

# In[48]:


from sklearn.tree import DecisionTreeRegressor


# In[49]:


model_test(DecisionTreeRegressor)


# In[50]:


cart = DecisionTreeRegressor()
cart_params = {"min_samples_split": range(2,20),
               "min_samples_leaf" : range(1,10),
               "max_leaf_nodes": range(2,10)}


# In[51]:


best_parameters = model_tuning(cart, cart_params)
best_parameters


# In[52]:


cart_tuned = model_optimization(DecisionTreeRegressor, best_parameters)
cart_tuned


# In[ ]:





# # Random Forest

# In[53]:


from sklearn.ensemble import RandomForestRegressor


# In[54]:


model_test(RandomForestRegressor)


# In[56]:


rf_model = RandomForestRegressor()
rf_params = {"max_features": [10,20,30],
             "min_samples_split" : [2,4,10],
             "min_samples_leaf" : [1,2,4],
            'n_estimators' : [500, 1000]}
best_parameters = model_tuning(rf_model, rf_params)
best_parameters


# In[57]:


rf_tuned = model_optimization(RandomForestRegressor, best_parameters)
rf_tuned


# In[ ]:





# # XGBoost

# In[58]:


from xgboost import XGBRegressor


# In[59]:


model_test(XGBRegressor)


# In[60]:


xgb_model = XGBRegressor()
xgb_params = {'colsample_bytree': [0.4,0.5,0.6,0.9,1], 
     'n_estimators':[300, 500, 1000],
     'max_depth': [2,3,5,6],
     'learning_rate': [0.1, 0.01, 0.3]}
best_parameters = model_tuning(xgb_model, xgb_params)
best_parameters


# In[61]:


xgb_tuned = model_optimization(XGBRegressor, best_parameters)
xgb_tuned


# In[ ]:





# # LightGBM

# In[62]:


from lightgbm import LGBMRegressor


# In[63]:


model_test(LGBMRegressor)


# In[64]:


lgb_model = LGBMRegressor()
lgb_params = {
    'colsample_bytree': [0.4, 0.5,0.6,0.9,1],
    'learning_rate': [0.01, 0.1, 0.2, 1],
    'n_estimators': [300, 500,1000],
    'max_depth': [3,5,10,20]}

best_parameters = model_tuning(lgb_model, lgb_params)
best_parameters


# In[65]:


lgb_tuned = model_optimization(LGBMRegressor, best_parameters)
lgb_tuned


# In[ ]:





# # Gradient Boosting

# In[66]:


from sklearn.ensemble import GradientBoostingRegressor


# In[67]:


model_test(GradientBoostingRegressor)


# In[68]:


gbm_model = GradientBoostingRegressor()
gbm_params = {'learning_rate': [0.001, 0.1, 0.01],
    'max_depth': [3,8,20],
    'n_estimators': [300, 500, 1000],
    'subsample': [1,0.5,0.75]}
best_parameters = model_tuning(gbm_model, gbm_params)
best_parameters


# In[69]:


gbm_tuned = model_optimization(GradientBoostingRegressor, best_parameters)
gbm_tuned


# In[ ]:





# In[ ]:





# Better performance is from XGBoost model

# ## Model Evaluate

# In[71]:


col_names = X_train.columns
col_names


# In[72]:


xgb_tuned.feature_importances_


# In[73]:


# feature importance data

importance = pd.Series(data = xgb_tuned.feature_importances_, index = col_names)
importance *= 100
importance


# In[74]:


# plotting

plt.figure(figsize = (10,8))

ax = importance.sort_values().plot(kind = "barh", color = "r");

ax.yaxis.set_tick_params(labelsize=15)

plt.title("Feature importance rank", fontsize = 15, fontweight = "bold", color = "b");


# In[ ]:





# In[75]:


y_pred = xgb_tuned.predict(X_test)
y_pred


# In[76]:


y_test.head()


# In[77]:


# plotting actual and predicted values as line plot

fig, ax = plt.subplots(figsize = (12,10))

ax = plt.plot(y_pred[:50], label = "Actual", color = "r")
ax = plt.plot(y_test.values[:50], label = "Predicted", color = "g")

legend_properties = {"weight":"bold", "size" : 15}
plt.legend(prop = legend_properties);

plt.title("M")


# In[ ]:





# In[78]:


# scatter plot between y_test and y_pred data

plt.figure(figsize = (10,8))
plt.scatter(y_test, y_pred, alpha=0.2)
plt.xlabel("Actual",size=20, color = "brown", fontweight = "bold")
plt.ylabel("Predicted",size=20, color = "brown", fontweight = "bold")
plt.show()


