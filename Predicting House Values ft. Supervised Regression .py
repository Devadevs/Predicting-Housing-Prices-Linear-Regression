#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): 
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()


# In[4]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH): 
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path)


# In[5]:


fetch_housing_data()


# In[6]:


housing = load_housing_data()
housing.head()


# In[7]:


housing.info()


# In[8]:


housing["ocean_proximity"].value_counts()


# In[9]:


housing.describe()


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
housing.hist(bins=50, figsize=(20, 15))
plt.show()


# In[11]:


import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[12]:


train_set, test_set = split_train_test(housing, 0.2)


# In[15]:


len(train_set)


# In[16]:


len(test_set)


# In[17]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[18]:


housing['income_cat'] = pd.cut(housing['median_income'],
                               bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[19]:


housing['income_cat'].hist()


# In[21]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    


# In[22]:


strat_test_set['income_cat'].value_counts() / len(strat_test_set)


# In[23]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
    


# In[24]:


housing = strat_train_set.copy()


# In[27]:


housing.plot(kind='scatter', x='longitude', y='latitude')


# In[28]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)


# In[30]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', figsize=(10,7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
            )
plt.legend()


# In[32]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[34]:


from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms',
             'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[35]:


housing.plot(kind='scatter', x='median_income', y='median_house_value',
             alpha=0.1)


# In[36]:


housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


# In[37]:


housing.dropna(subset=['total_bedrooms'])


# In[38]:


from sklearn.impute import SimpleImputer


# In[39]:


imputer = SimpleImputer(strategy='median')


# In[40]:


housing_num = housing.drop('ocean_proximity', axis=1)


# In[41]:


imputer.fit(housing_num)


# In[42]:


imputer.statistics_


# In[43]:


housing_num.median().values


# In[44]:


X = imputer.transform(housing_num)


# In[45]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()


# In[48]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[49]:


ordinal_encoder.categories_


# In[50]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[51]:


housing_cat_1hot.toarray()


# In[54]:


from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room 
    def fit(self, X, y=None):
        return self # nothing else to do 
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix] 
        population_per_household = X[:, population_ix] / X[:, households_ix] 
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[57]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[59]:


from sklearn.compose import ColumnTransformer 
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
             ("num", num_pipeline, num_attribs),
             ("cat", OneHotEncoder(), cat_attribs),
         ])
housing_prepared = full_pipeline.fit_transform(housing)


# In[60]:


# Training the Model 


# In[61]:


from sklearn.linear_model import LinearRegression


# In[62]:


lin_reg = LinearRegression()


# In[63]:


lin_reg.fit(housing_prepared, housing_labels)


# In[64]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)


# In[65]:


print("Predictions:", lin_reg.predict(some_data_prepared))


# In[66]:


print("Labels", list(some_labels))


# In[67]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[69]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

lin_rmse
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

