
# coding: utf-8

# In[2]:

import os 
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH="datasets/housing"
HOUSING_URL=DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    print("Downloaded file.")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    print("Extracted archive.")
    housing_tgz.close()
    print("Done.")

fetch_housing_data()


# In[3]:

import pandas as pd

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()


# In[4]:

housing.info()


# In[5]:

housing["ocean_proximity"].head()


# In[6]:

housing["ocean_proximity"].value_counts()


# In[7]:

housing.describe()


# In[8]:

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[9]:

import hashlib
import numpy as np # erratum, was missing

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

# Using this approach, we need to make sure that new data
# gets added at the end of the dataset!
housing_with_id = housing.reset_index()    # add id column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

print("Train set:")
train_set.info()

print("Test set:")
test_set.info()


# In[10]:

# Create income categories to be able to use stratified sampling 
# across data (avoid sampling bias).
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

housing["income_cat"].hist(bins=5, figsize=(8,8))
plt.show()


# In[11]:

# Create stratified test / train set split using income categories
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
housing["income_cat"].value_counts() / len(housing)


# In[12]:

# Remove temporary income category from sets
for set in (strat_test_set, strat_train_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# In[13]:

# Create working set and visualize
get_ipython().magic('matplotlib inline')
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(12,10))


# In[14]:

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
            s=housing["population"]/100, label="population",
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            figsize=(12,10))
plt.legend()


# In[15]:

# Check correlation between columns
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[16]:

# Create scatter matrix to visually check for correlations
from pandas.tools.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[17]:

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, figsize=(8,6))


# In[18]:

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[19]:

# reset data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Now handle n/a values using one of the following:

#housing.dropna(subset=["total_bedrooms"]) # 1) get rid of districts
#housing.drop(["total_bedrooms"], axis=2)  # 2) get rid of whole attribute
#median = housing["total_bedrooms"].median()
#housing["total_bedrooms"].fillna(median)  # 3) fill with median value

# This is 3) using sklearn's library
from sklearn.preprocessing import Imputer

housing_num = housing.drop("ocean_proximity", axis=1)
imputer = Imputer(strategy="median")
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.info()


# In[20]:

# Use one-hot encoding got near ocean attribute
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()

housing_cat = housing["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat)


# In[21]:

# Custom class for adding further attributes
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, population_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs


# In[23]:

# Creating a full pipeline with all transformations so far

# 1) Helper class to extract only given attributes
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

# 2) Define pipeline with interim steps
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attribs)),  # select only numeric attributes
    ("imputer", Imputer(strategy="median")),       # fill n/a values with median
    ("attribs_added", CombinedAttributesAdder()),  # add derived attributes
    ("std_scaler", StandardScaler()),              # scale values to [-1,1]
])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attribs)),  # select only category attributes
    ("label_binarizer", LabelBinarizer()),         # convert to 0/1 flags in columns
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape


# In[ ]:



