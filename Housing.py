import hashlib
#import matplotlib.pyplot as plt
import numpy as np 
import os 
import pandas as pd
import tarfile

#from pandas.tools.plotting import scatter_matrix
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH="datasets/housing"
HOUSING_URL=DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

# ---- CLASS DEFINITIONS ----

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

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

# ---- FUNCTION DEFINITIONS ----

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

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

# ---- PROCESSING ----

print("Fetching and extracting source data")
fetch_housing_data()
housing = load_housing_data()

print("create test and training sets using derived column")
print("Create column for income categories")
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

print("Create test and trainings sets using new column")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print("Remove temporary column again from sets")
for set in (strat_test_set, strat_train_set):
    set.drop(["income_cat"], axis=1, inplace=True)

print("Make sure we are using only the training set for our models")
housing = strat_train_set.copy()

print("add new derived columns on training set 'housing'")
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# drop() creates a copy and does not affect the source set, so this works
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

print("Set up processing pipeline")
# Recap:
# - housing        -> stratified training set
# - housing_num    -> only numerical attributes / features
# - housing_labels -> real result values, used to calculate error for models
housing_num = housing.drop("ocean_proximity", axis=1)

# Custom column indexes, required for CombinedAttributesAdder etc.
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

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

print("Fit & transform pipeline to data")
housing_prepared = full_pipeline.fit_transform(housing)

# # ==== 9) Train different models and evaluate error
# 
# # === 9.1) Linear Regression
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
# 
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)  # NO fit()!
# print("Predictions:\t", lin_reg.predict(some_data_prepared))
# print("Labels:\t\t", list(some_labels))
# 
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# 
# print("Linear regression: ")
# display_scores(lin_rmse_scores)
# 
# # === 9.2) Decision Tree Regressor
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# 
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_mrse = np.sqrt(tree_mse)
# print("MRSE (tree): ", tree_mrse, " (obviously overfitting)")
# 
# tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-tree_scores)
# 
# print("Decision tree: ")
# display_scores(tree_rmse_scores)
# 
# # === 9.3) Random Forest Regressor
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# 
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# forest_mrse = np.sqrt(forest_mse)
# print("MRSE (forest): ", forest_mrse)
# 
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# 
# print("Forest scores: ")
# display_scores(forest_rmse_scores)


print("Using grid search to find best parameters for RandomForestRegressor")
param_grid = [
    { 'n_estimators' : [3, 10, 30], 'max_features' : [2, 4, 6, 8] },
    { 'bootstrap' : [False], 'n_estimators' : [3, 10], 'max_features' : [2, 3, 4] }
]

print("Parameters are ", param_grid)

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

print("Grid search, best params:\n--------------\n", grid_search.best_params_)
print("Grid search, best estimator:\n--------------\n", grid_search.best_estimator_)
print("Grid search, cross-evaluation scores:\n--------------\n")

print("Results for all")
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# feature_importances = grid_search.best_estimator_.feature_importances_
# extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_one_hot_attribs = list(encoder.classes_)
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# sorted(zip(feature_importances, attributes), reverse=True)


print("Use best parameters / model to verify against the test set")
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)  # DO NOT fit()!

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final RMSE against test set: ", final_rmse)

