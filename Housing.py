#!/usr/bin/env python
# coding: utf-8

# 1.加载数据集

# In[ ]:


import os
import tarfile
import numpy as np
import pandas as pd
from six.moves import urllib
path = "D:\Desktop\机器学习平台\实验3\housing.csv"
housing = pd.read_csv(path)


# 快速查看数据结构

# In[2]:


#显示数据集前五行
housing.head()


# In[3]:


#使用 info() 获取数据集的简单描述。包括总行数、每个属性的类型和非空值的数量
housing.info()


# In[4]:


#使用 describe() 获取数值属性的描述，注意统计时的空值会被忽略。
housing.describe()


# In[5]:


#绘制每个数值的直方图。直方图横轴表示数值范围，纵轴表示实例数量。
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# 2. 划分测试集

# In[6]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[7]:


import hashlib
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[8]:


#使用索引行作为ID
housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[9]:


#将纬度、经度结合成一个ID
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[10]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[11]:


#收入数据分层
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[12]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[13]:


#查看收入分类比例：
housing["income_cat"].value_counts() / len(housing)


# In[14]:


#删除 income_cat 属性，使数据回到初始状态
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# 3.数据探索和可视化

# In[15]:


housing = strat_train_set.copy()


# In[16]:


#地理可视化
housing.plot(kind="scatter", x="longitude", y="latitude")


# In[17]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[18]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[19]:


#直接对信息进行归一化、标准化或机器学习
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
knn =  KNeighborsClassifier()
housing = pd.get_dummies(housing)
#再进行查找关联
corr_matrix = housing.corr()


# In[20]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[21]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[22]:


#锁定收入中位数
housing.plot(kind="scatter", x="median_income",y="median_house_value",alpha=0.1)


# In[23]:


#属性组合试验
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[24]:


corr_matrix = housing.corr()


# In[25]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[26]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# 数据清洗

# In[27]:


median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median) 


# In[28]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[29]:


housing_num = housing.drop("ocean_proximity", axis=1)


# In[30]:


#处理缺失值
imputer.fit(housing_num)


# In[31]:


imputer.statistics_


# In[32]:


housing_num.median().values


# In[33]:


X = imputer.transform(housing_num)


# In[34]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# 处理文本和类别属性

# In[35]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# In[36]:


print(encoder.classes_)


# In[37]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot


# In[38]:


housing_cat_1hot.toarray()


# In[39]:


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[40]:


#添加组合后的属性
from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix , bedrooms_ix, population_ix , households_ix =3,4,5,6
class  CombinedAttributesAdder(BaseEstimator, TransformerMixin ):
    def __init__ (self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X, y = None):
        return self
    def transform(self , X):
        rooms_per_household = X [: , rooms_ix] / X[:,households_ix]
        population_per_household = X[:,population_ix] / X [:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
            return np.c_[X,rooms_per_household , population_per_household,bedrooms_per_room ]
        else:
            return np.c_[X,rooms_per_household, population_per_household ]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room= False)
housing_extra_attribs= attr_adder.transform(housing.values)


# In[41]:


col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# In[42]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[43]:


#将所有转换应用到房屋数据
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num",num_pipeline , num_attribs ),
    ("cat" , OneHotEncoder(),cat_attribs ),
])
housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared
housing_prepared.shape


# In[44]:


#训练线性回归模型。
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

#训练集实例预测
some_data =housing.iloc[:5]
some_labels = housing_labels.iloc[: 5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:",lin_reg.predict(some_data_prepared))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse  = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[45]:


from sklearn.tree  import DecisionTreeRegressor      
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse 


# In[46]:


#交叉验证 过拟合
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,housing_prepared , housing_labels , 
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:",scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)      


# In[47]:


#随机森林
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[48]:


from sklearn.model_selection import GridSearchCV    

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_          
for mean_score ,params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)


# In[49]:


#随机搜索。
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

rnd_search.best_params_

#所有结果：
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[50]:


feature_importances = grid_search.best_estimator_.feature_importances_ 
feature_importances     


# In[51]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[52]:


final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test , final_predictions)
final_rmse =  np.sqrt(final_mse)  

final_rmse


# In[53]:


from scipy import stats    
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# In[54]:


full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)


# In[55]:


#保存训练好的模型
my_model = full_pipeline_with_predictor
import joblib
joblib.dump(my_model, "my_model.pkl") 
my_model_loaded = joblib.load("my_model.pkl") 


# In[56]:


#第一题
housing_labels_log = np.log(housing_labels)


# In[57]:


#交叉验证 过拟合
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,housing_prepared , housing_labels , 
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:",scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
#线性回归
display_scores(lin_rmse_scores)  


# In[58]:


#随机森林
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
#结果有改善！


# In[59]:


#第二题
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(housing_prepared, housing_labels)


# In[60]:


negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
grid_search.best_params_


# In[ ]:




