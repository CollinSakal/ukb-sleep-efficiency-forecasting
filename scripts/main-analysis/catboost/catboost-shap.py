# Imports
import shap
import random
import numpy as np
import polars as pl
from catboost import Pool, CatBoostClassifier

# Initializing stuff
random.seed(19970507)

# Model params
df_params_8hrs = pl.read_csv('results/catboost-params/params-8hrs.csv')
df_params_4hrs = pl.read_csv('results/catboost-params/params-4hrs.csv')

params_8hrs = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'random_seed': 19970507,
    'learning_rate': df_params_8hrs[0,'learning_rate'],
    'iterations': df_params_8hrs[0,'iterations'],
    'depth': df_params_8hrs[0,'depth'],
    'subsample': df_params_8hrs[0,'subsample'],
    'scale_pos_weight':3.3
}
params_4hrs = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'random_seed': 19970507,
    'learning_rate': df_params_4hrs[0,'learning_rate'],
    'iterations': df_params_4hrs[0,'iterations'],
    'depth': df_params_4hrs[0,'depth'],
    'subsample': df_params_4hrs[0,'subsample'],
    'scale_pos_weight':3.3
}

# Model features
X_features = pl.read_csv('results/cat-features.csv').get_column('feature_names').to_list()

# Data frames
df_8hrs = pl.read_csv('data/derived/df-final-8hrs.csv', null_values=['NA'], infer_schema_length=10000)
df_4hrs = pl.read_csv('data/derived/df-final-4hrs.csv', null_values=['NA'], infer_schema_length=10000)

X_8hrs = df_8hrs.select(X_features).to_numpy()
X_4hrs = df_4hrs.select(X_features).to_numpy()

y = df_8hrs.get_column('sleep_efficiency85').to_numpy()

pool_8hrs = Pool(X_8hrs, y, feature_names=X_features)
pool_4hrs = Pool(X_4hrs, y, feature_names=X_features)

# Models
model_8hrs = CatBoostClassifier(**params_8hrs)
model_4hrs = CatBoostClassifier(**params_4hrs)

model_8hrs.fit(pool_8hrs)
model_4hrs.fit(pool_4hrs)

preds_8hrs = model_8hrs.predict_proba(pool_8hrs)
preds_4hrs = model_4hrs.predict_proba(pool_4hrs)

# SHAP Values
explainer_8hrs = shap.TreeExplainer(model_8hrs, feature_names=X_features)
explainer_4hrs = shap.TreeExplainer(model_4hrs, feature_names=X_features)

shap_8hrs = explainer_8hrs.shap_values(pool_8hrs)
shap_4hrs = explainer_4hrs.shap_values(pool_4hrs)

# Plots
shap.plots.beeswarm(explainer_8hrs(X_8hrs), max_display=len(X_features))
shap.plots.beeswarm(explainer_4hrs(X_4hrs), max_display=len(X_features))

shap.plots.bar(explainer_8hrs(X_8hrs), max_display=len(X_features))
shap.plots.bar(explainer_4hrs(X_4hrs), max_display=len(X_features))
