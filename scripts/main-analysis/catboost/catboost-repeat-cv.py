# Imports
import random
import polars as pl
from catboost import Pool, CatBoostClassifier

# Initializing stuff
dir_train = 'results/repeat-cv/training-preds'
dir_valid = 'results/repeat-cv/validation-preds'
dir_valid_eids = 'results/repeat-cv/validation-preds/valid-preds-8hrs'

random.seed(19970507)
nfolds = 10
nrepeats = 10

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
y = 'sleep_efficiency85'

# Data frames
df_8hrs = pl.read_csv('data/derived/df-final-8hrs.csv', null_values=['NA'], infer_schema_length=10000)
df_4hrs = pl.read_csv('data/derived/df-final-4hrs.csv', null_values=['NA'], infer_schema_length=10000)

# Repeat cv
for fold in range(1,nfolds*nrepeats+1):

    valid_eids = pl.read_csv(f'{dir_valid_eids}/valid-preds-8hrs-fold{fold}.csv').get_column('eid').to_list() #EIDS same for all tasks

    X_train_8hrs = df_8hrs.filter(~pl.col('eid').is_in(valid_eids)).select(X_features).to_numpy()
    X_train_4hrs = df_4hrs.filter(~pl.col('eid').is_in(valid_eids)).select(X_features).to_numpy()

    X_valid_8hrs = df_8hrs.filter(pl.col('eid').is_in(valid_eids)).select(X_features).to_numpy()
    X_valid_4hrs = df_4hrs.filter(pl.col('eid').is_in(valid_eids)).select(X_features).to_numpy()

    y_train = df_8hrs.filter(~pl.col('eid').is_in(valid_eids)).get_column(y).to_numpy() # y's same for all tasks
    y_valid = df_8hrs.filter(pl.col('eid').is_in(valid_eids)).get_column(y).to_numpy()  # y's same for all tasks

    pool_train_8hrs = Pool(X_train_8hrs, y_train)
    pool_train_4hrs = Pool(X_train_4hrs, y_train)

    pool_valid_8hrs = Pool(X_valid_8hrs, y_valid)
    pool_valid_4hrs = Pool(X_valid_4hrs, y_valid)

    model_8hrs = CatBoostClassifier(**params_8hrs)
    model_4hrs = CatBoostClassifier(**params_4hrs)

    model_8hrs.fit(pool_train_8hrs)
    model_4hrs.fit(pool_train_4hrs)

    preds_valid_8hrs = model_8hrs.predict_proba(pool_valid_8hrs)[:,1]
    preds_valid_4hrs = model_4hrs.predict_proba(pool_valid_4hrs)[:,1]

    preds_valid_8hrs = pl.Series(name='catboost_preds', values=preds_valid_8hrs)
    preds_valid_4hrs = pl.Series(name='catboost_preds', values=preds_valid_4hrs)

    path_valid_8hrs = f'{dir_valid}/valid-preds-8hrs/valid-preds-8hrs-fold{fold}.csv'
    path_valid_4hrs = f'{dir_valid}/valid-preds-4hrs/valid-preds-4hrs-fold{fold}.csv'

    pl.read_csv(path_valid_8hrs).with_columns(preds_valid_8hrs.alias('catboost_pred')).write_csv(path_valid_8hrs)
    pl.read_csv(path_valid_4hrs).with_columns(preds_valid_4hrs.alias('catboost_pred')).write_csv(path_valid_4hrs)

    print(f'{fold} of {nfolds*nrepeats} files processed')