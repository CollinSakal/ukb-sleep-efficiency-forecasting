# Code for selecting features for the CatBoost models. Note that the selection procedure is for
#   the 8-hr window and all CatBoost models used features from that process.

# Imports
import random
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier

# Initialize stuff
random.seed(19970507)

# Initializing stuff
nfolds = 10
candidate_features = [
    'slp_eff_t1', 'slp_eff_t2', 'slp_eff_t3', 'slp_eff_t4', 'slp_eff_avg',
    'slp_eff_std', 'slp_onset_t1', 'slp_onset_t2', 'slp_onset_t3', 'slp_onset_t4', 'slp_onset_avg', 'slp_onset_std',
    'slp_offset_t1', 'slp_offset_t2', 'slp_offset_t3', 'slp_offset_t4', 'slp_offset_avg', 'slp_offset_std',
    'slp_dur_t1', 'slp_dur_t2', 'slp_dur_t3', 'slp_dur_t4', 'slp_dur_avg', 'slp_dur_std',
    'L5_act_t1', 'L5_act_t2', 'L5_act_t3', 'L5_act_t4', 'L5_act_avg', 'L5_act_std',
    'M10_act_t1', 'M10_act_t2', 'M10_act_t3', 'M10_act_t4', 'M10_act_avg', 'M10_act_std', 'day_t1', 'day_t2', 'day_t3',
    'day_t4', 'dof_weekday', 'dof_total_acc', 'dof_median_acc', 'dof_std_acc', 'dof_max_acc',
    'dof_min_acc', 'dof_q10_acc', 'dof_q25_acc', 'dof_q75_acc', 'dof_q90_acc', 'dof_skew_acc', 'dof_kurt_acc',
    'all_mean_acc', 'all_median_acc', 'all_std_acc', 'all_max_acc', 'all_min_acc', 'all_q10_acc', 'all_q25_acc',
    'all_q75_acc', 'all_q90_acc', 'all_skew_acc', 'all_kurt_acc'
]

params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'random_seed': 19970507,
    'learning_rate': 0.015,
    'iterations': 500,
    'depth': 4,
    'subsample': 0.80,
    'scale_pos_weight': 3.3
}

# Data (CC)
df = pl.read_csv('data/derived/df-final-8hrs.csv', null_values=['NA'], infer_schema_length=1500)
target = 'sleep_efficiency85'

# Train default model
x = df.select(candidate_features).to_numpy()
y = df.get_column(target).to_numpy()
pool = Pool(x, y, feature_names=candidate_features)

model = CatBoostClassifier(**params)
model.fit(pool, eval_set=pool)

# Get mean absolute shap values (returns objects x nfeatures +1, exclude last column)
importance_vals = model.get_feature_importance(pool, type='ShapValues')[:,:-1]
importance_vals = np.abs(importance_vals)
importance_vals = np.mean(importance_vals, axis=0)

df_importance = pl.DataFrame({
    'feature':candidate_features,
    'importance':importance_vals
}).sort(pl.col('importance'), descending=True)
df_importance.write_csv(f'data/derived/catboost-mean-shap-values.csv')

# Forward selection: 10-fold CV for default model adding one feature at a time
added_feature = []
auc_temp = []
auc_avg = []
auc_std = []

for feature in df_importance.get_column('feature').to_list():

    # Features that will be used to train the model
    added_feature.append(feature)

    for fold in range(1,nfolds):

        eids_train = pl.read_csv(f'results/repeat-cv/training-preds/train-preds-8hrs/train-preds-8hrs-fold{fold}.csv').get_column('eid')
        eids_valid = pl.read_csv(f'results/repeat-cv/validation-preds/valid-preds-8hrs/valid-preds-8hrs-fold{fold}.csv').get_column('eid')

        x_train = df.filter(pl.col('eid').is_in(eids_train)).select(added_feature).to_numpy()
        x_valid = df.filter(pl.col('eid').is_in(eids_valid)).select(added_feature).to_numpy()

        y_train = df.filter(pl.col('eid').is_in(eids_train)).get_column(target).to_numpy()
        y_valid = df.filter(pl.col('eid').is_in(eids_valid)).get_column(target).to_numpy()

        pool_train = Pool(x_train, y_train, feature_names=added_feature)
        pool_valid = Pool(x_valid, y_valid, feature_names=added_feature)

        cat = CatBoostClassifier(iterations=500,verbose=False)
        cat.fit(pool_train, eval_set=pool_valid)

        cat_preds_train = cat.predict_proba(pool_train)
        cat_preds_valid = cat.predict_proba(pool_valid)

        auc_temp.append(roc_auc_score(y_valid, cat_preds_valid[:,1]))

        if fold == nfolds-1:
            avg_auc_temp = np.mean(auc_temp)
            std_auc_temp = np.std(auc_temp)
            auc_avg.append(avg_auc_temp)
            auc_std.append(std_auc_temp)
            auc_temp = []

            print(f'Average AUC: {avg_auc_temp} .... std: {std_auc_temp}')

# Create data frame to save selection metrics
df_selection = pl.DataFrame({
    'added_feature':added_feature,
    'auc_avg':auc_avg,
    'auc_std':auc_std
})

df_selection.write_csv(f'data/derived/catboost-feature-selection.csv')
