from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def run_xgb(x_train, y_train, x_test):
    cv_folds = 5
    gs_metric = 'roc_auc'
    param_grid = {'max_depth': [5, 6, 7, 8],
                 'n_estimators': [200, 300],
                 'learning_rate': [0.3, 0.1, 0.05],
                 }
    
    est = xgb.XGBClassifier(verbosity=0, scale_pos_weight = (len(y_train) - sum(y_train))/sum(y_train), seed = 42,
                            tree_method='gpu_hist', gpu_id=0, eval_metric='logloss')
    
    gs = GridSearchCV(estimator = est, param_grid=param_grid, scoring=gs_metric, cv= cv_folds)
    gs.fit(x_train, y_train)

    y_pred_prob_train = gs.predict_proba(x_train)
    y_pred_train = gs.predict(x_train)

    y_pred_prob = gs.predict_proba(x_test)
    y_pred = gs.predict(x_test)

    return y_pred, y_pred_prob[:,1], y_pred_train, y_pred_prob_train[:,1]


 #classification scores

def get_scores_clf(y_true, y_pred, y_pred_prob):
    f1= metrics.f1_score(y_true, y_pred, average='macro')
    accu = metrics.accuracy_score(y_true, y_pred)
    accu_bl = metrics.balanced_accuracy_score(y_true, y_pred)
    auc =  metrics.roc_auc_score(y_true, y_pred_prob)
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    return auc, f1, accu, accu_bl, conf_matrix


def update_result(result, modality, model, auc, f1, accu, accu_bl ):
    result = result.append({
        'modality': modality,
        'model': model,
        'auc': auc,
        'f1': f1,
        'accuracy' :accu,
        'balanced_accuracy': accu_bl}, ignore_index=True)
    return result


def run_models(x_y, modality, model_method):
    pkl_list = df['haim_id'].unique().tolist()
    
    for seed in range(5):
        #train test split
        train_id, test_id = train_test_split(pkl_list, test_size=0.3, random_state=seed)
        #get the index for training and testing set
        train_idx = df[df['haim_id'].isin(train_id)]['haim_id'].tolist()
        test_idx = df[df['haim_id'].isin(test_id)]['haim_id'].tolist()
            
        model, auc, f1, accu, accu_bl, conf_matrix, auc_train, f1_train, accu_train, accu_bl_train, conf_matrix_train, y_pred_prob, y_pred_prob_train = run_single_model(x_y, train_idx, test_idx, model_method)
        
        result = pd.DataFrame(
        columns = ['Data Modality', 'Seed', 'Model', 'Train AUC', 'Train F1 Score', 'Train Accuracy', 'Train Balanced Accuracy', 
                   'Train Confusion Matrix',  'Test AUC', 'Test F1 Score', 'Test Accuracy', 'Test Balanced Accuracy', 
                   'Test Confusion Matrix'], 
        data = [[str(modality), seed, model_method.__name__, auc_train, f1_train, accu_train, accu_bl_train,
                 str(conf_matrix_train), auc, f1, accu, accu_bl, str(conf_matrix)]])  
        
        result.to_csv('mortality-result/{}-{}.csv'.format(ind, seed))
        pd.DataFrame(y_pred_prob).to_csv('mortality-result/y_pred_prob/{}-{}.csv'.format(ind, seed))
        pd.DataFrame(y_pred_prob_train).to_csv('mortality-result/y_pred_prob_train/{}-{}.csv'.format(ind, seed))


def run_single_model(x_y, train_idx, test_idx, model_method):
    x_y = x_y[~x_y.isna().any(axis=1)]
    #split train and test according to pkl list
    y_train = x_y[x_y['haim_id'].isin(train_idx)]['y']
    y_test = x_y[x_y['haim_id'].isin(test_idx)]['y']

    x_train = x_y[x_y['haim_id'].isin(train_idx)].drop(['y','haim_id'],axis=1)
    x_test = x_y[x_y['haim_id'].isin(test_idx)].drop(['y','haim_id'],axis=1)
    print('train, test shapes', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    print('train set, death outcome case = %s, percentage = %s' %(y_train.sum(),  y_train.sum()/len(y_train)))
    print('test set, death outcome case = %s, percentage = %s' %(y_test.sum(),  y_test.sum()/len(y_test)))
    
    y_pred, y_pred_prob, y_pred_train, y_pred_prob_train = model_method(x_train, y_train, x_test)
    
    auc, f1, accu, accu_bl, conf_matrix = get_scores_clf(y_test, y_pred, y_pred_prob)
    auc_train, f1_train, accu_train, accu_bl_train, conf_matrix_train = get_scores_clf(y_train, y_pred_train, y_pred_prob_train)
    
    return [model_method, auc, f1, accu, accu_bl, conf_matrix, auc_train, f1_train, accu_train, accu_bl_train, conf_matrix_train,
           y_pred_prob, y_pred_prob_train]



def data_fusion(type_list):
    df_other_cols = ['haim_id', 'y']
    em_all = data_type_dict[type_list[0]]
    for type_instance in type_list[1:]:
        em_all = em_all | data_type_dict[type_instance]
    return df[em_all | df_other_cols]


def get_data_dict(df):
	de_df = df.columns[df.columns.str.startswith('de_')]

	vd_df = df.columns[df.columns.str.startswith('vd_')]
	vp_df = df.columns[df.columns.str.startswith('vp_')]
	vmd_df = df.columns[df.columns.str.startswith('vmd_')]
	vmp_df = df.columns[df.columns.str.startswith('vmp_')]

	ts_ce_df = df.columns[df.columns.str.startswith('ts_ce_')]
	ts_le_df = df.columns[df.columns.str.startswith('ts_le_')]
	ts_pe_df = df.columns[df.columns.str.startswith('ts_pe_')]

	n_ecg_df = df.columns[df.columns.str.startswith('n_ecg')]
	n_ech_df = df.columns[df.columns.str.startswith('n_ech')]
	n_rad_df = df.columns[df.columns.str.startswith('n_rad')]

	data_type_dict = {
	    'demo': de_df,
	    'vd': vd_df, 
	    'vp': vp_df, 
	    'vmd': vmd_df, 
	    'vmp': vmp_df,
	    'ts_ce': ts_ce_df,
	    'ts_le': ts_le_df,
	    'ts_pe': ts_pe_df,
	    'n_ecg': n_ecg_df,
	    'n_ech': n_ech_df,
	    'n_rad': n_rad_df,
	}

	return data_type_dict

from itertools import combinations

def get_all_dtypes():
	individual_types = ['demo', 'vd', 'vp', 'vmd', 'vmp', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech', 'n_rad']
	combined_types = []
	n = len(individual_types)
	for i in range(n):
	    combined_types.extend(combinations(individual_types, i + 1))

	# Possible to add additional model types to here
	model_method_lis = [run_xgb]

	all_types_experiment = []
	for data_type in combined_types:
	    for model in model_method_lis:
	        all_types_experiment.append([data_type, model])

	return all_types_experiment








