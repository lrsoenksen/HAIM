###########################################################################################################
#                                       Pathology Diagnosis Modeling
###########################################################################################################
# 
# Licensed under the Apache License, Version 2.0**
# You may not use this file except in compliance with the License. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is 
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
# implied. See the License for the specific language governing permissions and limitations under the License.

#-> Authors: 
#      Luis R Soenksen (<soenksen@mit.edu>),
#      Yu Ma (<midsumer@mit.edu>),
#      Cynthia Zeng (<czeng12@mit.edu>),
#      Leonard David Jean Boussioux (<leobix@mit.edu>),
#      Kimberly M Villalobos Carballo (<kimvc@mit.edu>),
#      Liangyuan Na (<lyna@mit.edu>),
#      Holly Mika Wiberg (<hwiberg@mit.edu>),
#      Michael Lingzhi Li (<mlli@mit.edu>),
#      Ignacio Fuentes (<ifuentes@mit.edu>),
#      Dimitris J Bertsimas (<dbertsim@mit.edu>),
# -> Last Update: Dec 30th, 2021

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from itertools import product
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from scipy import interp
import matplotlib.pyplot as plt
from itertools import combinations

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
       'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
       'Pneumothorax', 'Support Devices', 'PerformedProcedureStepDescription',
       'ViewPosition']

LABELS_GOOD = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Lung Opacity', 'Pneumonia',
       'Pneumothorax']

VP_MAP = {'Atelectasis': 'vp_0',
         'Consolidation': 'vp_1',
         'Infiltration': 'vp_2',
         'Pneumothorax': 'vp_3',
         'Edema': 'vp_4',
         'Emphysema': 'vp_5',
         'Fibrosis': 'vp_6',
         'Effusion': 'vp_7',
         'Pneumonia': 'vp_8',
         'Pleural_Thickening': 'vp_9',
         'Cardiomegaly': 'vp_10',
         'Nodule': 'vp_11',
         'Mass': 'vp_12',
         'Hernia': 'vp_13',
         'Lung Lesion': 'vp_14',
         'Fracture': 'vp_15',
         'Lung Opacity': 'vp_16',
         'Enlarged Cardiomediastinum': 'vp_17'}

N_EST_SET, MAX_PARAM_SET, LR_SET = [200],[5],[0.3]#[200,300], [5,6,7], [0.3, 0.1, 0.05]
num_seeds = 3

VPs = list(VP_MAP.values())

cols_vp = [col for col in embeddings.columns if col[:2]=='vp']
dict_mode = {'vp': cols_vp}#, 'vd':cols_vd, 'ce':cols_ce, 'de': cols_de, 'vd':cols_le, 'pe':cols_pe,
            #'vmd': cols_vmd, 'vmp':cols_vmp, 'ts_le':cols_ts_le, 'ts_ce':cols_ts_ce}
dict_mode['vp']

from itertools import combinations
individual_types = ['de', 'vd', 'vp', 'vmd', 'vmp', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech']
combined_types = []
n = len(individual_types)
for i in range(n):
    combined_types.extend(combinations(individual_types, i + 1))
    
def transfo_combo(combo):
    s_ = combo[0]
    for s in combo[1:]:
        s_ = s_ + '+' + s
    return s_

l = list(map(transfo_combo, combined_types))

def get_X_y2(data, label, mode = 'vp'):
    df = data[data[label].isin([0, 1])]
    info = ['haim_id'] + [label]
    cols_vp = [col for col in data.columns if col[:3]=='vp_'] 
    cols_vd = [col for col in data.columns if col[:3]=='vd_']
    cols_ce = [col for col in data.columns if col[:3]=='ce_'] 
    cols_de = [col for col in data.columns if col[:3]=='de_'] 
    cols_le = [col for col in data.columns if col[:3]=='le_'] 
    cols_pe = [col for col in data.columns if col[:3]=='pe_']
    
    cols_vmd = [col for col in data.columns if col[:4]=='vmd_']
    cols_vmp = [col for col in data.columns if col[:4]=='vmp_']
    
    cols_ts_le = [col for col in data.columns if col[:6]=='ts_le_']
    cols_ts_ce = [col for col in data.columns if col[:6]=='ts_ce_']
    cols_ts_pe = [col for col in data.columns if col[:6]=='ts_pe_']
    cols_n_ecg = [col for col in data.columns if col[:6]=='n_ecg_']
    cols_n_ech = [col for col in data.columns if col[:6]=='n_ech_']
    dict_mode = {'vp': cols_vp, 'vd':cols_vd, 'ce':cols_ce, 'de': cols_de, 'le':cols_le, 'pe':cols_pe,
            'vmd': cols_vmd, 'vmp':cols_vmp, 'ts_le':cols_ts_le, 'ts_ce':cols_ts_ce, 'ts_pe':cols_ts_pe, 
                 'n_ecg':cols_n_ecg, 'n_ech':cols_n_ech}
    cols = list(info)
    mode_list = mode.split("+")
    for m in mode_list:
        cols+=dict_mode[m]
    return df[cols], df[info + cols_vp]


def train_vision_only(X_train, X_test, y_train, y_test, seed, method, n_est, max_param, lr):
    ######################################
    #X_train, X_test, y_train and y_test are pandas dataframes. The number of rows corresponds to
    #the data size, and the number of columns corresponds to the number of features per data point.
    ######################################
    clf = XGBClassifier(n_estimators = n_est, max_depth = max_param, learning_rate = lr, eval_metric='logloss',
                        tree_method='gpu_hist', gpu_id=0)
    clf.fit(X_train, y_train)
    y_pred = np.array(clf.predict_proba(X_test)[:,1])
  
    auc = roc_auc_score(y_test, y_pred)
    return y_pred, auc

def create_prob_files(probabilities, ground_truth, num_seeds, mode, sparse, labels, method):
    for seed in range(num_seeds):
        df_prob = pd.DataFrame()
        df_truth = pd.DataFrame()
        for label in labels:
            df_prob_new = pd.DataFrame(probabilities[(seed, label)], columns=[label])
            df_prob = pd.concat([df_prob, df_prob_new], axis = 1)
            df_truth_new = pd.DataFrame(ground_truth[(seed, label)], columns=[label])
            df_truth = pd.concat([df_truth, df_truth_new], axis = 1)
        df_prob.to_csv('predicted_probs_'+ str(seed) + mode + method + str(sparse)+'.csv', sep=',')
        df_truth.to_csv('ground_truth_'+ str(seed) + mode + method + str(sparse)+'.csv', sep=',')
        
        
def save_tpr_fpr(num_seeds, mode, sparse, method, HAIM_mode):
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'pink', 'green', 'gray', 'yellow']
    for i, color in zip(range(len(LABELS_GOOD)), colors):
        for seed in range(num_seeds):
            y_pred = pd.read_csv('disease-result/probs/'+ str(seed) + mode + method + str(sparse)+'.csv')#.fillna(0.5)
            y_truth = pd.read_csv('disease-result/ground_truth/'+ str(seed) + mode + method + str(sparse)+'.csv')#.fillna(0)
            
            y_pred = y_pred[LABELS_GOOD[i]].dropna()
            y_truth = y_truth[ LABELS_GOOD[i]].dropna()
            
            fpr, tpr, _ = roc_curve(y_truth, y_pred)
            np.savetxt("disease-result/FPR/"+HAIM_mode+"/fpr_" + LABELS_GOOD[i] + str(seed) + mode + method +  str(sparse)+'.csv', fpr, delimiter=',')
            np.savetxt("disease-result/TPR/" + HAIM_mode + "/tpr_" + LABELS_GOOD[i] + str(seed) + mode + method +  str(sparse)+'.csv', tpr, delimiter=',')

            
            
lw = 2
def plot_roc_curves(num_seeds, mode, sparse, method):
    
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'pink', 'green', 'gray', 'yellow']
    for i, color in zip(range(len(LABELS_GOOD)), colors):
    
        tprs = []
        aucs = []
        base_fpr = np.linspace(0, 1, 101)
    
        for seed in range(num_seeds):
        
            y_pred = pd.read_csv('disease-result/probs/'+ str(seed) + mode + method + str(sparse)+'.csv').fillna(0.5)
            y_truth = pd.read_csv('disease-result/ground_truth/'+ str(seed) + mode + method + str(sparse)+'.csv').fillna(0)
    
            fpr, tpr, _ = roc_curve(y_truth[LABELS_GOOD[i]], y_pred[LABELS_GOOD[i]])
            aucs.append(roc_auc_score(y_truth[LABELS_GOOD[i]], y_pred[LABELS_GOOD[i]]))
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
    
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        mean_auc = sum(aucs)/num_seeds
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std


        plt.plot(base_fpr, mean_tprs, color = color, lw=lw, label='Avg. ROC curve of {0} (area = {1:0.2f})'.format(LABELS_GOOD[i], mean_auc))
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=color, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC with Confidence Intervals - (modality:'+ mode + ', sparsity:' + str(sparse) + ', method:' + method +')')
    plt.legend(bbox_to_anchor=(1.85,0), loc="lower right")
    plt.show()

df_modalities = pd.read_csv("AUPRC_All_Modality_Sources.csv");

best_HAIM_AUPRC = [0 for i in range(len(LABELS_GOOD))]
best_HAIM_mode = [0 for i in range(len(LABELS_GOOD))]

best_not_HAIM_AUPRC = [0 for i in range(len(LABELS_GOOD))]
best_not_HAIM_mode = [0 for i in range(len(LABELS_GOOD))]

###### LAUNCHING ALL EXPERIMENTS #######
for mode in l:
	for sparse in [False]:#, True]:
		for method in ['xgb']:#, 'lightxgb']:
			gen_param.append((mode,sparse,method))
            
sparse = False
method = 'xgb'

for i in range(1023):
    params = gen_param[i]
    mode = params[0]
    
    for i in range(len(LABELS_GOOD)):
        disease = LABELS_GOOD[i]
        name = "AUPRC_" + mode +"_" + str(method) +"_"+ str(sparse)

        auprc = df_modalities[df_modalities["Unnamed: 0"] ==name][disease].values[0]
        num_modalities = df_modalities[df_modalities["Unnamed: 0"] ==name]["Number of Modalities"].values[0]

        is_HAIM = (num_modalities > 1)
        if is_HAIM and  (auprc > best_HAIM_AUPRC[i]):
            best_HAIM_AUPRC[i] = auprc
            best_HAIM_mode[i] = mode
        
        if  not is_HAIM and  (auprc > best_not_HAIM_AUPRC[i]):
            best_not_HAIM_AUPRC[i] = auprc
            best_not_HAIM_mode[i] = mode

for i in range(len(LABELS_GOOD)):
    save_tpr_fpr(5, best_HAIM_mode[i], sparse, method, "HAIM")
    save_tpr_fpr(5, best_not_HAIM_mode[i], sparse, method, "not_HAIM")

# Supply filename of the embedding file 
fname = fname
def train_all_vision(mode, method, num_seeds, num_folds, n_est_set=N_EST_SET, max_param_set=MAX_PARAM_SET, lr_set=LR_SET, test_size=0.2, cross_val_size=0.2, diseases = LABELS, sparse = True, s_low = 600, s_high = 810, space = 50):
    embeddings = pd.read_csv(fname)
    print('########## Method', method, '##########')
    results = {}
    probabilities = {}
    ground_truth = {}
    
    #create_prob_files(probabilities, ground_truth, num_seeds, mode, sparse, labels)
    
    for label in diseases:
        
        probs = []
        
        print('\n#### LABEL', label, '####')
        df, base = get_X_y2(embeddings, label, mode)
        print("Data size: ", df.shape[0])
        
        if True:
            print("Ratio Positives/Total:", np.round(np.sum(df[label])/df.shape[0], 2))
            #Initialize best parameters and results
            avg_test_auc = 0
            avg_test_auc_base = 0
            
            #Stratify by patient for main-test sets
            for seed in range(num_seeds):
                best_auc = 0
                best_n_est = 0
                best_max_param = 0
                best_lr = 0
            
                np.random.seed(seed)
                patient_labels = df.groupby(['haim_id']).agg(label_count = (label,'count'),
                                                                 label_ones = (label, 'sum')).reset_index()
                patient_labels['label'] = patient_labels.apply(lambda row: int(row['label_ones'] >=  row['label_count']/2), axis=1)
                haim_id = np.array(patient_labels['haim_id'])
                labels = np.array(patient_labels['label'])
                id_main, id_test, label_main, label_test = train_test_split(haim_id, labels, test_size = test_size, stratify = labels)
                X_test = df[df['haim_id'].isin(id_test)]
                y_test = X_test[label]
                #print(y_test.values.any())
                X_main = df[df['haim_id'].isin(id_main)]
                y_main = X_main[label]
                test_label_ratio = np.round(np.sum(X_test[label])/X_test.shape[0], 2)
                test_length = X_test.shape[0]
                main_label_ratio = np.round(np.sum(X_main[label])/X_main.shape[0], 2)
                main_length = X_main.shape[0]
                test_train_ratio = test_length/(test_length + main_length)
                X_main = X_main.drop([label, 'haim_id'], axis=1)
                X_test = X_test.drop([label, 'haim_id'], axis=1)
                if sparse:
                    feats = get_sparse_X(X_main, y_main, s_low, s_high, space)
                    X_main = X_main[feats]
                    X_test = X_test[feats]
                
                for n_est, max_param, lr in product(n_est_set, max_param_set, lr_set):
                    avg_auc = 0
                    for fold in range(num_folds): 

                        np.random.seed(fold)

                        #Stratify by patient for val-train sets
                        id_train, id_val, _, _ = train_test_split(id_main, label_main, test_size = cross_val_size, stratify = label_main)

                        #Get xrays train and val split
                        X_train = df[df['haim_id'].isin(id_train)]
                        y_train = X_train[label]
                        X_val = df[df['haim_id'].isin(id_val)]
                        y_val = X_val[label]
                        X_train = X_train.drop([label, 'haim_id'], axis=1)
                        X_val = X_val.drop([label, 'haim_id'], axis=1)
                        if sparse:
                            X_train = X_train[feats]
                            X_val = X_val[feats]

                        #Predict and Evaluate
                        y_pred, auc = train_vision_only(X_train, X_val, y_train, y_val, fold, method, n_est=n_est, max_param=max_param, lr=lr)
                        avg_auc += auc

                    avg_auc  = avg_auc/num_seeds

                    if avg_auc > best_auc:
                        best_auc, best_n_est, best_max_param, best_lr = avg_auc, n_est, max_param, lr

                #Retrain using main data set with best parameters
                print('n_est:',best_n_est, 'max_param:',best_max_param,'lr:', best_lr)
                y_pred, test_auc = train_vision_only(X_main, X_test, y_main, y_test, fold, method, n_est=best_n_est, max_param=best_max_param, lr=best_lr)
                avg_test_auc += test_auc
                probabilities[(seed, label)] = tuple(y_pred)
                ground_truth[(seed, label)] = tuple(y_test)
            
                #Get baseline test split
                X_test_base = base[base['haim_id'].isin(id_test)].drop([label, 'haim_id'], axis=1)
                y_pred_base = X_test_base[VP_MAP[label]]
                test_auc_base = roc_auc_score(y_test, y_pred_base)
                avg_test_auc_base += test_auc_base
                
            avg_test_auc = avg_test_auc/num_seeds
            avg_test_auc_base = avg_test_auc_base/num_seeds
            print("AUC Baseline:", np.round(avg_test_auc_base,3))
            print("AUC: ", np.round(avg_test_auc,3))
            results[label] = (avg_test_auc, avg_test_auc_base, df.shape[0], np.sum(df[label])/df.shape[0],
                              main_label_ratio, test_label_ratio, test_train_ratio)
    
    create_prob_files(probabilities, ground_truth, num_seeds, mode, sparse, diseases, method)
        
    return results


train_all_vision(mode = 'vp', method = 'xgb', num_seeds = 2, num_folds = 2, diseases = LABELS_GOOD, sparse=False)










