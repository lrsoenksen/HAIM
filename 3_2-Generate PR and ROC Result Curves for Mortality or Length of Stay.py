###########################################################################################################
#              Generate Generate PR and ROC Curves for Mortality and Length of Stay Models
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

df = pd.read_csv("result-table/AUC_All_mortality_pred.csv")
haim_auc = df[df['Number of Modalities'] == 4]['AUC_mean'].max()
haim_id = df[df['AUC_mean'] == haim_auc]['Unnamed: 0'].values(0)

single_auc = df[df['Number of Modalities'] == 1]['AUC_mean'].max()
single_id = df[df['AUC_mean'] == single_auc]['Unnamed: 0'].values(0)

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

def get_ft_pr_tables(y_true, y_pred, mode, seed):
    fpr, tpr, ft_thresholds = metrics.roc_curve(y_true, y_pred)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)

    df_result_ft = pd.DataFrame(columns = ['fpr', 'tpr', 'ft_thresholds'])
    df_result_ft['fpr'] = fpr
    df_result_ft['tpr'] = tpr
    df_result_ft['ft_thresholds'] = ft_thresholds
    df_result_ft.to_csv('result-table/df_result_mortality_ft-{}-{}.csv'.format(mode, seed))

    df_result_pr = pd.DataFrame(columns = ['precision', 'recall', 'pr_thresholds'])
    df_result_pr['precision'] = precision
    df_result_pr['recall'] = recall
 
    df_result_pr.to_csv('result-table/df_result_mortality_pr-{}-{}.csv'.format(mode, seed))
    
# Supple result folder name, potentially mortality-result, los-result
folder_name = ''   
# Could supple haim or single
mode = ''
# Could supple mortality or los
task = ''
# Could supply single_id or haim_id
id_record = 

df1 = pd.read_csv('{}/y_pred_prob/{}-0.csv'.format(folder_name, id_record))['0'].values
df2 = pd.read_csv('{}/y_pred_prob/{}-1.csv'.format(folder_name, id_record))['0'].values
df3 = pd.read_csv('{}/y_pred_prob/{}-2.csv'.format(folder_name, id_record))['0'].values
df4 = pd.read_csv('{}/y_pred_prob/{}-3.csv'.format(folder_name, id_record))['0'].values
df5 = pd.read_csv('{}/y_pred_prob/{}-4.csv'.format(folder_name, id_record))['0'].values

y1 = pd.read_csv('y_test_{}_{}-0.csv'.format(task, mode))['y'].values
y2 = pd.read_csv('y_test_{}_{}-1.csv'.format(task, mode))['y'].values
y3 = pd.read_csv('y_test_{}_{}-2.csv'.format(task, mode))['y'].values
y4 = pd.read_csv('y_test_{}_{}-3.csv'.format(task, mode))['y'].values
y5 = pd.read_csv('y_test_{}_{}-4.csv'.format(task, mode))['y'].values  

for seed in range(5):
    for mode in ['single', 'haim']:
        get_ft_pr_tables(y1, df1, 'single', seed)
    

        