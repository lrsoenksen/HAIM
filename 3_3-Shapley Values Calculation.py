###########################################################################################################
#                                       Shapley Values Calculation
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
import ast
from itertools import combinations
from math import comb

los_auc = pd.read_csv("AUC_All_los_pred.csv")
los_auc.txt = [1 if 'n_rad' in x else txt for x,txt in zip(los_auc.Sources,los_auc.txt)]
    

source_auc_dict = {}
for source, auc in zip(los_auc.Sources, los_auc.AUC_mean):
    source_sorted = tuple(sorted(ast.literal_eval(source)))
    source_auc_dict[source_sorted] = auc
source_auc_dict[()] = 0.5

sources = ['de', 'vd', 'vp', 'vmd', 'vmp', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech', 'n_rad']

def powerset(iterable,missing_el):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    xs = list(iterable)
    return [tuple(sorted(list(subset))) for i in range(0, len(xs) + 1) for subset in combinations(xs, i) if missing_el not in subset]


source_shap_dict = {}
for source in sources:
    source_shap = 0
    total_sets = powerset(sources, source)
    for single_set in total_sets:
        single_list = list(single_set)
        single_list.append(source)
        source_shap += 1 / (len(sources) * comb(len(sources)-1, len(single_set))) * (source_auc_dict[tuple(sorted(single_list))]-source_auc_dict[single_set])
    source_shap_dict[source] = source_shap
    
shap_values_sources = pd.DataFrame(source_shap_dict.items(), columns=['Source', 'Gain'])
shap_values_sources.to_csv("shap_vals_sources_los.csv")


los_auc_types = los_auc[['AUC_mean', 'viz','txt','tab','ts']].groupby(['viz','txt','tab','ts']).mean().reset_index()
type_auc_dict = {}
colnames = list(los_auc_types.columns.values)
for row_count, row in los_auc_types.iterrows():
    set_comb = []
    for i in range(4):
        if row[i] == 1:
            set_comb.append(colnames[i])
    
    type_auc_dict[tuple(sorted(set_comb))] = row[4]
type_auc_dict[()] = 0.5
    
types = ['viz','txt','tab','ts']
types_shap_dict = {}
for tpe in types:
    type_shap = 0
    total_sets = powerset(types, tpe)
    for single_set in total_sets:
        single_list = list(single_set)
        single_list.append(tpe)
        type_shap += 1 / (len(types) * comb(len(types)-1, len(single_set))) * (type_auc_dict[tuple(sorted(single_list))]-type_auc_dict[single_set])
    types_shap_dict[tpe] = type_shap

shap_values_types = pd.DataFrame(types_shap_dict.items(), columns=['Type', 'Gain'])
shap_values_types.to_csv("shap_vals_types_los.csv")


mortality_auc = pd.read_csv("AUC_All_mortality_pred.csv")
mortality_auc.txt = [1 if 'n_rad' in x else txt for x,txt in zip(mortality_auc.Sources,mortality_auc.txt)]

source_auc_dict = {}
for source, auc in zip(mortality_auc.Sources, mortality_auc.AUC_mean):
    source_sorted = tuple(sorted(ast.literal_eval(source)))
    source_auc_dict[source_sorted] = auc
source_auc_dict[()] = 0.5

sources = ['demo', 'vd', 'vp', 'vmd', 'vmp', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech', 'n_rad']

def powerset(iterable,missing_el):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    xs = list(iterable)
    return [tuple(sorted(list(subset))) for i in range(0, len(xs) + 1) for subset in combinations(xs, i) if missing_el not in subset]


source_shap_dict = {}
for source in sources:
    source_shap = 0
    total_sets = powerset(sources, source)
    for single_set in total_sets:
        single_list = list(single_set)
        single_list.append(source)
        source_shap += 1 / (len(sources) * comb(len(sources)-1, len(single_set))) * (source_auc_dict[tuple(sorted(single_list))]-source_auc_dict[single_set])
    source_shap_dict[source] = source_shap
    
shap_values_sources = pd.DataFrame(source_shap_dict.items(), columns=['Source', 'Gain'])
shap_values_sources.to_csv("shap_vals_sources_mortality.csv")


mortality_auc_types = mortality_auc[['AUC_mean', 'viz','txt','tab','ts']].groupby(['viz','txt','tab','ts']).mean().reset_index()
type_auc_dict = {}
colnames = list(mortality_auc_types.columns.values)
for row_count, row in mortality_auc_types.iterrows():
    set_comb = []
    for i in range(4):
        if row[i] == 1:
            set_comb.append(colnames[i])
    
    type_auc_dict[tuple(sorted(set_comb))] = row[4]
type_auc_dict[()] = 0.5
    
types = ['viz','txt','tab','ts']
types_shap_dict = {}
for tpe in types:
    type_shap = 0
    total_sets = powerset(types, tpe)
    for single_set in total_sets:
        single_list = list(single_set)
        single_list.append(tpe)
        type_shap += 1 / (len(types) * comb(len(types)-1, len(single_set))) * (type_auc_dict[tuple(sorted(single_list))]-type_auc_dict[single_set])
    types_shap_dict[tpe] = type_shap

shap_values_types = pd.DataFrame(types_shap_dict.items(), columns=['Type', 'Gain'])
shap_values_types.to_csv("shap_vals_types_mortality.csv")


all_modality_auc = pd.read_csv("AUC_All_Modality_Sources.csv")

diseases = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion', 'Lung Opacity', 'Pneumonia','Pneumothorax']

shap_values_sources_diseases = []
shap_values_types_diseases = []
for disease in diseases:
    source_auc_dict = {}
    for source, auc in zip(all_modality_auc.Sources, all_modality_auc[disease]):
        source_sorted = tuple(sorted(ast.literal_eval(source)))
        source_auc_dict[source_sorted] = auc
    source_auc_dict[()] = 0.5
    
    sources = ['de', 'vd', 'vp', 'vmd', 'vmp', 'ts_ce', 'ts_le', 'ts_pe', 'n_ecg', 'n_ech']
    
    def powerset(iterable,missing_el):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        xs = list(iterable)
        return [tuple(sorted(list(subset))) for i in range(0, len(xs) + 1) for subset in combinations(xs, i) if missing_el not in subset]
    
    
    source_shap_dict = {}
    for source in sources:
        source_shap = 0
        total_sets = powerset(sources, source)
        for single_set in total_sets:
            single_list = list(single_set)
            single_list.append(source)
            source_shap += 1 / (len(sources) * comb(len(sources)-1, len(single_set))) * (source_auc_dict[tuple(sorted(single_list))]-source_auc_dict[single_set])
        source_shap_dict[source] = source_shap
    
    column_naming = 'Gain_' + disease
    shap_values_sources = pd.DataFrame(source_shap_dict.items(), columns=['Source', column_naming])
    shap_values_sources = shap_values_sources.set_index('Source')
    shap_values_sources_diseases.append(shap_values_sources)
    
    selected_columns = ['viz','txt','tab','ts']
    selected_columns.append(disease)
    
    mortality_auc_types = all_modality_auc[selected_columns].groupby(['viz','txt','tab','ts']).mean().reset_index()
    type_auc_dict = {}
    colnames = list(mortality_auc_types.columns.values)
    for row_count, row in mortality_auc_types.iterrows():
        set_comb = []
        for i in range(4):
            if row[i] == 1:
                set_comb.append(colnames[i])
        
        type_auc_dict[tuple(sorted(set_comb))] = row[4]
    type_auc_dict[()] = 0.5
        
    types = ['viz','txt','tab','ts']
    types_shap_dict = {}
    for tpe in types:
        type_shap = 0
        total_sets = powerset(types, tpe)
        for single_set in total_sets:
            single_list = list(single_set)
            single_list.append(tpe)
            type_shap += 1 / (len(types) * comb(len(types)-1, len(single_set))) * (type_auc_dict[tuple(sorted(single_list))]-type_auc_dict[single_set])
        types_shap_dict[tpe] = type_shap
    
    shap_values_types = pd.DataFrame(types_shap_dict.items(), columns=['Type', column_naming])
    shap_values_types = shap_values_types.set_index('Type')
    shap_values_types_diseases.append(shap_values_types)
shap_values_sources_diseases_output = pd.concat(shap_values_sources_diseases, axis=1)
shap_values_types_diseases_output = pd.concat(shap_values_types_diseases, axis=1)

shap_values_sources_diseases_output.to_csv("shap_vals_sources_diseases.csv")
shap_values_types_diseases_output.to_csv("shap_vals_types_diseases.csv")