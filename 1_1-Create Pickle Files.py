###########################################################################################################
#                      Create HAIM-MIMIC-MM & pickle files from MIMIC-IV & MIMIC-CXR-JPG
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
# # Code for Patient parsing in pickle files from HAIM-MIMIC-MM creation ( MIMIC-IV + MIMIC-CXR-JPG)
# # This files describes how we generate the patient-admission-stay level pickle files. 

#HAIM
from MIMIC_IV_HAIM_API import *

# Display optiona
from IPython.display import Image # IPython display
pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('float_format', '{:f}'.format)
pd.options.mode.chained_assignment = None  # default='warn'
get_ipython().run_line_magic('matplotlib', 'inline')


# ### -> Initializations & Data Loading
# Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)

# Define MIMIC IV Data Location
core_mimiciv_path = 'data/HAIM/physionet/files/mimiciv/1.0/'

# Define MIMIC IV Image Data Location (usually external drive)
core_mimiciv_imgcxr_path = 'data/HAIM/physionet/files/mimiciv/1.0/mimic-cxr-jpg/2.0.0/'

df_admissions, df_patients, f_transfers, df_diagnoses_icd, df_drgcodes, df_emar, df_emar_detail, df_hcpcsevents, df_labevents, df_microbiologyevents, df_poe, df_poe_detail, df_prescriptions, df_procedures_icd, df_services, df_d_icd_diagnoses, df_d_icd_procedures, df_d_hcpcs, df_d_labitem, df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents, df_d_items, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_metadata, df_mimic_cxr_negbio, df_noteevents, df_dsnotes, df_ecgnotes, df_echonotes, df_radnotes = load_mimiciv(core_mimiciv_path)

# -> MASTER DICTIONARY of health items
# Generate dictionary for chartevents, labevents and HCPCS
df_patientevents_categorylabels_dict = pd.DataFrame(columns = ['eventtype', 'category', 'label'])

# Get Chartevent items with labels & category
df = df_d_items
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
	#print(category)
	category_list = df[df['category']==category]
	for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
		df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'chart', 'category': category, 'label': item}, ignore_index=True)
		
# Get Lab items with labels & category
df = df_d_labitems
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
	#print(category)
	category_list = df[df['category']==category]
	for item_idx, item in enumerate(sorted(category_list.label.astype(str).unique())):
		df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'lab', 'category': category, 'label': item}, ignore_index=True)
		
# Get HCPCS items with labels & category
df = df_d_hcpcs
for category_idx, category in enumerate(sorted((df.category.astype(str).unique()))):
	#print(category)
	category_list = df[df['category']==category]
	for item_idx, item in enumerate(sorted(category_list.long_description.astype(str).unique())):
		df_patientevents_categorylabels_dict = df_patientevents_categorylabels_dict.append({'eventtype': 'hcpcs', 'category': category, 'label': item}, ignore_index=True)
		
		
## CORE
print('- CORE > df_admissions')
print('--------------------------------')
print(df_admissions.dtypes)
print('\n\n')

print('- CORE > df_patients')
print('--------------------------------')
print(df_patients.dtypes)
print('\n\n')

print('- CORE > df_transfers')
print('--------------------------------')
print(df_transfers.dtypes)
print('\n\n')


## HOSP
print('- HOSP > df_d_labitems')
print('--------------------------------')
print(df_d_labitems.dtypes)
print('\n\n')

print('- HOSP > df_d_icd_procedures')
print('--------------------------------')
print(df_d_icd_procedures.dtypes)
print('\n\n')

print('- HOSP > df_d_icd_diagnoses')
print('--------------------------------')
print(df_d_icd_diagnoses.dtypes)
print('\n\n')

print('- HOSP > df_d_hcpcs')
print('--------------------------------')
print(df_d_hcpcs.dtypes)
print('\n\n')

print('- HOSP > df_diagnoses_icd')
print('--------------------------------')
print(df_diagnoses_icd.dtypes)
print('\n\n')

print('- HOSP > df_drgcodes')
print('--------------------------------')
print(df_drgcodes.dtypes)
print('\n\n')

print('- HOSP > df_emar')
print('--------------------------------')
print(df_emar.dtypes)
print('\n\n')

print('- HOSP > df_emar_detail')
print('--------------------------------')
print(df_emar_detail.dtypes)
print('\n\n')

print('- HOSP > df_hcpcsevents')
print('--------------------------------')
print(df_hcpcsevents.dtypes)
print('\n\n')

print('- HOSP > df_labevents')
print('--------------------------------')
print(df_labevents.dtypes)
print('\n\n')

print('- HOSP > df_microbiologyevents')
print('--------------------------------')
print(df_microbiologyevents.dtypes)
print('\n\n')

print('- HOSP > df_poe')
print('--------------------------------')
print(df_poe.dtypes)
print('\n\n')

print('- HOSP > df_poe_detail')
print('--------------------------------')
print(df_poe_detail.dtypes)
print('\n\n')

print('- HOSP > df_prescriptions')
print('--------------------------------')
print(df_prescriptions.dtypes)
print('\n\n')

print('- HOSP > df_procedures_icd')
print('--------------------------------')
print(df_procedures_icd.dtypes)
print('\n\n')

print('- HOSP > df_services')
print('--------------------------------')
print(df_services.dtypes)
print('\n\n')


## ICU
print('- ICU > df_procedureevents')
print('--------------------------------')
print(df_procedureevents.dtypes)
print('\n\n')

print('- ICU > df_outputevents')
print('--------------------------------')
print(df_outputevents.dtypes)
print('\n\n')

print('- ICU > df_inputevents')
print('--------------------------------')
print(df_inputevents.dtypes)
print('\n\n')

print('- ICU > df_icustays')
print('--------------------------------')
print(df_icustays.dtypes)
print('\n\n')

print('- ICU > df_datetimeevents')
print('--------------------------------')
print(df_datetimeevents.dtypes)
print('\n\n')

print('- ICU > df_d_items')
print('--------------------------------')
print(df_d_items.dtypes)
print('\n\n')

print('- ICU > df_chartevents')
print('--------------------------------')
print(df_chartevents.dtypes)
print('\n\n')


## CXR
print('- CXR > df_mimic_cxr_split')
print('--------------------------------')
print(df_mimic_cxr_split.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_chexpert')
print('--------------------------------')
print(df_mimic_cxr_chexpert.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_metadata')
print('--------------------------------')
print(df_mimic_cxr_metadata.dtypes)
print('\n\n')

print('- CXR > df_mimic_cxr_negbio')
print('--------------------------------')
print(df_mimic_cxr_negbio.dtypes)
print('\n\n')


## NOTES
print('- NOTES > df_noteevents')
print('--------------------------------')
print(df_noteevents.dtypes)
print('\n\n')

print('- NOTES > df_icunotes')
print('--------------------------------')
print(df_dsnotes.dtypes)
print('\n\n')

print('- NOTES > df_ecgnotes')
print('--------------------------------')
print(df_ecgnotes.dtypes)
print('\n\n')

print('- NOTES > df_echonotes')
print('--------------------------------')
print(df_echonotes.dtypes)
print('\n\n')

print('- NOTES > df_radnotes')
print('--------------------------------')
print(df_radnotes.dtypes)
print('\n\n')


# ## -> GET LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV (subject_id, hadm_id, stay_id)

# Get Unique Subject/HospAdmission/Stay Combinations
df_ids = pd.concat([pd.DataFrame(), df_procedureevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_outputevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_inputevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_icustays[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_datetimeevents[['subject_id','hadm_id','stay_id']]], sort=False).drop_duplicates()
df_ids = pd.concat([df_ids, df_chartevents[['subject_id','hadm_id','stay_id']]], sort=True).drop_duplicates()

# Get Unique Subjects with Chest Xrays
df_cxr_ids = pd.concat([pd.DataFrame(), df_mimic_cxr_chexpert[['subject_id']]], sort=True).drop_duplicates()

# Get Unique Subject/HospAdmission/Stay Combinations with Chest Xrays
df_haim_ids = df_ids[df_ids['subject_id'].isin(df_cxr_ids['subject_id'].unique())] 

# Save Unique Subject/HospAdmission/Stay Combinations with Chest Xrays    
df_haim_ids.to_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv', index=False)


print('Unique Subjects: ' + str(len(df_patients['subject_id'].unique())))
print('Unique Subjects/HospAdmissions/Stays Combinations: ' + str(len(df_ids)))
print('Unique Subjects with Chest Xrays Available: ' + str(len(df_cxr_ids)))


# Save Unique Subject/HospAdmission/Stay Combinations with Chest Xrays    
df_haim_ids = pd.read_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv')
print('Unique HAIM Records Available: ' + str(len(df_haim_ids)))

# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
nfiles = generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path)
