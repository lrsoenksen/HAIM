###########################################################################################################
#                                       Embeddings Generation
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

from MIMIC_IV_HAIM_API import *
import sys
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import gc

# Full Core MIMIC-IV database path
core_mimiciv_path = 'data/'
df_haim_ids = pd.read_csv(core_mimiciv_path + 'pickle/haim_mimiciv_key_ids.csv')
# Get Key dictionary
df_patientevents_categorylabels_dict = load_haim_event_dictionaries(core_mimiciv_path)
# Get core data
df_haim_ids_core_info = load_core_mimic_haim_info(core_mimiciv_path, df_haim_ids)

def process_cxr_embeddings_haim_id(haim_id, dt_patient, df_init):
    
    # DEMOGRAPHICS EMBEDDINGS EXTRACTION
    demo_embeddings = get_demographic_embeddings(dt_patient, verbose=0)
    gc.collect() #Clear memory
    
    # Time Series (TSFRESH-like) CHARTEVENT & LABEVENT EMBEDDINGS EXTRACTION
    aggregated_ts_ce_embeddings = get_ts_embeddings(dt_patient, event_type = 'chart')
    gc.collect() #Clear memory
    
    aggregated_ts_le_embeddings = get_ts_embeddings(dt_patient, event_type = 'lab')
    gc.collect() #Clear memory
    
    aggregated_ts_pe_embeddings = get_ts_embeddings(dt_patient, event_type = 'procedure')
    gc.collect() #Clear memory
    
    # CHEST XRAY VISION EMBEDDINGS EXTRACTION
    aggregated_densefeature_embeddings, _, aggregated_prediction_embeddings, _, _ = get_chest_xray_embeddings(dt_patient, verbose=0)
    gc.collect() #Clear memory
    
    # NOTES FROM ECGs
    aggregated_ecg_embeddings = get_notes_biobert_embeddings(patient, note_type = 'ecgnotes')
    gc.collect() #Clear memory
    
    # NOTES FROM ECOCARDIOGRAMs
    aggregated_echo_embeddings = get_notes_biobert_embeddings(patient, note_type = 'echonotes')
    gc.collect() #Clear memory
    
    # NOTES FROM RADIOLOGY
    aggregated_rad_embeddings = get_notes_biobert_embeddings(patient, note_type = 'radnotes')
    gc.collect() #Clear memory

    # CHEST XRAY VISION SINGLE-IMAGE EMBEDDINGS EXTRACTION
    print('getting xray')
    img = df_imcxr[idx]
    densefeature_embeddings, prediction_embeddings = get_single_chest_xray_embeddings(img)
    gc.collect() #Clear memory

    # Create Dataframes filteed by ordered sample number for Fusion
    df_haim_ids_fusion = pd.DataFrame([haim_id],columns=['haim_id'])
    df_demographics_embeddings_fusion = pd.DataFrame(demo_embeddings.reshape(1,-1), columns=['de_'+str(i) for i in range(demo_embeddings.shape[0])])
    df_ts_ce_embeddings_fusion = pd.DataFrame(aggregated_ts_ce_embeddings.values.reshape(1,-1), columns=['ts_ce_'+str(i) for i in range(aggregated_ts_ce_embeddings.values.shape[0])])
    df_ts_le_embeddings_fusion = pd.DataFrame(aggregated_ts_le_embeddings.values.reshape(1,-1), columns=['ts_le_'+str(i) for i in range(aggregated_ts_le_embeddings.values.shape[0])])
    df_ts_pe_embeddings_fusion = pd.DataFrame(aggregated_ts_pe_embeddings.values.reshape(1,-1), columns=['ts_pe_'+str(i) for i in range(aggregated_ts_pe_embeddings.values.shape[0])])
    
    df_vision_dense_embeddings_fusion = pd.DataFrame(densefeature_embeddings.reshape(1,-1), columns=['vd_'+str(i) for i in range(densefeature_embeddings.shape[0])])
    df_vision_predictions_embeddings_fusion = pd.DataFrame(prediction_embeddings.reshape(1,-1), columns=['vp_'+str(i) for i in range(prediction_embeddings.shape[0])])
    df_vision_multi_dense_embeddings_fusion = pd.DataFrame(aggregated_densefeature_embeddings.reshape(1,-1), columns=['vmd_'+str(i) for i in range(aggregated_densefeature_embeddings.shape[0])])
    df_vision_multi_predictions_embeddings_fusion = pd.DataFrame(aggregated_prediction_embeddings.reshape(1,-1), columns=['vmp_'+str(i) for i in range(aggregated_prediction_embeddings.shape[0])])
    df_ecgnotes_embeddings_fusion = pd.DataFrame(aggregated_ecg_embeddings.reshape(1,-1), columns=['n_ecg_'+str(i) for i in range(aggregated_ecg_embeddings.shape[0])])
    df_echonotes_embeddings_fusion = pd.DataFrame(aggregated_echo_embeddings.reshape(1,-1), columns=['n_ech_'+str(i) for i in range(aggregated_echo_embeddings.shape[0])])
    df_radnotes_embeddings_fusion = pd.DataFrame(aggregated_rad_embeddings.reshape(1,-1), columns=['n_rad_'+str(i) for i in range(aggregated_rad_embeddings.shape[0])])
    
    # Vision targets
    cxr_target_columns = ['split','Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices', 'PerformedProcedureStepDescription','ViewPosition']
    df_vision_targets_fusion = df_stay_cxr.loc[idx:idx][cxr_target_columns].reset_index(drop=True)

    # Embeddings FUSION
    df_fusion = df_haim_ids_fusion
    df_fusion = pd.concat([df_fusion, df_init], axis=1)
    df_fusion = pd.concat([df_fusion, df_demographics_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_vision_dense_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_vision_predictions_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_vision_multi_dense_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_vision_multi_predictions_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_ts_ce_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_ts_le_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_ts_pe_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_ecgnotes_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_echonotes_embeddings_fusion], axis=1)
    df_fusion = pd.concat([df_fusion, df_radnotes_embeddings_fusion], axis=1)
    
    #Add targets
    df_fusion = pd.concat([df_fusion, df_vision_targets_fusion], axis=1)
    gc.collect() #Clear memory
    
    return df_fusion


# Clean out process bar before starting
sys.stdout.flush()

# Define inclusion criteria
inclusion_criteria =[[''], ['']]

# Supply the haim id patient you would like to process
haim_id = n
# Supply name of file of which you would like to save all embeddings in
fname = filenmae

# Let's select a single HAIM Patient from pickle files and check if it fits inclusion criteria
haim_patient_idx = haim_id

#Load precomputed file
filename = f"{haim_patient_idx:08d}" + '.pkl'
folder = f"{haim_patient_idx:05d}"[:2] + "/"
patient = load_patient_object(core_mimiciv_path + 'pickle/folder' + folder + filename)

# Get information of chest x-rays conducted within this patiewnt stay
df_cxr = patient.cxr
df_imcxr = patient.imcxr
admittime = patient.admissions.admittime.values[0]
dischtime = patient.admissions.dischtime.values[0]
df_stay_cxr = df_cxr.loc[(df_cxr['charttime'] >= admittime) & (df_cxr['charttime'] <= dischtime)]

if not df_stay_cxr.empty:
    for idx, df_stay_cxr_row in df_stay_cxr.iterrows():
        # Get stay anchor times
        img_charttime = df_stay_cxr_row['charttime']
        img_deltacharttime = df_stay_cxr_row['deltacharttime']

        # Get time to discharge and discharge location/status
        img_id = df_stay_cxr_row["dicom_id"]
        img_length_of_stay = date_diff_hrs(dischtime, img_charttime)
        discharge_location = patient.core['discharge_location'][0]
        if discharge_location == "DIED": death_status = 1
        else: death_status = 0
            
        # Select allowed timestamp range
        start_hr = None
        end_hr = img_deltacharttime
        
        patient = load_patient_object(core_mimiciv_path + 'pickle/folder' + folder + filename)
        dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
        is_included = True

        if is_included:
            df_init = pd.DataFrame([[img_id, img_charttime, img_deltacharttime, discharge_location, img_length_of_stay, death_status]],columns=['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 'death_status'])
            df_fusion = process_cxr_embeddings_haim_id(haim_id, dt_patient, df_init)
            
            if os.path.isfile(fname):
                df_fusion.to_csv(fname, mode='a', index=False, header=False)
            else:
                df_fusion.to_csv(fname, mode='w', index=False)
                