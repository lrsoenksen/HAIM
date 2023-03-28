# %%

###########################################################################################################
#                                  __    __       ___       __  .___  ___. 
#                                 |  |  |  |     /   \     |  | |   \/   | 
#                                 |  |__|  |    /  ^  \    |  | |  \  /  | 
#                                 |   __   |   /  /_\  \   |  | |  |\/|  | 
#                                 |  |  |  |  /  _____  \  |  | |  |  |  | 
#                                 |__|  |__| /__/     \__\ |__| |__|  |__| 
#
#                               HOLISTIC ARTIFICIAL INTELLIGENCE IN MEDICINE
#
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
# -> Changes:
#       * Added embeddings extraction wrappers
#       * Added Code for Patient parsing towards Multi-Input AI/ML to predict value of Next Lab/X-Ray with MIMIC-IV
#       * Add Model helper functions

 
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                            PREREQUISITS                                         |
#                                                                                                 | 

## -> To run this code, we first need to install serveral required packages

#!pip install tensorflow
#!pip install tqdm==4.19.9
#!pip install dask
#!pip install sklearn
#!pip install tsfresh
#!pip install missingno
#!pip install transformers
#!pip install torch==1.0.1
#!pip install torchvision==0.2.2
#!pip install torchxrayvision


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                              IMPORTS                                            |

# System                                                                                           
import os
import sys

# Base
import cv2
import math
import copy
import pickle
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import datetime as dt
import plotly.express as px
import matplotlib.pyplot as plt
import missingno as msno
from tqdm import tqdm
from glob import glob
from shutil import copyfile

from dask import dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

# Core AI/ML
import tensorflow as tf
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
from torch.utils.data import Dataset, DataLoader

# Scipy
from scipy.stats import ks_2samp
from scipy.signal import find_peaks

# Scikit-learn
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

# TSFresh
# from tsfresh import extract_features, select_features
# from tsfresh.utilities.dataframe_functions import impute

#TS (Manual) module  
# from ts_embeddings import *  

# NLP
from transformers import AutoTokenizer, AutoModel, logging
logging.set_verbosity_error()
# biobert_path = '../pretrained_models/bio_clinical_bert/biobert_pretrain_output_all_notes_150000/'
biobert_path = 'pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'
biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
biobert_model = AutoModel.from_pretrained(biobert_path)
# biobert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# biobert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Computer Vision
import cv2
import skimage, skimage.io
import torchxrayvision as xrv

# Deep Fusion for MIMIC-IV

# Warning handling
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                Initializations & Data Loading                                   |
#                                                                                                 | 
"""
Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)
"""

# Define MIMIC IV Data Location
core_mimiciv_path = '../data/HAIM/physionet/files/mimiciv/1.0/'

# Define MIMIC IV Image Data Location (usually external drive)
core_mimiciv_imgcxr_path = '../data/HAIM/physionet/files/mimiciv/1.0/mimic-cxr-jpg/2.0.0.'

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                        Helper functions                                        |
#  



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                     HAIM-MIMICIV specific Patient Representation Function                       |
#                                                                                                 |

# MIMICIV PATIENT CLASS STRUCTURE
class Patient_ICU(object):
    def __init__(self, admissions, demographics, transfers, core,\
        diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents,\
        labevents, microbiologyevents, poe, poe_detail,\
        prescriptions, procedures_icd, services, procedureevents,\
        outputevents, inputevents, icustays, datetimeevents,\
        chartevents, cxr, imcxr, noteevents, dsnotes, ecgnotes, \
        echonotes, radnotes):
        
        ## CORE
        self.admissions = admissions
        self.demographics = demographics
        self.transfers = transfers
        self.core = core
        ## HOSP
        self.diagnoses_icd = diagnoses_icd
        self.drgcodes = drgcodes
        self.emar = emar
        self.emar_detail = emar_detail
        self.hcpcsevents = hcpcsevents
        self.labevents = labevents
        self.microbiologyevents = microbiologyevents
        self.poe = poe
        self.poe_detail = poe_detail
        self.prescriptions = prescriptions
        self.procedures_icd = procedures_icd
        self.services = services
        ## ICU
        self.procedureevents = procedureevents
        self.outputevents = outputevents
        self.inputevents = inputevents
        self.icustays = icustays
        self.datetimeevents = datetimeevents
        self.chartevents = chartevents
        ## CXR
        self.cxr = cxr
        self.imcxr = imcxr
        ## NOTES
        self.noteevents = noteevents
        self.dsnotes = dsnotes
        self.ecgnotes = ecgnotes
        self.echonotes = echonotes
        self.radnotes = radnotes


# GET FULL MIMIC IV PATIENT RECORD USING DATABASE KEYS
def get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_hadm_id    -> hadm_id is unique to a patient hospital stay
    #   key_stay_id    -> stay_id is unique to a patient ward stay
    #   
    #   NOTES: Identifiers which specify the patient. More information about 
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> ICU patient stay structure

    #-> FILTER data
    ##-> CORE
    f_df_base_core = df_base_core[(df_base_core.subject_id == key_subject_id) & (df_base_core.hadm_id == key_hadm_id)]
    f_df_admissions = df_admissions[(df_admissions.subject_id == key_subject_id) & (df_admissions.hadm_id == key_hadm_id)]
    f_df_patients = df_patients[(df_patients.subject_id == key_subject_id)]
    f_df_transfers = df_transfers[(df_transfers.subject_id == key_subject_id) & (df_transfers.hadm_id == key_hadm_id)]
    ###-> Merge data into single patient structure
    f_df_core = f_df_base_core
    f_df_core = f_df_core.merge(f_df_admissions, how='left')
    f_df_core = f_df_core.merge(f_df_patients, how='left')
    f_df_core = f_df_core.merge(f_df_transfers, how='left')

    ##-> HOSP
    f_df_diagnoses_icd = df_diagnoses_icd[(df_diagnoses_icd.subject_id == key_subject_id)]
    f_df_drgcodes = df_drgcodes[(df_drgcodes.subject_id == key_subject_id) & (df_drgcodes.hadm_id == key_hadm_id)]
    f_df_emar = df_emar[(df_emar.subject_id == key_subject_id) & (df_emar.hadm_id == key_hadm_id)]
    f_df_emar_detail = df_emar_detail[(df_emar_detail.subject_id == key_subject_id)]
    f_df_hcpcsevents = df_hcpcsevents[(df_hcpcsevents.subject_id == key_subject_id) & (df_hcpcsevents.hadm_id == key_hadm_id)]
    f_df_labevents = df_labevents[(df_labevents.subject_id == key_subject_id) & (df_labevents.hadm_id == key_hadm_id)]
    f_df_microbiologyevents = df_microbiologyevents[(df_microbiologyevents.subject_id == key_subject_id) & (df_microbiologyevents.hadm_id == key_hadm_id)]
    f_df_poe = df_poe[(df_poe.subject_id == key_subject_id) & (df_poe.hadm_id == key_hadm_id)]
    f_df_poe_detail = df_poe_detail[(df_poe_detail.subject_id == key_subject_id)]
    f_df_prescriptions = df_prescriptions[(df_prescriptions.subject_id == key_subject_id) & (df_prescriptions.hadm_id == key_hadm_id)]
    f_df_procedures_icd = df_procedures_icd[(df_procedures_icd.subject_id == key_subject_id) & (df_procedures_icd.hadm_id == key_hadm_id)]
    f_df_services = df_services[(df_services.subject_id == key_subject_id) & (df_services.hadm_id == key_hadm_id)]
    ###-> Merge content from dictionaries
    f_df_diagnoses_icd = f_df_diagnoses_icd.merge(df_d_icd_diagnoses, how='left') 
    f_df_procedures_icd = f_df_procedures_icd.merge(df_d_icd_procedures, how='left')
    f_df_hcpcsevents = f_df_hcpcsevents.merge(df_d_hcpcs, how='left')
    f_df_labevents = f_df_labevents.merge(df_d_labitems, how='left')

    ##-> ICU
    f_df_procedureevents = df_procedureevents[(df_procedureevents.subject_id == key_subject_id) & (df_procedureevents.hadm_id == key_hadm_id) & (df_procedureevents.stay_id == key_stay_id)]
    f_df_outputevents = df_outputevents[(df_outputevents.subject_id == key_subject_id) & (df_outputevents.hadm_id == key_hadm_id) & (df_outputevents.stay_id == key_stay_id)]
    f_df_inputevents = df_inputevents[(df_inputevents.subject_id == key_subject_id) & (df_inputevents.hadm_id == key_hadm_id) & (df_inputevents.stay_id == key_stay_id)]
    f_df_icustays = df_icustays[(df_icustays.subject_id == key_subject_id) & (df_icustays.hadm_id == key_hadm_id) & (df_icustays.stay_id == key_stay_id)]
    f_df_datetimeevents = df_datetimeevents[(df_datetimeevents.subject_id == key_subject_id) & (df_datetimeevents.hadm_id == key_hadm_id) & (df_datetimeevents.stay_id == key_stay_id)]
    f_df_chartevents = df_chartevents[(df_chartevents.subject_id == key_subject_id) & (df_chartevents.hadm_id == key_hadm_id) & (df_chartevents.stay_id == key_stay_id)]
    ###-> Merge content from dictionaries
    f_df_procedureevents = f_df_procedureevents.merge(df_d_items, how='left')
    f_df_outputevents = f_df_outputevents.merge(df_d_items, how='left')
    f_df_inputevents = f_df_inputevents.merge(df_d_items, how='left')
    f_df_datetimeevents = f_df_datetimeevents.merge(df_d_items, how='left')
    f_df_chartevents = f_df_chartevents.merge(df_d_items, how='left')       

    ##-> CXR
    f_df_mimic_cxr_split = df_mimic_cxr_split[(df_mimic_cxr_split.subject_id == key_subject_id)]
    f_df_mimic_cxr_chexpert = df_mimic_cxr_chexpert[(df_mimic_cxr_chexpert.subject_id == key_subject_id)]
    f_df_mimic_cxr_metadata = df_mimic_cxr_metadata[(df_mimic_cxr_metadata.subject_id == key_subject_id)]
    f_df_mimic_cxr_negbio = df_mimic_cxr_negbio[(df_mimic_cxr_negbio.subject_id == key_subject_id)]
    ###-> Merge data into single patient structure
    f_df_cxr = f_df_mimic_cxr_split
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_chexpert, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_metadata, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_negbio, how='left')
    ###-> Get images of that timebound patient
    f_df_imcxr = []
    for img_idx, img_row in f_df_cxr.iterrows():
        img_path = core_mimiciv_imgcxr_path + str(img_row['Img_Folder']) + '/' + str(img_row['Img_Filename'])
        img_cxr_shape = [224, 224]
        img_cxr = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (img_cxr_shape[0], img_cxr_shape[1]))
        f_df_imcxr.append(np.array(img_cxr))
      
    ##-> NOTES
    f_df_noteevents = df_noteevents[(df_noteevents.subject_id == key_subject_id) & (df_noteevents.hadm_id == key_hadm_id)]
    f_df_dsnotes = df_dsnotes[(df_dsnotes.subject_id == key_subject_id) & (df_dsnotes.hadm_id == key_hadm_id) & (df_dsnotes.stay_id == key_stay_id)]
    f_df_ecgnotes = df_ecgnotes[(df_ecgnotes.subject_id == key_subject_id) & (df_ecgnotes.hadm_id == key_hadm_id) & (df_ecgnotes.stay_id == key_stay_id)]
    f_df_echonotes = df_echonotes[(df_echonotes.subject_id == key_subject_id) & (df_echonotes.hadm_id == key_hadm_id) & (df_echonotes.stay_id == key_stay_id)]
    f_df_radnotes = df_radnotes[(df_radnotes.subject_id == key_subject_id) & (df_radnotes.hadm_id == key_hadm_id) & (df_radnotes.stay_id == key_stay_id)]

    ###-> Merge data into single patient structure
    #--None


    # -> Create & Populate patient structure
    ## CORE
    admissions = f_df_admissions
    demographics = f_df_patients
    transfers = f_df_transfers
    core = f_df_core

    ## HOSP
    diagnoses_icd = f_df_diagnoses_icd
    drgcodes = f_df_diagnoses_icd
    emar = f_df_emar
    emar_detail = f_df_emar_detail
    hcpcsevents = f_df_hcpcsevents
    labevents = f_df_labevents
    microbiologyevents = f_df_microbiologyevents
    poe = f_df_poe
    poe_detail = f_df_poe_detail
    prescriptions = f_df_prescriptions
    procedures_icd = f_df_procedures_icd
    services = f_df_services

    ## ICU
    procedureevents = f_df_procedureevents
    outputevents = f_df_outputevents
    inputevents = f_df_inputevents
    icustays = f_df_icustays
    datetimeevents = f_df_datetimeevents
    chartevents = f_df_chartevents

    ## CXR
    cxr = f_df_cxr 
    imcxr = f_df_imcxr

    ## NOTES
    noteevents = f_df_noteevents
    dsnotes = f_df_dsnotes
    ecgnotes = f_df_ecgnotes
    echonotes = f_df_echonotes
    radnotes = f_df_radnotes


    # Create patient object and return
    Patient_ICUstay = Patient_ICU(admissions, demographics, transfers, core, \
                                  diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents, \
                                  labevents, microbiologyevents, poe, poe_detail, \
                                  prescriptions, procedures_icd, services, procedureevents, \
                                  outputevents, inputevents, icustays, datetimeevents, \
                                  chartevents, cxr, imcxr, noteevents, dsnotes, ecgnotes, \
                                  echonotes, radnotes)

    return Patient_ICUstay


# DELTA TIME CALCULATOR FROM TWO TIMESTAMPS
def date_diff_hrs(t1, t0):
    # Inputs:
    #   t1 -> Final timestamp in a patient hospital stay
    #   t0 -> Initial timestamp in a patient hospital stay

    # Outputs:
    #   delta_t -> Patient stay structure bounded by allowed timestamps

    try:
        delta_t = (t1-t0).total_seconds()/3600 # Result in hrs
    except:
        delta_t = math.nan
    
    return delta_t


# GET TIMEBOUND MIMIC-IV PATIENT RECORD BY DATABASE KEYS AND TIMESTAMPS
def get_timebound_patient_icustay(Patient_ICUstay, start_hr = None, end_hr = None):
    # Inputs:
    #   Patient_ICUstay -> Patient ICU stay structure
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    #   NOTES: Identifiers which specify the patient. More information about 
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    
    # %% EXAMPLE OF USE
    ## Let's select a single patient
    '''
    key_subject_id = 10000032
    key_hadm_id = 29079034
    key_stay_id = 39553978
    start_hr = 0
    end_hr = 24
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    '''
    
    # Create a deep copy so that it is not the same object
    # Patient_ICUstay = copy.deepcopy(Patient_ICUstay)
    
    
    ## --> Process Event Structure Calculations
    admittime = Patient_ICUstay.core['admittime'].values[0]
    dischtime = Patient_ICUstay.core['dischtime'].values[0]
    Patient_ICUstay.emar['deltacharttime'] = Patient_ICUstay.emar.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.labevents['deltacharttime'] = Patient_ICUstay.labevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.microbiologyevents['deltacharttime'] = Patient_ICUstay.microbiologyevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.outputevents['deltacharttime'] = Patient_ICUstay.outputevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.datetimeevents['deltacharttime'] = Patient_ICUstay.datetimeevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.chartevents['deltacharttime'] = Patient_ICUstay.chartevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.noteevents['deltacharttime'] = Patient_ICUstay.noteevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.dsnotes['deltacharttime'] = Patient_ICUstay.dsnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.ecgnotes['deltacharttime'] = Patient_ICUstay.ecgnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.echonotes['deltacharttime'] = Patient_ICUstay.echonotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.radnotes['deltacharttime'] = Patient_ICUstay.radnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    
    # Re-calculate times of CXR database
    Patient_ICUstay.cxr['StudyDateForm'] = pd.to_datetime(Patient_ICUstay.cxr['StudyDate'], format='%Y%m%d')
    Patient_ICUstay.cxr['StudyTimeForm'] = Patient_ICUstay.cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
    Patient_ICUstay.cxr['StudyTimeForm'] = pd.to_datetime(Patient_ICUstay.cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
    Patient_ICUstay.cxr['charttime'] = Patient_ICUstay.cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)
    Patient_ICUstay.cxr['charttime'] = Patient_ICUstay.cxr['charttime'].dt.floor('Min')
    Patient_ICUstay.cxr['deltacharttime'] = Patient_ICUstay.cxr.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    
    ## --> Filter by allowable time stamps
    if not (start_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        #Notes
        Patient_ICUstay.noteevents = Patient_ICUstay.noteevents[(Patient_ICUstay.noteevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.noteevents.deltacharttime)]
        Patient_ICUstay.dsnotes = Patient_ICUstay.dsnotes[(Patient_ICUstay.dsnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.dsnotes.deltacharttime)]
        Patient_ICUstay.ecgnotes = Patient_ICUstay.ecgnotes[(Patient_ICUstay.ecgnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.ecgnotes.deltacharttime)]
        Patient_ICUstay.echonotes = Patient_ICUstay.echonotes[(Patient_ICUstay.echonotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.echonotes.deltacharttime)]
        Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
        
        
    if not (end_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        #Notes
        Patient_ICUstay.noteevents = Patient_ICUstay.noteevents[(Patient_ICUstay.noteevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.noteevents.deltacharttime)]
        Patient_ICUstay.dsnotes = Patient_ICUstay.dsnotes[(Patient_ICUstay.dsnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.dsnotes.deltacharttime)]
        Patient_ICUstay.ecgnotes = Patient_ICUstay.ecgnotes[(Patient_ICUstay.ecgnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.ecgnotes.deltacharttime)]
        Patient_ICUstay.echonotes = Patient_ICUstay.echonotes[(Patient_ICUstay.echonotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.echonotes.deltacharttime)]
        Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
        
        # Filter CXR to match allowable patient stay
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.charttime <= dischtime)]
    
    return Patient_ICUstay


# LOAD MASTER DICTIONARY OF MIMIC IV EVENTS
def load_haim_event_dictionaries(core_mimiciv_path):
    # Inputs:
    #   df_d_items -> MIMIC chartevent items dictionary
    #   df_d_labitems -> MIMIC labevent items dictionary
    #   df_d_hcpcs -> MIMIC hcpcs items dictionary
    #
    # Outputs:
    #   df_patientevents_categorylabels_dict -> Dictionary with all possible event types

    # Generate dictionary for chartevents, labevents and HCPCS
    df_patientevents_categorylabels_dict = pd.DataFrame(columns = ['eventtype', 'category', 'label'])
  
    # Load dictionaries
    df_d_items = pd.read_csv(core_mimiciv_path + 'icu/d_items.csv')
    df_d_labitems = pd.read_csv(core_mimiciv_path + 'hosp/d_labitems.csv')
    df_d_hcpcs = pd.read_csv(core_mimiciv_path + 'hosp/d_hcpcs.csv')

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
            

    return df_patientevents_categorylabels_dict



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                            Data filtering by condition and outcome                              |
#                                                                                                 | 
"""
Resources to identify tables and variables of interest can be found in the MIMIC-IV official API 
(https://mimic-iv.mit.edu/docs/)
"""

# QUERY IN ALL SINGLE PATIENT ICU STAY RECORD FOR KEYWORD MATCHING
def is_haim_patient_keyword_match(patient, keywords, verbose = 0):
    # Inputs:
    #   patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   keywords -> List of string keywords to attempt to match in an "OR" basis
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   is_key -> Boolean flag indicating if any of the input Keywords are present
    #   keyword_mask -> Array indicating which of the input Keywords are present (0-Absent, 1-Present)
  
    # Retrieve list of all the contents of patient datastructures
    patient_dfs_list = [## CORE
                        patient.core,
                        ## HOSP
                        patient.diagnoses_icd,
                        patient.drgcodes,
                        patient.emar,
                        patient.emar_detail,
                        patient.hcpcsevents,
                        patient.labevents,
                        patient.microbiologyevents,
                        patient.poe,
                        patient.poe_detail,
                        patient.prescriptions,
                        patient.procedures_icd,
                        patient.services,
                        ## ICU
                        patient.procedureevents,
                        patient.outputevents,
                        patient.inputevents,
                        patient.icustays,
                        patient.datetimeevents,
                        patient.chartevents,
                        ## CXR
                        patient.cxr,
                        patient.imcxr,
                        ## NOTES
                        patient.noteevents,
                        patient.dsnotes,
                        patient.ecgnotes,
                        patient.echonotes,
                        patient.radnotes
                        ]

    patient_dfs_dict = ['core', 'diagnoses_icd', 'drgcodes', 'emar', 'emar_detail', 'hcpcsevents', 'labevents', 'microbiologyevents', 'poe',
                        'poe_detail', 'prescriptions', 'procedures_icd', 'services', 'procedureevents', 'outputevents', 'inputevents', 'icustays',
                        'datetimeevents', 'chartevents', 'cxr', 'imcxr', 'noteevents', 'dsnotes', 'ecgnotes', 'echonotes', 'radnotes']
  
    # Initialize query mask
    keyword_mask = np.zeros([len(patient_dfs_list), len(keywords)])
    for idx_df, patient_df in enumerate(patient_dfs_list):
        for idx_keyword, keyword in enumerate(keywords):
            try:
                patient_df_text = patient_df.astype(str)
                is_df_key = patient_df_text.sum(axis=1).str.contains(keyword, case=False).any()

                if is_df_key:
                    keyword_mask[idx_df, idx_keyword]=1
                    if (verbose >= 2):
                        print('')
                        print('Keyword: ' + '"' + keyword + ' " ' +  '(Found in "' + patient_dfs_dict[idx_df] + '" table )')
                        print(patient_df_text)
                else:
                    keyword_mask[idx_df, idx_keyword]=0
              
            except:
                is_df_key = False
                keyword_mask[idx_df, idx_keyword]=0

    # Create final keyword mask
    if keyword_mask.any():
        is_key = True
    else:
        is_key = False
    
    return is_key, keyword_mask


# QUERY IN ALL SINGLE PATIENT ICU STAY RECORD FOR INCLUSION CRITERIA MATCHING
def is_haim_patient_inclusion_criteria_match(patient, inclusion_criteria, verbose = 0):
    # Inputs:
    #   patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   inclusion_criteria -> Inclusion criteria in groups of keywords. 
    #                         Keywords in groups are follow and "OR" logic,
    #                         while an "AND" logic is stablished among groups
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   is_included -> Boolean flag if inclusion criteria is found in patient
    #   inclusion_criteria_mask -> Binary mask of inclusion criteria found in patient
  
    # Clean out process bar before starting
    inclusion_criteria_mask = np.zeros(len(inclusion_criteria))
    for idx_keywords, keywords in enumerate(inclusion_criteria):
        is_included_flag, _ = is_haim_patient_keyword_match(patient, keywords, verbose)
        inclusion_criteria_mask[idx_keywords] = is_included_flag
    
    if inclusion_criteria_mask.all():
        is_included = True
    else:
        is_included = False

    # Print if patient has to be included
    if (verbose >=2):
        print('')
        print('Inclusion Criteria: ' + str(inclusion_criteria))
        print('Inclusion Vector: ' + str(inclusion_criteria_mask) + ' , To include: ' + str(is_included))
    
    return is_included, inclusion_criteria_mask



# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
def search_key_mimiciv_patients(df_haim_ids, core_mimiciv_path, inclusion_criteria, verbose = 0):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   core_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #
    # Outputs:
    #   nfiles -> Number of single patient HAIM files produced

    # Clean out process bar before starting
    sys.stdout.flush()

    # List of key patients
    key_haim_patient_ids = []

    # Extract information for patient
    nfiles = len(df_haim_ids)
    with tqdm(total = nfiles) as pbar:
        # Update process bar
        nbase= 0
        pbar.update(nbase)
        #Iterate through all patients
        for haim_patient_idx in range(nbase, nfiles):
            #Load precomputed patient file
            filename = f"{haim_patient_idx:08d}" + '.pkl'
            patient = load_patient_object(core_mimiciv_path + 'pickle/' + filename)
            #Check if patient fits keywords
            is_key, _ = is_haim_patient_inclusion_criteria_match(patient, keywords, verbose)
            if is_key:
                key_haim_patient_ids.append(haim_patient_idx)

            # Update process bar
            pbar.update(1)

    return key_haim_patient_ids



# GET MIMIC IV PATIENT LIST FILTERED BY DESIRED CONDITION
def get_haim_ids_only_by_condition(condition_tokens, core_mimiciv_path):
    # Inputs:
    #   condition_tokens     -> string identifier of the condition you want to isolate (condition_tokens= ['heart failure','chronic'])
    #   outcome_tokens       -> string identifier of the outcome you want to isolate
    #   core_mimiciv_path    -> path to folder where the base MIMIC IV dataset files are located
  
    # Outputs:
    #   condition_outcome_df -> Dataframe including patients IDs with desired Condition, indicating the Outcome.
  
    # Load necessary ICD diagnostic lists and general patient information
    d_icd_diagnoses = pd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv')
    d_icd_diagnoses['long_title'] = d_icd_diagnoses['long_title'].str.lower()
    diagnoses_icd['icd_code'] = diagnoses_icd['icd_code'].str.replace(' ', '')
  
    admissions = pd.read_csv(core_mimiciv_path + 'core/admissions.csv')
    patients = pd.read_csv(core_mimiciv_path + 'core/patients.csv')
    admissions = pd.merge(admissions, patients, on = 'subject_id')
  
    #list of unique hadm id with conditions specified
    condition_list = []
    condition_list = d_icd_diagnoses[d_icd_diagnoses['long_title'].str.contains(condition_keywords[0])]
    for i in condition_keywords[1:]:
        condition_list = condition_list[condition_list['long_title'].str.contains('chronic')]
      
    icd_list = condition_list['icd_code'].unique().tolist() 
    hid_list_chf = diagnoses_icd[diagnoses_icd['icd_code'].isin(icd_list) & 
                  (diagnoses_icd['seq_num']<=3)]['hadm_id'].unique().tolist()
  
    pkl_id = pd.read_csv(core_mimiciv_path + 'pickle/haim_mimiciv_key_ids.csv')
    id_hf = pkl_id[pkl_id['hadm_id'].isin(hid_list_chf)].drop_duplicates(subset='hadm_id')
  
    # delete all pkl files with only less than 1 day recorded
    pkl_list_adm = admissions[admissions['hadm_id'].isin(id_hf['hadm_id'])]
    pkl_list_adm['dischtime'] = pd.to_datetime(pkl_list_adm['dischtime'])
    pkl_list_adm['admittime'] = pd.to_datetime(pkl_list_adm['admittime'])
    pkl_list_adm['deltatime'] = (pkl_list_adm['dischtime'] - pkl_list_adm['admittime']).astype('timedelta64[D]').values
    pkl_no_zero = pkl_list_adm[pkl_list_adm['deltatime'] != 0]['hadm_id']
    no_zero_id = pkl_id[pkl_id['hadm_id'].isin(pkl_no_zero)].drop_duplicates(subset='hadm_id')
  
    haim_ids_list = no_zero_id['haim_id_pickle'].values
  
    return haim_ids_list



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                             Core embeddings for MIMIC-IV Deep Fusion                        |
#   

'''
'''
# LOAD CORE INFO OF MIMIC IV PATIENTS
def load_core_mimic_haim_info(core_mimiciv_path, df_haim_ids):
    # Inputs:
    #   core_mimiciv_path -> Base path of mimiciv
    #   df_haim_ids -> Table of HAIM ids and corresponding keys
    #
    # Outputs:
    #   df_haim_ids_core_info -> Updated dataframe with integer representations of core data

    # %% EXAMPLE OF USE
    # df_haim_ids_core_info = load_core_mimic_haim_info(core_mimiciv_path)

    # Load core table
    df_mimiciv_core = pd.read_csv(core_mimiciv_path + 'core/core.csv')

    # Generate integer representations of categorical variables in core
    core_var_select_list = ['gender', 'ethnicity', 'marital_status', 'language','insurance']
    core_var_select_int_list = ['gender_int', 'ethnicity_int', 'marital_status_int', 'language_int','insurance_int']
    df_mimiciv_core[core_var_select_list] = df_mimiciv_core[core_var_select_list].astype('category')
    df_mimiciv_core[core_var_select_int_list] = df_mimiciv_core[core_var_select_list].apply(lambda x: x.cat.codes)

    # Combine HAIM IDs with core data
    df_haim_ids_core_info = pd.merge(df_haim_ids, df_mimiciv_core, on=["subject_id", "hadm_id"])

    return df_haim_ids_core_info


# GET DEMOGRAPHICS EMBEDDINGS OF MIMIC IV PATIENT
def get_demographic_embeddings(dt_patient, verbose=0):
    # Inputs:
    #   dt_patient -> Timebound mimic patient structure
    #   verbose -> Flag to print found keyword outputs (0,1,2)
    #
    # Outputs:
    #   base_embeddings -> Core base embeddings for the selected patient

    # %% EXAMPLE OF USE
    # base_embeddings = get_demographic_embeddings(dt_patient, df_haim_ids_core_info, verbose=2)

    # Retrieve dt_patient and get embeddings 
    demo_embeddings =  dt_patient.core.loc[0, ['anchor_age', 'gender_int', 'ethnicity_int', 'marital_status_int', 'language_int', 'insurance_int']]

    if verbose >= 1:
        print(demo_embeddings)

    demo_embeddings = demo_embeddings.values

    return demo_embeddings
  

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                             TSFresh embeddings for MIMIC-IV Deep Fusion                        |
#   

'''
## -> DEEP FUSION REPRESENTATION OF MIMIC-IV EHR USING TSFRESH
      "tsfresh" is a python package. It automatically calculates a large number of time series 
      characteristics, the so called features. Further the package contains methods to evaluate 
      the explaining power and importance of such characteristics for regression or classification 
      tasks. https://tsfresh.readthedocs.io/en/latest/
'''

# TSFRESH FEATURE EXTRACTOR OF MIMICIV CHARTEVENTS
def get_chartevent_tsfresh_timeseries_embeddings(dt_patient, df_patientevents_categorylabels_dict, verbose=0):
    # Inputs:
    #   dt_patient -> Timebound Patient ICU stay structure
    #   df_patientevents_categorylabels_dict -> MIMIC IV Event dictionary
    #   verbose -> Flag to print generated outputs (0,1,2)
    #
    # Outputs:
    #   evs_features -> TSfresh generated chart event features for each timeseries
    
    # %% EXAMPLE OF USE
    # evs_features = extract_chartevent_tsfresh_timeseries_embeddings(dt_patient, df_patientevents_categorylabels_dict, event_type, verbose=1)
    
    # Stablish dynamics of progressbar
    if verbose <= 1: disable_progressbar=True
    else: disable_progressbar=False
    
    # Prep features of empty timeseries features from TSFresh in the context of clinical data
    fc_parameters = {"length": None,
                    "absolute_sum_of_changes": None, 
                    "maximum": None, 
                    "mean": None,
                    "mean_abs_change": None,
                    "mean_change": None,
                    "median": None,
                    "minimum": None,
                    "standard_deviation": None,
                    "variance": None,
                    "large_standard_deviation": [{"r": r * 0.2} for r in range(1, 5)],
                     
                     # Comment by Yu: don't think we need the 1 for quntile?
                    "quantile": [{"q": q} for q in [.25, .5, .75, 1]],
                    "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},{"attr": "slope"}, {"attr": "stderr"}]}
  
    x_hr =[0]
    y_hr = np.arange(len(x_hr))
  
    # Extract Features with TSFresh
    timeseries = pd.DataFrame({'id': np.zeros_like(y_hr), 'valnum': y_hr, 'time': x_hr}, columns=['id', 'valnum', 'time'])
    features_empty_timeseries = extract_features(timeseries, column_id="id", column_sort="time", disable_progressbar=True, default_fc_parameters=fc_parameters)
    
    
    #Get patient events by event type
    evs = dt_patient.chartevents
    
    #List all types of chart events (Charts, Labs and signals)
    for eventtype_idx, eventtype in enumerate(sorted((df_patientevents_categorylabels_dict.eventtype.unique()))):
        if verbose >= 3: print('* ' + eventtype)
        event_list = df_patientevents_categorylabels_dict[df_patientevents_categorylabels_dict['eventtype']==eventtype]
        for category_idx, category in enumerate(sorted((df_patientevents_categorylabels_dict.category.unique()))):
            if verbose >= 3: print('-> ' + category)
            category_list = df_patientevents_categorylabels_dict[df_patientevents_categorylabels_dict['category']==category]
            for item_idx, item in enumerate(sorted(category_list.label.unique())):
                if verbose >= 3: print('---> ' + item) 
                
                # POPULATE FEATURE SPACE FOR PATIENT
                item_chart = evs[evs['label']==item]
                empty_timeseries = False
                # Set x equal to the times
                x_hr = item_chart.deltacharttime[item_chart.label==item]
                if len(x_hr)==0: 
                    x_hr =[0]
                    empty_timeseries = True
                    
                y_hr = item_chart.valuenum[item_chart.label==item]
                y_hr = y_hr[~(np.isnan(y_hr))]
                x_hr = x_hr[0:len(y_hr)]
                if y_hr.empty: 
                    y_hr = np.arange(len(x_hr))
                    extracted_features = features_empty_timeseries
                else:                    
                    # Extract Features with TSFresh
                    timeseries = pd.DataFrame({'id': np.zeros_like(y_hr), 'valnum': y_hr, 'time': x_hr}, columns=['id', 'valnum', 'time'])
                    extracted_features = impute(extract_features(timeseries, column_id="id", column_sort="time", disable_progressbar=disable_progressbar, default_fc_parameters = fc_parameters))
                    
                if (eventtype_idx ==0) & (category_idx ==0) & (item_idx == 0):
                    evs_features = extracted_features
                else:
                    evs_features = evs_features.append(extracted_features) 
                    
    # Transform extracted features from 0-1
    transformer = QuantileTransformer().fit(evs_features)
    norm_evs_features = transformer.transform(evs_features)
    norm_evs_features = np.asarray(norm_evs_features)
    
    if verbose >= 1:
        # Plot feature representation
        plt.figure(figsize = (20,5))
        plt.imshow(X, cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(label="Patient Timeseries Features", orientation="vertical")
        plt.show()
        
    return norm_evs_features, evs_features


def pivot_chartevent(df, event_list):
    # create a new table with additional columns with label list  
    df1 = df[['subject_id', 'hadm_id', 'stay_id', 'charttime']] 
    for event in event_list: 
        df1[event] = np.nan
         #search in the abbreviations column  
        df1.loc[(df['label']==event), event] = df['valuenum'].astype(float)
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out 

def pivot_labevent(df, event_list):
    # create a new table with additional columns with label list  
    df1 = df[['subject_id', 'hadm_id',  'charttime']] 
    for event in event_list: 
        df1[event] = np.nan
        #search in the label column 
        df1.loc[(df['label']==event), event] = df['valuenum'].astype(float) 
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out 

def pivot_procedureevent(df, event_list):
    # create a new table with additional columns with label list  
    df1 = df[['subject_id', 'hadm_id',  'storetime']] 
    for event in event_list: 
        df1[event] = np.nan
        #search in the label column 
        df1.loc[(df['label']==event), event] = df['value'].astype(float)  #Yu: maybe if not label use abbreviation 
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out 


#FUNCTION TO COMPUTE A LIST OF TIME SERIES FEATURES
def get_ts_emb(df_pivot, event_list):
    # Inputs:
    #   df_pivot -> Pivoted table
    #   event_list -> MIMIC IV Type of Event
    #
    # Outputs:
    #   df_out -> Embeddings
    
    # %% EXAMPLE OF USE
    # df_out = get_ts_emb(df_pivot, event_list)
    
    # Initialize table
    try:
        df_out = df_pivot[['subject_id', 'hadm_id']].iloc[0]
    except:
#         print(df_pivot)
        df_out = pd.DataFrame(columns = ['subject_id', 'hadm_id'])
#         df_out = df_pivot[['subject_id', 'hadm_id']]
        
     #Adding a row of zeros to df_pivot in case there is no value
    df_pivot = df_pivot.append(pd.Series(0, index=df_pivot.columns), ignore_index=True)
    
    #Compute the following features
    for event in event_list:
        series = df_pivot[event].dropna() #dropna rows
        if len(series) >0: #if there is any event
            df_out[event+'_max'] = series.max()
            df_out[event+'_min'] = series.min()
            df_out[event+'_mean'] = series.mean(skipna=True)
            df_out[event+'_variance'] = series.var(skipna=True)
            df_out[event+'_meandiff'] = series.diff().mean() #average change
            df_out[event+'_meanabsdiff'] =series.diff().abs().mean()
            df_out[event+'_maxdiff'] = series.diff().abs().max()
            df_out[event+'_sumabsdiff'] =series.diff().abs().sum()
            df_out[event+'_diff'] = series.iloc[-1]-series.iloc[0]
            #Compute the n_peaks
            peaks,_ = find_peaks(series) #, threshold=series.median()
            df_out[event+'_npeaks'] = len(peaks)
            #Compute the trend (linear slope)
            if len(series)>1:
                df_out[event+'_trend']= np.polyfit(np.arange(len(series)), series, 1)[0] #fit deg-1 poly
            else:
                 df_out[event+'_trend'] = 0
    return df_out


def get_ts_embeddings(dt_patient, event_type):
    # Inputs:
    #   dt_patient -> Timebound Patient ICU stay structure
    #
    # Outputs:
    #   ts_emb -> TSfresh-like generated Lab event features for each timeseries
    #
    # %% EXAMPLE OF USE
    # ts_emb = get_labevent_ts_embeddings(dt_patient)
    
    #Get chartevents
    
    if(event_type == 'procedure'):
        df = dt_patient.procedureevents
        #Define chart events of interest
        event_list = ['Foley Catheter', 'PICC Line', 'Intubation', 'Peritoneal Dialysis', 
                            'Bronchoscopy', 'EEG', 'Dialysis - CRRT', 'Dialysis Catheter', 
                            'Chest Tube Removed', 'Hemodialysis']
        df_pivot = pivot_procedureevent(df, event_list)
        
    elif(event_type == 'lab'):
        df = dt_patient.labevents
        #Define chart events of interest
        event_list = ['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
           'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
           'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
           'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW', 
                      'Platelet Count', 'Neutrophils', 'Vancomycin']
        df_pivot = pivot_labevent(df, event_list)
        
    elif(event_type == 'chart'):
        df = dt_patient.chartevents
        #Define chart events of interest
        event_list = ['Heart Rate','Non Invasive Blood Pressure systolic',
                    'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean', 
                    'Respiratory Rate','O2 saturation pulseoxymetry', 
                    'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response'] 
        df_pivot = pivot_chartevent(df, event_list)
    
    #Pivote df to record these values
    
    ts_emb = get_ts_emb(df_pivot, event_list)
    try:
        ts_emb = ts_emb.drop(['subject_id', 'hadm_id']).fillna(value=0)
    except:
        ts_emb = pd.Series(0, index=ts_emb.columns).drop(['subject_id', 'hadm_id']).fillna(value=0)

    return ts_emb

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                     Biobert Chart Event embeddings for MIMIC-IV Deep Fusion                     |
#   

'''
## -> NLP REPRESENTATION OF MIMIC-IV EHR USING TRANSFORMERS
The Transformers era originally started from the work of [(Vaswani & al., 2017)](https://arxiv.org/abs/1706.03762) who
demonstrated its superiority over [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network)
on translation tasks but it quickly extended to almost all the tasks RNNs were State-of-the-Art at that time.

One advantage of Transformer over its RNN counterpart was its non sequential attention model. Remember, the RNNs had to
iterate over each element of the input sequence one-by-one and carry an "updatable-state" between each hop. With Transformer, the model is able to look at every position in the sequence, at the same time, in one operation.

For a deep-dive into the Transformer architecture, [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder-and-decoder-stacks) 
will drive you along all the details of the paper.
    
![transformer-encoder-decoder](https://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png)
'''

# CONVERT SINGLE CHART EVENT DURING ICU STAY TO STRING
def chart_event_to_string(chart_event):
    # Inputs:
    #   chart_event -> Chart_event in the form of a dataframe row
    #
    # Outputs:
    #   chart_event_string -> String of chart event
    #   deltacharttime -> Time of chart event from admission
    
    # %% EXAMPLE OF USE
    # event_string, deltacharttime = chart_event_to_string(chart_event)
    
    deltacharttime = str(round(chart_event['deltacharttime'].values[0],3))
    category = str(chart_event['category'].values[0])
    itemid = int(chart_event['itemid'].values[0])
    label = str(chart_event['label'].values[0])
    value = str(chart_event['value'].values[0])
    units = '' if str(chart_event['valueuom'].values[0]) == 'NaN' else str(chart_event['valueuom'].values[0])
    warning = ', Warning: outside normal' if int(chart_event['warning'].values[0]) > 0 else ''
    rangeval = '' if str(chart_event['lownormalvalue'].values[0]) == 'nan' else 'range: [' + str(chart_event['lownormalvalue'].values[0]) + ' - ' + '' if str(chart_event['highnormalvalue'].values[0]) == 'nan' else str(chart_event['highnormalvalue'].values[0]) + ']'
    
    chart_event_string = label + ': ' + value + ' ' + units + warning + rangeval
    chart_event_string =  chart_event_string.replace('nan','').replace('NaN','')
    
    return chart_event_string, deltacharttime


# CONVERT SINGLE LAB EVENT DURING ICU STAY TO STRING
def lab_event_to_string(lab_event):
    # Inputs:
    #   lab_event -> Lab_event in the form of a dataframe row
    #
    # Outputs:
    #   lab_event_string -> String of lab event
    #   deltacharttime -> Time of chart event from admission
    
    # %% EXAMPLE OF USE
    # lab_event_string, deltacharttime = lab_event_to_string(lab_event)
    
    deltacharttime = str(round(lab_event['deltacharttime'].values[0],3))
    category = str(lab_event['category'].values[0])
    itemid = int(lab_event['itemid'].values[0])
    label = str(lab_event['label'].values[0])
    value = str(lab_event['value'].values[0])
    units = '' if str(lab_event['valueuom'].values[0]) == 'NaN' else str(lab_event['valueuom'].values[0])
    warning = ', Warning: outside normal' if not pd.isna(lab_event['flag'].values[0]) else ''
    # rangeval = '' if str(lab_event['ref_range_lower'].values[0]) == 'nan' else 'range: [' + str(lab_event['ref_range_lower'].values[0]) + ' - ' + '' if str(lab_event['ref_range_upper'].values[0]) == 'nan' else str(lab_event['ref_range_upper'].values[0]) + ']'
    lab_event_string = label + ': ' + value + ' ' + units + warning

    return lab_event_string, deltacharttime


# CONVERT SINGLE PRESCRIPTION EVENT DURING ICU STAY TO STRING
def prescription_event_to_string(prescription_event):
    # Inputs:
    #   prescription_event -> prescription_event in the form of a dataframe row
    #
    # Outputs:
    #   prescription_event_string -> String of prescription event
    #   deltacharttime -> Time of chart event from admission
    
    # %% EXAMPLE OF USE
    # prescription_event_string, deltacharttime = prescription_event_to_string(prescription_event)
    
    deltacharttime = str(round(prescription_event['deltacharttime'].values[0],3))
    label = str(prescription_event['drug'].values[0])
    value = str(prescription_event['dose_val_rx'].values[0])
    units = '' if str(prescription_event['dose_unit_rx'].values[0]) == 'NaN' else str(prescription_event['dose_unit_rx'].values[0])
    prescription_event_string = label + ': ' + value + ' ' + units

    return prescription_event_string, deltacharttime


# OBTAIN LIST OF ALL EVENTS FROM CHART OF TIMEBOUND PATIENT DURING ICU STAY
def get_events_list(dt_patient, event_type, verbose):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   event_type -> Event type string
    #   verbose ->  Visualization setting for printed outputs
    #
    # Outputs:
    #   full_events_list -> List of all chart events of a single timebound patient
    #   chart_weights -> Weights of all chart events by time of occurance
    
    # %% EXAMPLE OF USE
    # full_events_list, event_weights = get_events_list(dt_patient, event_type, verbose)
    
    full_events_list = []
    event_weights = []
    
    if event_type == 'chartevents':
        events = dt_patient.chartevents
    elif event_type == 'labevents':
        events = dt_patient.labevents
    elif event_type == 'prescriptions':
        events = dt_patient.prescriptions
        # Get proxi for deltachartime in prescriptions (Stop date - admition)
        admittime = dt_patient.core['admittime'][0]
        dt_patient.prescriptions['deltacharttime'] = dt_patient.prescriptions.apply(lambda x: date_diff_hrs(x['stoptime'],admittime) if not x.empty else None, axis=1)
        
    #Sort events
    events = events.sort_values(by=['deltacharttime'])
    
    for idx in range(len(events)):
        #Gather chart event data
        event = events.iloc[[idx]]
        if event_type == 'chartevents':
            event_string, deltacharttime = chart_event_to_string(event)
        elif event_type == 'labevents':
            event_string, deltacharttime = lab_event_to_string(event)
        elif event_type == 'prescriptions':
            event_string, deltacharttime = prescription_event_to_string(event)
            
        if verbose>=3: print(event_string)
        
        if idx==0: 
            full_events_list = [event_string]
            event_weights = [float(deltacharttime)]
        else: 
            full_events_list.append(event_string)
            event_weights.append(float(deltacharttime))
            
    return full_events_list, event_weights



# OBTAIN BIOBERT EMBEDDINGS OF TEXT STRING
def get_biobert_embeddings(text):
    # Inputs:
    #   text -> Input text (str)
    #
    # Outputs:
    #   embeddings -> Final Biobert embeddings with vector dimensionality = (1,768)
    #   hidden_embeddings -> Last hidden layer in Biobert model with vector dimensionality = (token_size,768)
  
    # %% EXAMPLE OF USE
    # embeddings, hidden_embeddings = get_biobert_embeddings(text)
  
    tokens_pt = biobert_tokenizer(text, return_tensors="pt")
    outputs = biobert_model(**tokens_pt)
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    hidden_embeddings = last_hidden_state.detach().numpy()
    embeddings = pooler_output.detach().numpy()

    return embeddings, hidden_embeddings



# OBTAIN FIXED-SIZED BIOBERT EMBEDDINGS FOR ALL EVENTS OF A SINGLE TIMEBOUND PATIENT ICU STAY BY ANALYZING THE CHART LINE BY LINE
def get_lined_events_biobert_embeddings(dt_patient, event_type, verbose = 0):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   verbose -> Level of printed output of function
    #
    # Outputs:
    #   aggregated_embeddings -> Biobert event features for all events
    #   full_embeddings -> Biobert event features across each event line
    #   event_weights -> Used weights for aggregation of features in final embeddings
  
    # %% EXAMPLE OF USE
    # aggregated_embeddings, full_embeddings, event_weights = get_biobert_embeddings_from_events(dt_patient, 'chartevents', verbose=1)
  
    # Import BioBERT from local path
    #biobert_path = '../pretrained_models/bio_clinical_bert/biobert_pretrain_output_all_notes_150000/'
    #biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
    #biobert_model = AutoModel.from_pretrained(biobert_path)
  
    # Get all chart events of a single patient
    assert event_type in ['chartevents','labevents','prescriptions'], "Unsupported event type: %s" % event_type
    full_events_list, event_weights = get_events_list(dt_patient, event_type, verbose)
    
    #Normalize event_weights
    orig_event_weights = np.asarray(event_weights)
    adj_event_weights = orig_event_weights - orig_event_weights.min()
    event_weights = (adj_event_weights) / (adj_event_weights).max()
  
    # Process null biobert embeddings (to adjust)
    text = ''
    null_embeddings, null_hidden_embeddings = get_biobert_embeddings(text)
  
    # Initialize processbar
    if (verbose >= 2):
        sys.stdout.flush()
        pbar = tqdm(total=len(full_events_list))
      
    # Init embeddings
    full_embeddings, full_hidden_embeddings = get_biobert_embeddings('')
    
    # Process full chart embeddings of a single patient
    for idx, event_string in enumerate(full_events_list):
        #Extract biobert embeddings
        embeddings, hidden_embeddings = get_biobert_embeddings(event_string)
        #Normalize
        embeddings = embeddings - null_embeddings
        #Concatenate
        if idx==0:
            full_embeddings = embeddings
        else: 
            full_embeddings = np.concatenate((full_embeddings, embeddings), axis=0)
        # Update processbar
        if (verbose >= 2):
            pbar.update(1)
      
    # Return the weighted average of ebedding vector across temporal dimension
    try:
        aggregated_embeddings = np.average(full_embeddings, axis=0, weights=np.array(event_weights))
    except:
        aggregated_embeddings = null_embeddings
        
    aggregated_embeddings = aggregated_embeddings.numpy()
    
    # Close processbar    
    if (verbose >= 2):
        pbar.close()
    
    if (verbose >= 1):
        features = aggregated_embeddings.squeeze()
        print('Hidden State embeddings Shape:' + str(full_embeddings.shape) + ', Pooled Vector embeddings: ' + str(len(full_embeddings)))
        plt.bar(np.arange(0, len(features)), features, color='mediumaquamarine', width=0.3)
        plt.title('Feature Vector of Patient Event Chart')
    
    return aggregated_embeddings, full_embeddings, event_weights




# OBTAIN FIXED-SIZED BIOBERT EMBEDDINGS FOR ALL CHART EVENTS OF A SINGLE TIMEBOUND PATIENT ICU STAY WITH A LONG WINDOW
def get_events_biobert_embeddings(dt_patient, event_type, verbose=0):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   event_type -> Type of event to get embeddings from (e.g. event_type = 'chartevents', 'labevents' or 'prescriptions')
    #   verbose -> Level of printed output of function
    #
    # Outputs:
    #   aggregated_embeddings -> Biobert event features for all events
    #   full_embeddings -> Biobert event features across each event chunk
    #   event_weights -> Used weights for aggregation of features in final embeddings
    
    # %% EXAMPLE OF USE
    # aggregated_chart_embeddings, full_chart_embeddings, chart_weights = get_events_biobert_embeddings(dt_patient, verbose=1)
  
  
    # Get all selected events of a single patient
    assert event_type in ['chartevents','labevents','prescriptions'], "Unsupported event type: %s" % event_type
    full_events_list, event_weights = get_events_list(dt_patient, event_type, verbose)
    text = '\n'.join(full_events_list)
  
    # Tokenize this longer piece of text
    tokens = biobert_tokenizer.encode_plus(text, add_special_tokens=False)
    # Break our tokenized dictionary into input_ids and attention_mask variables.
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
  
    # initialize probabilities list
    probs_list = []
  
    start = 0
    window_size = 512  # Define core window size
    window_size = window_size -2  # we take 2 off here so that we can fit in our [CLS] and [SEP] tokens
  
    # get the total length of our tokens
    total_len = len(input_ids)
    loop = True
  
    # Initialize processbar
    if (verbose >= 2):
        sys.stdout.flush()
        pbar = tqdm(total = math.ceil(total_len/window_size))
    
    # Process full embeddings of selected event type of a single patient  
    while loop:
        end = start + window_size
        if end >= total_len:
            loop = False
            end = total_len
        # (1) extract window from input_ids and attention_mask
        input_ids_chunk = input_ids[start:end]
        attention_mask_chunk = attention_mask[start:end]
        # (2) add [CLS] and [SEP]
        input_ids_chunk = [101] + input_ids_chunk + [102]
        attention_mask_chunk = [1] + attention_mask_chunk + [1]
        # (3) add padding upto window_size + 2 (512) tokens
        input_ids_chunk += [0] * (window_size - len(input_ids_chunk) + 2)
        attention_mask_chunk += [0] * (window_size - len(attention_mask_chunk) + 2)
        # (4) format into PyTorch tensors dictionary
        input_dict = {'input_ids': torch.Tensor([input_ids_chunk]).long(),
                    'attention_mask': torch.Tensor([attention_mask_chunk]).int()}
        # (5) make logits prediction
        outputs = biobert_model(**input_dict)
        # (6) calculate softmax and append to list
        probs = torch.nn.functional.softmax(outputs[0], dim=-1)
        probs_list.append(probs)
        # (7) Concatenate hidden embeddings
        if start==0:
            full_embeddings = probs.detach().numpy()
        else: 
            full_embeddings = np.concatenate((full_embeddings, probs.detach().numpy()), axis=0)
        # (8) prep for next loop and update processbar
        start = end

        # Update processbar
        if (verbose >= 2):
            pbar.update(1)
        
    # Return the weighted average of ebedding vector across temporal dimension
    try:
        # calculate the average vector across the full text
        with torch.no_grad():
            # we must include our stacks operation in here too
            biobert_stacks = torch.stack(probs_list)
            # now resize
            biobert_stacks = biobert_stacks.resize_(biobert_stacks.shape[0], biobert_stacks.shape[2])
            # finally, we can calculate the mean value for each sentiment class
            aggregated_embeddings = biobert_stacks.mean(dim=2)
            # Convert tensor to array
            aggregated_embeddings = aggregated_embeddings.numpy()

    except:
        # Process null biobert embeddings (to adjust)
        text = ''
        null_embeddings, _ = get_biobert_embeddings(text)
        aggregated_embeddings = null_embeddings
    
    # Close processbar    
    if (verbose >= 2):
        pbar.close()    
      
    if (verbose >= 1):
        features = aggregated_embeddings.squeeze()
        print('Hidden State embeddings Shape:' + str(full_embeddings.shape) + ', Pooled Vector embeddings: ' + str(len(full_embeddings)))
        plt.bar(np.arange(0, len(features)), features, color='mediumaquamarine', width=0.3)
        plt.title('Feature Vector of Patient Event Chart')
      
    return aggregated_embeddings, full_embeddings, event_weights
  

# FOR NOTES OBTAIN FIXED-SIZED BIOBERT EMBEDDINGS FOR ALL NOTE EVENTS OF A SINGLE TIMEBOUND PATIENT ICU STAY
def get_biobert_embedding_from_events_list(full_events_list, event_weights, verbose = 0):
    # Inputs:
    #   full_events_list -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   event_weights ->  Weights for aggregation of features in final embeddings
    #   verbose -> Level of printed output of function
    #
    # Outputs:
    #   aggregated_embeddings -> Biobert event features for all events
    #   full_embeddings -> Biobert event features across each event line
    #   event_weights -> Finally used weights for aggregation of features in final embeddings
  
    # %% EXAMPLE OF USE
    # aggregated_embeddings, full_embeddings, event_weights = get_biobert_embedding_from_events_list(full_events_list, event_weights, verbose=1)
  
    event_weights_exp = []
    for idx, event_string in enumerate(full_events_list):   
        weight = event_weights.values[idx]
        string_list, lengths = split_note_document(event_string)
        for idx_sub, event_string_sub in enumerate(string_list):
            #Extract biobert embedding
            embedding, hidden_embedding = get_biobert_embeddings(event_string_sub)
            #Concatenate
            if (idx==0) & (idx_sub==0):
                full_embedding = embedding
            else: 
                full_embedding = np.concatenate((full_embedding, embedding), axis=0)
            event_weights_exp.append(weight)
          
    # Return the weighted average of ebedding vector across temporal dimension
    try:
        #aggregated_embedding = np.dot(np.transpose(full_embedding), np.array(event_weights_exp))
        aggregated_embedding = np.average(full_embedding, axis=0, weights=np.array(event_weights_exp))
    except:
        aggregated_embedding = np.zeros(768)
      
    return aggregated_embedding, full_embedding, event_weights


# FOR NOTES SPLIT TEXT IF TOO LONG FOR NOTE EMBEDDING EXTRACTION
def split_note_document(text, min_length = 15):
    # Inputs:
    #   text -> String of text to be processed into an embedding. BioBERT can only process a string with â‰¤ 512 tokens. If the 
    #           input text exceeds this token count, we split it based on line breaks (driven from the discharge summary syntax). 
    #   min_length ->  When parsing the text into its subsections, remove text strings below a minimum length. These are generally 
    #                  very short and encode minimal information (e.g. 'Name: ___'). 
    #
    # Outputs:
    #   chunk_parse -> A list of "chunks", i.e. text strings, that breaks up the original text into strings with â‰¤ 512 tokens
    #   chunk_length -> A list of the token counts for each "chunk"
  
    # %% EXAMPLE OF USE
    # chunk_parse, chunk_length = split_note_document(ext, min_length = 15)
  
    tokens_list_0 = biobert_tokenizer.tokenize(text)
  
    if len(tokens_list_0) <= 510:
        return [text], [1]
    #print("Text exceeds 512 tokens - splitting into sections")
  
    chunk_parse = []
    chunk_length = []
    chunk = text
  
    ## Go through text and aggregate in groups up to 510 tokens (+ padding)
    tokens_list = biobert_tokenizer.tokenize(chunk)
    if len(tokens_list) >= 510:
        temp = chunk.split('\n')
        ind_start = 0
        len_sub = 0
        for i in range(len(temp)):
            temp_tk = biobert_tokenizer.tokenize(temp[i])
            if len_sub + len(temp_tk) >  510:
                chunk_parse.append(' '.join(temp[ind_start:i]))
                chunk_length.append(len_sub)
                # reset for next chunk
                ind_start = i
                len_sub = len(temp_tk)
            else: 
                len_sub += len(temp_tk)
    elif len(tokens_list) >= min_length:
        chunk_parse.append(chunk)
        chunk_length.append(len(tokens_list))
    #print("Parsed lengths: ", chunk_length)
      
    return chunk_parse, chunk_length


# FOR NOTES EMBEDDING EXTRACTION
def get_notes_biobert_embeddings(dt_patient, note_type):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   note_type -> Type of note to get
    #
    # Outputs:
    #   aggregated_embeddings -> Biobert event features for selected note
  
    # %% EXAMPLE OF USE
    # aggregated_embeddings = get_notes_biobert_embeddings(dt_patient, note_type = 'ecgnotes')
  
    admittime = dt_patient.core['admittime'].values[0]
    note_table = getattr(dt_patient, note_type).copy()
    note_table['deltacharttime'] = note_table['charttime'].apply(lambda x: (x.replace(tzinfo=None) - admittime).total_seconds()/3600)
    try:
        aggregated_embeddings, __, __ = get_biobert_embedding_from_events_list(note_table['text'], note_table['deltacharttime'])
    except:
        aggregated_embeddings, __, __ = get_biobert_embedding_from_events_list(pd.Series([""]), pd.Series([1]))
  
    return aggregated_embeddings


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                           Vision CXR embeddings for MIMIC-IV Deep Fusion                       |
#   

'''
## -> VISION REPRESENTATION OF MIMIC-IV EHR USING CNNs
A library for chest X-ray datasets and models. Including pre-trained models.
Motivation: While there are many publications focusing on the prediction of radiological and clinical findings from chest X-ray images much of this work is inaccessible to other researchers.

In the case of researchers addressing clinical questions it is a waste of time for them to train models from scratch. To address this, TorchXRayVision provides pre-trained models which are trained on large cohorts of data and enables 1) rapid analysis of large datasets 2) feature reuse for few-shot learning.
In the case of researchers developing algorithms it is important to robustly evaluate models using multiple external datasets. Metadata associated with each dataset can vary greatly which makes it difficult to apply methods to multiple datasets. TorchXRayVision provides access to many datasets in a uniform way so that they can be swapped out with a single line of code. These datasets can also be merged and filtered to construct specific distributional shifts for studying generalization. https://github.com/mlmed/torchxrayvision
'''

def get_single_chest_xray_embeddings(img):
    # Inputs:
    #   img -> Image array
    #
    # Outputs:
    #   densefeature_embeddings ->  CXR dense feature embeddings for image
    #   prediction_embeddings ->  CXR embeddings of predictions for image
    
    
    # %% EXAMPLE OF USE
    # densefeature_embeddings, prediction_embeddings = get_single_chest_xray_embeddings(img)
    
    # Clean out process bar before starting
    sys.stdout.flush()
    
    # Select if you want to use CUDA support for GPU (optional as it is usually pretty fast even in CPUT)
    cuda = False
    
    # Select model with a String that determines the model to use for Chest Xrays according to https://github.com/mlmed/torchxrayvision
    #model_weights_name = "densenet121-res224-all" # Every output trained for all models
    #model_weights_name = "densenet121-res224-rsna" # RSNA Pneumonia Challenge
    #model_weights_name = "densenet121-res224-nih" # NIH chest X-ray8
    #model_weights_name = "densenet121-res224-pc") # PadChest (University of Alicante)
    model_weights_name = "densenet121-res224-chex" # CheXpert (Stanford)
    #model_weights_name = "densenet121-res224-mimic_nb" # MIMIC-CXR (MIT)
    #model_weights_name = "densenet121-res224-mimic_ch" # MIMIC-CXR (MIT)
    #model_weights_name = "resnet50-res512-all" # Resnet only for 512x512 inputs
    # NOTE: The all model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset.
    
    # Extract chest x-ray image embeddings and preddictions
    densefeature_embeddings = []
    prediction_embeddings = []
    
    #img = skimage.io.imread(img_path) # If importing from path use this
    img = xrv.datasets.normalize(img, 255)

    # For each image check if they are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("Error: Dimension lower than 2 for image!")
    
    # Add color channel for prediction
    #Resize using OpenCV
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)   
    img = img[None, :, :]

    #Or resize using core resizer (thows error sometime)
    #transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    #img = transform(img)
    model = xrv.models.DenseNet(weights = model_weights_name)
    # model = xrv.models.ResNet(weights="resnet50-res512-all") # ResNet is also available

    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cuda:
            img = img.cuda()
            model = model.cuda()
          
        # Extract dense features
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        densefeatures = feats.cpu().detach().numpy().reshape(-1)
        densefeature_embeddings = densefeatures

        # Extract predicted probabilities of considered 18 classes:
        # Get by calling "xrv.datasets.default_pathologies" or "dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))"
        # ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',Fibrosis',
        #  'Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule',Mass','Hernia',
        #  'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
        preds = model(img).cpu()
        predictions = preds[0].detach().numpy()
        prediction_embeddings = predictions  

    # Return embeddings
    return densefeature_embeddings, prediction_embeddings



def get_chest_xray_embeddings(dt_patient, verbose=0):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   verbose -> Level of printed output of function
    #
    # Outputs:
    #   aggregated_densefeature_embeddings -> CXR aggregated dense feature embeddings for all images in timebound patient
    #   densefeature_embeddings ->  List of CXR dense feature embeddings for all images
    #   aggregated_prediction_embeddings -> CXR aggregated embeddings of predictions for all images in timebound patient
    #   prediction_embeddings ->  List of CXR embeddings of predictions for all images
    #   imgs_weights ->  Array of weights for embedding aggregation


    # %% EXAMPLE OF USE
    # aggregated_densefeature_embeddings, densefeature_embeddings, aggregated_prediction_embeddings, prediction_embeddings, imgs_weights = get_chest_xray_embeddings(dt_patient, verbose=2)

    # Clean out process bar before starting
    sys.stdout.flush()

    # Select if you want to use CUDA support for GPU (optional as it is usually pretty fast even in CPUT)
    cuda = False

    # Select model with a String that determines the model to use for Chest Xrays according to https://github.com/mlmed/torchxrayvision
    #   model_weights_name = "densenet121-res224-all" # Every output trained for all models
    #   model_weights_name = "densenet121-res224-rsna" # RSNA Pneumonia Challenge
    #model_weights_name = "densenet121-res224-nih" # NIH chest X-ray8
    #model_weights_name = "densenet121-res224-pc") # PadChest (University of Alicante)
    model_weights_name = "densenet121-res224-chex" # CheXpert (Stanford)
    #   model_weights_name = "densenet121-res224-mimic_nb" # MIMIC-CXR (MIT)
    #model_weights_name = "densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)
    #model_weights_name = "resnet50-res512-all" # Resnet only for 512x512 inputs
    # NOTE: The all model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset.


    # Extract chest x-ray images from timebound patient and iterate through them
    imgs = dt_patient.imcxr
    densefeature_embeddings = []
    prediction_embeddings = []

    # Iterate
    nImgs = len(imgs)
    with tqdm(total = nImgs) as pbar:
        for idx, img in enumerate(imgs):
            #img = skimage.io.imread(img_path) # If importing from path use this
            img = xrv.datasets.normalize(img, 255)
          
            # For each image check if they are 2D arrays
            if len(img.shape) > 2:
                img = img[:, :, 0]
            if len(img.shape) < 2:
                print("Error: Dimension lower than 2 for image!")

            # Add color channel for prediction
            #Resize using OpenCV
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)   
            img = img[None, :, :]
            
            #Or resize using core resizer (thows error sometime)
            #transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
            #img = transform(img)
            model = xrv.models.DenseNet(weights = model_weights_name)
            # model = xrv.models.ResNet(weights="resnet50-res512-all") # ResNet is also available
            
            output = {}
            with torch.no_grad():
                img = torch.from_numpy(img).unsqueeze(0)
                if cuda:
                    img = img.cuda()
                    model = model.cuda()
              
                # Extract dense features
                feats = model.features(img)
                feats = F.relu(feats, inplace=True)
                feats = F.adaptive_avg_pool2d(feats, (1, 1))
                densefeatures = feats.cpu().detach().numpy().reshape(-1)
                densefeature_embeddings.append(densefeatures) # append to list of dense features for all images
                
                # Extract predicted probabilities of considered 18 classes:
                # Get by calling "xrv.datasets.default_pathologies" or "dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))"
                # ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',Fibrosis',
                #  'Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule',Mass','Hernia',
                #  'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
                preds = model(img).cpu()
                predictions = preds[0].detach().numpy()
                prediction_embeddings.append(predictions) # append to list of predictions for all images
            
                if verbose >=1:
                    # Update process bar
                    pbar.update(1)
        
        
    # Get image weights by hours passed from current time to image
    orig_imgs_weights = np.asarray(dt_patient.cxr.deltacharttime.values)
    adj_imgs_weights = orig_imgs_weights - orig_imgs_weights.min()
    imgs_weights = (adj_imgs_weights) / (adj_imgs_weights).max()
  
    # Aggregate with weighted average of ebedding vector across temporal dimension
    try:
        aggregated_densefeature_embeddings = np.average(densefeature_embeddings, axis=0, weights=imgs_weights)
        if np.isnan(np.sum(aggregated_densefeature_embeddings)):
            aggregated_densefeature_embeddings = np.zeros_like(densefeature_embeddings[0])
    except:
        aggregated_densefeature_embeddings = np.zeros_like(densefeature_embeddings[0])
      
    try:
        aggregated_prediction_embeddings = np.average(prediction_embeddings, axis=0, weights=imgs_weights)
        if np.isnan(np.sum(aggregated_prediction_embeddings)):
            aggregated_prediction_embeddings = np.zeros_like(prediction_embeddings[0])
    except:
        aggregated_prediction_embeddings = np.zeros_like(prediction_embeddings[0])
      
      
    if verbose >=2:
        x = orig_imgs_weights
        y = prediction_embeddings
        plt.xlabel("Time [hrs]")
        plt.ylabel("Disease probability [0-1]")
        plt.title("A test graph")
        for i in range(len(y[0])):
            plt.plot(x,[pt[i] for pt in y],'o', label = xrv.datasets.default_pathologies[i])
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.show()

    # Return embeddings
    return aggregated_densefeature_embeddings, densefeature_embeddings, aggregated_prediction_embeddings, prediction_embeddings, imgs_weights





#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                  Preprocessing MIMIC-IV Dataset                                 |
#

# SAVE SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV 
def save_patient_object(obj, filepath):
    # Inputs:
    #   obj -> Timebound ICU patient stay object
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   VOID -> Object is saved in filename path
    # Overwrites any existing file.
    with open(filepath, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# LOAD SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV
def load_patient_object(filepath):
    # Inputs:
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   obj -> Loaded timebound ICU patient stay object

    # Overwrites any existing file.
    with open(filepath, 'rb') as input:  
        return pickle.load(input)

    
# BUILD DATAFRAME OF IMAGES AND NOTES FOR MIMIC-IV CXR
def build_mimic_cxr_jpg_dataframe(core_mimiciv_imgcxr_path, do_save=False):
    # Inputs:
    #   core_mimiciv_imgcxr_path -> Directory of CXR images and image notes
    #   do_save -> Flag to save dataframe
    #
    # Outputs:
    #   df_mimic_cxr_jpg -> CXR images and image notes Dataframe
    df_mimic_cxr_jpg = pd.DataFrame()
    mimic_cxr_jpg_dir = core_mimiciv_imgcxr_path
    
    #Figure out how many files we will read
    file_count = 0
    for subdir, dirs, files in os.walk(mimic_cxr_jpg_dir):
        for file in files:
            # Extract filename and extension to filter by CSV only
            filename, extension = os.path.splitext(file)
            if extension=='.txt':
                file_count = file_count + 1
                
    #Setup progress bar
    pbar = tqdm(total=file_count)
    
    #Iterate
    for subdir, dirs, files in os.walk(mimic_cxr_jpg_dir):
        for file in files:
            # Extract filename and extension to filter by CSV only
            filename, extension = os.path.splitext(file)
            if extension=='.txt':
                note = open(subdir + '/' + filename + extension, "r", errors='ignore')
                img_note_text = note.read()
                note.close()
                img_folder = subdir + '/' + filename
                
                for img_subdir, img_dirs, img_files in os.walk(img_folder):
                    for img_file in img_files:
                        # Extract filename and extension to filter by CSV only
                        img_filename, img_extension = os.path.splitext(img_file)
                        if img_extension=='.jpg':
                            df_mimic_cxr_jpg = df_mimic_cxr_jpg.append({'Note_folder': subdir.replace(core_mimiciv_imgcxr_path,''), 'Note_file': filename + extension , 'Note': img_note_text, 'Img_Folder': img_folder.replace(core_mimiciv_imgcxr_path,''), 'Img_Filename': img_filename + img_extension, 'dicom_id': img_filename}, ignore_index=True)
                            
        #Update progress bar
        pbar.update(1)
        
    #Save
    if do_save:
        df_mimic_cxr_jpg.to_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-jpeg-txt.csv')
        
    return df_mimic_cxr_jpg


# LOAD ALL MIMIC IV TABLES IN MEMORY (warning: High memory lengthy process)
def load_mimiciv(core_mimiciv_path):
    # Inputs:
    #   core_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #   filename -> Pickle filename to save object to
    #
    # Outputs:
    #   df's -> Many dataframes with all loaded MIMIC IV tables 
    
    ### -> Initializations & Data Loading
    ###    Resources to identify tables and variables of interest can be found in the MIMIC-IV official API (https://mimic-iv.mit.edu/docs/)
    
    ## CORE
    df_admissions = dd.read_csv(core_mimiciv_path + 'core/admissions.csv', assume_missing=True, dtype={'admission_location': 'object','deathtime': 'object','edouttime': 'object','edregtime': 'object'})
    df_patients = dd.read_csv(core_mimiciv_path + 'core/patients.csv', assume_missing=True, dtype={'dod': 'object'})  
    df_transfers = dd.read_csv(core_mimiciv_path + 'core/transfers.csv', assume_missing=True, dtype={'careunit': 'object'})
  
    ## HOSP
    df_d_labitems = dd.read_csv(core_mimiciv_path + 'hosp/d_labitems.csv', assume_missing=True, dtype={'loinc_code': 'object'})
    df_d_icd_procedures = dd.read_csv(core_mimiciv_path + 'hosp/d_icd_procedures.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_d_icd_diagnoses = dd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_d_hcpcs = dd.read_csv(core_mimiciv_path + 'hosp/d_hcpcs.csv', assume_missing=True, dtype={'category': 'object'})
    df_diagnoses_icd = dd.read_csv(core_mimiciv_path + 'hosp/diagnoses_icd.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_drgcodes = dd.read_csv(core_mimiciv_path + 'hosp/drgcodes.csv', assume_missing=True)
    df_emar = dd.read_csv(core_mimiciv_path + 'hosp/emar.csv', assume_missing=True)
    df_emar_detail = dd.read_csv(core_mimiciv_path + 'hosp/emar_detail.csv', assume_missing=True, low_memory=False, dtype={'completion_interval': 'object','dose_due': 'object','dose_given': 'object','infusion_complete': 'object','infusion_rate_adjustment': 'object','infusion_rate_unit': 'object','new_iv_bag_hung': 'object','product_description_other': 'object','reason_for_no_barcode': 'object','restart_interval': 'object','route': 'object','side': 'object','site': 'object','continued_infusion_in_other_location': 'object','infusion_rate': 'object','non_formulary_visual_verification': 'object','prior_infusion_rate': 'object','product_amount_given': 'object', 'infusion_rate_adjustment_amount': 'object'})
    df_hcpcsevents = dd.read_csv(core_mimiciv_path + 'hosp/hcpcsevents.csv', assume_missing=True, dtype={'hcpcs_cd': 'object'})
    df_labevents = dd.read_csv(core_mimiciv_path + 'hosp/labevents.csv', assume_missing=True, dtype={'storetime': 'object', 'value': 'object', 'valueuom': 'object', 'flag': 'object', 'priority': 'object', 'comments': 'object'})
    df_microbiologyevents = dd.read_csv(core_mimiciv_path + 'hosp/microbiologyevents.csv', assume_missing=True, dtype={'comments': 'object', 'quantity': 'object'})
    df_poe = dd.read_csv(core_mimiciv_path + 'hosp/poe.csv', assume_missing=True, dtype={'discontinue_of_poe_id': 'object','discontinued_by_poe_id': 'object','order_status': 'object'})
    df_poe_detail = dd.read_csv(core_mimiciv_path + 'hosp/poe_detail.csv', assume_missing=True)
    df_prescriptions = dd.read_csv(core_mimiciv_path + 'hosp/prescriptions.csv', assume_missing=True, dtype={'form_rx': 'object','gsn': 'object'})
    df_procedures_icd = dd.read_csv(core_mimiciv_path + 'hosp/procedures_icd.csv', assume_missing=True, dtype={'icd_code': 'object', 'icd_version': 'object'})
    df_services = dd.read_csv(core_mimiciv_path + 'hosp/services.csv', assume_missing=True, dtype={'prev_service': 'object'})
  
    ## ICU
    df_d_items = dd.read_csv(core_mimiciv_path + 'icu/d_items.csv', assume_missing=True)
    df_procedureevents = dd.read_csv(core_mimiciv_path + 'icu/procedureevents.csv', assume_missing=True, dtype={'value': 'object', 'secondaryordercategoryname': 'object', 'totalamountuom': 'object'})
    df_outputevents = dd.read_csv(core_mimiciv_path + 'icu/outputevents.csv', assume_missing=True, dtype={'value': 'object'})
    df_inputevents = dd.read_csv(core_mimiciv_path + 'icu/inputevents.csv', assume_missing=True, dtype={'value': 'object', 'secondaryordercategoryname': 'object', 'totalamountuom': 'object'})
    df_icustays = dd.read_csv(core_mimiciv_path + 'icu/icustays.csv', assume_missing=True)
    df_datetimeevents = dd.read_csv(core_mimiciv_path + 'icu/datetimeevents.csv', assume_missing=True, dtype={'value': 'object'})
    df_chartevents = dd.read_csv(core_mimiciv_path + 'icu/chartevents.csv', assume_missing=True, low_memory=False, dtype={'value': 'object', 'valueuom': 'object'})
  
    ## CXR
    df_mimic_cxr_split = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv', assume_missing=True)
    df_mimic_cxr_chexpert = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv', assume_missing=True)
    try:
        df_mimic_cxr_metadata = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object'}, blocksize=None)
    except:
        df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object'})
        df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
    df_mimic_cxr_negbio = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-negbio.csv', assume_missing=True)
  
    ## NOTES
    df_noteevents = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/noteevents.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    df_dsnotes = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/ds_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    df_ecgnotes = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/ecg_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    df_echonotes = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/echo_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    df_radnotes = dd.from_pandas(pd.read_csv(core_mimiciv_path + 'note/rad_icustay.csv', dtype={'charttime': 'object', 'storetime': 'object', 'text': 'object'}), chunksize=8)
    
    
    ### -> Data Preparation (Create full database in dask format)
    ### Fix data type issues to allow for merging
    ## CORE
    df_admissions['admittime'] = dd.to_datetime(df_admissions['admittime'])
    df_admissions['dischtime'] = dd.to_datetime(df_admissions['dischtime'])
    df_admissions['deathtime'] = dd.to_datetime(df_admissions['deathtime'])
    df_admissions['edregtime'] = dd.to_datetime(df_admissions['edregtime'])
    df_admissions['edouttime'] = dd.to_datetime(df_admissions['edouttime'])
    
    df_transfers['intime'] = dd.to_datetime(df_transfers['intime'])
    df_transfers['outtime'] = dd.to_datetime(df_transfers['outtime'])
    
    ## HOSP
    df_diagnoses_icd.icd_code = df_diagnoses_icd.icd_code.str.strip()
    df_diagnoses_icd.icd_version = df_diagnoses_icd.icd_version.str.strip()
    df_d_icd_diagnoses.icd_code = df_d_icd_diagnoses.icd_code.str.strip()
    df_d_icd_diagnoses.icd_version = df_d_icd_diagnoses.icd_version.str.strip()
    
    df_procedures_icd.icd_code = df_procedures_icd.icd_code.str.strip()
    df_procedures_icd.icd_version = df_procedures_icd.icd_version.str.strip()
    df_d_icd_procedures.icd_code = df_d_icd_procedures.icd_code.str.strip()
    df_d_icd_procedures.icd_version = df_d_icd_procedures.icd_version.str.strip()
    
    df_hcpcsevents.hcpcs_cd = df_hcpcsevents.hcpcs_cd.str.strip()
    df_d_hcpcs.code = df_d_hcpcs.code.str.strip()
    
    df_prescriptions['starttime'] = dd.to_datetime(df_prescriptions['starttime'])
    df_prescriptions['stoptime'] = dd.to_datetime(df_prescriptions['stoptime'])
    
    df_emar['charttime'] = dd.to_datetime(df_emar['charttime'])
    df_emar['scheduletime'] = dd.to_datetime(df_emar['scheduletime'])
    df_emar['storetime'] = dd.to_datetime(df_emar['storetime'])
    
    df_labevents['charttime'] = dd.to_datetime(df_labevents['charttime'])
    df_labevents['storetime'] = dd.to_datetime(df_labevents['storetime'])
    
    df_microbiologyevents['chartdate'] = dd.to_datetime(df_microbiologyevents['chartdate'])
    df_microbiologyevents['charttime'] = dd.to_datetime(df_microbiologyevents['charttime'])
    df_microbiologyevents['storedate'] = dd.to_datetime(df_microbiologyevents['storedate'])
    df_microbiologyevents['storetime'] = dd.to_datetime(df_microbiologyevents['storetime'])
    
    df_poe['ordertime'] = dd.to_datetime(df_poe['ordertime'])
    df_services['transfertime'] = dd.to_datetime(df_services['transfertime'])
    
    ## ICU
    df_procedureevents['starttime'] = dd.to_datetime(df_procedureevents['starttime'])
    df_procedureevents['endtime'] = dd.to_datetime(df_procedureevents['endtime'])
    df_procedureevents['storetime'] = dd.to_datetime(df_procedureevents['storetime'])
    df_procedureevents['comments_date'] = dd.to_datetime(df_procedureevents['comments_date'])
    
    df_outputevents['charttime'] = dd.to_datetime(df_outputevents['charttime'])
    df_outputevents['storetime'] = dd.to_datetime(df_outputevents['storetime'])
    
    df_inputevents['starttime'] = dd.to_datetime(df_inputevents['starttime'])
    df_inputevents['endtime'] = dd.to_datetime(df_inputevents['endtime'])
    df_inputevents['storetime'] = dd.to_datetime(df_inputevents['storetime'])
    
    df_icustays['intime'] = dd.to_datetime(df_icustays['intime'])
    df_icustays['outtime'] = dd.to_datetime(df_icustays['outtime'])
    
    df_datetimeevents['charttime'] = dd.to_datetime(df_datetimeevents['charttime'])
    df_datetimeevents['storetime'] = dd.to_datetime(df_datetimeevents['storetime'])
    
    df_chartevents['charttime'] = dd.to_datetime(df_chartevents['charttime'])
    df_chartevents['storetime'] = dd.to_datetime(df_chartevents['storetime'])
    
    ## CXR
    if (not 'cxrtime' in df_mimic_cxr_metadata.columns) or (not 'Img_Filename' in df_mimic_cxr_metadata.columns):
        # Create CXRTime variable if it does not exist already
        print("Processing CXRtime stamps")
        df_cxr = df_mimic_cxr_metadata.compute()
        df_cxr['StudyDateForm'] = pd.to_datetime(df_cxr['StudyDate'], format='%Y%m%d')
        df_cxr['StudyTimeForm'] = df_cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
        df_cxr['StudyTimeForm'] = pd.to_datetime(df_cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
        df_cxr['cxrtime'] = df_cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)
        # Add paths and info to images in cxr
        df_mimic_cxr_jpg =pd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-jpeg-txt.csv')
        df_cxr = pd.merge(df_mimic_cxr_jpg, df_cxr, on='dicom_id')
        # Save
        df_cxr.to_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', index=False)
        #Read back the dataframe
        try:
            df_mimic_cxr_metadata = dd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object', 'Note': 'object'}, blocksize=None)
        except:
            df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object', 'Note': 'object'})
            df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
    df_mimic_cxr_metadata['cxrtime'] = dd.to_datetime(df_mimic_cxr_metadata['cxrtime'])
    
    ## NOTES
    df_noteevents['chartdate'] = dd.to_datetime(df_noteevents['chartdate'])
    df_noteevents['charttime'] = dd.to_datetime(df_noteevents['charttime'])
    df_noteevents['storetime'] = dd.to_datetime(df_noteevents['storetime'])
  
    df_dsnotes['charttime'] = dd.to_datetime(df_dsnotes['charttime'])
    df_dsnotes['storetime'] = dd.to_datetime(df_dsnotes['storetime'])
  
    df_ecgnotes['charttime'] = dd.to_datetime(df_ecgnotes['charttime'])
    df_ecgnotes['storetime'] = dd.to_datetime(df_ecgnotes['storetime'])
  
    df_echonotes['charttime'] = dd.to_datetime(df_echonotes['charttime'])
    df_echonotes['storetime'] = dd.to_datetime(df_echonotes['storetime'])
  
    df_radnotes['charttime'] = dd.to_datetime(df_radnotes['charttime'])
    df_radnotes['storetime'] = dd.to_datetime(df_radnotes['storetime'])
    
    
    ### -> SORT data
    ## CORE
    print('PROCESSING "CORE" DB...')
    df_admissions = df_admissions.compute().sort_values(by=['subject_id','hadm_id'])
    df_patients = df_patients.compute().sort_values(by=['subject_id'])
    df_transfers = df_transfers.compute().sort_values(by=['subject_id','hadm_id'])
    
    ## HOSP
    print('PROCESSING "HOSP" DB...')
    df_diagnoses_icd = df_diagnoses_icd.compute().sort_values(by=['subject_id'])
    df_drgcodes = df_drgcodes.compute().sort_values(by=['subject_id','hadm_id'])
    df_emar = df_emar.compute().sort_values(by=['subject_id','hadm_id'])
    df_emar_detail = df_emar_detail.compute().sort_values(by=['subject_id'])
    df_hcpcsevents = df_hcpcsevents.compute().sort_values(by=['subject_id','hadm_id'])
    df_labevents = df_labevents.compute().sort_values(by=['subject_id','hadm_id'])
    df_microbiologyevents = df_microbiologyevents.compute().sort_values(by=['subject_id','hadm_id'])
    df_poe = df_poe.compute().sort_values(by=['subject_id','hadm_id'])
    df_poe_detail = df_poe_detail.compute().sort_values(by=['subject_id'])
    df_prescriptions = df_prescriptions.compute().sort_values(by=['subject_id','hadm_id'])
    df_procedures_icd = df_procedures_icd.compute().sort_values(by=['subject_id','hadm_id'])
    df_services = df_services.compute().sort_values(by=['subject_id','hadm_id'])
    #--> Unwrap dictionaries
    df_d_icd_diagnoses = df_d_icd_diagnoses.compute()
    df_d_icd_procedures = df_d_icd_procedures.compute()
    df_d_hcpcs = df_d_hcpcs.compute()
    df_d_labitems = df_d_labitems.compute()
    
    ## ICU
    print('PROCESSING "ICU" DB...')
    df_procedureevents = df_procedureevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_outputevents = df_outputevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_inputevents = df_inputevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_icustays = df_icustays.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_datetimeevents = df_datetimeevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_chartevents = df_chartevents.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    #--> Unwrap dictionaries
    df_d_items = df_d_items.compute()
    
    ## CXR
    print('PROCESSING "CXR" DB...')
    df_mimic_cxr_split = df_mimic_cxr_split.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_chexpert = df_mimic_cxr_chexpert.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_metadata = df_mimic_cxr_metadata.compute().sort_values(by=['subject_id'])
    df_mimic_cxr_negbio = df_mimic_cxr_negbio.compute().sort_values(by=['subject_id'])
    
    ## NOTES
    print('PROCESSING "NOTES" DB...')
    df_noteevents = df_noteevents.compute().sort_values(by=['subject_id','hadm_id'])
    df_dsnotes = df_dsnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_ecgnotes = df_ecgnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_echonotes = df_echonotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    df_radnotes = df_radnotes.compute().sort_values(by=['subject_id','hadm_id','stay_id'])
    
    # Return
    return df_admissions, df_patients, f_transfers, df_diagnoses_icd, df_drgcodes, df_emar, df_emar_detail, df_hcpcsevents, df_labevents, df_microbiologyevents, df_poe, df_poe_detail, df_prescriptions, df_procedures_icd, df_services, df_d_icd_diagnoses, df_d_icd_procedures, df_d_hcpcs, df_d_labitem, df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents, df_d_items, df_mimic_cxr_split, df_mimic_cxr_chexpert, df_mimic_cxr_metadata, df_mimic_cxr_negbio, df_noteevents, df_dsnotes, df_ecgnotes, df_echonotes, df_radnotes


# GET LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV (subject_id, hadm_id, stay_id)
def get_unique_available_HAIM_MIMICIV_records(df_procedureevents, df_outputevents, df_inputevents, df_icustays, df_datetimeevents, df_chartevents):
    # Inputs:
    #   df's -> Many dataframes with all loaded MIMIC IV tables 
    #
    # Outputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    
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
    print('Unique Subjects/Hospital Admissions/Stays Combinations: ' + str(len(df_ids)))
    print('Unique Subjects with Chest Xrays Available: ' + str(len(df_cxr_ids)))
    print('Unique HAIM Records Available: ' + str(len(df_haim_ids)))
    
    return df_haim_ids


# SAVE LIST OF ALL UNIQUE ID COMBINATIONS IN MIMIC-IV
def save_unique_available_HAIM_MIMICIV_records (df_haim_ids, core_mimiciv_path):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers 
    #   core_mimiciv_path -> Path to MIMIC IV Dataset
    #
    # Outputs:
    #   Saved dataframe in location
    
    # Save Unique Subject/HospAdmission/Stay Combinations with Chest Xrays    
    df_haim_ids.to_csv(core_mimiciv_path + 'haim_mimiciv_key_ids.csv', index=False)
    return print('Saved')


# EXTRACT ALL INFO OF A SINGLE PATIENT FROM MIMIC-IV DATASET USING HAIM ID
def extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr):
    # Inputs:
    #   haim_patient_idx -> Ordered number of HAIM patient
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    # Outputs:
    #   key_subject_id -> MIMIC-IV Subject ID of selected patient
    #   key_hadm_id -> MIMIC-IV Hospital Admission ID of selected patient
    #   key_stay_id -> MIMIC-IV ICU Stay ID of selected patient
    #   patient -> Full ICU patient ICU stay structure
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    
    # Extract information for patient
    key_subject_id = df_haim_ids.iloc[haim_patient_idx].subject_id
    key_hadm_id = df_haim_ids.iloc[haim_patient_idx].hadm_id
    key_stay_id = df_haim_ids.iloc[haim_patient_idx].stay_id
    start_hr = start_hr # Select timestamps
    end_hr = end_hr   # Select timestamps
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    
    return key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient


# GET ALL DEMOGRAPHCS DATA OF A TIMEBOUND PATIENT RECORD
def get_demographics(dt_patient):
    dem_info = dt_patient.demographics[['gender', 'anchor_age', 'anchor_year']] 
    dem_info['gender'] = (dem_info['gender'] == 'M').astype(int)
    return dem_info.values[0]


# GENERATE ALL SINGLE PATIENT ICU STAY RECORDS FOR ENTIRE MIMIC-IV DATABASE
def generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path):
    # Inputs:
    #   df_haim_ids -> Dataframe with all unique available HAIM_MIMICIV records by key identifiers
    #   core_mimiciv_path -> Path to structured MIMIC IV databases in CSV files
    #
    # Outputs:
    #   nfiles -> Number of single patient HAIM files produced
    
    # Extract information for patient
    nfiles = len(df_haim_ids)
    with tqdm(total = nfiles) as pbar:
        #Iterate through all patients
        for haim_patient_idx in range(nfiles):
            # Let's select each single patient and extract patient object
            start_hr = None # Select timestamps
            end_hr = None   # Select timestamps
            key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient = extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr)
            
            # Save
            filename = f"{haim_patient_idx:08d}" + '.pkl'
            save_patient_object(dt_patient, core_mimiciv_path + 'pickle/' + filename)
            # Update process bar
            pbar.update(1)
    return nfiles


