# Integrated multimodal artificial intelligence framework for healthcare applications

This repository contains the code to replicate the data processing, modeling and reporting of our Holistic AI in Medicine (HAIM) in Nature's NPJ Digital Medicine. [Soenksen, L.R., Ma, Y., Zeng, C. et al. Integrated multimodal artificial intelligence framework for healthcare applications. npj Digit. Med. 5, 149 (2022). https://doi.org/10.1038/s41746-022-00689-4](https://www.nature.com/articles/s41746-022-00689-4).

## Authors:
Luis R. Soenksen, Yu Ma, Cynthia Zeng, Léonard Boussioux, Kimberly Villalobos Carballo, Liangyuan Na, Holly M. Wiberg, Michael L. Li, Ignacio Fuentes, Dimitris Bertsimas

Artificial intelligence (AI) systems hold great promise to improve healthcare over the next decades. Specifically, AI systems leveraging multiple data sources and input modalities are poised to become a viable method to deliver more accurate results and deployable pipelines across a wide range of applications. In this work, we propose and evaluate a unified Holistic AI in Medicine (HAIM) framework to facilitate the generation and testing of AI systems that leverage multimodal inputs. Our approach uses generalizable data pre-processing and machine learning modeling stages that can be readily adapted for research and deployment in healthcare environments. We evaluate our HAIM framework by training and characterizing 14,324 independent models based on HAIM-MIMIC-MM, a multimodal clinical database (N=34,537 samples) containing 7,279 unique hospitalizations and 6,485 patients, spanning all possible input combinations of 4 data modalities (i.e., tabular, time-series, text, and images), 11 unique data sources and 12 predictive tasks. We show that this framework can consistently and robustly produce models that outperform similar single-source approaches across various healthcare demonstrations (by 6-33%), including 10 distinct chest pathology diagnoses, along with length-of-stay and 48-hour mortality predictions. We also quantify the contribution of each modality and data source using Shapley values, which demonstrates the heterogeneity in data modality importance and the necessity of multimodal inputs across different healthcare-relevant tasks. The generalizable properties and flexibility of our Holistic AI in Medicine (HAIM) framework could offer a promising pathway for future multimodal predictive systems in clinical and operational healthcare settings.

## Code

The code uses Python3.6.9 and is separated into four sections:

0 - Software Package requirement

1 - Data Preprocessing. Noteevents.csv are public and available for download at Physionet.org; however, other "NOTES" data requires pre-release direct permission from Physionet.org for download as "discharge notes", "radiology notes", "ECG notes" and "ECHO notes" are not yet publicly released for MIMIC-IV as of Sep 2022, these files are: ds_icustay.csv, ecg_icustay.csv, echo_icustay.csv, rad_icustay.csv). To run our code without them just comment import and usage of these notes.

2 - Modeling of our three tasks: mortality prediction, length of stay prediction, chest pathology classification

3 - Result Generating: Including reporting of the AUROC, AUPRC, F1 scores, as well as code to generate the plots reported in the paper.


Please be advised that sufficient RAM or cluster access to parallel processing is needed to run these experiments.

### UPDATE (Jan. 6, 2023)
The radiology and the discharge notes for MIMIC-IV have been officially released on:
https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel

### UPDATE (Jun. 12, 2023)
For the publication, our team generated the file 'mimic-cxr-2.0.0-jpeg-txt.csv' by compiling an early-release version of participant notes and text from the images in CXR corresponding to MIMIC-IV. We wanted to add these to this repository, but the data policy from PhysioNet.org states we cannot directly share this compiled data via Git Hub. Physionet is the only one with permission to do so or subsets of the data. This means users need to generate their own mimic-cxr-2.0.0-jpeg-txt.csv based on the released notes and CXR files from Physionet.org once all notes are released. The dataset structure can be inferred from the code. As of June 12, 2023, Physionet has not fully released these notes, but it is likely they are planning to do so as part of their full release of MIMIC-IV. We are very sorry for any inconvenience this may cause.
