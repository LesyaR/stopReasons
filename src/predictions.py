
# Libraries
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel
from transformers import BertTokenizer
from common_classes import BertClassifier
from common_classes import text_preprocessing
from common_classes import preprocessing_for_bert
from numpy import argmax
from common_classes import get_class
from common_classes import bert_predict
import csv
import torch.nn as nn
import logging


 
# modify the model path to load the model    
model=torch.load('/Users/olesyar/Documents/data/bert_trials')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


logging.basicConfig(level=logging.ERROR)
names_all = ["nct_id","phase","name","mesh_term","why_stopped","pmid"]
names_studies = ['nct_id','nlm_download_date_description',
                 'study_first_submitted_date','results_first_submitted_date','disposition_first_submitted_date',
                 'last_update_submitted_date','study_first_submitted_qc_date','study_first_posted_date',
                 'study_first_posted_date_type','results_first_submitted_qc_date','results_first_posted_date',
                 'results_first_posted_date_type','disposition_first_submitted_qc_date',
                 'disposition_first_posted_date','disposition_first_posted_date_type',
                 'last_update_submitted_qc_date','last_update_posted_date','last_update_posted_date_type',
                 'start_month_year','start_date_type','start_date','verification_month_year',
                 'verification_date','completion_month_year','completion_date_type','completion_date',
                 'primary_completion_month_year','primary_completion_date_type','primary_completion_date',
                 'target_duration','study_type','acronym','baseline_population','brief_title','official_title',
                 'overall_status','last_known_status','phase','enrollment','enrollment_type','source',
                 'limitations_and_caveats','number_of_arms','number_of_groups','why_stopped','has_expanded_access',
                 'expanded_access_type_individual','expanded_access_type_intermediate',
                 'expanded_access_type_treatment','has_dmc','is_fda_regulated_drug','is_fda_regulated_device',
                 'is_unapproved_device','is_ppsd','is_us_export','biospec_retention','biospec_description',
                 'ipd_time_frame','ipd_access_criteria','ipd_url','plan_to_share_ipd','plan_to_share_ipd_description',
                 'created_at','updated_at']

def create_predictions(df):
    # Display 5 samples from the test data
    print('The test set is loaded')
    # Run `preprocessing_for_bert` on the test set
    test_inputs, test_masks = preprocessing_for_bert(df[df["why_stopped"].notnull()].why_stopped)
    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    print('The test set is ready')
    return test_dataset

# =============================================================================
# make predictions
# =============================================================================

if __name__ == '__main__':
    studies_file = "/Users/olesyar/Documents/data/studies.txt"
    reader = pd.read_csv(studies_file, skiprows=1, names=names_studies, delimiter='|')
    reader=(reader[['why_stopped','phase','nct_id', 'start_date', 'overall_status', 'last_update_posted_date', 'completion_date']]).drop_duplicates()
    probs = bert_predict(model, create_predictions(reader))
    csv_file1=open('/Users/olesyar/Documents/data/stopped_predictions2.tsv', "w")
    writer1 = csv.writer(csv_file1, delimiter='\t', lineterminator='\n')
    i=0
    not_stopped=reader[reader["why_stopped"].notnull()]    
    for ind,dat in not_stopped.iterrows():
        writer1.writerow([dat['why_stopped'].replace('\r~', ''),dat['phase'],dat['nct_id'],dat['start_date'], dat['overall_status'],dat['last_update_posted_date'],dat['completion_date'],get_class(argmax(probs[i]))])
        i=i+1
    csv_file2=open('/Users/olesyar/Documents/data/notstopped_predictions2.tsv', "w")
    writer2 = csv.writer(csv_file2, delimiter='\t')
    i=0
    stopped=reader[reader["why_stopped"].isnull()]
    for ind,dat in stopped.iterrows():
        writer2.writerow([dat['why_stopped'],dat['phase'],dat['nct_id'],dat['start_date'],dat['overall_status'],dat['last_update_posted_date'],dat['completion_date'],''])