# stopReasons
Analysis on stop reasons. 

To run the model, please run the run_prediction.py script and specify the following four arguments in the main method:
-path to the studies.txt file;
-path fine-tuned model; 
-path to the output file with predictions;
-path to where you wish to store the rest of the trials that have not stopped (may be needed for statistical analysis).

The studies.txt file can be downloaded from clinicaltrials.gov or can be found here: gs://ot-team/olesya/Stop reasons/data/studies.txt
The latest BERT model to use: gs://ot-team/olesya/Stop reasons/bert_stop_reasons_revised


You will need to download BertClassifier.py and predict.py files and place them in the same directory as run_prediction.py.


