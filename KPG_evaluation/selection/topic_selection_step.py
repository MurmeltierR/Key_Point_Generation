from azureml.core import Run

# Get the run context
new_run = Run.get_context()

# Get the workspace from the run
ws = new_run.experiment.workspace

#Import Enviroment Libraries
import pandas as pd
import numpy as np
import json
import os
import joblib
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--topic_query', type=str)
parser.add_argument('--topic_selection', type=str)
parser.add_argument('--stance_selection', type=int)
args = parser.parse_args()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

topic_query = str(args.topic_query)
selection_output = str(args.topic_selection)
stance = int(args.stance_selection)

os.makedirs(os.path.dirname(selection_output), exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(topic_query)
print(device)
print(stance)

argkp_df = new_run.input_datasets['argkp_corpus'].to_pandas_dataframe()
argkp_df = argkp_df[argkp_df['label']==1]

print(len(argkp_df))
print(argkp_df.info())

selected_topic = argkp_df[(argkp_df.topic.str.contains(topic_query)) & (argkp_df.stance==stance)]

print(len(argkp_df))

with open(os.path.join(selection_output, 'selected_topic.csv'), 'w') as f:
    f.write(selected_topic.to_csv(index=False))
    f.close()

new_run.complete()