# --------------------------------------------------------------
# Training Script for the Hyperdrive Job
# --------------------------------------------------------------

from azureml.core import Run, Model, Datastore, Experiment, run, Dataset
from azureml.pipeline.steps import HyperDriveStepRun

# Get the run context
new_run = Run.get_context()

# Get the workspace from the run
ws = new_run.experiment.workspace

# -------------------------------------------------
#Import Enviroment Libraries
# -------------------------------------------------

from ast import literal_eval
import numpy as np
import pandas as pd
import argparse
import os
from statistics import mean
import torch
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sentence_transformers import SentenceTransformer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# -------------------------------------------------
#Get parameters
# -------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--min_length', type=int)
parser.add_argument('--top_k', type=int)
parser.add_argument('--top_p', type=float)
parser.add_argument('--output_keypoint_data', type=str)

args = parser.parse_args()

output_keypoint_data = str(args.output_keypoint_data)
min_length = int(args.min_length)
top_k = int(args.top_k)
top_p = float(args.top_k)

pipeline_runid = new_run.get_details()['runDefinition']['outputData']['combined_df']['additionalOptions']['registrationOptions']['properties']['azureml.pipelineRunId']
pipeline_run = run.get_run(Experiment(ws, 'keypoint_pipeline'), pipeline_runid) 
hyperdrive_run_id = pipeline_run.find_step_run('topic_optimizer')[0]
get_hyperdrive_run = HyperDriveStepRun(step_run=hyperdrive_run_id)
best_run = get_hyperdrive_run.get_best_run_by_primary_metric() 

os.makedirs(os.path.dirname(output_keypoint_data), exist_ok=True)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

keypoint_storage = Datastore.get(ws, datastore_name='kp_datastore')
topic_output_path = (keypoint_storage, os.path.join('topic_output/', best_run.id, 'output_topic_data.csv'))
arguments_output_path = (keypoint_storage, os.path.join('arguments/', best_run.id, 'arguments.csv'))
topiclabel_output_path = (keypoint_storage, os.path.join('topic_labels/', best_run.id, 'topic_labels.csv'))
tm_output = Dataset.Tabular.from_delimited_files(path=topic_output_path, support_multi_line=True, separator=',').to_pandas_dataframe()
raw_arguments = Dataset.Tabular.from_delimited_files(path=arguments_output_path, support_multi_line=True, separator=',').to_pandas_dataframe()
print(len(tm_output))
print(tm_output.info())
print(len(raw_arguments))
print(raw_arguments.info())

raw_arguments['topic_assignment']=raw_arguments['topic_assignment'].astype(int)
tm_output['text_processed_cluster'] = tm_output['text_processed_cluster'].apply(literal_eval)

pegasus_model_files = Model.get_model_path(model_name='kp_hyper', version=1, _workspace=ws) # keypoint_hyperdrive

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
sum_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_files)
sum_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_files).to(device)

def get_representative_arguments(argus):

    concatenated_text = ''
    concatenated_length = 0
    input_args = ''

    arguments_df = pd.DataFrame(argus, columns=['text_processed_cluster'])
    arguments_df['arg_enc'] = arguments_df['text_processed_cluster'].apply(lambda x: model.encode(x, convert_to_numpy=True))
    argument_similarity_dict = {}

    for argument_identifier in arguments_df.index:
        argument_similarity = []
        argument_enc = arguments_df[arguments_df.index == argument_identifier].arg_enc.iloc[0]
        for index, argument in arguments_df.iterrows():
            if argument_identifier != index:
                argument_enc_2 = argument['arg_enc'] 
                argument_similarity.append(cosine_similarity([argument_enc], [argument_enc_2])[0][0])
        argument_similarity_dict[argument_identifier] = mean(argument_similarity)
    
    argument_similarity_dict_sorted = dict(sorted(argument_similarity_dict.items(), key=lambda x:x[1], reverse=True))
    
    for key, value in argument_similarity_dict_sorted.items():
        text = arguments_df[arguments_df.index == key].text_processed_cluster.iloc[0]
        text_length = len(text.split())
        
        if concatenated_length + text_length <= 512:
            concatenated_text += text + " "
            input_args += text + '. '
            concatenated_length += text_length
        else:
            break

    return input_args

def summarize_keypoints(arguments):
    
    input_raw = get_representative_arguments(arguments)

    input_ids = sum_tokenizer.encode(str(input_raw), return_tensors='pt', truncation=True, max_length=512).to(device)
    output = sum_model.generate(input_ids,
                            min_length=min_length,
                            max_length=25,
                            no_repeat_ngram_size=1,
                            remove_invalid_values=True,
                            do_sample=True,
                            top_k=top_k, 
                            top_p=top_p,
                            ) 
    key_point = sum_tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    key_point_capitalized = key_point.capitalize()

    return key_point_capitalized

def create_keypoints(raw_arguments, tm_output):

    tm_output['key_point'] = tm_output['text_processed_cluster'].apply(lambda x: summarize_keypoints(x))

    tm_output = tm_output.rename(columns={'topic_assignment':'topic_size', 'text_processed_cluster':'clustered_args'})
    tm_output = tm_output[['topic_size', 'key_point', 'clustered_args']]

    tm_output['topic'] = tm_output.index
    combined_df = pd.merge(raw_arguments, tm_output[['key_point','topic']], left_on='topic_assignment', right_on='topic')

    return combined_df

combined_df = create_keypoints(raw_arguments, tm_output)
new_run.log('generated_keypoints', combined_df.loc[:, ['key_point', 'topic']].drop_duplicates().values)

def calculate_davies_bouldin(df):
    
    df = df.sort_values(by=['topic'])
    intra_distances_dict = {}
    unique_keypoints = df.loc[:, ['key_point', 'topic']].drop_duplicates().values
    for key_point in unique_keypoints:
        intra_distances = []
        
        key_point_enc = model.encode(key_point[0], convert_to_numpy=True)
        df_cluster = df[(df.key_point == key_point[0]) & (df.topic == key_point[1])]
        topic_index = df[(df.key_point == key_point[0]) & (df.topic == key_point[1])].topic.iloc[0]
        for index, row in df_cluster.iterrows():
            argument_enc = model.encode(row['text_processed_cluster'], convert_to_numpy=True)
            intra_distances.append(cosine_distances([argument_enc], [key_point_enc])[0][0])
        mean_intra_distances = mean(intra_distances)

        intra_distances_dict[topic_index] = mean_intra_distances

    intra_distances_df = pd.DataFrame(intra_distances_dict.items(), columns=['topic', 'intra_score'])

    df = df.sort_values(by=['topic'])
    inter_distances_dict = {}

    for key_point in unique_keypoints:
        inter_keypoint_dist_dict = {}
        topic_index = df[(df.key_point == key_point[0]) & (df.topic == key_point[1])].topic.iloc[0]
        key_point_inter = model.encode(key_point[0], convert_to_numpy=True)
        for kp in unique_keypoints:
            topic_index_2 = df[(df.key_point == kp[0]) & (df.topic == kp[1])].topic.iloc[0]
            key_point_inter_2 = model.encode(kp[0], convert_to_numpy=True)
            inter_dist = cosine_distances([key_point_inter], [key_point_inter_2])[0][0]
            inter_keypoint_dist_dict[topic_index_2] = inter_dist
        inter_distances_dict[topic_index] = inter_keypoint_dist_dict

    inter_distances_df = pd.DataFrame.from_dict(inter_distances_dict)

    cluster_similarity_dict = {}
    for index, cluster in intra_distances_df.iterrows():
        cluster_score = cluster['intra_score']
        between_cluster_similarity = {}
        for index_2, cluster_2 in intra_distances_df.iterrows():
            if index != index_2:
                cluster_similarity = (cluster_score + cluster_2['intra_score'])/inter_distances_df.loc[index, index_2]
                between_cluster_similarity[index_2] = cluster_similarity
        cluster_similarity_dict[index] = between_cluster_similarity

    cluster_similarity_df = pd.DataFrame.from_dict(cluster_similarity_dict)

    cluster_similarity_df = cluster_similarity_df.sort_index()
    davies_bouldin_index = mean(cluster_similarity_df.max(axis=1))

    return davies_bouldin_index

with open(os.path.join(output_keypoint_data, 'output_keypoint_data.csv'), 'w') as f:
    f.write(combined_df.to_csv())
    f.close()

db_score = calculate_davies_bouldin(combined_df)
new_run.log('DB_Index_modified', db_score)

new_run.complete()