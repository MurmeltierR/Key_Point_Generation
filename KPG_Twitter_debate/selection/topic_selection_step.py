from azureml.core import Run, Model

# Get the run context
new_run = Run.get_context()

# Get the workspace from the run
ws = new_run.experiment.workspace

import os
import torch
from ast import literal_eval
from bertopic import BERTopic

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--topic_query', type=str)
parser.add_argument('--tweet_selection', type=str)
parser.add_argument('--party_selection', type=str)
args = parser.parse_args()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

topic_query = str(args.topic_query)
selection_output = str(args.tweet_selection)
party = literal_eval(args.party_selection)

os.makedirs(os.path.dirname(selection_output), exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(topic_query)
print(device)

tweets = new_run.input_datasets['tweet_corpus'].to_pandas_dataframe()

print(len(tweets))
print(tweets.info())

bertopic_model_files = Model.get_model_path(model_name='topic_model_2percent', version=1, _workspace=ws)

topic_model = BERTopic.load(bertopic_model_files)

topics, similarity = topic_model.find_topics(topic_query, top_n=1)

selected_tweets = tweets[(tweets['probabilities']==topics[0])]
selected_tweets = selected_tweets[(selected_tweets['Party'].isin(party))]

print(topic_model.get_topic_info(topics[0]))
print(selected_tweets.info())
with open(os.path.join(selection_output, 'selected_tweets.csv'), 'w') as f:
    f.write(selected_tweets.to_csv(index=False))
    f.close()

new_run.complete()