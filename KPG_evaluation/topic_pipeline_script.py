# --------------------------------------------------------------
# Training Script for the Hyperdrive Job
# --------------------------------------------------------------

from azureml.core import Run

# Get the run context
new_run = Run.get_context()

# Get the workspace from the run
ws = new_run.experiment.workspace

#Import Enviroment Libraries
import pandas as pd
import numpy as np
from bertopic import BERTopic

from statistics import mean
import torch
import matplotlib.pyplot as plt
import hdbscan
import os
import io
from umap import UMAP
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

#Get parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_neighbors', type=int)
parser.add_argument('--n_components', type=int)
parser.add_argument('--min_samples', type=float)
parser.add_argument('--selected_topic', type=str)
parser.add_argument('--output_topic_data', type=str)
parser.add_argument('--output_arguments_data', type=str)
args = parser.parse_args()

n_neighbors = int(args.n_neighbors)
n_components = int(args.n_components)
output_topic_data = str(args.output_topic_data)
output_arguments_data = str(args.output_arguments_data)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

path = os.path.join(args.selected_topic, 'selected_topic.csv')
arguments = pd.read_csv(path)
new_run.log('nr_tweets', len(arguments))
print(arguments.info())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def topic_modeling(df):
    
    cluster_size = int(len(df)/20)
    if cluster_size < 3:
        cluster_size = 3

    print(cluster_size)

    min_samples = int(args.min_samples*cluster_size)

    if min_samples < 2:
        min_samples = 2

    print(min_samples)

    text_list = df['argument'].tolist()
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = embedding_model.encode(text_list, show_progress_bar=True)
    umap_model = UMAP(random_state=42, n_neighbors=n_neighbors, n_components=n_components, min_dist=0.00, metric='cosine')
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=cluster_size,
                                    metric='euclidean',
                                    gen_min_span_tree=True,
                                    cluster_selection_method='leaf',
                                    min_samples=min_samples,
                                    prediction_data=True,
                                    ) 
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english')

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=None,
        verbose=True,
        calculate_probabilities=True
    )

    topics, probabilities = topic_model.fit_transform(documents=text_list, embeddings=embeddings)
    
    topic_array = np.array(topics)
    embeddings_double_t = embeddings.astype('double')
    dbcv = hdbscan.validity_index(embeddings_double_t, labels=topic_array, metric='cosine')

    df['topic_assignment'] = topics

    new_run.log('nr_topics', len(df['topic_assignment'].unique()))
    new_run.log('nr_outliers', len(df[df.topic_assignment == -1]))
    new_run.log('DBCV', dbcv)

    topic_labels_fig = topic_model.generate_topic_labels(nr_words=3,
                                                  topic_prefix=False,
                                                  word_length=50,
                                                  separator=', ')
    topic_model.set_topic_labels(topic_labels_fig)
    
    figure = topic_model.visualize_documents(docs=text_list,
        embeddings=embeddings,
        hide_annotations=False, 
        custom_labels=True)
        
    os.makedirs(os.path.dirname(output_topic_data), exist_ok=True)
    figure.write_image(os.path.join(output_topic_data, 'cluster_plot.svg'))
    figure.write_html(os.path.join(output_topic_data, 'cluster_plot.html'))

    bytes_figure = figure.to_image(format='png', scale=4)
    buf = io.BytesIO(bytes_figure)
    img = plt.imread(fname=buf, format='png')
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis('off')
    new_run.log_image('Reduced_Clusters', plot=fig)
    plt.close()

    return df, embeddings

def intra_silhouette_cluster_distance(distance_matrix, labels, index):
    
    same_label = [list_index for (list_index, item) in enumerate(labels) if item == labels[index] if list_index != index]
    
    return mean(distance_matrix[index][same_label])

def inter_silhouette_cluster_distance(distance_matrix, labels, index): 
    topic_list = list(set(labels))
    cluster_distance = 1
    gen = (topic for topic in topic_list if topic != -1 if topic != labels[index])
    for topic in gen:
        topic_cluster = [list_index for (list_index, item) in enumerate(labels) if item == topic]
        inter_distance = mean(distance_matrix[index][topic_cluster])
        if inter_distance < cluster_distance:
            cluster_distance = inter_distance
    
    return cluster_distance

def silhouette_score(distance_matrix, labels):
    score = {}

    for index in range(len(distance_matrix)):
        if labels[index] == -1:
            score[index] = {'topic' : labels[index], 'score' : 0}
        
        else:
            a_i = intra_silhouette_cluster_distance(distance_matrix, labels, index)
            b_i = inter_silhouette_cluster_distance(distance_matrix, labels, index)
            s_i = (b_i - a_i)/np.max([b_i, a_i])
            score[index] = {'topic' : labels[index], 'score' : s_i}

    return score

combined_df, embeddings = topic_modeling(arguments)
labels = combined_df['topic_assignment'].to_list()
distance_matrix = cosine_distances(np.array(embeddings)[:, :])
result = silhouette_score(distance_matrix, labels)
result_df = pd.DataFrame.from_dict(result, orient='index')
sil_score = mean(result_df['score'])
new_run.log('Silhouette_Score', sil_score)

eval_df = combined_df
combined_df = combined_df[combined_df.topic_assignment != -1]

os.makedirs(os.path.dirname(output_arguments_data), exist_ok=True)
with open(os.path.join(output_arguments_data, 'arguments.csv'), 'w') as f:
    f.write(eval_df.to_csv(index=False, lineterminator='\n',sep=','))
    f.close()

documents_per_topic = combined_df.groupby(['topic_assignment'])['argument'].apply(list)
sizes = combined_df['topic_assignment'].value_counts()
topic_df = pd.concat([documents_per_topic, sizes], axis=1)

new_run.log('sizes', sizes)
topic_df['topic'] = topic_df.index
with open(os.path.join(output_topic_data, 'output_topic_data.csv'), 'w') as f:
    f.write(topic_df.to_csv(index=False, lineterminator='\n',sep=','))
    f.close()

new_run.complete()






