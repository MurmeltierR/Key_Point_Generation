# ------------------------------------------------------------
# Run a Hyperdrive Experiment in an Azureml environment
# ------------------------------------------------------------

# -------------------------------------------------
# Import the Azure ML classes
# -------------------------------------------------

from azureml.core import Workspace, Experiment, Dataset, Datastore

# -------------------------------------------------
# Access the workspace using config.json
# -------------------------------------------------

print('Accessing the workspace from job....')
ws = Workspace.from_config('./config')

# -------------------------------------------------
# Initialize Experiment 
# -------------------------------------------------

new_experiment = Experiment(workspace=ws, name='keypoint_evaluation_pipeline')

# -------------------------------------------------
# Get the input dataset
# -------------------------------------------------

print('Accessing the datasets...')
datastore = Datastore(ws, name='kp_datastore')
input_ds = Dataset.get_by_name(ws, name='ArgKP_Dataset', version=1)

# -------------------------------------------------
# Create custom environment
# -------------------------------------------------

from azureml.core import Environment

print('Activating environment...')

myenv = Environment.get(ws, 'key_gen', version=13)

# --------------------------------------------------------------------
# Create the compute Cluster 
# --------------------------------------------------------------------

from azureml.core.compute import AmlCompute

cluster_name = 'hyperdrive-topic-compute'

print('Accessing the compute cluster...')

if cluster_name not in ws.compute_targets:
    print('Creating the compute cluster with name: ', cluster_name)
    compute_config = AmlCompute.provisioning_configuration(
                                     vm_size='STANDARD_NC6S_V2',
                                     max_nodes=4, vm_priority='lowpriority')

    topic_cluster = AmlCompute.create(ws, cluster_name, compute_config)
    topic_cluster.wait_for_completion()
else:
    topic_cluster = ws.compute_targets[cluster_name]
    print(cluster_name, ", compute cluster found. Using it...")

# ---------------------------------------------------------------------
# Define Input/Output Data
# ---------------------------------------------------------------------

from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline, PipelineData

selected_topic = PipelineData(name='selected_topic', 
    datastore=datastore,
    output_mode='upload',
    output_path_on_compute='/selected_topic/',
    pipeline_output_name='selected_topic'
    )

topic_model_output = OutputFileDatasetConfig(name='topic_model_output', destination=(datastore, 'topic_output/{run-id}/')).as_upload()
arguments = OutputFileDatasetConfig(name='arguments', destination=(datastore, 'arguments/{run-id}/')).as_upload()
combined_df = OutputFileDatasetConfig(name='combined_df', destination=(datastore, 'result_df/{run-id}/')).as_upload()

# ---------------------------------------------------------------------
# Create a script configuration for custom environment of myenv
# ---------------------------------------------------------------------

from azureml.core import ScriptRunConfig, RunConfiguration
from azureml.core.runconfig import PyTorchConfiguration

distr_config = PyTorchConfiguration(process_count=1, node_count=1)

script_config_selection = RunConfiguration() 
script_config_selection.environment = myenv

script_config_topic = ScriptRunConfig(source_directory='.',
                                script='topic_pipeline_script.py',
                                environment=myenv,
                                compute_target=topic_cluster,
                                distributed_job_config=distr_config
                                )

script_config_keypoint = ScriptRunConfig(source_directory='./keypoint_hyperdrive/',
                                script='keypoint_pipeline_script.py',
                                environment=myenv,
                                compute_target=topic_cluster,
                                distributed_job_config=distr_config
                                )

# ---------------------------------------------------------------------
# Create Hyper drive parameters
# ---------------------------------------------------------------------

from azureml.train.hyperdrive import BayesianParameterSampling
from azureml.train.hyperdrive import uniform, choice

hyper_params_topic = BayesianParameterSampling( {
        'n_neighbors': choice(range(3, 21)),
        'n_components': choice(range(3, 21)),
        'min_samples': uniform(min_value=0.05, max_value=1.0)
    }
)

hyper_params_keypoint = BayesianParameterSampling( {
        #'min_length': choice(range(8, 20)),
        'top_k': choice(range(20, 100)),
        'top_p': uniform(min_value=0.00, max_value=1),
    }
)

# ---------------------------------------------------------------------
# Configure the Hyperdrive class
# ---------------------------------------------------------------------

from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal

hyper_config_topic = HyperDriveConfig(run_config=script_config_topic,
                                hyperparameter_sampling=hyper_params_topic,
                                policy=None,
                                primary_metric_name='DBCV',
                                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                max_total_runs=40,
                                max_concurrent_runs=4)

hyper_config_keypoint = HyperDriveConfig(run_config=script_config_keypoint,
                                hyperparameter_sampling=hyper_params_keypoint,
                                policy=None,
                                primary_metric_name='DB_Index_modified',
                                primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                max_total_runs=20,
                                max_concurrent_runs=4)

# ---------------------------------------------------------------------
# Create the experiment and run
# ---------------------------------------------------------------------

from azureml.pipeline.steps import HyperDriveStep, PythonScriptStep

# ['USA', 'Social media', 'vaccinations', 'celibacy', 'Assisted suicide', 'Homeschooling']

topic_query = 'celibacy'

# [1, -1]

stance_party = -1

topic_selection_step = PythonScriptStep(
    source_directory='./selection/',
    name='topic_selection',
    script_name='topic_selection_step.py',
    compute_target=topic_cluster,
    runconfig=script_config_selection,
    arguments=['--topic_query', topic_query,
    '--topic_selection', selected_topic,
    '--stance_selection', stance_party,
    ],
    inputs=[input_ds.as_named_input(name='argkp_corpus')],
    outputs=[selected_topic],
    allow_reuse=False
    )

hd_topic_step = HyperDriveStep(
    name='topic_optimizer',
    hyperdrive_config=hyper_config_topic,
    estimator_entry_script_arguments=['--output_topic_data', topic_model_output,
    '--selected_topic', selected_topic,
    '--output_arguments_data', arguments],
    inputs=[selected_topic],
    outputs=[topic_model_output, arguments]
    )

hd_topic_step.run_after(topic_selection_step)

hd_keypoint_step = HyperDriveStep(
    name='keypoint_optimizer',
    hyperdrive_config=hyper_config_keypoint,
    estimator_entry_script_arguments=['--output_keypoint_data', combined_df],
    inputs=[topic_model_output.read_delimited_files().as_input(name='topic_model_input'),
    arguments.read_delimited_files().as_input(name='raw_arguments')
    ],
    outputs=[combined_df]
    )

hd_keypoint_step.run_after(hd_topic_step)

# ---------------------------------------------------------------------
# Create the experiment and run
# ---------------------------------------------------------------------

pipeline = Pipeline(workspace=ws, steps=[topic_selection_step, hd_topic_step, hd_keypoint_step])
pipeline_run = new_experiment.submit(pipeline)

# ------------------------------------------------------------
# Best hyperdrive run with best combination of hyperparameter
# ------------------------------------------------------------

pipeline_run.wait_for_completion(show_output=True)













