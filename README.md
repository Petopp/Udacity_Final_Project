# Final Project
In this final project, all the possibilities and knowledge acquired in the Nanodegree are still being used. In order to evaluate the heart failure prediction data in AutoML as well as HyperDrive and to determine the best model as well as to compare the different technologies.
Here you can see a representation of what is being implemented in this project with the help of MS Azure and its technological possibilities.

<kbd>![image](https://user-images.githubusercontent.com/41972011/117867884-35f63380-b299-11eb-87c9-b03fc5d561ef.png)</kbd>

## Dataset

### Source of the data set 
For this project we are using files from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). In this dataset are data on cardiovascular diseases (CVDs). Which are the number one cause of death worldwide, claiming the lives of an estimated 17 million people each year. This represents approximately 31% of all deaths worldwide.
Heart failure is one of the common events caused by CVDs. This dataset contains 12 characteristics that can be used to predict mortality from heart failure.
In order for people with cardiovascular disease or at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or established disease) to receive early detection and treatment, these datasets attempt to improve prediction.

### Content of the data set
The dataset contains 12 features that can be used to predict mortality from heart failure:
- age: Age of the patient
- amaemia: Decrease of red blood cells or hemoglobin
- creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L)
- diabetes: If the patient has diabetes
- ejection_fraction: Percentage of blood leaving the heart at each contraction
- high_blood_pressure: If the patient has hypertension
- platelets: Platelets in the blood (kiloplatelets/mL)
- serum_creatinine: Level of serum creatinine in the blood (mg/dL)
- serum_sodium: Level of serum sodium in the blood (mEq/L)
- sex: Woman or man (Gender at birth)
- smoking: patient smokes or not
- time: Follow-up period (days)

### Target
Our goal is to develop a machine learning algorithm that can detect whether a person is likely to die from heart failure. This will help in diagnosis and early prevention. For this, the above mentioned 12 features in the dataset are used to develop a model for the detection.

### Attention!
This is an experiment that was developed in the course of a test for the Udacity learning platform. Do not use this model in a medical environment or for acute indications. Always consult your doctor for medical questions or the medical emergency service in acute cases!

## Hyperparameter Tuning
The model used here is a logistic regression model that is trained with a custom script train.py. 
The dataset is fetched from [HERE](https://raw.githubusercontent.com/Petopp/Udacity_Final_Project/main/heart_failure_clinical_records_dataset.csv) as a dataset. The hyperparameters chosen for the scikit-learn model are regularisation strength (C) and maximum iterations (max_iter). The trained model is evaluated against 25% data selected from the original dataset. The remaining data is used to train the model.

Hyperparameter tuning with HyperDrive requires several steps: 
- define the parameter search space
- define a sampling method
- selecting a primary metric for optimisation 
- selecting an early stop policy.

The parameter sampling method used for this project is Random Sampling. It randomly selects the best hyperparameters for the model so that the entire search space does not need to be searched. The Random Sampling method saves time and is much faster than Grid Sampling and Bayesian Sampling, which are only recommended if you have a budget to explore the entire search space.

The early stop policy used in this project is the Bandit policy, which is based on a slack factor (in this case 0.1) and a scoring interval (in this case 1). This policy stops runs where the primary metric is not within the specified slip factor, compared to the run with the best performance. This will save time and resources as runs that may not produce good results would be terminated early.

### Paramerters

in the Jupyter Notebook
```python
# Create the different params that will be needed during training
param_sampling = RandomParameterSampling(
    {
        "--C": uniform(0.001,100),
        "--max_iter": choice(50, 90, 125, 170)
    }
)
```

and in the train.py
```python
# Path to dataset 
    path_to_data="https://raw.githubusercontent.com/Petopp/Udacity_Final_Project/main/heart_failure_clinical_records_dataset.csv"
```

```python
# Split data into train and test sets.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25)
```



### Results

Details from Jupyter Notebook
<kbd>![image](https://user-images.githubusercontent.com/41972011/117972722-c2e9cd00-b32b-11eb-848b-c585a6db04ab.png)</kbd>

Details from Azure Experiments
<kbd>![image](https://user-images.githubusercontent.com/41972011/117972864-f3ca0200-b32b-11eb-98e1-863ae7c2239a.png)</kbd>

Experiment completed
<kbd>![image](https://user-images.githubusercontent.com/41972011/117973039-2aa01800-b32c-11eb-85d1-c449ec8bcdd2.png)</kbd>

The result in detail 
```python
['--C', '97.2861169940756', '--max_iter', '125']
['azureml-logs/55_azureml-execution-tvmps_b3d8a370fdab6acc496b1fa398220948b9ae8dd605d8df21bbd0582f1cc744bc_d.txt', 'azureml-logs/65_job_prep-tvmps_b3d8a370fdab6acc496b1fa398220948b9ae8dd605d8df21bbd0582f1cc744bc_d.txt', 'azureml-logs/70_driver_log.txt', 'azureml-logs/75_job_post-tvmps_b3d8a370fdab6acc496b1fa398220948b9ae8dd605d8df21bbd0582f1cc744bc_d.txt', 'azureml-logs/process_info.json', 'azureml-logs/process_status.json', 'logs/azureml/106_azureml.log', 'logs/azureml/job_prep_azureml.log', 'logs/azureml/job_release_azureml.log', 'outputs/model.joblib']
Best Run Accuracy: 0.84

```



## Automated ML
The AutomatedML run was created using an instance of AutoML Config. The AutoML Config class is a way to use the AutoML SDK to automate machine learning. The following parameters were used for the AutoML run.

| Parameter        | Value          | Description  |
| :----- |:-----:| :---------------|
| task     | 'classification' | Classification is selected since we are performing binary classification, i.e whether or not a death event occurs |
| debug.log      | 'automl_errors.log"  | The debug information is written to this file instead of the automl.log file |
| training_data | train_data    | train_data is passed that which contains the data to be used for training |
| label_column_name | 'DEATH_EVENT' | Since the DEATH_EVENT column contains what we need to predict, it is passed |
| compute_target | computcluster    | The compute target on which we want this AutoML experiment to run is specified |
| experiment_timeout_minutes | 30  | Specifies the time that all iterations combined can take. Due to the lack of resources this is selected as 30 |
| primary_metric | 'accuracy'    | This is the metric that AutoML will optimize for model_selection. Accuracy is selected as it is well suited to problems involving binary classification. |
| enable_early_stopping | True | Early Stopping is enabled to terminate a run in case the score is not improving in short term. This allows AutoML to explore more better models in less time |
| featurization | 'auto'   | Featurization is set to auto so that the featurization step is done automatically |
| n_cross_validations | 4  | This is specified so that there are 4 different trainings and each training uses 1/4 of data for validation |
| verbosity | logging.INFO   | This specifies the verbosity level for writing to the log file |
| enable_onnx_compatible_models | True   | Export to ONNX format from Azure ML is enabled for later export, more about ONNX can be found [HERE](https://onnx.ai/about.html) |

The paramerters in code:
```python
# automl settings 
automl_settings = {
    "enable_early_stopping" : True,
    "experiment_timeout_minutes": 30,
    "n_cross_validations": 4,
    "featurization": "auto",
    "primary_metric": "accuracy",
    "verbosity": logging.INFO
}

# automl config (with onnx compatible modus)
automl_config = AutoMLConfig(
    task="classification",
    debug_log = "automl_errors.log",
    training_data=train_data,
    label_column_name="DEATH_EVENT",
    compute_target=compute_cluster,
    enable_onnx_compatible_models=True,
    **automl_settings
)
```


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Save the best model in ONNX

```python
from azureml.automl.runtime.onnx_convert import OnnxConverter
automl_best_run_onnx, automl_fitted_model_onnx = remote_run.get_output(return_onnx_model=True)
OnnxConverter.save_onnx_model(automl_fitted_model_onnx, './outputs/AutoML.onnx' )
```
So that the calculations can be understood by other systems. The best result is stored in the onnx format. 
With the ONNX, AI developers can exchange models between different tools and choose the best combination of these tools for them.


## Screen Recording
See on [youtube](https://youtu.be/w7i0fTQ_AeU)

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
