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

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

#### Save the best model in ONNX
```python
from azureml.automl.runtime.onnx_convert import OnnxConverter
automl_best_run_onnx, automl_fitted_model_onnx = remote_run.get_output(return_onnx_model=True)
OnnxConverter.save_onnx_model(automl_fitted_model_onnx, './outputs/AutoML.onnx' )
```
So that the calculations can be understood by other systems. The best result is stored in the onnx format. 
With the ONNX, AI developers can exchange models between different tools and choose the best combination of these tools for them.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
