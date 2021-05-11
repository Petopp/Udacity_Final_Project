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

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

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
