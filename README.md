# WalkingDetection

This is an example code to show the implementation of CNN-based and ResNet-based walking detection models.
Currently, the testing data is fixed as ADL data while training data could be lab or ADL data.


## Requirement
* To use the early-stopping package, please visit https://github.com/Bjarten/early-stopping-pytorch/tree/master.
* Python >=3.8
* Sklearn
* Pandas =2.0.2
* Numpy = 1.23
* Matplotlib = 3.7
* Seaborn = 0.11.2
* Scipy = 1.10.1
* os
* copy
* collections

## Check the path of the data & pre-trained model in main.py
```python
config = {...
'load_path':'./Lab/', #lab-based data
'load_path_ADL': './ADL/', # ADL data
'flip_net_path': './model_check_point/mtl_best.mdl', # pre-trained model
...}
```

### Path of Data Folder

* **lab data**  
```shell
path = '/Lab'
```

* **ADL data**  
```shell
path = '/ADL'
```

* **ADL Label**  
```shell
path = main_path + '/ADL/Labeling_final'
```
* **Pre-trained model**  
```shell
path = '/model_check_point/mtl_best.mdl', 
```
## Directory Structure
```shell
-main_binary.py
	|_ CrossValidation_SpecifyActivityTrain_ADLTest_simpleversion.LSOCV
		|_ data_preprocessing.py
			|_create_segments_and_labels
				|_ SlidindWindow_and_lables
			|_create_segments_and_labels_ADL
				|_ SlidindWindow_and_lables_ADL
		|_ models.py
			|_ ResNet
			|_ CNNc3f1
		|_ utilities.py
			|_ plot2
			|_ PerformanceSave
			|_ ConfusionSave
		|_ pytorchtools.py (pip install -r requirements.txt, https://github.com/Bjarten/early-stopping-pytorch/tree/master)
		|_ train
	|_ validation
	|_ test
```
## Run and test
RUN “python3 main_binary.py”

Test
All testing data are ADL data

Train on lab data
```python
config = {…,
'SpecifiedActivity_train': True,
'ADL_train':False, … }
```

Train on ADL data
```python
config = {…,
'SpecifiedActivity_train': False,
'ADL_train': True, … }
```

Train on lab & ADL data
```python
config = {…,
'SpecifiedActivity_train': True,
'ADL_train': True, … }
```

Train from scratch
```python
config = {…,
'load_weights': False,
'freeze_weight': False, … }
```

Fine-tuning
```python
config = {…,
'load_weights': True,
'freeze_weight': False, … }
```

Linear probing 
```python
config = {…,
'load_weights': True,
'freeze_weight': True, … }
```

## Check Results
Please visit FOLDER:
```shell
DATE_Model_outputs
```
FIND the file
```shell
WindowsResults_WS.._... .csv
```

