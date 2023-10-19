# WalkingDetection

This an example code to show the implementation of CNN-based and ResNet-based walking detection model.
Currently, the testing data is fixed as ADL data while training data could be lab or ADL data.


## Requirement
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

## Check the path of the data in main.py
```python
config = {...
'load_path':'/Users/kai-chunliu/Documents/UMass/AHHA/Shirley Ryan AbilityLab/Preprocessing/agOnly/', #lab-based data
'load_path_ADL': '/Users/kai-chunliu/Documents/UMass/AHHA/Shirley Ryan AbilityLab/Labeling/ADL/', # ADL data
...}
```

### Path of Data Folder
* **lab data**  
```shell
/Users/kai-chunliu/Documents/UMass/AHHA/Shirley Ryan AbilityLab/Preprocessing/agOnly/
```

* **ADL data**  
```shell
/Users/kai-chunliu/Documents/UMass/AHHA/Shirley Ryan AbilityLab/Labeling/ADL/csvVersion(ShiftOnly)_ADL
```
* **ADL Label**  
```shell
/Users/kai-chunliu/Documents/UMass/AHHA/Shirley Ryan AbilityLab/Labeling/ADL/Labeling_final
```
* **Pre-trained model**  
```shell
'flip_net_path': '/Users/kai-chunliu/Documents/code/ssl-wearables-main/model_check_point/mtl_best.mdl', 
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
OverallResultsWindowWise.csv
```

